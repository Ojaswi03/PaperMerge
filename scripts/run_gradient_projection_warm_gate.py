#!/usr/bin/env python3
"""Fresh three-round CE/projection validation diagnostic, never a queue job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_gradient_projection_gate import save_json, validation_split, verification_path

NAME = 'clean_training_validation_warm_gate_3r_v1'
RESULT = ROOT/'newResults/nonIID/gradient_projection'/NAME
PLOTS = ROOT/'newPlots/nonIID/gradient_projection'/NAME
ROUNDS = 3
SOURCE_FILES = ('scripts/run_gradient_projection_warm_gate.py',
    'scripts/run_gradient_projection_gate.py', 'basil_core/gradient_projection_retention.py',
    'basil_core/research_protocol.py', 'basil_core/research_audit.py',
    'basil_core/iid_study.py', 'basil_core/models.py', 'basil_core/trainer.py')


def config_for(device):
    from basil_core.gradient_projection_retention import REVISION, ENERGY, REPRESENTATION_IMAGES, MAX_PATCHES
    return dict(experimentProtocol='sequential_basil_one_class_v1', methodRevision=REVISION,
        purpose='three_round_training_validation_retention_diagnostic', researchValid=False,
        seed=2025, nNodes=10, nRounds=ROUNDS, localEpochs=5, batchSize=512,
        trainingSamplesPerNode=4500, validationSamplesPerClass=500,
        modelArchitecture='basil_paper_cnn', learningRate=.05,
        learningRateSchedule='basil_round_decay', optimizer='sgd', momentum=0.,
        optimizerStateMode='reset_each_activation', gradientClipNorm=None,
        weightDecayCoefficient=0., aggregationMode='strict_sequential_handoff',
        basilMemorySize=5, assumedByzantineCount=4, attackerCount=0,
        resolvedAttackerIds=[], useChannelNoise=False, ebmMode='none',
        snapshotAnchorMu=0., liveTrackingEnabled=False, snapshotSelection=True,
        snapshotSelectionRule='lowest_receiver_local_loss', snapshotPlausibilityGuard='none',
        executionDevice=device, projectionEnergy=ENERGY,
        representationImages=REPRESENTATION_IMAGES, maximumPatchesPerLayer=MAX_PATCHES,
        testUsage='not evaluated or used; all evaluation is held-out training validation',
        screenPolicy='Warmed R1-R2: >=80% activations own-class >5%; >=50% activations '
            'retain >=2 other classes >5%; mean validation exceeds paired CE by >=5pp. '
            'Descriptive screen only; no automatic run promotion or tuning.',
        checkpointScope='snapshot memories only, not resumable research state')


def learning_screen(controls):
    """Fixed before execution; omit cold-start R0, never tune training from it."""
    import numpy as np
    warm = {key: [r for r in rows if r['round'] in (1, 2)] for key, rows in controls.items()}
    if any(len(rows) != 20 for rows in warm.values()):
        raise ValueError('Screen requires 20 warmed activations per control.')
    projection = warm['projection']
    own_fraction = sum(r['epochs'][-1]['perClassValidationAccuracy'][r['nodeId']] > .05
                       for r in projection)/20
    retained_fraction = sum(sum(a > .05 for c, a in enumerate(r['epochs'][-1]['perClassValidationAccuracy'])
                               if c != r['nodeId']) >= 2 for r in projection)/20
    improvement = float(np.mean([r['epochs'][-1]['globalValidationAccuracy'] for r in projection]) -
                        np.mean([r['epochs'][-1]['globalValidationAccuracy'] for r in warm['ce']]))
    checks = dict(ownClassLearnedInAtLeast80Percent=own_fraction >= .8,
        twoOtherClassesRetainedInAtLeast50Percent=retained_fraction >= .5,
        meanValidationImprovementAtLeast5pp=improvement >= .05)
    return dict(passed=all(checks.values()), checks=checks, evaluatedRounds=[1, 2],
        ownClassLearnedFraction=own_fraction, twoOtherClassesRetainedFraction=retained_fraction,
        meanValidationImprovement=improvement, testSetUsed=False, automaticPromotion=False)


def expected_candidates(round_id, node):
    return [(sender, round_id if sender < node else round_id-1)
            for sender in ((node-distance) % 10 for distance in range(5, 0, -1))]


def execute(tf, device):
    import numpy as np
    from basil_core.data.cifar import loadCifar10, _datasetArrays
    from basil_core.gradient_projection_retention import (GradientProjectionWorker,
        save_memory, load_memory, state_hash, validate_state)
    from basil_core.research_protocol import (build_model, initialize_memories, select_snapshot,
        Snapshot, broadcast, keyed_rng, params_hash, basil_round_learning_rate)
    from basil_core.research_audit import ProtocolAudit, array_hash
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    from basil_core.iid_runtime import runtime_metadata

    verified = json.loads(verification_path(device).read_text())
    if verified.get('status') != 'passed' or verified['objectiveSha256'] != hashlib.sha256(
            (ROOT/'basil_core/gradient_projection_retention.py').read_bytes()).hexdigest():
        raise ValueError('Current projection implementation requires passed device verification.')
    if RESULT.exists() or PLOTS.exists():
        raise ValueError('Fresh diagnostic refuses existing result/plot directories; no guessed resume.')
    RESULT.mkdir(parents=True)
    config = config_for(device)
    save_json(RESULT/'config.json', config)
    manifest = dict(status='initializing', config=config, **runtime_metadata(tf, config),
        wandbEnabled=False, multiprocessingUsed=False,
        implementationSha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES},
        completedActivations=0, completedRoundsByControl={'ce': 0, 'projection': 0},
        thirtyRoundRunStarted=False, productionRunStarted=False)
    stop = False
    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    guard = RuntimeMemoryGuard(tf, device)
    started = time.perf_counter()
    context, controls = {}, {}
    try:
        check_host_memory()
        train, _unused_test = loadCifar10(cacheDir=ROOT/'experiments/cache/cifar10')
        images, labels = _datasetArrays(train)
        training, validation = validation_split(labels, config['seed'])
        partition_hash = params_hash([np.asarray(a, np.float32) for a in training+validation])
        indices = np.concatenate(validation)
        val_x, val_y = images[indices], labels[indices]
        save_json(RESULT/'partition_audit.json', dict(trainingCounts=[len(a) for a in training],
            validationCounts=[len(a) for a in validation], partitionHash=partition_hash,
            trainingValidationDisjoint=True, assignedTrainingUniverse=50000, testEvaluated=False,
            validationImageHash=array_hash(val_x), validationLabelHash=array_hash(val_y)))
        with tf.device('/CPU:0'):
            initial = [v.numpy() for v in build_model('basil_paper_cnn', config['seed']).trainable_variables]
        manifest.update(status='running', partitionHash=partition_hash,
            initialModelHash=params_hash(initial), parameterCount=sum(w.size for w in initial))
        save_json(RESULT/'run.json', manifest)

        class GateAudit(ProtocolAudit):
            def gradients(self, gradients, values):
                super().gradients(gradients, values)
                self.current.update(zip(GradientProjectionWorker.extra_metric_names, map(float, values[6:])))
            def failure(self, worker, error):
                self.write('failure.json', {**getattr(self, 'current', {}), 'error': str(error),
                    'gradientRule': config['methodRevision'], 'graphReplaySkipped': True})
                if self.pending:
                    np.savez_compressed(self.directory/'failure_batch.npz', x=self.pending['x'],
                        y=self.pending['y'], key=self.pending['key'],
                        **{f'weight_{i}': w for i, w in enumerate(worker.export())},
                        **{f'basis_{i}': q.numpy() for i, q in enumerate(worker.bases)})

        for label, enabled in [('ce', False), ('projection', True)]:
            directory = RESULT/label
            directory.mkdir()
            audit = GateAudit(directory/'audit')
            try:
                with tf.device(f'/{device}:0'):
                    worker = GradientProjectionWorker(build_model('basil_paper_cnn', config['seed']), enabled=enabled)
                    worker.audit = audit
                    worker.load(initial)
                    if params_hash(worker.export()) != manifest['initialModelHash']:
                        raise ValueError('Paired canonical initialization changed.')
                    memories = initialize_memories(initial)
                    records = []
                    for round_id in range(ROUNDS):
                        for node in range(10):
                            context = dict(condition=label, round=round_id, node=node)
                            if stop:
                                raise InterruptedError('Diagnostic stopped; snapshot-only files cannot resume it.')
                            audit.activation(round_id, node)
                            candidates = memories[node].candidates()
                            if [(s.sender_id, s.round_id) for s in candidates] != expected_candidates(round_id, node):
                                raise ValueError('Rolling five-predecessor round contract failed.')
                            if round_id > 0 and enabled and any(not s.retention_state['seenClasses'] for s in candidates):
                                raise ValueError('Warmed projection candidates unexpectedly lack trained history.')
                            local = training[node]
                            probe = local[keyed_rng(config['seed'], 'selection_batch', node).permutation(len(local))[:512]]
                            selected, diagnostics = select_snapshot(node, candidates,
                                lambda p: worker.probe(p, images[probe], labels[probe]))
                            selected_hash = params_hash(selected.params)
                            for d, snapshot in zip(diagnostics, candidates):
                                d.update(paramsHash=params_hash(snapshot.params),
                                    representationClasses=validate_state(snapshot.retention_state, worker.dimensions)['seenClasses'])
                            worker.receive_retention_state(selected.retention_state)
                            def evaluation():
                                confusion = np.zeros((10, 10), np.int64)
                                for start in range(0, len(val_y), 512):
                                    logits = worker.predict(tf.convert_to_tensor(val_x[start:start+512]))
                                    tf.debugging.assert_all_finite(logits, 'Non-finite validation logits')
                                    predictions = tf.argmax(logits, axis=1).numpy()
                                    np.add.at(confusion, (val_y[start:start+512], predictions), 1)
                                return dict(globalValidationAccuracy=float(np.trace(confusion)/len(val_y)),
                                    perClassValidationAccuracy=(np.diag(confusion)/500).tolist(),
                                    validationConfusion=confusion.tolist())
                            worker.load(selected.params)
                            if params_hash(worker.export()) != selected_hash:
                                raise ValueError('Strict complete parameter handoff failed.')
                            rate = basil_round_learning_rate(.05, round_id)
                            trained, record = worker.train(selected.params, images[local], labels[local],
                                seed=config['seed'], round_id=round_id, node_id=node, epochs=5, batch_size=512,
                                learning_rate=rate, ebm_mode='none', semantics='paper_absolute_gaussian',
                                sigma=0., evaluation=evaluation, should_stop=lambda: stop)
                            if record['steps'] != 45 or any(e['batchSizes'] != [512]*8+[404] or e['samplesSeen'] != 4500
                                                          for e in record['epochs']):
                                raise ValueError('Five complete validation-diagnostic epochs failed.')
                            if worker.optimizer.momentum != 0 or worker.optimizer.clipnorm is not None or worker.optimizer.weight_decay is not None:
                                raise ValueError('Unapproved optimizer option enabled.')
                            record.update(round=round_id, nodeId=node, learningRate=rate,
                                selectedSender=selected.sender_id, selectedSnapshotRound=selected.round_id,
                                selectedParamsHash=selected_hash, candidates=diagnostics,
                                trainedOutputHash=params_hash(trained))
                            records.append(record)
                            broadcast(memories, Snapshot(node, round_id, trained, retention_state=worker.output_state))
                            save_memory(directory/'snapshot_memory.npz', memories, worker.dimensions)
                            restored = load_memory(directory/'snapshot_memory.npz', worker.dimensions)
                            for left, right in zip(memories, restored):
                                for a, b in zip(left.candidates(), right.candidates()):
                                    if params_hash(a.params) != params_hash(b.params) or state_hash(
                                            validate_state(a.retention_state, worker.dimensions)) != state_hash(b.retention_state):
                                        raise ValueError('Parameter/basis checkpoint round-trip failed.')
                            save_json(directory/'activations.json', records)
                            manifest.update(currentCondition=label, currentRound=round_id,
                                completedActivations=sum(len(rows) for rows in controls.values())+len(records),
                                memorySafety=guard.checkpoint())
                            manifest['completedRoundsByControl'][label] = (len(records)//10)
                            save_json(RESULT/'run.json', manifest)
                            print(json.dumps(dict(condition=label, round=round_id, node=node,
                                completedActivations=len(records))), flush=True)
                    controls[label] = records
                    del worker
            finally:
                audit.close()
        decision = learning_screen(controls)
        save_json(RESULT/'learning_screen.json', decision)
        manifest.update(status='completed', optimizerSteps=2700,
            runtimeSeconds=time.perf_counter()-started, screen=decision, memorySafety=guard.checkpoint())
        save_json(RESULT/'run.json', manifest)
        generate_report()
        print(json.dumps(dict(status=manifest['status'], runtimeSeconds=manifest['runtimeSeconds'], screen=decision)), flush=True)
    except Exception as error:
        manifest.update(status='stopped' if isinstance(error, InterruptedError) else 'failed',
            error=str(error), failureContext=context, runtimeSeconds=time.perf_counter()-started)
        save_json(RESULT/'run.json', manifest)
        raise


def generate_report():
    from scripts.report_gradient_projection_warm_gate import main
    main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--execute', action='store_true')
    action.add_argument('--report', action='store_true', help='Saved-only verification and plots; no training.')
    args = parser.parse_args()
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled',
        MPLCONFIGDIR='/tmp/papermerge-projection-warm-mpl')
    if args.report:
        generate_report()
        return
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf, args.device)
    execute(tf, args.device)


if __name__ == '__main__':
    main()
