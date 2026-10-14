#!/usr/bin/env python3
"""Bounded full-data CE/fixed-anchor/strict-CART diagnostic; never auto-promotes."""
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
NAME = 'clean_full_data_three_round_v1'
RESULT = ROOT/'newResults/nonIID/strict_cart'/NAME
PLOTS = ROOT/'newPlots/nonIID/strict_cart'/NAME
SOURCES = ('basil_core/strict_cart.py', 'basil_core/class_registry.py', 'basil_core/noniid_study.py',
           'basil_core/research_protocol.py', 'basil_core/iid_study.py', 'basil_core/research_audit.py',
           'basil_core/models.py', 'basil_core/trainer.py', 'basil_core/lazy_imports.py',
           'scripts/run_strict_cart_diagnostic.py')


def source_hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def config(device):
    return dict(experimentProtocol='sequential_basil_one_class_v1',
        methodRevision='strict_cart_receiver_supported_confidence_gap_v1', purpose='bounded_full_data_diagnostic',
        researchValid=False, executionDevice=device, seed=2025, nNodes=10, nRounds=3,
        partitionStrategy='one_class_per_node', trainingSamplesPerNode=5000, localEpochs=5,
        batchSize=512, epochSemantics='full_local_dataset', modelArchitecture='basil_paper_cnn',
        optimizer='sgd', momentum=0., optimizerStateMode='reset_each_activation', learningRate=.05,
        learningRateSchedule='basil_round_decay', gradientClipNorm=None, weightDecayCoefficient=0.,
        aggregationMode='strict_sequential_handoff', basilMemorySize=5, assumedByzantineCount=4,
        attackerCount=0, resolvedAttackerIds=[], useChannelNoise=False, channelNoiseSigmaAbsolute=0.,
        channelNoiseSemantics='paper_absolute_gaussian', ebmMode='none', liveTrackingEnabled=False,
        snapshotSelection=True, snapshotSelectionRule='lowest_receiver_local_loss', snapshotPlausibilityGuard='none',
        fixedAnchorMu=.01, cartMuMax=.01, checkpointScope='evidence only; not resumable research state',
        testUsage='full 10000-image set, evaluation only; never control, selection, or tuning')


def expected_candidates(round_id, node):
    return [(s, round_id if s < node else round_id-1) for s in ((node-d) % 10 for d in range(5, 0, -1))]


def verification_path(device):
    return ROOT/f'gui/worker_state/STRICT_CART_{device}_VERIFICATION.json'


def verify(tf, device):
    import numpy as np
    from basil_core.strict_cart import StrictCartWorker
    from basil_core.research_protocol import build_model
    from scripts.run_gradient_projection_gate import save_json
    rng = np.random.default_rng(77)
    x = rng.uniform(0, 1, (512, 32, 32, 3)).astype(np.float32)
    y = np.zeros(512, np.int32)
    with tf.device('/CPU:0'):
        initial = [w.numpy() for w in build_model('basil_paper_cnn', 2025).trainable_variables]
    outcomes = {}
    for target in dict.fromkeys(('CPU', device)):
        with tf.device(f'/{target}:0'):
            worker = StrictCartWorker(build_model('basil_paper_cnn', 2025), device=target)
            worker.load(initial)
            for ref, value in zip(worker.reference, initial):
                ref.assign(value + .001)
            worker.anchor_mu.assign(.007)
            modes = {}
            for mode in ('none', 'gradient_norm_objective'):
                a = worker._gradients(tf.constant(x), tf.constant(y), tf.constant([7, 8]), mode,
                    'paper_absolute_gaussian', tf.constant(.010), tf.constant(1.))
                b = worker._gradients(tf.constant(x), tf.constant(y), tf.constant([7, 8]), mode,
                    'paper_absolute_gaussian', tf.constant(.010), tf.constant(999999.))
                for p, q in zip(a[0], b[0]): np.testing.assert_array_equal(p.numpy(), q.numpy())
                np.testing.assert_array_equal([float(v) for v in a[1]], [float(v) for v in b[1]])
                partial, _ = worker._gradients(tf.constant(x[:392]), tf.constant(y[:392]), tf.constant([7, 8]),
                    mode, 'paper_absolute_gaussian', tf.constant(.010), tf.constant(1.))
                assert all(np.isfinite(g.numpy()).all() for g in partial)
                tf.keras.optimizers.SGD(.05).apply_gradients(zip(a[0], worker.model.trainable_variables))
                modes[mode] = ([g.numpy() for g in a[0]], worker.export())
                worker.load(initial)
            outcomes[target] = modes
    errors = {}
    for mode in ('none', 'gradient_norm_objective'):
        a, au = outcomes['CPU'][mode]; b, bu = outcomes[device][mode]
        relative = float(np.linalg.norm(np.concatenate([(p-q).ravel() for p, q in zip(a, b)])) /
                         max(np.linalg.norm(np.concatenate([p.ravel() for p in a])), 1e-12))
        update = max(float(np.max(np.abs(p-q))) for p, q in zip(au, bu))
        if relative > .01 or update > 1e-4:
            raise ValueError('CPU/device objective tolerance failed.')
        errors[mode] = dict(gradientRelativeDifference=relative, maxUpdateDifference=update)
    report = dict(status='passed', device=device, numericalTolerance=dict(gradientRelative=.01, updateAbsolute=1e-4),
        checks=errors, legacyLambdaIgnored=True, batchSizes=[512, 392],
        sourceHashes=source_hashes(),
        sourceHash=hashlib.sha256((ROOT/'basil_core/strict_cart.py').read_bytes()).hexdigest())
    save_json(verification_path(device), report)
    print(json.dumps(report), flush=True)


def execute(tf, device):
    import numpy as np
    from basil_core.strict_cart import StrictCartWorker
    from basil_core.class_registry import ClassRegistry
    from basil_core.noniid_study import SnapshotAnchorWorker, partition_audit
    from basil_core.data.cifar import loadCifar10, _datasetArrays
    from basil_core.research_protocol import (build_model, initialize_memories, select_snapshot,
        broadcast, Snapshot, keyed_rng, params_hash, basil_round_learning_rate)
    from basil_core.research_audit import ProtocolAudit
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    from basil_core.iid_runtime import runtime_metadata
    from scripts.run_gradient_projection_gate import save_json
    checked = json.loads(verification_path(device).read_text())
    if checked['status'] != 'passed' or checked.get('sourceHashes') != source_hashes():
        raise ValueError('Current CART device verification required.')
    if RESULT.exists() or PLOTS.exists():
        raise ValueError('Refusing overwrite/resume of existing diagnostic evidence.')
    RESULT.mkdir(parents=True)
    cfg = config(device)
    save_json(RESULT/'config.json', cfg)
    manifest = dict(status='initializing', config=cfg, **runtime_metadata(tf, cfg),
        implementationSha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
        wandbEnabled=False, multiprocessingUsed=False, completedActivations=0,
        completedRoundsByControl={}, productionRunStarted=False)
    stop = False
    def request_stop(_sig, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    guard = RuntimeMemoryGuard(tf, device)
    context = {}
    try:
        check_host_memory()
        train, test = loadCifar10(cacheDir=ROOT/'experiments/cache/cifar10')
        images, labels = _datasetArrays(train)
        test_x, test_y = _datasetArrays(test)
        partition, training = partition_audit(labels, test_y)
        save_json(RESULT/'partition_audit.json', partition)
        with tf.device('/CPU:0'):
            initial = [w.numpy() for w in build_model('basil_paper_cnn', cfg['seed']).trainable_variables]
        manifest.update(status='running', partitionHash=partition['partitionHash'],
            initialModelHash=params_hash(initial), parameterCount=sum(w.size for w in initial))
        save_json(RESULT/'run.json', manifest)
        for condition in ('ce', 'fixed_anchor', 'cart'):
            directory = RESULT/condition; directory.mkdir()
            audit = ProtocolAudit(directory/'audit')
            try:
                with tf.device(f'/{device}:0'):
                    worker = (StrictCartWorker(build_model('basil_paper_cnn', cfg['seed']), mu_max=.01, device=device)
                              if condition == 'cart' else SnapshotAnchorWorker(
                                  build_model('basil_paper_cnn', cfg['seed']), mu=.01 if condition == 'fixed_anchor' else 0., device=device))
                    worker.audit = audit
                    memories = initialize_memories(initial)
                    registries = [ClassRegistry() for _ in range(10)]
                    records = []
                    for r in range(3):
                        for node in range(10):
                            context = dict(condition=condition, round=r, node=node)
                            if stop: raise InterruptedError('Stopped bounded diagnostic; no guessed resume.')
                            audit.activation(r, node)
                            candidates = memories[node].candidates()
                            if [(s.sender_id, s.round_id) for s in candidates] != expected_candidates(r, node):
                                raise ValueError('BASIL rolling-memory contract failed.')
                            local = training[node]
                            probe = local[keyed_rng(cfg['seed'], 'selection_batch', node).permutation(len(local))[:512]]
                            selected, diagnostics = select_snapshot(node, candidates,
                                lambda p: worker.probe(p, images[probe], labels[probe]))
                            selected_hash = params_hash(selected.params)
                            for row, s in zip(diagnostics, candidates): row['paramsHash'] = params_hash(s.params)
                            worker.load(selected.params)
                            if params_hash(worker.export()) != selected_hash: raise ValueError('Strict handoff failed.')
                            def evaluation():
                                confusion = np.zeros((10, 10), np.int64)
                                for start in range(0, len(test_y), 512):
                                    logits = worker.predict(tf.convert_to_tensor(test_x[start:start+512]))
                                    tf.debugging.assert_all_finite(logits, 'Non-finite test logits')
                                    predicted = tf.argmax(logits, axis=1).numpy()
                                    np.add.at(confusion, (test_y[start:start+512], predicted), 1)
                                return dict(globalTestAccuracy=float(np.trace(confusion)/10000),
                                    perClassTestAccuracy=(np.diag(confusion)/1000).tolist(), testConfusion=confusion.tolist())
                            kwargs = dict(seed=cfg['seed'], round_id=r, node_id=node, epochs=5, batch_size=512,
                                learning_rate=basil_round_learning_rate(.05, r), ebm_mode='none',
                                semantics='paper_absolute_gaussian', sigma=0., evaluation=evaluation, should_stop=lambda: stop)
                            if condition == 'cart':
                                incoming = ClassRegistry.fromDict(selected.retention_state) if selected.retention_state else ClassRegistry()
                                loss = next(d['receiverLocalLoss'] for d in diagnostics if d['senderId'] == selected.sender_id)
                                kwargs.update(local_registry=registries[node], received_registry=incoming,
                                    selection_x=images[probe], selection_y=labels[probe], selected_loss=loss)
                            trained, record = worker.train(selected.params, images[local], labels[local], **kwargs)
                            if record['steps'] != 50 or any(e['batchSizes'] != [512]*9+[392] or e['samplesSeen'] != 5000 for e in record['epochs']):
                                raise ValueError('Five full epochs/partial batch contract failed.')
                            if worker.optimizer.momentum != 0 or worker.optimizer.clipnorm is not None or worker.optimizer.weight_decay is not None:
                                raise ValueError('Unexpected optimizer option.')
                            record.update(round=r, nodeId=node, candidates=diagnostics, selectedSender=selected.sender_id,
                                selectedSnapshotRound=selected.round_id, selectedParamsHash=selected_hash,
                                trainedOutputHash=params_hash(trained), learningRate=kwargs['learning_rate'])
                            records.append(record)
                            broadcast(memories, Snapshot(node, r, trained, retention_state=record.get('outputRegistry')))
                            # Evidence checkpoint: all 50 snapshot weights and exact carried registries.
                            payload = {}; metadata = []
                            for receiver, memory in enumerate(memories):
                                for snap in memory.candidates():
                                    prefix = f'r{receiver}s{snap.sender_id}'
                                    for j, value in enumerate(snap.params): payload[f'{prefix}w{j}'] = value
                                    metadata.append(dict(receiver=receiver, sender=snap.sender_id, round=snap.round_id,
                                        hash=params_hash(snap.params), registry=snap.retention_state))
                            payload['metadata'] = np.array(json.dumps(metadata))
                            tmp = directory/'snapshot_memory.tmp.npz'
                            np.savez_compressed(tmp, **payload); tmp.replace(directory/'snapshot_memory.npz')
                            save_json(directory/'activations.json', records)
                            manifest['completedActivations'] += 1
                            manifest.update(currentCondition=condition, currentRound=r, memorySafety=guard.checkpoint())
                            manifest['completedRoundsByControl'][condition] = len(records)//10
                            save_json(RESULT/'run.json', manifest)
                            print(json.dumps(dict(**context, completedActivations=manifest['completedActivations'])), flush=True)
                    del worker
            finally:
                audit.close()
        manifest.update(status='completed', optimizerSteps=4500, runtimeSeconds=time.perf_counter()-started,
            memorySafety=guard.checkpoint())
        save_json(RESULT/'run.json', manifest)
        report()
    except Exception as error:
        manifest.update(status='stopped' if isinstance(error, InterruptedError) else 'failed',
            error=str(error), failureContext=context, runtimeSeconds=time.perf_counter()-started)
        save_json(RESULT/'run.json', manifest)
        raise


def report():
    from scripts.report_strict_cart_diagnostic import main
    main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-device', action='store_true')
    action.add_argument('--execute', action='store_true')
    action.add_argument('--report', action='store_true')
    args = parser.parse_args()
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/papermerge-cart-mpl')
    if args.report: report(); return
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf, args.device)
    verify(tf, args.device) if args.verify_device else execute(tf, args.device)


if __name__ == '__main__': main()
