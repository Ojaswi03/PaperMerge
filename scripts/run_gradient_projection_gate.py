#!/usr/bin/env python3
"""Clean CE/projection training-validation gate; no production queue or W&B."""
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


def validation_split(labels, seed=2025):
    import numpy as np
    from basil_core.research_protocol import keyed_rng
    labels = np.asarray(labels)
    if not np.array_equal(np.bincount(labels, minlength=10), [5000]*10):
        raise ValueError('Gate requires all 50,000 CIFAR training examples.')
    training, validation = [], []
    for node in range(10):
        order = keyed_rng(seed, 'projection_validation_split', node).permutation(np.flatnonzero(labels == node))
        validation.append(order[:500]); training.append(order[500:])
    if len(np.unique(np.concatenate(training + validation))) != 50000:
        raise ValueError('Gate split overlaps or omits training images.')
    return training, validation


def gate_decision(controls):
    """Predeclared validation trend screen, not a software accuracy assertion.

    Each of the first four nodes must improve its own held-out class by 5pp.
    After Node 3, at least two earlier classes must exceed 5% validation accuracy
    and their mean must exceed paired CE by 5pp. These modest screens reject
    preserving a constant-class predictor while preventing new learning.
    """
    ce, projection = controls['ce'], controls['projection']
    own_gains = [r['epochs'][-1]['perClassValidationAccuracy'][i] -
                 r['beforeTraining']['perClassValidationAccuracy'][i]
                 for i, r in enumerate(projection[:4])]
    previous = projection[3]['epochs'][-1]['perClassValidationAccuracy'][:3]
    reference = ce[3]['epochs'][-1]['perClassValidationAccuracy'][:3]
    checks = dict(allFirstFourClassesLearned=min(own_gains) >= .05,
        atLeastTwoEarlierClassesRetained=sum(a > .05 for a in previous) >= 2,
        previousClassImprovementOverCE=sum(previous)/3 - sum(reference)/3 >= .05)
    return dict(passed=all(checks.values()), checks=checks, ownClassGains=own_gains,
        previousClassAccuracy=previous, cePreviousClassAccuracy=reference,
        validationOnly=True, testSetUsed=False)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verification_path(device):
    return ROOT / f'gui/worker_state/GRADIENT_PROJECTION_{device}_VERIFICATION.json'


def verify(tf, device):
    import numpy as np
    from basil_core.gradient_projection_retention import GradientProjectionWorker, REVISION
    from basil_core.research_protocol import build_model
    with tf.device('/CPU:0'):
        initial = [v.numpy() for v in build_model('basil_paper_cnn', 2025).trainable_variables]
    rng = np.random.default_rng(77)
    images = rng.uniform(0, 1, (512, 32, 32, 3)).astype(np.float32)
    labels = np.zeros(512, np.int32)
    outcomes = {}
    for target in dict.fromkeys(('CPU', device)):
        with tf.device(f'/{target}:0'):
            worker = GradientProjectionWorker(build_model('basil_paper_cnn', 2025))
            worker.load(initial)
            basis_rng = np.random.default_rng(81)
            for variable, dim in zip(worker.bases, worker.dimensions):
                q, _ = np.linalg.qr(basis_rng.normal(size=(dim, 3)))
                variable.assign(q.astype(np.float32))
            results = []
            for legacy in (1., 999999.):
                grads, values = worker._gradients(tf.constant(images), tf.constant(labels),
                    tf.constant([7, 8]), 'none', 'paper_absolute_gaussian', tf.constant(0.), tf.constant(legacy))
                results.append(([g.numpy() for g in grads], [float(v) for v in values]))
            for a, b in zip(results[0][0], results[1][0]):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(results[0][1], results[1][1])
            if float(values[-1]) > 1e-5:
                raise ValueError('Projected gradient is not orthogonal to protected inputs.')
            partial, _ = worker._gradients(tf.constant(images[:392]), tf.constant(labels[:392]),
                tf.constant([7, 8]), 'none', 'paper_absolute_gaussian', tf.constant(0.), tf.constant(0.))
            if any(not np.isfinite(g.numpy()).all() for g in partial):
                raise ValueError('Non-finite partial-batch projection gradient.')
            tf.keras.optimizers.SGD(.05, momentum=0.).apply_gradients(zip(grads, worker.model.trainable_variables))
            outcomes[target] = results[0], worker.export()
    (a, av), au = outcomes['CPU']; (b, bv), bu = outcomes[device]
    relative = float(np.linalg.norm(np.concatenate([(x-y).ravel() for x, y in zip(a, b)])) /
                     max(np.linalg.norm(np.concatenate([x.ravel() for x in a])), 1e-12))
    update = max(float(np.max(np.abs(x-y))) for x, y in zip(au, bu))
    if relative > .01 or update > 1e-4:
        raise ValueError(f'CPU/device gradient/update tolerance failed: {relative}, {update}')
    result = dict(status='passed', revision=REVISION, device=device, legacyLambdaIgnored=True,
        gradientRelativeL2Difference=relative, gradientTolerance=.01,
        maxParameterUpdateDifference=update, updateTolerance=1e-4,
        batchSizes=[512, 392], researchRunsStarted=0,
        objectiveSha256=hashlib.sha256((ROOT/'basil_core/gradient_projection_retention.py').read_bytes()).hexdigest())
    if device == 'GPU':
        result['gpuMemory'] = tf.config.experimental.get_memory_info('GPU:0')
    save_json(verification_path(device), result)
    print(json.dumps(result), flush=True)
    return result


def execute(tf, device):
    import numpy as np
    from basil_core.data.cifar import loadCifar10, _datasetArrays
    from basil_core.gradient_projection_retention import (GradientProjectionWorker, REVISION,
        ENERGY, REPRESENTATION_IMAGES, MAX_PATCHES, save_memory, load_memory, state_hash)
    from basil_core.research_protocol import (build_model, initialize_memories, select_snapshot,
        Snapshot, broadcast, keyed_rng, params_hash, basil_round_learning_rate)
    from basil_core.research_audit import ProtocolAudit, array_hash
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    from basil_core.iid_runtime import runtime_metadata

    verification = json.loads(verification_path(device).read_text())
    if (verification.get('status') != 'passed' or verification['objectiveSha256'] !=
            hashlib.sha256((ROOT/'basil_core/gradient_projection_retention.py').read_bytes()).hexdigest()):
        raise ValueError('Current projection implementation must pass device verification first.')
    output = ROOT / 'newResults/nonIID/gradient_projection/clean_training_validation_gate_v1'
    output.mkdir(parents=True, exist_ok=False)  # Never replace previous evidence.
    seed = 2025
    config = dict(experimentProtocol='sequential_basil_one_class_v1', methodRevision=REVISION,
        purpose='training_validation_retention_gate', researchValid=False, seed=seed,
        nNodes=10, nRounds=1, localEpochs=5, batchSize=512, trainingSamplesPerNode=4500,
        validationSamplesPerClass=500, learningRate=.05, learningRateSchedule='basil_round_decay',
        optimizer='sgd', momentum=0., optimizerStateMode='reset_each_activation',
        gradientClipNorm=None, weightDecayCoefficient=0., modelArchitecture='basil_paper_cnn',
        aggregationMode='strict_sequential_handoff', basilMemorySize=5, assumedByzantineCount=4,
        attackerCount=0, resolvedAttackerIds=[], useChannelNoise=False, ebmMode='none',
        snapshotSelection=True, snapshotSelectionRule='lowest_receiver_local_loss',
        snapshotPlausibilityGuard='none', snapshotAnchorMu=0., liveTrackingEnabled=False,
        executionDevice=device, projectionEnergy=ENERGY,
        representationImages=REPRESENTATION_IMAGES, maximumPatchesPerLayer=MAX_PATCHES,
        testUsage='not evaluated or used by this gate',
        gatePolicy='first-four own gain >=5pp each; >=2 earlier classes >5%; prior mean >CE by >=5pp')
    save_json(output/'config.json', config)
    runtime = runtime_metadata(tf, config)
    manifest = dict(status='initializing', config=config, **runtime, wandbEnabled=False,
        multiprocessingUsed=False, implementationSha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
            for p in ('basil_core/gradient_projection_retention.py', 'scripts/run_gradient_projection_gate.py',
                      'basil_core/research_protocol.py', 'basil_core/models.py')})
    stop = False
    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    all_records = {}
    guard = RuntimeMemoryGuard(tf, device)
    started = time.perf_counter()
    context = {}
    try:
        check_host_memory()
        train, _unused_test = loadCifar10(cacheDir=ROOT/'experiments/cache/cifar10')
        images, labels = _datasetArrays(train)
        training, validation = validation_split(labels, seed)
        split_hash = params_hash([np.asarray(a, np.float32) for a in training+validation])
        # All evaluation inputs are held-out TRAINING images; never test data.
        val_indices = np.concatenate(validation)
        val_x, val_y = images[val_indices], labels[val_indices]
        save_json(output/'partition_audit.json', dict(trainingCounts=[len(a) for a in training],
            validationCounts=[len(a) for a in validation], partitionHash=split_hash,
            assignedTrainingUniverse=50000, trainingValidationDisjoint=True,
            testEvaluated=False, validationImageHash=array_hash(val_x), validationLabelHash=array_hash(val_y)))
        with tf.device('/CPU:0'):
            initial = [v.numpy() for v in build_model('basil_paper_cnn', seed).trainable_variables]
        manifest.update(status='running', partitionHash=split_hash, initialModelHash=params_hash(initial))
        save_json(output/'run.json', manifest)

        class GateAudit(ProtocolAudit):
            def gradients(self, gradients, values):
                super().gradients(gradients, values)
                self.current.update(zip(GradientProjectionWorker.extra_metric_names, map(float, values[6:])))
            def failure(self, worker, error):
                self.write('failure.json', {**getattr(self, 'current', {}), 'error': str(error),
                    'gradientRule': REVISION, 'graphReplaySkipped': True})
                if self.pending:
                    np.savez_compressed(self.directory/'failure_batch.npz', x=self.pending['x'],
                        y=self.pending['y'], key=self.pending['key'],
                        **{f'weight_{i}': w for i, w in enumerate(worker.export())},
                        **{f'basis_{i}': q.numpy() for i, q in enumerate(worker.bases)})

        for label, enabled in [('ce', False), ('projection', True)]:
            directory = output/label; directory.mkdir()
            audit = GateAudit(directory/'audit')
            try:
                with tf.device(f'/{device}:0'):
                    worker = GradientProjectionWorker(build_model('basil_paper_cnn', seed), enabled=enabled)
                    worker.audit = audit
                    memories = initialize_memories(initial)
                    records = []
                    for node in range(10):
                        context = dict(condition=label, round=0, node=node)
                        if stop:
                            raise InterruptedError('Gate stopped; no automatic or guessed continuation.')
                        audit.activation(0, node)
                        local = training[node]
                        probe = local[keyed_rng(seed, 'selection_batch', node).permutation(len(local))[:512]]
                        selected, diagnostics = select_snapshot(node, memories[node].candidates(),
                            lambda p: worker.probe(p, images[probe], labels[probe]))
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
                        if params_hash(worker.export()) != params_hash(selected.params):
                            raise ValueError('Strict complete handoff failed before SGD.')
                        trained, record = worker.train(selected.params, images[local], labels[local],
                            seed=seed, round_id=0, node_id=node, epochs=5, batch_size=512,
                            learning_rate=basil_round_learning_rate(.05, 0), ebm_mode='none',
                            semantics='paper_absolute_gaussian', sigma=0., evaluation=evaluation,
                            should_stop=lambda: stop)
                        if record['steps'] != 45 or any(e['batchSizes'] != [512]*8+[404] or e['samplesSeen'] != 4500 for e in record['epochs']):
                            raise ValueError('Five complete gate epochs or partial batch contract failed.')
                        record.update(round=0, nodeId=node, selectedSender=selected.sender_id,
                            selectedSnapshotRound=selected.round_id, candidates=diagnostics,
                            trainedOutputHash=params_hash(trained))
                        if worker.optimizer.momentum != 0 or worker.optimizer.clipnorm is not None or worker.optimizer.weight_decay is not None:
                            raise ValueError('Unapproved SGD option enabled.')
                        records.append(record)
                        broadcast(memories, Snapshot(node, 0, trained, retention_state=worker.output_state))
                        save_memory(directory/'snapshot_memory.npz', memories, worker.dimensions)
                        restored = load_memory(directory/'snapshot_memory.npz', worker.dimensions)
                        for expected, actual in zip(memories, restored):
                            for a, b in zip(expected.candidates(), actual.candidates()):
                                if params_hash(a.params) != params_hash(b.params):
                                    raise ValueError('Parameter transfer checkpoint changed.')
                                from basil_core.gradient_projection_retention import validate_state
                                if state_hash(validate_state(a.retention_state, worker.dimensions)) != state_hash(b.retention_state):
                                    raise ValueError('Projection state transfer checkpoint changed.')
                        save_json(directory/'activations.json', records)
                        manifest.update(currentCondition=label, completedActivations=len(records), memorySafety=guard.checkpoint())
                        save_json(output/'run.json', manifest)
                        print(json.dumps(dict(condition=label, completedActivations=len(records),
                            validationAccuracy=record['epochs'][-1]['globalValidationAccuracy'])), flush=True)
                    all_records[label] = records
                    del worker
            finally:
                audit.close()
        decision = gate_decision(all_records)
        save_json(output/'gate_decision.json', decision)
        manifest.update(status='completed', completedActivations=20, completedRoundsPerControl=1,
            optimizerSteps=900, runtimeSeconds=time.perf_counter()-started, gate=decision,
            thirtyRoundRunStarted=False, memorySafety=guard.checkpoint())
        save_json(output/'run.json', manifest)
        report(output, all_records, decision)
        print(json.dumps(manifest), flush=True)
    except Exception as error:
        manifest.update(status='stopped' if isinstance(error, InterruptedError) else 'failed',
            error=str(error), failureContext=context, runtimeSeconds=time.perf_counter()-started)
        save_json(output/'run.json', manifest)
        raise


def report(output, controls, decision):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plots = ROOT/'newPlots/nonIID/gradient_projection/clean_training_validation_gate_v1'
    plots.mkdir(parents=True, exist_ok=False)
    for label, records in controls.items():
        trajectory = np.asarray([r['epochs'][-1]['perClassValidationAccuracy'] for r in records])
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(trajectory.T, vmin=0, vmax=1, aspect='auto', cmap='viridis')
        ax.set(xlabel='Active node, first traversal', ylabel='CIFAR class',
               title=f'{label}: held-out training-class accuracy (not test accuracy)')
        fig.colorbar(im, ax=ax, label='Accuracy'); fig.tight_layout()
        fig.savefig(plots/f'{label}_validation_retention.png'); plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        r = records[1]
        stages = [r['beforeTraining'], *r['epochs']]
        for ax, stage, title in zip(axes.flat, stages, ['Before', 'Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']):
            ax.imshow(np.asarray(stage['validationConfusion'])/500, vmin=0, vmax=1, cmap='viridis')
            ax.set(title=title, xlabel='Prediction', ylabel='True class')
        fig.suptitle(f'{label}, Node 1: held-out training-validation confusions')
        fig.tight_layout(); fig.savefig(plots/f'{label}_node1_epoch_confusions.png'); plt.close(fig)
    rows = ['| Node | CE own class | Projection own class | CE prior-class mean | Projection prior-class mean | Projection basis ranks |',
            '|---:|---:|---:|---:|---:|---|']
    for node, (c, p) in enumerate(zip(controls['ce'], controls['projection'])):
        ca, pa = c['epochs'][-1]['perClassValidationAccuracy'], p['epochs'][-1]['perClassValidationAccuracy']
        cm, pm = (np.mean(ca[:node]), np.mean(pa[:node])) if node else (float('nan'), float('nan'))
        rows.append(f'| {node} | {ca[node]:.2%} | {pa[node]:.2%} | {cm:.2%} | {pm:.2%} | {p["gradientProjection"]["ranks"]} |')
    status = 'PASSED' if decision['passed'] else 'DID NOT PASS'
    text = f'''# Gradient-projection clean retention gate

**Learning-trend gate: {status}.** Completed both paired one-traversal controls.
No 30-round or production run, attack/noise condition, W&B run or queue change.

This is a separately labelled GPM-inspired, bias-aware single-head adaptation.
The source's CIFAR experiments used multi-head classifiers and no bias units;
our common ten-class head and protected biases are important differences.
Source: [Saha et al., Sections 5–6](https://arxiv.org/html/2103.09762v1).

## Design and metric definitions

All 50,000 training images are deterministically split into 4,500 SGD images
and 500 held-out training-validation images per class. No CIFAR test images
are evaluated, selected or trained on. Gate data is NOT a full-data research
run: five full gate epochs mean 45 SGD steps/activation, with last batch 404.
The original 5,000-image protocol remains unchanged (50 steps, last batch 392).
Both controls use identical initialization/split/order/augmentation, the same
117,706-parameter CNN, fixed ring, BASIL S=5, minimum local-CE selection,
complete handoff and plain reset SGD at round-0 LR=0.05.

Prior-class mean is the evaluated accuracy on classes of earlier activated
nodes, regardless of whether BASIL selected a snapshot carrying those classes.
Seen-class metadata records representation exposure, not retained knowledge.

{chr(10).join(rows)}

## Predeclared gate checks

```json
{json.dumps(decision, indent=2)}
```

The 5pp/5% thresholds are modest validation trend screens, not mathematical
correctness assertions or a tuned target. No parameter was changed after
observing them. A method preserving an old constant-class predictor while
failing to learn the next class is not considered a successful retention gate.

## Update and limitations

For each kernel/bias pair, flatten the kernel to input-by-output layout and
append the bias as the last row: G_affine. With orthonormal input basis Q:
`G_applied = G_affine - Q @ (Q.T @ G_affine)`.
Five-epoch SGD uses this constrained CE direction, not a new proximal loss.
After training, 32 receiver-training images (unaugmented) supply layer-input
representations; convolutional patches are deterministically bounded at 2,048
per layer. Residual covariance eigenvectors expand Q to capture 99% of sampled
representation energy. This is SVD-equivalent, not an estimate of test accuracy.
The basis is frozen during each activation and travels with exactly the
selected snapshot; other memories are not merged. No raw images are sent.

Projection can suppress new-class learning as capacity becomes protected.
Preserving a previously collapsed predictor does not create discrimination.
BASIL's local-CE selection can select a snapshot without the older history.
One seed and this adaptation cannot establish universal success/failure of GPM.
Auxiliary bases have no formal privacy or Byzantine-authentication guarantee.
Gate memory files restore all five predecessor snapshots and bases but are
NOT full resumable research-run checkpoints. Interruption stops this gate;
it is not silently resumed from final weights.

## Plots

![CE retention](../../../../newPlots/nonIID/gradient_projection/clean_training_validation_gate_v1/ce_validation_retention.png)
![Projection retention](../../../../newPlots/nonIID/gradient_projection/clean_training_validation_gate_v1/projection_validation_retention.png)
![Projection Node-1 epochs](../../../../newPlots/nonIID/gradient_projection/clean_training_validation_gate_v1/projection_node1_epoch_confusions.png)
'''
    (output/'GRADIENT_PROJECTION_GATE_REVIEW.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-device', action='store_true')
    action.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/papermerge-projection-mpl')
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf, args.device)
    if args.verify_device:
        verify(tf, args.device)
    else:
        execute(tf, args.device)


if __name__ == '__main__':
    main()
