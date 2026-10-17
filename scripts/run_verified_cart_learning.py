#!/usr/bin/env python3
"""One clean full-data ring traversal on the isolated verified CART backend.

Default is dry-run. No GUI queue, cloud logging, horizon tuning or A/B/C/D.
This explicitly replicates TRAIN data; it is not the exclusive-data BASIL model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT/'newResults/nonIID/cart_verified/clean_full_data_one_round_v1'
PLOTS = ROOT/'newPlots/nonIID/cart_verified/clean_full_data_one_round_v1'
RUN_ID = 'cart_verified_clean_full_data_one_round_v1'


def configuration():
    return dict(runId=RUN_ID, researchValid=False, purpose='bounded_full_data_learning_diagnostic',
        rounds=1, nodes=10, samplesPerNode=5000, localEpochs=5, batchSize=512,
        stepsPerActivation=50, finalBatch=392, model='basil_paper_cnn', seed=2025,
        optimizer='sgd', momentum=0., learningRate=.05, learningRateSchedule='basil_round_decay',
        gradientClipNorm=None, weightDecay=0., snapshotAnchorMu=.01,
        objective='mean_available_owner_ce_plus_fixed_snapshot_anchor',
        ebm=False, sigma_e=0., attackerIds=[], helperFaults={}, memorySize=5,
        snapshotSelection='certified_balanced_available_owner_train_ce',
        informationModel='replicated_train_trusted_coordinator', device='GPU', helperDevice='CPU',
        evaluationOnlyTestCount=10000, evaluationBatchSize=256, selectionBatchSize=512,
        wandb=False, fullResearchReady=False)


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.diagnostic-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, separators=(',', ':'), allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(name): os.unlink(name)


def require_proof():
    from basil_core.execution_profile import profile_path
    from basil_core.verified_cart_runner import fingerprints
    path = profile_path(ROOT/'gui/worker_state/CART_RESOURCE_LIMITED_RECOVERY_GPU.json')
    proof = json.loads(path.read_text())
    checker = ROOT/'scripts/check_isolated_replicas.py'
    if (proof.get('status') != 'passed' or proof.get('device') != 'GPU'
            or proof.get('sourceHashes') != fingerprints()
            or proof.get('checkerSha256') != hashlib.sha256(checker.read_bytes()).hexdigest()
            or not proof.get('resumeOutputBitIdentical')
            or not proof.get('explicitTrainerRecoveryMatchesHonest')):
        raise ValueError('Current integrated GPU correctness/recovery proof required.')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resources(pool):
    values = {}
    for identity, endpoint in pool.endpoints.items():
        root = Path('/sys/fs/cgroup')/endpoint.resources['cgroup'].lstrip('/')
        events = {k: int(v) for k, v in (s.split() for s in (root/'memory.events').read_text().splitlines())}
        values[str(identity)] = dict(peakBytes=int((root/'memory.peak').read_text()), events=events)
        if events.get('oom', 0) or events.get('oom_kill', 0):
            raise RuntimeError(f'Replica {identity} encountered OOM; diagnostic aborted.')
    return values


def evaluate(model, params, images, labels, *, node, epoch):
    import numpy as np
    from basil_core.research_protocol import evaluate_params, params_hash
    def load(weights):
        for variable, value in zip(model.trainable_variables, weights): variable.assign(value)
    worker = SimpleNamespace(load=load, predict=lambda x: model(x, training=False), audit=None)
    # Capture confusion without using ProtocolAudit's write/training interfaces.
    class ConfusionCapture:
        last_confusion = None
        def evaluation_failure(self, *args): pass  # evaluate_params re-raises the error.
    audit = ConfusionCapture(); worker.audit = audit
    accuracy, classes, loss = evaluate_params(worker, params, images, labels, batch_size=256, details=True)
    return dict(node=node, epoch=epoch, accuracy=accuracy, perClass=classes.tolist(),
        loss=loss, ownClass=float(classes[node]), otherClasses=float(np.mean(np.delete(classes, node))),
        confusion=audit.last_confusion.tolist(), parameterHash=params_hash(params))


def commit(runner, state, directory):
    """Manifest is the commit pointer; an orphan newer checkpoint is never resumed."""
    generation = state.get('generation', -1) + 1
    checkpoint = directory/f'checkpoint_{generation:04d}.json'
    if checkpoint.exists(): raise FileExistsError('Uncommitted checkpoint exists; preserve it for investigation.')
    if sum(p.stat().st_size for p in directory.glob('checkpoint_*.json')) > 2*1024**3:
        raise OSError('Bounded checkpoint archive exceeded 2 GiB; do not delete evidence or extend.')
    runner.save(checkpoint)
    payload = dict(state, generation=generation, checkpoint=checkpoint.name,
                   cursor=[runner.round, runner.node, runner.step], steps=len(runner.records))
    atomic_json(directory/'progress.json', payload)
    state.update(payload)


def execute(*, resume=False):
    import numpy as np
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow, runtime_metadata
    proof_hash = require_proof()
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled')
    prepare_environment('GPU')
    import tensorflow as tf
    configure_tensorflow(tf, 'GPU')
    from basil_core.research_protocol import build_model, params_hash, keyed_rng
    from basil_core.replicated_training_service import TrainingReference
    from basil_core.isolated_replicas import IsolatedReplicaPool
    from basil_core.verified_cart_runner import VerifiedCartRunner, fingerprints
    cache = ROOT/'experiments/cache/cifar10'
    images = np.load(cache/'x_train_float32.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(cache/'y_train_int32.npy', mmap_mode='r', allow_pickle=False)
    reference = TrainingReference(images, labels, source_split='train')
    test_x = np.load(cache/'x_test_float32.npy', mmap_mode='r', allow_pickle=False)
    test_y = np.load(cache/'y_test_int32.npy', mmap_mode='r', allow_pickle=False)
    if test_x.shape != (10000, 32, 32, 3) or not np.array_equal(np.bincount(test_y, minlength=10), [1000]*10):
        raise ValueError('Full balanced evaluation-only TEST cache required.')
    selections = {i: tuple(map(int, keyed_rng(2025, 'selection_batch', i).permutation(np.flatnonzero(labels == i))[:512]))
                  for i in range(10)}
    with tf.device('/CPU:0'):
        canonical = build_model('basil_paper_cnn', 2025)
        initial = [w.numpy() for w in canonical.trainable_variables]
    with tf.device('/GPU:0'):
        model = build_model('basil_paper_cnn', 2025)
        for w, p in zip(model.trainable_variables, initial): w.assign(p)
    if sum(p.size for p in initial) != 117706:
        raise ValueError('Approved 117706-parameter architecture required.')
    manifest = dict(config=configuration(), dataRoot=reference.root, initialHash=params_hash(initial),
        parameterCount=sum(p.size for p in initial), sourceHashes=fingerprints(),
        scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        verificationSha256=proof_hash, runtime=runtime_metadata(tf, {'executionDevice': 'GPU'}),
        selectionIndices=selections, testClassCounts=[1000]*10)
    # JSON-normalize keys before comparing a saved run.
    manifest = json.loads(json.dumps(manifest))
    if resume:
        if json.loads((OUTPUT/'manifest.json').read_text()) != manifest:
            raise ValueError('Resume manifest/source/data mismatch.')
        state = json.loads((OUTPUT/'progress.json').read_text())
        if state['status'] == 'completed': raise ValueError('Already complete; no extra training authorized.')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        atomic_json(OUTPUT/'manifest.json', manifest)
        state = dict(status='running', evaluations=[], resourceHistory=[], generation=-1, elapsedSeconds=0.)
    started = time.monotonic(); previous_elapsed = state['elapsedSeconds']
    runner = None
    try:
        if shutil.disk_usage('/tmp').free < 4*1024**3: raise OSError('4 GiB temporary disk headroom required.')
        with tempfile.TemporaryDirectory(prefix='cart-learning-train-') as temp:
            fixture = Path(temp)/'train.npz'
            np.savez(fixture, images=images, labels=labels, synthetic=np.asarray(False))
            # Only TRAIN is mounted in each helper. TEST stays evaluation-only at coordinator.
            with IsolatedReplicaPool(fixture, timeout=15.) as pool:
                runner = VerifiedCartRunner(model, reference, pool.service, run_id=RUN_ID, device='GPU',
                    trainer_failure_policy='verified_coordinator_recompute')
                if resume: runner.restore(OUTPUT/state['checkpoint'])
                else: commit(runner, state, OUTPUT)
                while runner.round < 1:
                    node = runner.node
                    if runner.anchor is None:
                        runner.select({i: selections[i] for i in range(node+1)})
                        state['evaluations'].append(evaluate(model, runner.params, test_x, test_y, node=node, epoch=0))
                        commit(runner, state, OUTPUT)
                    while runner.step < 50:
                        receipt = runner.advance()
                        if runner.step % 10 == 0:
                            epoch = runner.step//10
                            observation = evaluate(model, runner.params, test_x, test_y, node=node, epoch=epoch)
                            before = next(e for e in reversed(state['evaluations']) if e['node']==node and e['epoch']==0)
                            observation['forgettingPerClass'] = (np.asarray(before['perClass'])-observation['perClass']).tolist()
                            observation['meanOtherClassForgetting'] = float(np.mean(np.delete(observation['forgettingPerClass'], node)))
                            observation['snapshotDistanceSquared'] = sum(float(np.sum((p.astype(np.float64)-a)**2))
                                for p, a in zip(runner.params, runner.anchor))
                            observation['meanEpochBalancedCE'] = float(np.mean([r['crossEntropy'] for r in runner.records[-10:]]))
                            state['evaluations'].append(observation)
                            state['resourceHistory'].append(dict(node=node, epoch=epoch, replicas=resources(pool)))
                            state['elapsedSeconds'] = previous_elapsed+time.monotonic()-started
                            commit(runner, state, OUTPUT)
                            print(json.dumps(dict(node=node, epoch=epoch, steps=len(runner.records),
                                accuracy=observation['accuracy'], objective=receipt['objective'],
                                seconds=state['elapsedSeconds'])), flush=True)
                    runner.finish_activation()
                    commit(runner, state, OUTPUT)
                assert len(runner.records)==500 and len(state['evaluations'])==60
                assert [r['samples'] for r in runner.records] == ([512]*9+[392])*50
                state.update(status='completed', elapsedSeconds=previous_elapsed+time.monotonic()-started,
                    helperCalls=pool.service.calls, fullResearchReady=False)
                commit(runner, state, OUTPUT)
    except BaseException as error:
        failure = dict(status='failed', exceptionType=type(error).__name__, message=str(error),
            cursor=[runner.round, runner.node, runner.step] if runner else None,
            lastCommittedCheckpoint=state.get('checkpoint'), elapsedSeconds=previous_elapsed+time.monotonic()-started)
        atomic_json(OUTPUT/'failure.json', failure)
        print(json.dumps(failure), flush=True)
        raise
    report()


def report():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    state = json.loads((OUTPUT/'progress.json').read_text())
    if state['status'] != 'completed': raise ValueError('Do not report partial evidence as complete.')
    finals = [e for e in state['evaluations'] if e['epoch']==5]
    matrix = np.asarray([e['perClass'] for e in finals]); accuracies = [e['accuracy'] for e in finals]
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5)); im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
    ax.set(xlabel='CIFAR-10 class', ylabel='Node after five epochs', xticks=range(10), yticks=range(10),
        title='One-round clean verified CART: class retention (diagnostic)')
    fig.colorbar(im, ax=ax, label='Accuracy'); fig.tight_layout(); fig.savefig(PLOTS/'node_class_accuracy.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True)
    for node, ax in enumerate(axes.flat):
        observations = [e for e in state['evaluations'] if e['node']==node]
        for c in range(10): ax.plot(range(6), [e['perClass'][c] for e in observations], color=plt.get_cmap('tab10')(c), label=str(c))
        ax.set(title=f'Node {node}', ylim=(0, 1), xlabel='Epoch (0 = selected input)')
    axes[0,0].legend(title='Class', ncol=2, fontsize=7); fig.tight_layout()
    fig.savefig(PLOTS/'five_epoch_class_retention.png'); plt.close(fig)
    lines = ['# Verified CART: one-round full-data learning diagnostic', '',
        'Clean, fixed μ=0.01; no attack/noise/EBM/W&B. GPU coordinator, ten CPU verifier replicas.',
        'This uses replicated TRAIN data and a trusted coordinator, not exclusive-data BASIL.', '',
        '| Node | Selected-input accuracy | After five epochs | Own class | Other classes | Other-class forgetting (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for e in finals:
        before = next(v for v in state['evaluations'] if v['node']==e['node'] and v['epoch']==0)
        lines.append(f"| {e['node']} | {100*before['accuracy']:.2f}% | {100*e['accuracy']:.2f}% | "
            f"{100*e['ownClass']:.2f}% | {100*e['otherClasses']:.2f}% | {100*e['meanOtherClassForgetting']:.2f} |")
    lines += ['', f'Node 9: {100*accuracies[-1]:.2f}%; node mean: {100*np.mean(accuracies):.2f}%; '
        f'worst node: {100*min(accuracies):.2f}%.',
        f"Runtime: {state['elapsedSeconds']/60:.2f} minutes. 500 updates; 10 complete activations; 50 full local epochs.",
        '', 'One traversal is not evidence of convergence, attack robustness or an EBM benefit. No accuracy threshold '
        'or test-set feedback changed the schedule. Early-node states saw fewer available classes than later-node states.',
        'Previous trusted-helper trajectories are not a paired control for this mixed GPU-coordinator/CPU-helper backend.',
        '', 'Detailed per-epoch confusion matrices, losses and hashes: `progress.json`; signed helper certificates '
        'and complete final state: the checkpoint referenced by `progress.json`.',
        '', '**Not a production-readiness claim. A/B/C/D remain unstarted.**']
    (OUTPUT/'LEARNING_REVIEW.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true')
    action.add_argument('--resume', action='store_true')
    action.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.execute or args.resume: execute(resume=args.resume)
    elif args.report: report()
    else: print(json.dumps(dict(config=configuration(), trainingRunsStarted=0), indent=2))


if __name__ == '__main__': main()
