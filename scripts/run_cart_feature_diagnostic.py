#!/usr/bin/env python3
"""One clean traversal per CART control; defaults to a zero-training plan.

Explicit --execute only after matching --verify-device proof. No queue, W&B,
test data, attack/noise study, production promotion or automatic retries.
"""
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
RESULT = ROOT/'newResults/nonIID/cart_feature_memory/clean_one_traversal_beta1_v1'
PLOTS = ROOT/'newPlots/nonIID/cart_feature_memory/clean_one_traversal_beta1_v1'
SOURCES = ('basil_core/cart_feature_memory.py', 'scripts/run_cart_feature_diagnostic.py',
    'basil_core/strict_cart.py', 'basil_core/noniid_study.py', 'basil_core/research_protocol.py',
    'basil_core/class_registry.py', 'basil_core/iid_study.py', 'basil_core/research_audit.py',
    'basil_core/models.py', 'basil_core/trainer.py', 'basil_core/iid_runtime.py')


def config(device):
    return dict(methodRevision='strict_cart_activation_feature_replay_v1',
        experimentProtocol='sequential_basil_one_class_v1', purpose='bounded_clean_feature_memory_diagnostic',
        researchValid=False, device=device, seed=2025, nNodes=10, nRounds=1,
        controls=['cart', 'cart_feature_memory'], localEpochs=5, samplesPerNode=5000,
        batchSize=512, optimizer='sgd', learningRate=.05, learningRateSchedule='basil_round_decay',
        momentum=0., weightDecayCoefficient=0., gradientClipNorm=None,
        optimizerStateMode='reset_each_activation', aggregationMode='strict_sequential_handoff',
        modelArchitecture='basil_paper_cnn', basilMemorySize=5, assumedByzantineCount=4,
        snapshotSelectionRule='lowest_receiver_local_loss', snapshotPlausibilityGuard='none',
        actualAttackers=[], channelNoise=False, ebm=False, cartMuMax=.01, featureReplayBeta=1.,
        featureReplayBetaSource='explicit_operator_choice_before_diagnostic',
        featureRowsPerClass=16, ownerEncodingBatch=512, featureRefresh='selected_encoder_per_activation',
        featureThreatScope='clean diagnostic only; no Byzantine/privacy guarantee',
        featureChannel='additional clean owner-response channel, not noisy parameter links',
        featureCommunication='previously visited owners encode selected model; no raw image transfer',
        encoderFrozen=False, featureMemoryFrozenDuringActivation=True, testSetUsed=False,
        analysisProbePerClass=64, analysisUsage='centralized TRAINING probe; not held out; analysis only',
        liveTrackingEnabled=False, queueStarted=False, automaticPromotion=False,
        checkpointScope='diagnostic evidence only, not production resume state')


def fingerprints():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def proof_path(device):
    return ROOT/f'gui/worker_state/CART_FEATURE_{device}_VERIFICATION.json'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify(tf, device):
    """One synthetic batch per CE/EBM mode; no CIFAR or research trajectory."""
    import numpy as np
    from basil_core.cart_feature_memory import FeatureReplayWorker, ActivationFeatureMemory, packet_from_features, encoder_hash
    from basil_core.research_protocol import build_model, params_hash
    with tf.device('/CPU:0'):
        initial = [w.numpy() for w in build_model('basil_paper_cnn', 2025).trainable_variables]
    h, e = params_hash(initial), encoder_hash(initial)
    p = packet_from_features(np.random.default_rng(11).uniform(0, .1, (16, 192)),
        owner=1, round_id=0, receiver=0, model_hash=h, encoder_id=e)
    memory = ActivationFeatureMemory(0, 0, h, e, 192, (1,), (p,))
    x = np.random.default_rng(7).uniform(0, 1, (512, 32, 32, 3)).astype(np.float32)
    outcomes = {}
    for target in dict.fromkeys(('CPU', device)):
        with tf.device(f'/{target}:0'):
            worker = FeatureReplayWorker(build_model('basil_paper_cnn', 2025), beta=1., device=target)
            worker.load(initial); worker.set_memory(memory); worker.anchor_mu.assign(.007)
            for ref, value in zip(worker.reference, initial): ref.assign(value+.001)
            modes = {}
            for mode in ('none', 'gradient_norm_objective'):
                a = worker._gradients(tf.constant(x), tf.zeros(512, tf.int32), tf.constant([7, 8]),
                    mode, 'paper_absolute_gaussian', tf.constant(.010), tf.constant(1.))
                b = worker._gradients(tf.constant(x), tf.zeros(512, tf.int32), tf.constant([7, 8]),
                    mode, 'paper_absolute_gaussian', tf.constant(.010), tf.constant(999999.))
                for u, v in zip(a[0], b[0]): np.testing.assert_array_equal(u.numpy(), v.numpy())
                np.testing.assert_array_equal([float(v) for v in a[1]], [float(v) for v in b[1]])
                partial, _ = worker._gradients(tf.constant(x[:392]), tf.zeros(392, tf.int32),
                    tf.constant([7, 8]), mode, 'paper_absolute_gaussian', tf.constant(.010), tf.constant(1.))
                assert all(np.isfinite(g.numpy()).all() for g in partial)
                tf.keras.optimizers.SGD(.05).apply_gradients(zip(a[0], worker.model.trainable_variables))
                modes[mode] = ([g.numpy() for g in a[0]], worker.export())
                worker.load(initial)
            outcomes[target] = modes
            del worker
    comparisons = {}
    for mode in ('none', 'gradient_norm_objective'):
        ga, wa = outcomes['CPU'][mode]; gb, wb = outcomes[device][mode]
        relative = float(np.linalg.norm(np.concatenate([(a-b).ravel() for a, b in zip(ga, gb)])) /
                         max(np.linalg.norm(np.concatenate([a.ravel() for a in ga])), 1e-12))
        update = max(float(np.max(np.abs(a-b))) for a, b in zip(wa, wb))
        if relative > .01 or update > 1e-4: raise ValueError('CPU/device feature objective tolerance failed.')
        comparisons[mode] = dict(gradientRelativeDifference=relative, maxUpdateDifference=update)
    report = dict(status='passed', device=device, tensorflowVersion=tf.__version__,
        comparisons=comparisons, tolerance=dict(gradientRelative=.01, updateAbsolute=1e-4),
        cpuGpuBitIdentityClaim=False, noLambda=True, testedBatchSizes=[512, 392], sourceHashes=fingerprints())
    write_json(proof_path(device), report)
    print(json.dumps(report), flush=True)


def execute(tf, device):
    import numpy as np
    from basil_core.cart_feature_memory import FeatureReplayWorker, owner_refresh_memory
    from basil_core.class_registry import ClassRegistry
    from basil_core.strict_cart import StrictCartWorker
    from basil_core.research_protocol import build_model, initialize_memories, keyed_rng, select_snapshot, params_hash, Snapshot, broadcast
    from basil_core.research_audit import ProtocolAudit
    from basil_core.forgetting_audit import classification_metrics
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    from basil_core.iid_runtime import runtime_metadata
    from scripts.run_strict_cart_diagnostic import expected_candidates
    proof = json.loads(proof_path(device).read_text())
    if proof['status'] != 'passed' or proof['sourceHashes'] != fingerprints():
        raise ValueError('Matching current feature device verification is required.')
    if RESULT.exists() or PLOTS.exists(): raise ValueError('Refusing overwrite/resume of diagnostic evidence.')
    check_host_memory()
    cache = ROOT/'experiments/cache/cifar10'
    images = np.load(cache/'x_train_float32.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(cache/'y_train_int32.npy', mmap_mode='r', allow_pickle=False)
    if images.shape != (50000, 32, 32, 3) or not np.array_equal(np.bincount(labels, minlength=10), [5000]*10):
        raise ValueError('Full one-class training cache required.')
    assignments = [np.flatnonzero(labels == i) for i in range(10)]
    selections = [keyed_rng(2025, 'selection_batch', i).permutation(chunk)[:512] for i, chunk in enumerate(assignments)]
    probes = []
    owner_pools = []
    for i, chunk in enumerate(assignments):
        pool = np.setdiff1d(chunk, selections[i])
        probe = keyed_rng(2025, 'cart_feature_analysis_probe', i).permutation(pool)[:64]
        probes.extend(map(int, probe))
        owner_pools.append(np.setdiff1d(pool, probe))
    probe_indices = np.asarray(probes)
    cfg = config(device)
    RESULT.mkdir(parents=True)
    write_json(RESULT/'config.json', cfg)
    with tf.device('/CPU:0'):
        initial = [w.numpy() for w in build_model('basil_paper_cnn', 2025).trainable_variables]
    digest = hashlib.sha256()
    for chunk in assignments: digest.update(np.asarray(chunk, dtype='<i8').tobytes())
    partition_hash = digest.hexdigest()
    manifest = dict(status='running', config=cfg, **runtime_metadata(tf, {'executionDevice': device}),
        sourceHashes=fingerprints(), initialModelHash=params_hash(initial), partitionHash=partition_hash,
        parameterCount=sum(w.size for w in initial), controlsCompleted=[], activationsCompleted=0,
        optimizerSteps=0, wandbEnabled=False, productionQueueStarted=False)
    write_json(RESULT/'run.json', manifest)
    write_json(RESULT/'partition_audit.json', dict(samplesPerNode=[len(c) for c in assignments],
        nodeClasses=list(range(10)), coverage=int(sum(len(c) for c in assignments)),
        trainingOnly=True, allTrainingStillUsedForSGD=True, analysisProbe=list(map(int, probe_indices)),
        ownerEncodingAndSSExcludeAnalysisProbe=True, partitionHash=partition_hash))
    stop = False
    def request_stop(_sig, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    guard = RuntimeMemoryGuard(tf, device)
    started = time.perf_counter()
    context = {}
    try:
        for condition in cfg['controls']:
            directory = RESULT/condition; directory.mkdir()
            audit = ProtocolAudit(directory/'audit')
            try:
                with tf.device(f'/{device}:0'):
                    model = build_model('basil_paper_cnn', 2025)
                    worker = FeatureReplayWorker(model, beta=1., device=device) if condition == 'cart_feature_memory' else StrictCartWorker(model, device=device)
                    worker.audit = audit
                    memories = initialize_memories(initial)
                    registries = [ClassRegistry() for _ in range(10)]
                    visited = set(); records = []; final_weights = {}
                    def evaluate():
                        logits = []
                        for start in range(0, 640, 512):
                            z = worker.predict(tf.convert_to_tensor(images[probe_indices[start:start+512]]))
                            tf.debugging.assert_all_finite(z, 'Non-finite analysis-only predictions')
                            logits.append(z.numpy())
                        metrics = classification_metrics(np.concatenate(logits), labels[probe_indices])
                        extra = {}
                        if isinstance(worker, FeatureReplayWorker):
                            present = worker.feature_weights.numpy() > 0
                            if present.any():
                                network = getattr(worker.model, 'model', worker.model)
                                z = network.layers[-1](tf.boolean_mask(worker.feature_rows, present), training=False)
                                tf.debugging.assert_all_finite(z, 'Non-finite frozen-feature analysis logits')
                                classes = worker.feature_labels.numpy()[present]
                                row_weights = worker.feature_weights.numpy()[present]
                                logits = z.numpy()
                                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes, logits=z).numpy()
                                extra = dict(frozenFeatureReplayAccuracy=float(np.sum(row_weights*(logits.argmax(axis=1) == classes))),
                                    frozenFeatureReplayLoss=float(np.sum(row_weights*losses)))
                        return dict(trainingAnalysis=metrics, **extra)
                    for node in range(10):
                        context = dict(condition=condition, round=0, node=node)
                        audit.activation(0, node)
                        candidates = memories[node].candidates()
                        if [(s.sender_id, s.round_id) for s in candidates] != expected_candidates(0, node):
                            raise ValueError('Rolling predecessor contract differs.')
                        indices = selections[node]
                        selected, diagnostics = select_snapshot(node, candidates,
                            lambda p: worker.probe(p, images[indices], labels[indices]))
                        worker.load(selected.params)
                        selected_loss = next(d['receiverLocalLoss'] for d in diagnostics
                            if (d['senderId'], d['snapshotRound']) == (selected.sender_id, selected.round_id))
                        selected_hash = params_hash(worker.export())
                        if selected_hash != params_hash(selected.params): raise ValueError('Strict handoff hash mismatch.')
                        kwargs = {}
                        if condition == 'cart_feature_memory':
                            memory = owner_refresh_memory(worker.model, images, labels, owner_pools,
                                receiver=node, round_id=0, seen_classes=visited, seed=2025)
                            kwargs['feature_memory'] = memory
                        params, record = worker.train(selected.params, images[assignments[node]], labels[assignments[node]],
                            seed=2025, round_id=0, node_id=node, epochs=5, batch_size=512, learning_rate=.05,
                            ebm_mode='none', semantics='paper_absolute_gaussian', sigma=0.,
                            local_registry=registries[node], received_registry=ClassRegistry.fromDict(selected.retention_state) if selected.retention_state else ClassRegistry(),
                            selection_x=images[indices], selection_y=labels[indices], selected_loss=selected_loss,
                            evaluation=evaluate, should_stop=lambda: stop, **kwargs)
                        if record['steps'] != 50 or any(e['samplesSeen'] != 5000 or e['batchSizes'] != [512]*9+[392] for e in record['epochs']):
                            raise ValueError('Five-full-epoch contract failed.')
                        record.update(round=0, node=node, selectedSender=selected.sender_id,
                            selectedSnapshotRound=selected.round_id, inputHash=selected_hash,
                            outputHash=params_hash(params), selection=diagnostics)
                        registry = ClassRegistry.fromDict(record['outputRegistry'])
                        broadcast(memories, Snapshot(node, 0, params, retention_state=registry.toDict()))
                        visited.add(node); records.append(record)
                        for j, value in enumerate(params): final_weights[f'node{node}_w{j}'] = value
                        write_json(directory/'activations.json', records)
                        manifest['activationsCompleted'] += 1; manifest['optimizerSteps'] += 50
                        manifest['context'] = context; manifest['memorySafety'] = guard.checkpoint()
                        write_json(RESULT/'run.json', manifest)
                        print(json.dumps(dict(condition=condition, nodeCompleted=node,
                            optimizerSteps=record['steps'], status='finite', elapsedSeconds=time.perf_counter()-started)), flush=True)
                    np.savez_compressed(directory/'final_nodes.npz', **final_weights)
                    manifest['controlsCompleted'].append(condition)
                    del worker, model, memories, final_weights
            finally:
                audit.close()
        manifest.update(status='completed', runtimeSeconds=time.perf_counter()-started,
            numericalFailures=0, oomEvents=0, memorySafety=guard.checkpoint())
    except Exception as error:
        manifest.update(status='interrupted' if isinstance(error, InterruptedError) else 'failed',
            context=context, error=str(error), runtimeSeconds=time.perf_counter()-started)
        write_json(RESULT/'run.json', manifest)
        raise
    write_json(RESULT/'run.json', manifest)
    print(json.dumps(dict(status='completed', result=str(RESULT), runtimeSeconds=manifest['runtimeSeconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify-device', action='store_true')
    action.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute and not args.verify_device:
        print(json.dumps(dict(config=config(args.device), trainingRunsStarted=0), indent=2)); return
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled')
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf, args.device)
    verify(tf, args.device) if args.verify_device else execute(tf, args.device)


if __name__ == '__main__': main()
