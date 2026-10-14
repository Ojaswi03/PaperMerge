#!/usr/bin/env python3
"""Isolated CART representation audit/diagnostic; default is zero training.

No automatic research queue, retries, W&B, test-data access or promotion.
Learning requires explicit fixed gamma OR a predeclared training-only rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT/'newResults/nonIID/cart_feature_memory/clean_one_traversal_beta1_v1'
AUDIT = ROOT/'newResults/nonIID/cart_representation/same_image_saved_v1'
RESULT = ROOT/'newResults/nonIID/cart_representation/clean_one_traversal_fixed_gamma_v1'
PLOTS = ROOT/'newPlots/nonIID/cart_representation/clean_one_traversal_fixed_gamma_v1'


def write_json(path, value):
    from scripts.run_cart_feature_diagnostic import write_json as save
    save(path, value)


def fingerprints():
    from scripts.run_cart_feature_diagnostic import SOURCES
    paths = (*SOURCES, 'basil_core/cart_representation.py', 'scripts/run_cart_representation_diagnostic.py')
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def output_paths(rounds=1):
    if rounds not in {1, 30}: raise ValueError('Only the authorized one-/30-round horizons are supported.')
    name = 'clean_one_traversal_fixed_gamma_v1' if rounds == 1 else 'clean_30_rounds_fixed_gamma_v1'
    return ROOT/f'newResults/nonIID/cart_representation/{name}', ROOT/f'newPlots/nonIID/cart_representation/{name}'


def config(device, gamma=None, ratio=None, rounds=1):
    output_paths(rounds)
    from scripts.run_cart_feature_diagnostic import config as feature_config
    cfg = feature_config(device)
    cfg.update(methodRevision='strict_cart_selected_encoder_preservation_v1',
        purpose='bounded_clean_representation_preservation_diagnostic',
        dataset='cifar10', partitionStrategy='one_class_per_node', snapshotSelection=True,
        controls=['feature_replay', 'feature_replay_preserve'], representationGamma=gamma,
        representationGammaRule='explicit_fixed' if gamma is not None else
            'one_time_training_virtual_CE_step' if ratio is not None else 'unresolved; training blocked',
        representationCalibrationRatio=ratio, representationStudentFrozen=False,
        representationTeacher='exact selected encoder; immutable for all five epochs',
        representationData='same augmented receiver-local batch only',
        sameImageAudit='same 640 TRAIN images at all six stages; no training/control feedback',
        nRounds=rounds, testSetUsed=rounds == 30,
        testUsage='full 10000-image test set, evaluation only after each activation' if rounds == 30 else 'not loaded',
        fullTestAccuracyDefinition='fixed Node-9 honest trained state at end of ring; not ensemble',
        meanNodeAccuracyDefinition='mean across ten trained node states; full-test evaluation',
        representationCalibrationApproval='operator approved training-only 10% rule before learning' if ratio == .1 else 'explicit operator-specified rule/value')
    return cfg


def load_training():
    import numpy as np
    cache = ROOT/'experiments/cache/cifar10'
    images = np.load(cache/'x_train_float32.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(cache/'y_train_int32.npy', mmap_mode='r', allow_pickle=False)
    if images.shape != (50000, 32, 32, 3) or labels.shape != (50000,) or not np.array_equal(np.bincount(labels, minlength=10), [5000]*10):
        raise ValueError('Full read-only one-class training cache required.')
    return images, labels


def saved_audit(tf, device):
    """No SGD: reconstruct each exact selected model and compare final state."""
    import numpy as np
    from basil_core.cart_representation import FrozenSelectedEncoder, same_image_audit
    from basil_core.research_protocol import build_model, params_hash
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    prior = json.loads((SOURCE/'run.json').read_text())
    if prior['status'] != 'completed' or prior['activationsCompleted'] != 20 or prior['optimizerSteps'] != 1000:
        raise ValueError('Complete saved feature-control evidence required.')
    if AUDIT.exists(): raise ValueError('Refusing to overwrite saved same-image audit.')
    check_host_memory()
    images, labels = load_training()
    indices = np.asarray(json.loads((SOURCE/'partition_audit.json').read_text())['analysisProbe'], np.int64)
    if len(indices) != 640 or len(np.unique(indices)) != 640 or np.any((indices < 0) | (indices >= 50000)):
        raise ValueError('Invalid saved training probe.')
    x, y = np.asarray(images[indices]), np.asarray(labels[indices])
    with tf.device('/CPU:0'):
        initial = [w.numpy() for w in build_model('basil_paper_cnn', 2025).trainable_variables]
    if params_hash(initial) != prior['initialModelHash']:
        raise ValueError('Canonical initialization differs from source evidence.')
    AUDIT.mkdir(parents=True)
    guard = RuntimeMemoryGuard(tf, device)
    rows = []
    started = time.perf_counter()
    source_hashes = {}
    try:
        with tf.device(f'/{device}:0'):
            model = build_model('basil_paper_cnn', 2025)
            encoder = FrozenSelectedEncoder(model)
            for condition in ('cart', 'cart_feature_memory'):
                directory = SOURCE/condition
                records = json.loads((directory/'activations.json').read_text())
                for path in (directory/'activations.json', directory/'final_nodes.npz'):
                    source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(directory/'final_nodes.npz', allow_pickle=False) as saved:
                    for record in records:
                        node = record['node']; sender = record['selectedSender']
                        if record['selectedSnapshotRound'] == -1:
                            selected = initial
                        elif record['selectedSnapshotRound'] == 0 and sender < node:
                            selected = [saved[f'node{sender}_w{j}'].copy() for j in range(10)]
                        else: raise ValueError('Saved first-traversal selection cannot be reconstructed exactly.')
                        final = [saved[f'node{node}_w{j}'].copy() for j in range(10)]
                        if params_hash(selected) != record['inputHash'] or params_hash(final) != record['outputHash']:
                            raise ValueError('Reconstructed selected/final parameter hash mismatch.')
                        encoder.freeze(selected)
                        for stage, params in ((0, selected), (5, final)):
                            for w, p in zip(model.trainable_variables, params): w.assign(p)
                            metrics = same_image_audit(model, encoder, x, y, indices, selected)
                            rows.append(dict(condition=condition, node=node, stage=stage, **metrics))
                        guard.checkpoint()
        report = dict(status='completed', sourceRun=str(SOURCE.relative_to(ROOT)),
            sourceManifestHash=hashlib.sha256((SOURCE/'run.json').read_bytes()).hexdigest(),
            sourceArtifactHashes=source_hashes, sourceHashes=fingerprints(),
            stagesAvailable=[0, 5], intermediateEpochsUnavailable=True,
            trainingRunsStarted=0, optimizerSteps=0, testSetUsed=False,
            device=device, tensorflowVersion=tf.__version__, records=rows,
            runtimeSeconds=time.perf_counter()-started, memorySafety=guard.checkpoint())
        write_json(AUDIT/'audit.json', report)
        print(json.dumps(dict(status='completed', observations=len(rows), trainingRunsStarted=0,
            result=str(AUDIT/'audit.json'))), flush=True)
    except Exception as error:
        write_json(AUDIT/'failure.json', dict(status='failed', error=str(error), trainingRunsStarted=0))
        raise


def proof_path(device):
    return ROOT/f'gui/worker_state/CART_REPRESENTATION_{device}_VERIFICATION.json'


def verify(tf, device):
    """Synthetic full/partial batch, source EBM, frozen teacher and no-lambda."""
    import numpy as np
    from basil_core.cart_representation import RepresentationReplayWorker
    from basil_core.cart_feature_memory import ActivationFeatureMemory, packet_from_features, encoder_hash
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
            worker = RepresentationReplayWorker(build_model('basil_paper_cnn', 2025), beta=1., gamma=.01, device=target)
            worker.load(initial); worker.set_memory(memory)
            worker.selected_encoder.freeze(initial); worker.anchor_mu.assign(.007)
            for ref, value in zip(worker.reference, initial): ref.assign(value)
            for w, value in zip(worker.model.trainable_variables, initial): w.assign(value+.001)
            shifted = worker.export()
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
                worker.selected_encoder.assert_frozen()
                modes[mode] = ([g.numpy() for g in a[0]], worker.export())
                worker.load(shifted)
            outcomes[target] = modes
            del worker
    comparisons = {}
    for mode in ('none', 'gradient_norm_objective'):
        ga, wa = outcomes['CPU'][mode]; gb, wb = outcomes[device][mode]
        relative = float(np.linalg.norm(np.concatenate([(a-b).ravel() for a, b in zip(ga, gb)])) /
            max(np.linalg.norm(np.concatenate([a.ravel() for a in ga])), 1e-12))
        update = max(float(np.max(np.abs(a-b))) for a, b in zip(wa, wb))
        if relative > .01 or update > 1e-4: raise ValueError('CPU/device representation objective tolerance failed.')
        comparisons[mode] = dict(gradientRelativeDifference=relative, maxUpdateDifference=update)
    report = dict(status='passed', device=device, tensorflowVersion=tf.__version__,
        comparisons=comparisons, tolerance=dict(gradientRelative=.01, updateAbsolute=1e-4),
        cpuGpuBitIdentityClaim=False, multiprocessingVerified=False, teacherFrozen=True,
        noLambda=True, gamma=.01, testedBatchSizes=[512, 392], sourceHashes=fingerprints())
    write_json(proof_path(device), report)
    print(json.dumps(report), flush=True)


def execute(tf, device, gamma, ratio, rounds=1):
    result, plots = output_paths(rounds)
    import numpy as np
    from basil_core.cart_representation import (RepresentationReplayWorker, FrozenSelectedEncoder,
        same_image_audit, calibrate_gamma)
    from basil_core.cart_feature_memory import FeatureReplayWorker, owner_refresh_memory
    from basil_core.class_registry import ClassRegistry
    from basil_core.research_protocol import (build_model, initialize_memories, keyed_rng,
        select_snapshot, params_hash, Snapshot, broadcast, basil_round_learning_rate)
    from basil_core.research_audit import ProtocolAudit, augment_batch
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    from basil_core.iid_runtime import runtime_metadata
    from scripts.run_strict_cart_diagnostic import expected_candidates
    if (gamma is None) == (ratio is None): raise ValueError('Specify fixed gamma OR explicit training-only calibration.')
    if gamma is not None and (not np.isfinite(gamma) or gamma <= 0): raise ValueError('Learning requires positive fixed gamma.')
    proof = json.loads(proof_path(device).read_text())
    if proof['status'] != 'passed' or proof['sourceHashes'] != fingerprints():
        raise ValueError('Matching current representation device proof required.')
    if result.exists() or plots.exists(): raise ValueError('Refusing overwrite/resume/retry of diagnostic evidence.')
    preflight = None
    if rounds == 30:
        preflight = json.loads((RESULT/'run.json').read_text())
        if preflight['status'] != 'completed' or preflight['activationsCompleted'] != 20 or preflight['optimizerSteps'] != 1000:
            raise ValueError('Finite, completed one-round execution preflight required; no accuracy threshold.')
        if preflight['sourceHashes'] != fingerprints() or preflight.get('numericalFailures') != 0 or preflight.get('oomEvents') != 0:
            raise ValueError('Current-source numerical/memory preflight required.')
    check_host_memory()
    images, labels = load_training()
    assignments = [np.flatnonzero(labels == i) for i in range(10)]
    selections = [keyed_rng(2025, 'selection_batch', i).permutation(chunk)[:512] for i, chunk in enumerate(assignments)]
    probes, owner_pools = [], []
    for i, chunk in enumerate(assignments):
        pool = np.setdiff1d(chunk, selections[i])
        probe = keyed_rng(2025, 'cart_feature_analysis_probe', i).permutation(pool)[:64]
        probes.extend(map(int, probe)); owner_pools.append(np.setdiff1d(pool, probe))
    probe_indices = np.asarray(probes, np.int64)
    probe_x, probe_y = np.asarray(images[probe_indices]), np.asarray(labels[probe_indices])
    test_x = test_y = None
    if rounds == 30:
        cache = ROOT/'experiments/cache/cifar10'
        test_x = np.load(cache/'x_test_float32.npy', mmap_mode='r', allow_pickle=False)
        test_y = np.load(cache/'y_test_int32.npy', mmap_mode='r', allow_pickle=False)
        if test_x.shape != (10000, 32, 32, 3) or not np.array_equal(np.bincount(test_y, minlength=10), [1000]*10):
            raise ValueError('Full balanced 10000-image evaluation-only test cache required.')
    with tf.device('/CPU:0'):
        initial = [w.numpy() for w in build_model('basil_paper_cnn', 2025).trainable_variables]
    calibration = None
    if ratio is not None:
        with tf.device(f'/{device}:0'):
            model = build_model('basil_paper_cnn', 2025)
            for w, p in zip(model.trainable_variables, initial): w.assign(p)
            selected_encoder = FrozenSelectedEncoder(model); selected_encoder.freeze(initial)
            # A disjoint training-only calibration batch, never the analysis probe.
            calibration_indices = keyed_rng(2025, 'cart_representation_calibration', 0).permutation(owner_pools[0])[:512]
            key = keyed_rng(2025, 'cart_representation_calibration_augmentation').integers(0, 2**30, size=2, dtype=np.int32)
            with tf.device('/CPU:0'): calibration_x = augment_batch(images[calibration_indices], key)
            calibration = calibrate_gamma(model, selected_encoder, calibration_x,
                tf.convert_to_tensor(labels[calibration_indices], tf.int32), target_ratio=ratio)
            calibration.update(trainingIndices=calibration_indices.tolist(),
                analysisProbeExcluded=True, snapshotSelectionBatchExcluded=True,
                augmentationKey=key.tolist())
            gamma = calibration['gamma']
            del model, selected_encoder, calibration_x
    if preflight is not None and (preflight['config']['representationGamma'] != gamma or
        preflight['initialModelHash'] != params_hash(initial) or preflight['config']['representationCalibrationRatio'] != ratio):
        raise ValueError('30-round coefficient/initialization must exactly match preflight.')
    cfg = config(device, gamma if ratio is None else None, ratio, rounds)
    cfg['representationGamma'] = gamma
    result.mkdir(parents=True)
    write_json(result/'config.json', cfg)
    digest = hashlib.sha256()
    for chunk in assignments: digest.update(np.asarray(chunk, dtype='<i8').tobytes())
    manifest = dict(status='running', config=cfg, **runtime_metadata(tf, {'executionDevice': device}),
        sourceHashes=fingerprints(), initialModelHash=params_hash(initial), partitionHash=digest.hexdigest(),
        parameterCount=sum(w.size for w in initial), controlsCompleted=[], activationsCompleted=0,
        optimizerSteps=0, calibration=calibration, wandbEnabled=False, productionQueueStarted=False,
        roundsCompletedByControl={c: 0 for c in cfg['controls']},
        executionPreflightHash=hashlib.sha256((RESULT/'run.json').read_bytes()).hexdigest() if preflight else None)
    write_json(result/'run.json', manifest)
    write_json(result/'partition_audit.json', dict(samplesPerNode=[len(c) for c in assignments],
        nodeClasses=list(range(10)), coverage=50000, trainingOnly=True, allTrainingStillUsedForSGD=True,
        analysisProbe=probe_indices.tolist(), ownerEncodingAndSSExcludeAnalysisProbe=True,
        partitionHash=digest.hexdigest()))
    stop = False
    def request_stop(_sig, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    guard = RuntimeMemoryGuard(tf, device)
    started = time.perf_counter(); context = {}
    try:
        for condition in cfg['controls']:
            directory = result/condition; directory.mkdir()
            audit = ProtocolAudit(directory/'audit')
            try:
                with tf.device(f'/{device}:0'):
                    model = build_model('basil_paper_cnn', 2025)
                    if condition == 'feature_replay_preserve':
                        worker = RepresentationReplayWorker(model, beta=1., gamma=gamma, device=device)
                        encoder = worker.selected_encoder
                    else:
                        worker = FeatureReplayWorker(model, beta=1., device=device)
                        encoder = FrozenSelectedEncoder(model)
                    worker.audit = audit
                    memories = initialize_memories(initial)
                    registries = [ClassRegistry() for _ in range(10)]
                    visited = set(); records = []; final_weights = {}; selected_params = None
                    def evaluate():
                        try:
                            metrics = same_image_audit(model, encoder, probe_x, probe_y,
                                probe_indices, selected_params)
                        except Exception as error:
                            audit.evaluation_failure(worker, probe_x, probe_y, 0, error)
                            raise
                        audit.last_confusion = np.asarray(metrics['interventions']['current_encoder_current_head']['confusion'])
                        audit.evaluation()
                        return dict(sameImageAudit=metrics,
                            trainingAnalysis=metrics['interventions']['current_encoder_current_head'])
                    def evaluate_test():
                        from basil_core.forgetting_audit import classification_metrics
                        logits = []
                        for start in range(0, 10000, 512):
                            batch = test_x[start:start+512]
                            try:
                                z = worker.predict(tf.convert_to_tensor(batch))
                                tf.debugging.assert_all_finite(z, 'Non-finite evaluation-only global test logits')
                            except Exception as error:
                                audit.evaluation_failure(worker, batch, test_y[start:start+512], start, error)
                                raise
                            logits.append(z.numpy())
                        return classification_metrics(np.concatenate(logits), np.asarray(test_y))
                    for activation in range(rounds*10):
                        round_id, node = divmod(activation, 10)
                        context = dict(condition=condition, round=round_id, node=node)
                        audit.activation(round_id, node)
                        candidates = memories[node].candidates()
                        if [(s.sender_id, s.round_id) for s in candidates] != expected_candidates(round_id, node):
                            raise ValueError('Rolling predecessor contract differs.')
                        ss = selections[node]
                        selected, diagnostics = select_snapshot(node, candidates,
                            lambda p: worker.probe(p, images[ss], labels[ss]))
                        worker.load(selected.params)
                        selected_params = selected.params
                        encoder.freeze(selected_params)
                        selected_loss = next(d['receiverLocalLoss'] for d in diagnostics
                            if (d['senderId'], d['snapshotRound']) == (selected.sender_id, selected.round_id))
                        selected_hash = params_hash(worker.export())
                        if selected_hash != params_hash(selected.params): raise ValueError('Strict handoff hash mismatch.')
                        memory = owner_refresh_memory(model, images, labels, owner_pools,
                            receiver=node, round_id=round_id, seen_classes=visited, seed=2025)
                        params, record = worker.train(selected.params, images[assignments[node]], labels[assignments[node]],
                            feature_memory=memory, seed=2025, round_id=round_id, node_id=node,
                            epochs=5, batch_size=512, learning_rate=basil_round_learning_rate(.05, round_id), ebm_mode='none',
                            semantics='paper_absolute_gaussian', sigma=0., local_registry=registries[node],
                            received_registry=ClassRegistry.fromDict(selected.retention_state) if selected.retention_state else ClassRegistry(),
                            selection_x=images[ss], selection_y=labels[ss], selected_loss=selected_loss,
                            evaluation=evaluate, should_stop=lambda: stop)
                        encoder.assert_frozen()
                        if record['steps'] != 50 or any(e['samplesSeen'] != 5000 or e['batchSizes'] != [512]*9+[392] for e in record['epochs']):
                            raise ValueError('Five-full-epoch contract failed.')
                        if test_x is not None:
                            record['globalTestEvaluationOnly'] = evaluate_test()
                            if params_hash(worker.export()) != params_hash(params):
                                raise ValueError('Evaluation changed trained weights.')
                        record.update(round=round_id, node=node, selectedSender=selected.sender_id,
                            selectedSnapshotRound=selected.round_id, inputHash=selected_hash,
                            outputHash=params_hash(params), selection=diagnostics,
                            learningRate=basil_round_learning_rate(.05, round_id))
                        registry = ClassRegistry.fromDict(record['outputRegistry'])
                        broadcast(memories, Snapshot(node, round_id, params, retention_state=registry.toDict()))
                        visited.add(node); records.append(record)
                        for j, value in enumerate(params): final_weights[f'node{node}_w{j}'] = value
                        write_json(directory/'activations.json', records)
                        manifest['activationsCompleted'] += 1; manifest['optimizerSteps'] += 50
                        manifest['context'] = context; manifest['memorySafety'] = guard.checkpoint()
                        if node == 9:
                            manifest['roundsCompletedByControl'][condition] = round_id+1
                            if test_x is not None:
                                last = records[-10:]
                                full = last[-1]['globalTestEvaluationOnly']
                                values = [r['globalTestEvaluationOnly']['accuracy'] for r in last]
                                row = dict(round=round_id, full_test_accuracy=full['accuracy'],
                                    mean_node_accuracy=float(np.mean(values)), worst_node_accuracy=min(values),
                                    full_test_loss=full['crossEntropy'],
                                    **{f'class_{c}_accuracy': a for c, a in enumerate(full['perClassAccuracy'])})
                                path = directory/'round_metrics.csv'
                                existed = path.exists()
                                with path.open('a', newline='') as handle:
                                    writer = csv.DictWriter(handle, fieldnames=list(row))
                                    if not existed: writer.writeheader()
                                    writer.writerow(row)
                            # Evidence checkpoint retains every latest predecessor
                            # and logical registry. No unverified automatic resume.
                            payload = dict(final_weights)
                            metadata = []
                            for receiver, rolling in enumerate(memories):
                                for snapshot in rolling.candidates():
                                    key = f'r{receiver}s{snapshot.sender_id}'
                                    payload.update({f'{key}w{j}': p for j, p in enumerate(snapshot.params)})
                                    metadata.append(dict(receiver=receiver, sender=snapshot.sender_id,
                                        round=snapshot.round_id, hash=params_hash(snapshot.params),
                                        registry=snapshot.retention_state))
                            payload['metadata'] = json.dumps(dict(completedRounds=round_id+1,
                                snapshots=metadata, registries=[r.toDict() for r in registries],
                                visitedClasses=sorted(visited), initialModelHash=params_hash(initial),
                                sourceHashes=fingerprints(), config=cfg))
                            temporary = directory/'round_checkpoint.tmp.npz'
                            np.savez_compressed(temporary, **payload)
                            temporary.replace(directory/'round_checkpoint.npz')
                            del payload, metadata
                        write_json(result/'run.json', manifest)
                        print(json.dumps(dict(condition=condition, round=round_id, nodeCompleted=node,
                            optimizerSteps=50, status='finite', elapsedSeconds=time.perf_counter()-started)), flush=True)
                    np.savez_compressed(directory/'final_nodes.npz', **final_weights)
                    manifest['controlsCompleted'].append(condition)
                    del worker, encoder, model, memories, final_weights
            finally: audit.close()
        manifest.update(status='completed', runtimeSeconds=time.perf_counter()-started,
            numericalFailures=0, oomEvents=0, memorySafety=guard.checkpoint())
    except Exception as error:
        manifest.update(status='interrupted' if isinstance(error, InterruptedError) else 'failed',
            context=context, error=str(error), runtimeSeconds=time.perf_counter()-started)
        write_json(result/'run.json', manifest)
        raise
    write_json(result/'run.json', manifest)
    print(json.dumps(dict(status='completed', result=str(result), runtimeSeconds=manifest['runtimeSeconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--rounds', type=int, choices=[1, 30], default=1)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--audit-saved', action='store_true')
    action.add_argument('--verify-device', action='store_true')
    action.add_argument('--execute', action='store_true')
    strength = parser.add_mutually_exclusive_group()
    strength.add_argument('--gamma', type=float)
    strength.add_argument('--calibrate-ratio', type=float)
    args = parser.parse_args()
    if not any((args.execute, args.verify_device, args.audit_saved)):
        print(json.dumps(dict(config=config(args.device, args.gamma, args.calibrate_ratio, args.rounds),
            trainingRunsStarted=0, nextStep='same-image saved audit; explicit gamma required for learning'), indent=2)); return
    if args.execute and args.gamma is None and args.calibrate_ratio is None:
        parser.error('Learning requires --gamma OR a predeclared --calibrate-ratio; no inferred coefficient.')
    os.environ.update(PAPERMERGE_LIVE_TRACKING='disabled', WANDB_MODE='disabled')
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf, args.device)
    if args.audit_saved: saved_audit(tf, args.device)
    elif args.verify_device: verify(tf, args.device)
    else: execute(tf, args.device, args.gamma, args.calibrate_ratio, args.rounds)


if __name__ == '__main__': main()
