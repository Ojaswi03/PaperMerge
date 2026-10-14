#!/usr/bin/env python3
"""Three bounded warmed activations, not a research continuation or new method.

One Node-0 activation per preserved clean CE/fixed-anchor/CART checkpoint.
No global test access, W&B, attacks, channel noise or queue changes.
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
SOURCE = ROOT/'newResults/nonIID/strict_cart/clean_full_data_three_round_v1'
RESULT = ROOT/'newResults/nonIID/forgetting_audit/warmed_node0_head_backbone_v1'
PLOTS = ROOT/'newPlots/nonIID/forgetting_audit/warmed_node0_head_backbone_v1'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def diagnostic_config(device):
    return dict(purpose='warmed_single_activation_mechanism_audit', researchValid=False,
        revision='head_backbone_training_probe_v1', executionDevice=device,
        controls=['ce', 'fixed_anchor', 'cart'], activationsPerControl=1, roundIndex=3, nodeId=0,
        localEpochs=5, trainingSamplesPerNode=5000, batchSize=512, optimizerStepsPerControl=50,
        seed=2025, learningRateSchedule='basil_round_decay', initialLearningRate=.05,
        momentum=0., weightDecayCoefficient=0., gradientClipNorm=None,
        aggregationMode='strict_sequential_handoff', basilMemorySize=5,
        actualAttackers=[], channelNoiseEnabled=False, ebmEnabled=False,
        liveTrackingEnabled=False, fixedAnchorMu=.01, cartMuMax=.01,
        analysisProbePerClass=64, selectionProbeSize=512,
        probeUsage='centralized analysis-only TRAINING probe, not held-out validation; excludes Node-0 SS batch',
        testSetUsed=False, checkpointScope='audit evidence only; not exact production resume')


def audit_indices(labels, selection_indices, seed=2025):
    import numpy as np
    from basil_core.research_protocol import keyed_rng
    labels = np.asarray(labels)
    excluded = set(map(int, selection_indices))
    chosen = []
    for c in range(10):
        pool = np.array([i for i in np.flatnonzero(labels == c) if int(i) not in excluded], np.int64)
        if len(pool) < 64:
            raise ValueError('Insufficient training-only audit support.')
        chosen.extend(keyed_rng(seed, 'head_backbone_audit_probe', c).permutation(pool)[:64])
    result = np.asarray(chosen, np.int64)
    if len(np.unique(result)) != 640 or set(map(int, result)) & excluded:
        raise ValueError('Audit probe overlaps SS or repeats examples.')
    return result


def read_candidates(directory):
    import numpy as np
    from basil_core.research_protocol import Snapshot, params_hash
    with np.load(directory/'snapshot_memory.npz', allow_pickle=False) as payload:
        metadata = json.loads(str(payload['metadata']))
        candidates = []
        for entry in metadata:
            if entry['receiver'] != 0:
                continue
            weights = [payload[f"r0s{entry['sender']}w{j}"].copy() for j in range(10)]
            if params_hash(weights) != entry['hash']:
                raise ValueError('Saved snapshot checksum differs.')
            candidates.append(Snapshot(entry['sender'], entry['round'], weights, retention_state=entry['registry']))
    if [(c.sender_id, c.round_id) for c in candidates] != [(s, 2) for s in range(5, 10)]:
        raise ValueError('Expected five distinct warmed predecessors 5:R2 through 9:R2.')
    return candidates


def execute(tf, device):
    import numpy as np
    from basil_core.class_registry import ClassRegistry
    from basil_core.forgetting_audit import (classification_metrics, parameter_drift,
        representation_metrics, split_parameters, INTERVENTIONS)
    from basil_core.iid_runtime import runtime_metadata
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    from basil_core.noniid_study import SnapshotAnchorWorker
    from basil_core.research_audit import ProtocolAudit, array_hash
    from basil_core.research_protocol import build_model, keyed_rng, select_snapshot, params_hash, basil_round_learning_rate
    from basil_core.strict_cart import StrictCartWorker
    from scripts.run_strict_cart_diagnostic import source_hashes, verification_path
    # Reject stale synthetic device verification before doing any training.
    proof = json.loads(verification_path(device).read_text())
    if proof['status'] != 'passed' or proof.get('sourceHashes') != source_hashes():
        raise ValueError('Current CE/EBM/anchor device verification required.')
    if RESULT.exists() or PLOTS.exists():
        raise ValueError('Refusing to overwrite mechanism-audit evidence.')
    prior = json.loads((SOURCE/'run.json').read_text())
    if prior['status'] != 'completed' or prior['completedActivations'] != 90:
        raise ValueError('Prior completed clean evidence required.')
    check_host_memory()
    RESULT.mkdir(parents=True)
    cfg = diagnostic_config(device)
    write_json(RESULT/'config.json', cfg)
    manifest = dict(status='initializing', config=cfg, **runtime_metadata(tf, {'executionDevice': device}),
        sourceRun=str(SOURCE.relative_to(ROOT)), sourceRunSHA256=hashlib.sha256((SOURCE/'run.json').read_bytes()).hexdigest(),
        sourceFilesSHA256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('scripts/run_forgetting_mechanism_audit.py', 'basil_core/forgetting_audit.py', *source_hashes().keys())},
        controlsCompleted=[], optimizerSteps=0, productionQueueStarted=False, wandbEnabled=False)
    started = time.perf_counter()
    stop = False
    def request_stop(_sig, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    guard = RuntimeMemoryGuard(tf, device)
    try:
        # Read-only existing normalized training cache. Do not load/download test data.
        cache = ROOT/'experiments/cache/cifar10'
        images = np.load(cache/'x_train_float32.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(cache/'y_train_int32.npy', mmap_mode='r', allow_pickle=False)
        if images.shape != (50000, 32, 32, 3) or not np.array_equal(np.bincount(labels, minlength=10), [5000]*10):
            raise ValueError('Full CIFAR training cache required.')
        local = np.flatnonzero(labels == 0)
        selection = local[keyed_rng(2025, 'selection_batch', 0).permutation(5000)[:512]]
        probe = audit_indices(labels, selection)
        x, y = np.asarray(images[probe]), np.asarray(labels[probe])
        write_json(RESULT/'probe.json', dict(indices=probe.tolist(), selectionIndices=selection.tolist(),
            classSupport=np.bincount(y, minlength=10).tolist(), imagesHash=array_hash(x), labelsHash=array_hash(y),
            warning=cfg['probeUsage']))
        manifest['status'] = 'running'
        write_json(RESULT/'run.json', manifest)
        for name in cfg['controls']:
            directory = RESULT/name
            directory.mkdir()
            candidates = read_candidates(SOURCE/name)
            old = json.loads((SOURCE/name/'activations.json').read_text())
            with tf.device(f'/{device}:0'):
                model = build_model('basil_paper_cnn', 2025)
                worker = (StrictCartWorker(model, mu_max=.01, device=device) if name == 'cart' else
                          SnapshotAnchorWorker(model, mu=.01 if name == 'fixed_anchor' else 0., device=device))
                # Shape pin prevents silently moving the split to another architecture.
                if [tuple(w.shape) for w in model.trainable_variables[-2:]] != [(192, 10), (10,)]:
                    raise ValueError('Expected approved 117706-parameter CNN head.')
                feature_model = tf.keras.Model(model.model.inputs, model.model.layers[-2].output)
                feature_predict = tf.function(lambda batch: feature_model(batch, training=False), reduce_retracing=True)
                def predict_features():
                    result = np.concatenate([feature_predict(tf.convert_to_tensor(x[i:i+128])).numpy()
                                             for i in range(0, len(y), 128)])
                    if not np.isfinite(result).all():
                        raise FloatingPointError('Non-finite audit features.')
                    return result
                selected, diagnostics = select_snapshot(0, candidates,
                    lambda p: worker.probe(p, images[selection], labels[selection]))
                reference = [v.copy() for v in selected.params]
                worker.load(reference)
                reference_features = predict_features()
                stages = []
                def evaluation():
                    current = worker.export()
                    current_hash = params_hash(current)
                    stage = dict(stage=len(stages), parameterHash=current_hash,
                        parameterDrift=parameter_drift(current, reference),
                        representation=representation_metrics(predict_features(), reference_features, y), interventions={})
                    try:
                        for intervention in INTERVENTIONS:
                            worker.load(split_parameters(current, reference, intervention))
                            logits = np.concatenate([worker.predict(tf.convert_to_tensor(x[i:i+128])).numpy()
                                                     for i in range(0, len(y), 128)])
                            stage['interventions'][intervention] = classification_metrics(logits, y)
                            if intervention == 'trained':
                                # The CE signs on the active node's class, not a balanced objective.
                                stage['ownClassTrainingProbe'] = classification_metrics(logits[y == 0], y[y == 0])
                    finally:
                        worker.load(current)
                    if params_hash(worker.export()) != current_hash:
                        raise ValueError('Analysis-only intervention changed live weights.')
                    np.savez_compressed(directory/f'epoch_{len(stages)}_params.npz',
                                        **{f'w{j}': v for j, v in enumerate(current)})
                    stages.append(stage)
                    write_json(directory/'stages.json', stages)
                    print(json.dumps(dict(condition=name, stage=stage['stage'],
                        trainingProbeAccuracy=stage['interventions']['trained']['accuracy'])), flush=True)
                    return dict(auditTrainingProbeAccuracy=stage['interventions']['trained']['accuracy'])
                audit = ProtocolAudit(directory/'audit')
                worker.audit = audit
                audit.activation(3, 0)
                kwargs = dict(seed=2025, round_id=3, node_id=0, epochs=5, batch_size=512,
                    learning_rate=basil_round_learning_rate(.05, 3), ebm_mode='none',
                    semantics='paper_absolute_gaussian', sigma=0., evaluation=evaluation, should_stop=lambda: stop)
                if name == 'cart':
                    registry = ClassRegistry.fromDict(old[20]['registryAfter'])
                    kwargs.update(local_registry=registry,
                        received_registry=ClassRegistry.fromDict(selected.retention_state),
                        selection_x=images[selection], selection_y=labels[selection],
                        selected_loss=next(d['receiverLocalLoss'] for d in diagnostics if d['senderId'] == selected.sender_id))
                try:
                    _, record = worker.train(reference, images[local], labels[local], **kwargs)
                finally:
                    audit.close()
                if record['steps'] != 50 or len(stages) != 6 or any(
                    e['samplesSeen'] != 5000 or e['batchSizes'] != [512]*9+[392] for e in record['epochs']):
                    raise ValueError('Five full epochs or partial-batch contract failed.')
                record.update(selectedSender=selected.sender_id, selectedRound=selected.round_id,
                    selectedHash=params_hash(reference), candidates=diagnostics, round=3, node=0)
                write_json(directory/'activation.json', record)
                manifest['controlsCompleted'].append(name)
                manifest['optimizerSteps'] += record['steps']
                manifest.update(memorySafety=guard.checkpoint(), runtimeSeconds=time.perf_counter()-started)
                write_json(RESULT/'run.json', manifest)
                del worker, model, feature_model, feature_predict
        manifest.update(status='completed', runtimeSeconds=time.perf_counter()-started, memorySafety=guard.checkpoint())
        write_json(RESULT/'run.json', manifest)
    except Exception as error:
        manifest.update(status='stopped' if isinstance(error, InterruptedError) else 'failed', error=str(error),
            runtimeSeconds=time.perf_counter()-started)
        write_json(RESULT/'run.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--execute', action='store_true', help='Explicitly authorize the three bounded activations.')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(diagnostic_config(args.device), indent=2))
        return
    os.environ.update(WANDB_MODE='disabled', PAPERMERGE_LIVE_TRACKING='disabled')
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    prepare_environment(args.device)
    import tensorflow as tf
    configure_tensorflow(tf, args.device)
    execute(tf, args.device)


if __name__ == '__main__':
    main()
