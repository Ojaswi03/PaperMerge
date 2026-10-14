#!/usr/bin/env python3
"""Print a zero-training CART integration/readiness plan; never executes a queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from basil_core.cart_owner_protocol import integration_plan, FULL_NODE, TRUSTED_AUXILIARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper-scope', choices=(FULL_NODE, TRUSTED_AUXILIARY), default=FULL_NODE)
    parser.add_argument('--verify-device', choices=('CPU', 'GPU'), help='Synthetic derivative/single-update proof only; no CIFAR trajectory.')
    args = parser.parse_args()
    print(json.dumps(verify_device(args.verify_device) if args.verify_device else integration_plan(args.helper_scope), indent=2))


def verify_device(device):
    """Synthetic 512/392 batches, nine helpers; CPU/GPU and no-lambda checks."""
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow, runtime_metadata
    from basil_core.cart_owner_protocol import source_hashes, FULL_NODE
    root = Path(__file__).resolve().parents[1]
    path = root/f'gui/worker_state/CART_OWNER_INTEGRATION_{device}_VERIFICATION.json'
    if path.exists():
        raise ValueError('Refusing to replace an existing device proof.')
    prepare_environment(device)
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf, device)
    from basil_core.research_protocol import build_model, params_hash
    from basil_core.cart_owner_ebm import make_owner_ebm_worker
    from basil_core.research_audit import augment_batch
    runtime = runtime_metadata(tf, {'executionDevice': device})
    with tf.device('/CPU:0'):
        initial = [w.numpy() for w in build_model('basil_paper_cnn', 2025).trainable_variables]
    assert sum(p.size for p in initial) == 117706
    rng = np.random.default_rng(8)
    helper_x = rng.uniform(0, 1, (144, 32, 32, 3)).astype(np.float32)
    helper_y = np.repeat(np.arange(1, 10, dtype=np.int32), 16)
    local_x = rng.uniform(0, 1, (512, 32, 32, 3)).astype(np.float32)
    results = {}
    updates = 0
    for target in dict.fromkeys(('CPU', device)):
        results[target] = {}
        with tf.device(f'/{target}:0'):
            worker = make_owner_ebm_worker(build_model('basil_paper_cnn', 2025),
                build_model('basil_paper_cnn', 2025), helper_scope=FULL_NODE, device=target)
            worker.source_x, worker.source_y = helper_x, helper_y
            worker.owner_pools = [np.flatnonzero(helper_y == c) for c in range(10)]
            worker.older_classes = tuple(range(1, 10))
            worker.receiver, worker.seed, worker.round_id = 0, 2025, 1
            worker.anchor_mu.assign(.007)
            for a, p in zip(worker.reference, initial):
                a.assign(p+.001)
            frozen = params_hash([a.numpy() for a in worker.reference])
            for size in (512, 392):
                for mode, sigma, label in (('none', 0., 'clean_ce'), ('none', .01, 'noisy_ce'),
                                           ('gradient_norm_objective', .01, 'noisy_ebm')):
                    worker.load(initial)
                    worker.owner_records = []
                    x, y, key = tf.constant(local_x[:size]), tf.zeros(size, tf.int32), tf.constant([7, 8])
                    actual, values = worker._gradients(x, y, key, mode, 'paper_absolute_gaussian',
                                                     tf.constant(sigma), tf.constant(1.))
                    trace = worker.owner_records[0]
                    with tf.device('/CPU:0'):
                        batches = [(augment_batch(x, key), y)]
                        for owner in trace['owners']:
                            ids = owner['trainingIndices']
                            batches.append((augment_batch(helper_x[ids], np.asarray(owner['augmentationKey'], np.int32)),
                                            tf.constant(helper_y[ids])))
                    with tf.GradientTape() as outer:
                        with tf.GradientTape() as inner:
                            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=y, logits=worker.model(batches[0][0])))
                        base = inner.gradient(ce, worker.model.trainable_variables)
                        terms = [ce]+[tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels, logits=worker.model(images))) for images, labels in batches[1:]]
                        coefficient = .0001 if mode == 'gradient_norm_objective' else 0.
                        loss = tf.add_n(terms)/10+coefficient*tf.add_n([tf.reduce_sum(g*g) for g in base])
                        loss += worker.anchor_mu/2*tf.add_n([tf.reduce_sum((w-a)**2)
                            for w, a in zip(worker.model.trainable_variables, worker.reference)])
                    expected = outer.gradient(loss, worker.model.trainable_variables)
                    for a, b in zip(actual, expected):
                        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-3, atol=2e-6)
                    np.testing.assert_allclose(float(values[2]), float(loss), rtol=1e-6)
                    worker.owner_records = []
                    repeat, repeated = worker._gradients(x, y, key, mode, 'paper_absolute_gaussian',
                                                        tf.constant(sigma), tf.constant(999999.))
                    for a, b in zip(actual, repeat):
                        np.testing.assert_array_equal(a.numpy(), b.numpy())
                    np.testing.assert_array_equal([float(v) for v in values], [float(v) for v in repeated])
                    tf.keras.optimizers.SGD(.05, momentum=0.).apply_gradients(zip(actual, worker.model.trainable_variables))
                    updates += 1
                    assert params_hash([a.numpy() for a in worker.reference]) == frozen
                    results[target][f'{size}_{label}'] = dict(gradients=[g.numpy() for g in actual],
                        weights=worker.export(), loss=float(loss), coefficient=float(values[3]),
                        correctionNorm=float(values[6]))
            del worker
    comparisons = {}
    if device == 'GPU':
        for key, cpu in results['CPU'].items():
            gpu = results['GPU'][key]
            a = np.concatenate([g.ravel().astype(np.float64) for g in cpu['gradients']])
            b = np.concatenate([g.ravel().astype(np.float64) for g in gpu['gradients']])
            relative = float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-12))
            update = max(float(np.max(np.abs(x-y))) for x, y in zip(cpu['weights'], gpu['weights']))
            assert relative <= .01 and update <= .0001
            comparisons[key] = dict(relativeGradientDifference=relative, maximumUpdateDifference=update)
    proof = dict(status='passed', device=device, runtime=runtime, sourceHashes=source_hashes(),
        scope='synthetic actual worker, ten-class CE/local EBM/anchor, 512/392 batch and single SGD updates',
        researchRunsStarted=0, syntheticOptimizerUpdates=updates, fullLocalEpochsExecuted=0,
        legacyLambdaIgnored=True, helperOptimizerSteps=0, helperClasses=list(range(1, 10)),
        parameterCount=117706, comparisons=comparisons,
        tolerance=dict(relativeGradient=.01, maximumUpdate=.0001),
        numericalPolicy='deterministic float32; TF32 off on GPU; 6144 MiB GPU cap')
    if device == 'GPU':
        proof['gpuMemory'] = tf.config.experimental.get_memory_info('GPU:0')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proof, indent=2)+'\n')
    return proof


if __name__ == '__main__':
    main()
