#!/usr/bin/env python3
"""Synthetic fixed-anchor CPU/GPU/no-lambda check; no CIFAR training."""
from pathlib import Path
import json
import hashlib
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core.iid_runtime import prepare_environment, configure_tensorflow, GPU_MEMORY_LIMIT_MB
from basil_core.noniid_study import SnapshotAnchorWorker
from basil_core.research_protocol import build_model, params_hash


def main():
    prepare_environment('GPU')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf, 'GPU')
    with tf.device('/CPU:0'):
        canonical = build_model('basil_paper_cnn', 2025)
        initial = [w.numpy() for w in canonical.trainable_variables]
    rng = np.random.default_rng(17)
    images = rng.normal(0, .2, (512, 32, 32, 3)).astype(np.float32)
    labels = np.arange(512, dtype=np.int32) % 10
    outputs = {}
    for device in ('CPU', 'GPU'):
        with tf.device(f'/{device}:0'):
            worker = SnapshotAnchorWorker(build_model('basil_paper_cnn', 2025), mu=.01, device=device)
            worker.load(initial)
            for ref, w in zip(worker.reference, initial): ref.assign(w + .001)
            reference_hash = params_hash([r.numpy() for r in worker.reference])
            outputs[device] = {}
            for mode in ('none', 'gradient_norm_objective'):
                worker.load(initial)
                repeats = []
                for legacy in (1., 999999.):
                    grads, values = worker._gradients(tf.constant(images), tf.constant(labels), tf.constant([7, 8]),
                        mode, 'paper_absolute_gaussian', tf.constant(.01), tf.constant(legacy))
                    repeats.append(([g.numpy() for g in grads], np.asarray([float(v) for v in values])))
                for a, b in zip(repeats[0][0], repeats[1][0]): np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(repeats[0][1], repeats[1][1])
                # Include every actual α=.2 partial batch, not just the IID 392.
                for size in (19,37,160,183,311,378,385,392,444,477,502):
                    partial, _ = worker._gradients(tf.constant(images[:size]), tf.constant(labels[:size]), tf.constant([7, 8]),
                        mode, 'paper_absolute_gaussian', tf.constant(.01), tf.constant(0.))
                    assert all(np.isfinite(g.numpy()).all() for g in partial)
                assert all(device in g.device for g in grads)
                tf.keras.optimizers.SGD(.05, momentum=0.).apply_gradients(zip(grads, worker.model.trainable_variables))
                assert params_hash([r.numpy() for r in worker.reference]) == reference_hash
                outputs[device][mode] = repeats[0]
    comparisons = {}
    for mode in ('none', 'gradient_norm_objective'):
        a, av = outputs['CPU'][mode]; b, bv = outputs['GPU'][mode]
        difference = np.concatenate([(x-y).reshape(-1) for x, y in zip(a, b)])
        reference = np.concatenate([x.reshape(-1) for x in a])
        relative = float(np.linalg.norm(difference) / np.linalg.norm(reference))
        assert relative < .01, relative
        np.testing.assert_allclose(av[:4], bv[:4], atol=5e-7, rtol=1e-4)
        comparisons[mode] = {'gradientRelativeL2Difference': relative, 'legacyLambdaIgnored': True}
    memory = tf.config.experimental.get_memory_info('GPU:0')
    assert memory['peak'] < .75 * GPU_MEMORY_LIMIT_MB * 1024**2
    result = dict(status='passed', researchRunsStarted=0, comparisons=comparisons,
        mu=.01, sigma_e=.01, coefficient=.0001, referenceUnchanged=True,
        batchSizes=[512,19,37,160,183,311,378,385,392,444,477,502], gpuMemory=memory, initialModelHash=params_hash(initial),
        objectiveSha256=hashlib.sha256((ROOT/'basil_core/noniid_study.py').read_bytes()).hexdigest())
    path = ROOT/'gui/worker_state/SNAPSHOT_ANCHOR_GPU_VERIFICATION.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
