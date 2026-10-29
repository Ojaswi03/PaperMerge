#!/usr/bin/env python3
"""Fresh CPU inference of complete local pool-diagnostic checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    from basil_core.cart_campaign_io import read_json, write_json
    from basil_core.cart_comparison_study import source, effective_config, partitions_for
    from basil_core.sharp_data import training_data, digest
    from basil_core.sharp_protocol import evaluate, vector_hash
    from basil_core.sharp_worker import SharpWorker
    root = Path(root).resolve()
    plan = read_json(root / 'plan.json')
    if source() != plan['producerSource']:
        raise ValueError('Required producer/inference code changed.')
    for name, expected in {**plan['producerSource'], **plan['diagnosticSource']}.items():
        if sha(root / 'source_snapshots' / name) != expected:
            raise ValueError('Frozen source snapshot changed: ' + name)
    x, y = training_data('cifar10')
    data = dict(imagesSha256=digest(x), labelsSha256=digest(y))
    results = []
    for case in plan['cases']:
        folder = root / case['mode']
        if not (folder / 'result.json').exists():
            continue
        row = read_json(folder / 'result.json')
        if row['status'] != 'COMPLETED' or row['contract']['case'] != case:
            raise ValueError('Diagnostic is incomplete or mismatched.')
        view = case['view']; cfg = effective_config(view)
        if row['contract'].get('config', cfg) != cfg:
            raise ValueError('Checkpoint configuration differs from its declared case.')
        chunks, held, audit = partitions_for(view, y)
        if audit != row['contract']['partition'] or data != row['contract']['data']:
            raise ValueError('Fitting, sharing or evaluation data differ.')
        if np.intersect1d(np.concatenate(chunks), held).size:
            raise ValueError('Evaluation examples leaked into fitting data.')
        model = SharpWorker(cfg, 'CPU')
        if vector_hash(model.export()) != row['contract']['initialHash']:
            raise ValueError('Initialization differs from the saved run.')
        # Initial diagnostic receipts nested the config under case/view rather
        # than the standard worker contract key. Verify their original bytes
        # directly; do not rewrite a checkpoint to fit another schema.
        with np.load(folder / 'checkpoint.npz', allow_pickle=False) as saved:
            state = json.loads(saved['metadata'].tobytes())
            if state['contract'] != row['contract']:
                raise ValueError('Checkpoint contract differs from the completed receipt.')
            state['nodeParams'] = saved['nodeParams'].copy()
            if (state['nodeParams'].shape != (cfg['nodes'], model.dimension) or
                    not np.isfinite(state['nodeParams']).all() or
                    saved['outbound'].shape != state['nodeParams'].shape or
                    not np.isfinite(saved['outbound']).all() or
                    not np.array_equal(saved['senderRounds'], np.full(cfg['nodes'], cfg['rounds'] - 1))):
                raise ValueError('Checkpoint model or sender arrays are invalid.')
        if state['cursor'] != cfg['nodes'] * cfg['rounds'] or len(state['roundMetrics']) != cfg['rounds']:
            raise ValueError('Final checkpoint is incomplete.')
        if len(state['telemetry']) != state['cursor'] or any(
                (r['round'], r['node'], r['activation']) != (i // cfg['nodes'], i % cfg['nodes'], i + 1)
                for i, r in enumerate(state['telemetry'])):
            raise ValueError('Activation telemetry is incomplete or unordered.')
        expected_steps = cfg['nodes'] * cfg['rounds'] * cfg['localSteps']
        if sum(r['training']['optimizerSteps'] for r in state['telemetry']) != expected_steps:
            raise ValueError('Optimizer-step count differs.')
        with np.load(folder / 'metrics.npz', allow_pickle=False) as arrays:
            accuracy = arrays['roundNodeAccuracy']; classes = arrays['roundNodePerClassAccuracy']
        np.testing.assert_array_equal(accuracy, [r['nodeAccuracy'] for r in state['roundMetrics']])
        fresh = [evaluate(model, flat, x[held], y[held], cfg['classes']) for flat in state['nodeParams']]
        actual = np.asarray([a for a, _ in fresh]); pc = np.asarray([c for _, c in fresh])
        difference = float(np.max(abs(actual - accuracy[-1])))
        if difference > 1 / len(held) + 1e-9:
            raise ValueError('Fresh CPU accuracy differs by more than one example per client.')
        counts = np.bincount(y[held], minlength=cfg['classes'])
        if np.any(abs(pc - classes[-1]) > 1 / counts[None] + 1e-9):
            raise ValueError('Fresh class recall differs by more than one example per class.')
        results.append(dict(mode=case['mode'], clientsVerified=len(actual), finalAccuracy=float(actual.mean()),
            maximumAccuracyDifference=difference, optimizerSteps=expected_steps,
            checkpointSha256=sha(folder / 'checkpoint.npz'), metricSha256=sha(folder / 'metrics.npz')))
        print(f'CPU verified {case["mode"]}: {100*actual.mean():.3f}%, max difference {difference:g}', flush=True)
    if not results:
        raise ValueError('No completed cases to verify.')
    receipt = dict(status='COMPLETED_CASES_INDEPENDENTLY_VERIFIED', results=results,
        incompleteModes=[c['mode'] for c in plan['cases'] if c['mode'] not in {r['mode'] for r in results}],
        verifierSha256=sha(Path(__file__)), producerSource=plan['producerSource'])
    write_json(root / 'independent_validation.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
