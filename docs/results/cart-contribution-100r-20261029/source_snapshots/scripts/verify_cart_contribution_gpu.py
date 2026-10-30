#!/usr/bin/env python3
"""GPU equivalence gates plus warm informative CART decisions under the saved cap."""
import argparse
import gc
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def warm(view, device='GPU'):
    """Synthetic binary recognition fixture, never exported as benchmark evidence."""
    import numpy as np
    from basil_core import cart_contribution_study as study
    from basil_core.cart_comparison_study import effective_config
    from basil_core.sharp_protocol import initial_state
    import basil_core.sharp_cart_v2_protocol as protocol
    import basil_core.sharp_cart_v2 as repair
    cfg = effective_config(view)
    worker = study.configure_worker(view, device)
    shape = (28, 28, 1) if cfg['dataset'] == 'mnist' else (32, 32, 3)
    x = np.random.default_rng(983).uniform(size=(100, *shape)).astype(np.float32)
    features = worker.outputs(worker.export(), x, features=True)
    column = int(features.std(axis=0).argmax())
    threshold = float(np.median(features[:, column]))
    weights, bias = worker.model.trainable_variables[-2:]
    kernel = np.zeros(weights.shape, np.float32)
    scale = 1 / max(float(features[:, column].std()), 1e-6)
    kernel[column, 0], kernel[column, 1] = scale, -scale
    offsets = np.full(bias.shape, -100., np.float32)
    offsets[0], offsets[1] = -scale * threshold, scale * threshold
    weights.assign(kernel)
    bias.assign(offsets)
    own = worker.export()
    y = worker.outputs(own, x).argmax(axis=1).astype(np.int32)
    if len(np.unique(y)) != 2 or np.bincount(y).min() < 32:
        raise AssertionError('Informative recognition fixture failed.')
    # Give every peer a poor head so the allowance branch sees a real deficit.
    bias.assign(np.array([0., 0., 100., *([0.] * (cfg['classes'] - 3))], np.float32))
    bad = worker.export()
    state = initial_state(cfg, own)
    state['senderRounds'][:] = 0
    state['outbound'][:] = bad
    state['cursor'] = cfg['nodes']
    batch = np.arange(min(cfg['batchSize'], len(y)))
    with study.intervention(view, worker) as actual:
        admitted, row = protocol.select_input(actual, state, worker, x, y, 0, 1, batch)
        context = getattr(worker, 'quality_context', None)
        coefficient = actual['sigma'] ** 2 / (2 * row['selectedCount'] * (
            actual['learningRate'] / (1 + actual['learningRate'])))
        proposal, training = worker.train(admitted, x, y, 1, 0, coefficient, batch)
        evidence = {}
        if actual['cart']:
            chosen, _, evidence = repair.transport(worker, proposal, None, x, y, 0, cfg['nodes'])
        else:
            chosen = proposal
    if training['optimizerSteps'] != cfg['localSteps'] or not np.isfinite(chosen).all():
        raise AssertionError('Invalid warm update or changed SGD budget.')
    name = study.variant(view)
    if name.startswith('cart_') and context is None:
        raise AssertionError('Informative fixture did not activate the quality context.')
    if name == 'cart_no_calibration' and row['continuity'].get('calibrationDraws') != 0:
        raise AssertionError('Calibration-disabled intervention sampled noise.')
    if name == 'cart_macro' and context['penalty']:
        raise AssertionError('Macro ablation retained the class deficit penalty.')
    if name == 'cart_no_repair' and not evidence.get('headRepairDisabled'):
        raise AssertionError('Head repair was not disabled.')
    if name == 'cart_ingress' and 'qualityCommit' in evidence:
        raise AssertionError('Ingress-only intervention applied the update check.')
    if name.startswith('cart_') and name != 'cart_ingress' and not evidence['qualityCommit']['applied']:
        raise AssertionError('Full-update check did not run.')
    composed = False
    if name.startswith('cart_') and name not in ('cart_ingress', 'cart_commit'):
        from basil_core.cart_class_skill import class_skill
        from basil_core.cart_quality_routing_candidate import utility
        indices, split = context['indices'], context['split']
        logits = worker.outputs(own, x[indices])
        own_score = utility(class_skill(logits[:split], y[indices[:split]], context['classes']),
            class_skill(logits[split:], y[indices[split:]], context['classes']), context['target'],
            context['qualified'], context['allowance'], penalty=context['penalty'])
        if (context['baseline'] < own_score - .01 - 1e-12 or
                evidence['qualityCommit']['committedScore'] < own_score - .02 - 2e-12):
            raise AssertionError('Composed receiver-probe utility bound failed.')
        composed = True
    return dict(variant=name, dataset=cfg['dataset'], syntheticFixture=True,
        optimizerSteps=training['optimizerSteps'], recognitionContext=context is not None,
        calibrationDraws=row.get('continuity', {}).get('calibrationDraws'),
        commitScale=evidence.get('qualityCommit', {}).get('scale'),
        checkedPooling=row.get('poolQuality', {}).get('checked'),
        composedEmpiricalBoundVerified=composed)


def verify(root, gpu, lease_fd):
    from basil_core import cart_contribution_study as study
    from basil_core.cart_campaign_io import read_json, write_json
    from scripts.verify_cart_comparison_gpu import verify as base_verify
    plan = read_json(root / 'plan.json')
    study.check_source(plan['source'])
    # The original objective/transfer checks remain mandatory on each GPU.
    with study.installed():
        base_verify(root, gpu, lease_fd)
    path = root / f'revision_gpu_{gpu}.json'
    receipt = read_json(path)
    # An interrupted warm gate must never leave an apparently passed receipt.
    receipt['status'] = 'VERIFYING_INTERVENTIONS'
    write_json(path, receipt)
    seen = set()
    for view in plan['configs']:
        key = (view['dataset'], study.variant(view))
        if key not in seen:
            receipt['checks'].append(warm(view))
            seen.add(key)
            gc.collect()
    import tensorflow as tf
    import resource
    receipt.update(status='PASSED', interventionGate=True,
        tensorflowMemory=tf.config.experimental.get_memory_info('GPU:0'),
        peakProcessRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    write_json(path, receipt)
    print(f'GPU {gpu}: all contribution gates PASSED.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--lease-fd', type=int, required=True)
    args = p.parse_args()
    verify(args.output.resolve(), args.gpu, args.lease_fd)
