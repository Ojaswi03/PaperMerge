"""Fresh noise-only controls using the frozen FedSharp/FedSharp+ algorithms."""
from contextlib import contextmanager
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from basil_core import basil_sharp_thesis as base

ROOT = base.ROOT
REVISION = 'fedsharp_noise_only_500r_20261114_v1'
BASE_REVISION = base.REVISION
BASE_SELECTED = base.selected
BASE_EFFECTIVE = base.effective_config
BASE_SOURCE = base.source
BASE_INSTALLED = base.installed


def source():
    from basil_core.fedsharp_runtime_optimization import source as execution_source
    names = ('basil_core/fedsharp_noise_only.py', 'basil_core/round_timing.py',
             'scripts/run_fedsharp_noise_only.py', 'scripts/verify_fedsharp_noise_only_gpu.py',
             'configs/research/hal9k_fedsharp_noise_only_500r.json')
    return {**BASE_SOURCE(), **execution_source(),
            **{n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}}


def check_source(expected):
    if source() != expected:
        raise ValueError('Noise-only producer changed; preserve the old output.')


def runtime_policy():
    from basil_core import fedsharp_runtime_optimization as execution
    return dict(revision=execution.REVISION, source=execution.source(), transitionId=REVISION)


def references(args):
    if tuple(args.families) != ('sharp',) or tuple(args.environments) != ('noise_only',):
        raise ValueError('Only FedSharp-family noise-only controls are supported.')
    options = copy.copy(args)
    options.environments = ['joint']
    with patch.object(base, 'REVISION', BASE_REVISION):
        return BASE_SELECTED(options)


def decorate(reference):
    view = copy.deepcopy(reference)
    case = view['comparisonCase']
    case.update(environment='noise_only', effectiveAttackers=0, caseRevision=REVISION,
                arm='sharp_cart' if case['addon'] == 'cart' else 'sharp_noise_only')
    view['noiseOnlyControl'] = dict(revision=REVISION, referenceRunId=reference['runId'],
                                  start='initialization', actualAttackers=0)
    view.update(actualAttackers=0, comparisonLabel=case['arm'])
    view['switches'] = dict(view['switches'], attack=False)
    # The standard inner protocol is reference metadata. The effective checkpoint
    # config below declares the actual attack-free environment explicitly.
    view['condition'] = 'noise_only_defense_rla_cart' if case['addon'] else 'noise_only_defense_rla'
    key = hashlib.sha256(json.dumps(view, sort_keys=True).encode()).hexdigest()[:12]
    view['runId'] = f"{view['dataset']}-{view['partition']}-{view['seed']}-{case['arm']}-{key}"
    return view


def selected(args):
    return [decorate(view) for view in references(args)]


def reference_for(view):
    inner = view['comparisonCase']['protocolConfig']
    args = SimpleNamespace(datasets=[inner['dataset']], partitions=[inner['partition']],
        seeds=[inner['seed']], families=['sharp'], environments=['noise_only'],
        nodes=inner['nodes'], rounds=inner['rounds'], local_steps=inner['localSteps'],
        alpha=inner['dirichletAlpha'], sigma=inner['sigma'], attack_start=inner['attackStart'],
        attack_type=inner['attackType'], gaussian_std=inner.get('gaussianAttackStd', 1.))
    reference = next(v for v in references(args)
                     if v['comparisonCase']['addon'] == view['comparisonCase']['addon'])
    if view != decorate(reference):
        raise ValueError('Noise-only identity or physical settings differ.')
    return reference


def effective_config(view):
    reference = reference_for(view)
    with patch.object(base, 'REVISION', BASE_REVISION):
        cfg = BASE_EFFECTIVE(reference)
    cfg.update(actualAttackers=0, condition=view['condition'], noiseOnlyRevision=REVISION)
    cfg['switches'] = dict(cfg['switches'], attack=False)
    if cfg['sigma'] <= 0 or not cfg['switches']['noise']:
        raise ValueError('Noise-only controls require a positive physical channel sigma.')
    if cfg['cart'] and cfg['partition'] != 'dirichlet':
        raise ValueError('FedSharp+ remains restricted to non-IID data.')
    return cfg


@contextmanager
def installed():
    with patch.object(base, 'REVISION', REVISION), patch.object(base, 'selected', selected), \
         patch.object(base, 'source', source), patch.object(base, 'check_source', check_source), \
         patch.object(base, 'effective_config', effective_config), BASE_INSTALLED():
        yield
