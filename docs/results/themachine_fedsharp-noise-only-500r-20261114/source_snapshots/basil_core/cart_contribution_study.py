"""Matched CART component interventions, isolated from historical producers."""
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from basil_core import cart_comparison_study as base

ROOT = base.ROOT
REVISION = 'cart_contribution_20261029_v1'
BASE_SOURCE = base.source
BASE_CONFIGURE = base.configure_worker
BASE_INTERVENTION = base.intervention
VARIANTS = dict(cart_full='prior_fallback', acds_checked='acds', sharp_checked='sharp',
    cart_macro='prior_fallback', cart_ingress='prior_fallback', cart_commit='prior_fallback',
    cart_no_prior='prior_fallback', cart_no_calibration='prior_fallback',
    cart_no_repair='prior_fallback', acds_single='acds', local_only='local_only')
MAIN = tuple(VARIANTS)[:9]


def source():
    names = ('basil_core/cart_contribution_study.py', 'basil_core/sharp_checked_pool.py',
        'scripts/cart_contribution_worker.py', 'scripts/run_cart_contribution.py',
        'scripts/verify_cart_contribution_gpu.py', 'scripts/run_cart_adaptive.py',
        'basil_core/cart_worker_admission.py')
    return {**BASE_SOURCE(), **{n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}}


def check_source(expected):
    if source() != expected:
        raise ValueError('Contribution producer changed; use a fresh output.')


def variant(view):
    case = view['comparisonCase']
    spec = case['studyVariant']
    name = spec['name']
    if spec != dict(name=name, revision=REVISION) or VARIANTS.get(name) != case['arm']:
        raise ValueError('Unknown or inconsistent contribution intervention.')
    return name


def selected(args):
    """66 cases: nine CIFAR variants and two MNIST methods, three paired seeds."""
    result = []
    for dataset in args.datasets:
        modes = args.variants or (MAIN if dataset == 'cifar10' and args.preset == 'contribution'
                                 else ('cart_full', 'acds_checked'))
        for mode in modes:
            options = SimpleNamespace(**vars(args))
            options.datasets = [dataset]
            options.arms = [VARIANTS[mode]]
            _, views = base.selected(options)
            for view in views:
                case = dict(view['comparisonCase'], studyVariant=dict(name=mode, revision=REVISION))
                key = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:12]
                result.append(dict(view, comparisonCase=case, comparisonLabel=mode,
                    runId=f'{dataset}-{view["partition"]}-{view["seed"]}-{mode}-{key}'))
    return result


def configure_worker(view, device='GPU'):
    name = variant(view)
    if name == 'cart_no_prior':
        from basil_core.cart_batched_scoring import BatchedAdmittedWorker
        return BatchedAdmittedWorker(base.effective_config(view), device)
    return BASE_CONFIGURE(view, device)


@contextmanager
def intervention(view, worker):
    import basil_core.sharp_cart_v2_protocol as protocol
    import basil_core.sharp_cart_v2 as repair
    import basil_core.cart_prior_neutral_commit as quality
    from basil_core.cart_batched_scoring import select_input as pooled
    from basil_core.sharp_checked_pool import selector as checked
    name = variant(view)
    original_repair = repair.transport
    factory = quality.selector

    def matching_selector(original, mode, **options):
        return factory(original, 'macro' if name == 'cart_macro' else mode, **options)

    with patch.object(quality, 'selector', matching_selector):
        with BASE_INTERVENTION(view, worker) as cfg:
            current = protocol.select_input
            if name in ('acds_checked', 'sharp_checked', 'acds_single'):
                protocol.select_input = checked(current, 'single' if name == 'acds_single' else 'checked')
            elif name == 'cart_no_calibration':
                def uncalibrated(c, *args):
                    return current(dict(c, calibrationDraws=0), *args)
                protocol.select_input = uncalibrated
            elif name == 'cart_commit':
                def commit_only(c, state, w, x, y, node, r, batch):
                    w.quality_context = None
                    value, row = pooled(c, state, w, x, y, node, r, batch)
                    w.admitted_route = 'pool'
                    w.admitted_has_channel = bool(row['availableCopies'])
                    w.continuity_anchor = value.copy()
                    row['continuity'] = dict(actualRoute='pool')
                    base._commit_context(c, state, w, x, y, node, value, row)
                    return value, row
                protocol.select_input = commit_only
            if name == 'cart_ingress':
                # Retain the same head repair and ingress; remove only the full-update gate.
                repair.transport = original_repair
            elif name == 'cart_no_repair':
                def no_head(w, proposal, previous, x, y, node, activation):
                    context = getattr(w, 'quality_context', None)
                    chosen, record = quality.commit(w, proposal, context, x, y) if context else (
                        proposal, dict(applied=False, reason='receiver recognition not established'))
                    w.load(chosen)
                    return chosen, previous, dict(referenceAdmitted=False,
                        headRepairDisabled=True, qualityCommit=record)
                repair.transport = no_head
            yield cfg


@contextmanager
def installed():
    """Use the unchanged checkpoint/export worker with this declared adapter."""
    with patch.object(base, 'source', source), patch.object(base, 'check_source', check_source), \
            patch.object(base, 'configure_worker', configure_worker), patch.object(base, 'intervention', intervention):
        yield
