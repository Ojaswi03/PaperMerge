"""Explicit scientific identity for isolated CART admission revisions.

The inner configuration retains the frozen SHARP equations/physical settings.
The outer revision and extended source map identify the changed CART method.
"""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('prior_neutral_class','prior_neutral_macro','prior_neutral_no_commit','transactional_class','transactional_macro','quality_class','quality_macro','discriminative','guard_only','capped','uncapped','semantic','admitted','fresh')


def revision(variant):
    if variant not in VARIANTS:raise ValueError('Unknown CART revision.')
    return 'cart_admission_20261028_'+variant+'_dev1'


def source():
    from basil_core.sharp_cart_v2_protocol import fingerprint
    from scripts.run_cart_v2_transfer import program_hashes
    names=('basil_core/cart_noise_selection.py','basil_core/cart_revision_study.py','basil_core/cart_continuity_candidate.py',
        'basil_core/cart_qualified_continuity_candidate.py','basil_core/cart_admitted_continuity_candidate.py',
        'basil_core/cart_discriminative_continuity_candidate.py','basil_core/cart_discriminative_admission_candidate.py',
        'basil_core/cart_quality_routing_candidate.py','basil_core/cart_transactional_quality_candidate.py',
        'basil_core/cart_quality_commit_candidate.py','basil_core/cart_batched_scoring.py',
        'basil_core/cart_class_skill.py','basil_core/cart_prior_neutral_routing.py','basil_core/cart_prior_neutral_commit.py',
        'basil_core/cart_semantic_admission_candidate.py','basil_core/cart_uncertainty_admission_candidate.py',
        'basil_core/sharp_compiled_transfer.py','scripts/run_cart_revision_study.py','scripts/cart_revision_worker.py',
        'scripts/verify_cart_revision_gpu.py')
    return {**fingerprint(),**program_hashes(),**{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}}


def check_source(expected):
    if source()!=expected:raise ValueError('CART revision source changed; preserve this run and use a new output.')


def selected(args):
    from basil_core.cart_noise_selection import configurations
    noise,configs=configurations(args)
    return noise,configs


def configure_worker(cfg,variant,device='GPU'):
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    if not cfg['cart']:return CompiledTransferWorker(cfg,device),None
    from basil_core.cart_batched_scoring import BatchedAdmittedWorker
    worker=BatchedAdmittedWorker(cfg,device)
    if variant.startswith('prior_neutral_'):
        from basil_core.cart_prior_neutral_commit import selector
        mode='macro' if variant=='prior_neutral_macro' else 'class'
    elif variant in ('transactional_class','transactional_macro'):
        from basil_core.cart_quality_commit_candidate import selector
        mode='class' if variant=='transactional_class' else 'macro'
    elif variant in ('quality_class','quality_macro'):
        from basil_core.cart_quality_routing_candidate import selector
        mode='class' if variant=='quality_class' else 'macro'
    elif variant in ('discriminative','guard_only'):
        from basil_core.cart_discriminative_admission_candidate import selector
        mode='class' if variant=='discriminative' else 'guard_only'
    elif variant in ('capped','uncapped'):
        from basil_core.cart_uncertainty_admission_candidate import selector
        mode=variant
    elif variant=='semantic':
        from basil_core.cart_semantic_admission_candidate import selector
        mode='qualified'
    else:
        from basil_core.cart_admitted_continuity_candidate import selector
        mode='always' if variant=='fresh' else 'qualified'
    from basil_core.cart_batched_scoring import select_input as batched_select
    return worker,lambda original:selector(batched_select,mode)
