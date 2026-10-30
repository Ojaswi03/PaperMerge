"""Explicit matched BASIL/SHARP conditions for the requested thesis figures."""
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from basil_core import cart_comparison_study as comparison
from basil_core.cart_contribution_study import source as previous_source

ROOT = comparison.ROOT
REVISION = 'basil_sharp_thesis_20261029_v1'
ENVIRONMENTS = ('clean', 'attack_only', 'joint')
_cart_validate = None


def source():
    names = ('basil_core/basil_sharp_thesis.py', 'scripts/run_basil_sharp_thesis.py',
        'scripts/basil_sharp_thesis_worker.py', 'scripts/verify_basil_sharp_thesis_gpu.py',
        'scripts/verify_basil_sharp_thesis_results.py', 'reporting/basil_sharp_thesis.py',
        'basil_core/cart_display_labels.py', 'basil_core/thesis_raw_data.py')
    return {**previous_source(), **{n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def check_source(expected):
    if source() != expected:
        raise ValueError('Thesis producer changed; use a new output.')


def variant(view):
    return view['comparisonCase']['arm']


def selected(args):
    from basil_core.sharp_study import configuration
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration as cart_configuration
    views = []
    for dataset in args.datasets:
        for partition in args.partitions:
            for seed in args.seeds:
                for family in ('basil', 'sharp'):
                    cases = [(env, None) for env in ENVIRONMENTS]
                    if partition == 'dirichlet':
                        cases.append(('joint', 'acds' if family == 'basil' else 'cart'))
                    for environment, addon in cases:
                        noisy = environment == 'joint'
                        condition = ('attack_noise_defense_rla_cart' if addon == 'cart'
                                     else 'attack_noise_defense_rla' if family == 'sharp'
                                     else 'attack_noise_defense')
                        # Standard validated inner configuration, with separately
                        # declared physical overrides for clean/attack-only runs.
                        inner = configuration(dataset, partition, seed, condition, args.sigma,
                            method=family, nodes=args.nodes, rounds=args.rounds,
                            local_steps=args.local_steps, alpha=args.alpha, attack_start=args.attack_start)
                        if family == 'sharp':
                            inner = cart_configuration(guarded_configuration(inner))
                        arm = f'{family}_{addon or environment}'
                        case = dict(arm=arm, family=family, environment=environment,
                            addon=addon, protocolConfig=inner, caseRevision=REVISION,
                            effectiveSigma=args.sigma if noisy else 0.,
                            effectiveAttackers=0 if environment=='clean' else inner['byzantineBudget'],
                            shareFraction=.05 if addon=='acds' else 0., acdsBatches=5,
                            selection='performance_single' if family=='basil' else 'checked_guarded_pool')
                        key = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:12]
                        views.append(dict(inner, comparisonCase=case, comparisonLabel=arm,
                            sigma=case['effectiveSigma'], runId=f'{dataset}-{partition}-{seed}-{arm}-{key}'))
    if len({v['runId'] for v in views}) != len(views):
        raise ValueError('Duplicate requested cases.')
    return views


def effective_config(view):
    from basil_core.sharp_study import validate_config as validate_basil
    from basil_core.sharp_cart_v2_protocol import validate_config as validate_cart
    case = view['comparisonCase']
    inner, family, addon, env = (case[k] for k in ('protocolConfig', 'family', 'addon', 'environment'))
    if case['caseRevision'] != REVISION or family not in ('basil','sharp') or env not in ENVIRONMENTS:
        raise ValueError('Unknown thesis case.')
    if family=='basil':
        validate_basil(inner)
    else:
        (_cart_validate or validate_cart)(inner)
    if addon is not None and (inner['partition']!='dirichlet' or env!='joint' or
            addon != ('acds' if family=='basil' else 'cart')):
        raise ValueError('Extra mechanisms are restricted to their declared non-IID cases.')
    if case['arm'] != f'{family}_{addon or env}' or case['effectiveAttackers'] != (
            0 if env=='clean' else inner['byzantineBudget']):
        raise ValueError('Thesis case identity or attacker count differs.')
    if (case['effectiveSigma']>0) != (env=='joint') or case['shareFraction'] != (.05 if addon=='acds' else 0.):
        raise ValueError('Thesis noise/sharing switch differs.')
    if case['selection'] != ('performance_single' if family=='basil' else 'checked_guarded_pool'):
        raise ValueError('Thesis selection revision differs.')
    cfg = dict(inner, sigma=case['effectiveSigma'], actualAttackers=case['effectiveAttackers'])
    cfg['switches'] = dict(inner['switches'], attack=env!='clean', noise=env=='joint',
        byzantine_mitigation=True, noise_mitigation=family=='sharp', cart=addon=='cart')
    return cfg


def partitions_for(view, y):
    from basil_core.sharp_data import prepare_partition
    cfg = effective_config(view)
    chunks, held, audit = prepare_partition(y, cfg)
    if view['comparisonCase']['addon']=='acds':
        from basil_core.acds import cyclic_training_share
        chunks, sharing = cyclic_training_share(chunks, y, fraction=.05, batches=5, seed=cfg['seed'])
        audit = dict(audit, acds=sharing)
    return chunks, held, audit


def data_contract(configs):
    from basil_core.sharp_data import digest
    from basil_core.thesis_raw_data import training_data
    data, audits, cache = {}, {}, {}
    for view in configs:
        dataset = view['dataset']
        if dataset not in cache:
            x,y=training_data(dataset);cache[dataset]=y
            data[dataset]=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
        audits[view['runId']]=partitions_for(view,cache[dataset])[2]
    return data, audits


def configure_worker(view, device='GPU'):
    cfg = effective_config(view)
    if view['comparisonCase']['addon']=='cart':
        from basil_core.cart_local_prior import LocalPriorWorker
        return LocalPriorWorker(cfg, device)
    from basil_core.cart_batched_scoring import BatchedAdmittedWorker
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    class Worker(BatchedAdmittedWorker):
        train = CompiledTransferWorker.train
    return Worker(cfg, device)


def basil_select(cfg, state, worker, x, y, node, r, batch):
    import numpy as np
    from basil_core.sharp_protocol import link_vector
    from basil_core.sharp_math import ranked_indices
    senders=[(node-h)%cfg['nodes'] for h in range(1,cfg['window']+1)
             if state['senderRounds'][(node-h)%cfg['nodes']]>=0]
    if not senders:
        return state['nodeParams'][node].copy(), dict(selectedSenders=[],selectedCount=1,
            availableCopies=0,bootstrap=True,screenApplied=False,screenAllFailFallback=False,ranking=[])
    vectors=np.stack([link_vector(state['outbound'][s],cfg['seed'],int(state['senderRounds'][s]),
        s,node,cfg['sigma']) for s in senders])
    losses=worker.score_many(vectors,x[batch],y[batch])
    chosen=ranked_indices(losses,senders,[state['senderRounds'][s] for s in senders],node,cfg['nodes'],1)
    index=int(chosen[0])
    return vectors[index].copy(),dict(selectedSenders=[senders[index]],selectedCount=1,targetCount=1,
        availableCopies=len(senders),bootstrap=False,screenApplied=False,screenAllFailFallback=False,
        ranking=[dict(sender=s,loss=float(loss) if np.isfinite(loss) else None) for s,loss in zip(senders,losses)])


@contextmanager
def intervention(view, worker):
    import basil_core.sharp_cart_v2_protocol as protocol
    import basil_core.sharp_cart_v2 as repair
    from basil_core.cart_batched_scoring import select_input
    from basil_core.sharp_checked_pool import selector as checked_pool
    from basil_core.cart_prior_neutral_commit import selector, transport
    global _cart_validate
    original_validate = protocol.validate_config
    _cart_validate = original_validate
    cfg = effective_config(view)
    choose = basil_select if view['comparisonCase']['family']=='basil' else checked_pool(select_input)
    if cfg['cart']:
        choose = selector(choose, 'class', peer_budget=8, guard_fallback=True)
    def validate(actual):
        if actual != cfg:
            raise ValueError('Runtime differs from the declared thesis case.')
    try:
        with patch.object(protocol,'validate_config',validate), patch.object(protocol,'select_input',choose), \
                patch.object(repair,'transport',transport(repair.transport) if cfg['cart'] else repair.transport):
            yield cfg
    finally:
        _cart_validate = None


@contextmanager
def installed():
    """Scope the existing worker/inference adapter to this independent family."""
    from basil_core.thesis_raw_data import direct_inputs
    with direct_inputs(), patch.object(comparison,'source',source), patch.object(comparison,'check_source',check_source), \
            patch.object(comparison,'effective_config',effective_config), \
            patch.object(comparison,'partitions_for',partitions_for), patch.object(comparison,'data_contract',data_contract), \
            patch.object(comparison,'configure_worker',configure_worker), patch.object(comparison,'intervention',intervention):
        yield
