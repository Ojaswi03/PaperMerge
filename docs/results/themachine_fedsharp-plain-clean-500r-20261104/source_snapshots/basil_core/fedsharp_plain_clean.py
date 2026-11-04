"""Separate clean ring control: immediate predecessor and unregularized SGD."""
from contextlib import contextmanager
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from basil_core import basil_sharp_thesis as base

ROOT=base.ROOT
REVISION='fedsharp_plain_clean_ring_20261104_v1'
BASE_REVISION=base.REVISION
BASE_SELECTED=base.selected
BASE_EFFECTIVE=base.effective_config
BASE_SOURCE=base.source
BASE_INSTALLED=base.installed


def source():
    names=('basil_core/fedsharp_plain_clean.py','scripts/run_fedsharp_plain_clean.py',
           'scripts/verify_fedsharp_plain_clean_gpu.py','configs/research/themachine_fedsharp_plain_clean_500r.json')
    return {**BASE_SOURCE(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def check_source(expected):
    if source()!=expected:raise ValueError('Plain clean implementation changed; preserve this plan and use a new output.')


def references(args):
    if tuple(args.families)!=('sharp',) or tuple(args.environments)!=('clean',):
        raise ValueError('Only plain base FedSharp clean controls are supported.')
    with patch.object(base,'REVISION',BASE_REVISION):return BASE_SELECTED(args)


def decorate(reference):
    view=copy.deepcopy(reference)
    view['plainCleanControl']=dict(revision=REVISION,referenceRunId=reference['runId'],
        selection='immediate_predecessor',proximalMu=0.,screening=False,ranking=False,
        averaging=False,noiseMitigation=False,start='initialization')
    view.update(proximalMu=0.,window=1,byzantineBudget=0)
    digest=hashlib.sha256(json.dumps(view,sort_keys=True).encode()).hexdigest()[:12]
    view['runId']=f"{view['dataset']}-{view['partition']}-{view['seed']}-fedsharp-plain-clean-{digest}"
    return view


def selected(args):return [decorate(v) for v in references(args)]


def effective_config(view):
    inner=view['comparisonCase']['protocolConfig']
    args=SimpleNamespace(datasets=[view['dataset']],partitions=[view['partition']],seeds=[view['seed']],
        families=['sharp'],environments=['clean'],nodes=inner['nodes'],rounds=inner['rounds'],
        local_steps=inner['localSteps'],alpha=inner['dirichletAlpha'],sigma=inner['sigma'],
        attack_start=inner['attackStart'],attack_type=inner['attackType'],gaussian_std=inner.get('gaussianAttackStd',1.))
    reference=references(args)[0]
    if view!=decorate(reference):raise ValueError('Plain clean identity or declared transfer/objective differs.')
    with patch.object(base,'REVISION',BASE_REVISION):cfg=BASE_EFFECTIVE(reference)
    cfg.update(method='fedsharp_plain_clean_ring',proximalMu=0.,window=1,byzantineBudget=0,
               receiverRule='immediate_predecessor',plainCleanRevision=REVISION,
               startup='first_activation_initial_model_then_immediate_predecessor')
    cfg['switches']={k:False for k in cfg['switches']}
    if cfg['sigma']!=0 or cfg['actualAttackers']!=0 or cfg['cart']:
        raise ValueError('Plain clean control cannot use attacks, noise or CART.')
    return cfg


def select_input(cfg,state,worker,x,y,receiver,round_id,first_batch):
    """Copy the preceding node verbatim; only the first activation bootstraps."""
    import numpy as np
    if (cfg['sigma']!=0 or cfg['actualAttackers']!=0 or cfg['proximalMu']!=0 or
            cfg['window']!=1 or cfg['byzantineBudget']!=0 or cfg['cart'] or any(cfg['switches'].values())):
        raise ValueError('Plain clean selector requires the declared disabled mechanisms.')
    if state['cursor']!=round_id*cfg['nodes']+receiver:
        raise ValueError('Receiver does not match the sequential activation cursor.')
    sender=(receiver-1)%cfg['nodes'];version=int(state['senderRounds'][sender])
    bootstrap=state['cursor']==0
    if bootstrap:
        if version!=-1:raise ValueError('Unexpected predecessor at initialization.')
        incoming=state['nodeParams'][receiver].copy();senders=[]
    else:
        expected=round_id if receiver else round_id-1
        if version!=expected:raise ValueError('Missing or stale immediate predecessor; no alternative sender is selected.')
        incoming=state['outbound'][sender].copy();senders=[sender]
    if not np.isfinite(incoming).all():raise FloatingPointError('Nonfinite predecessor parameters.')
    return incoming,dict(selectedSenders=senders,selectedCount=1,targetCount=1,
        availableCopies=len(senders),bootstrap=bootstrap,screenApplied=False,
        screenAllFailFallback=False,screenGuardFallback=False,ranking=[],
        transferRule='immediate_predecessor',senderRound=None if bootstrap else version,
        continuity=dict(actualRoute='initialization' if bootstrap else 'predecessor'))


def configure_worker(view,device='GPU'):
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    return CompiledTransferWorker(effective_config(view),device)


@contextmanager
def intervention(view,worker):
    import basil_core.sharp_cart_v2_protocol as protocol
    cfg=effective_config(view)
    def validate(actual):
        if actual!=cfg:raise ValueError('Runtime differs from the declared plain clean control.')
    with patch.object(protocol,'validate_config',validate),patch.object(protocol,'select_input',select_input):
        yield cfg


@contextmanager
def installed():
    with patch.object(base,'REVISION',REVISION),patch.object(base,'selected',selected),\
            patch.object(base,'source',source),patch.object(base,'check_source',check_source),\
            patch.object(base,'effective_config',effective_config),patch.object(base,'configure_worker',configure_worker),\
            patch.object(base,'intervention',intervention),BASE_INSTALLED():
        yield
