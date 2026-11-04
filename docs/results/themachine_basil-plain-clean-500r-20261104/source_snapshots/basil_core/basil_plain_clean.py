"""Fresh BASIL clean ring controls matched to the direct-transfer FedSharp run."""
from contextlib import contextmanager
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from basil_core import basil_sharp_thesis as base,fedsharp_plain_clean as sharp

ROOT=base.ROOT
REVISION='basil_plain_clean_ring_20261104_v1'
BASE_REVISION=base.REVISION
BASE_SELECTED=base.selected
BASE_EFFECTIVE=base.effective_config
BASE_INSTALLED=base.installed


def source():
    names=('basil_core/basil_plain_clean.py','scripts/run_basil_plain_clean.py',
           'scripts/verify_basil_plain_clean_gpu.py','configs/research/themachine_basil_plain_clean_500r.json')
    return {**sharp.source(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def check_source(expected):
    if source()!=expected:raise ValueError('BASIL clean implementation changed; preserve this plan and use a new output.')


def references(args):
    if tuple(args.families)!=('basil',) or tuple(args.environments)!=('clean',):
        raise ValueError('Only direct-transfer BASIL clean controls are supported.')
    with patch.object(base,'REVISION',BASE_REVISION):return BASE_SELECTED(args)


def options_for(view,family):
    inner=view['comparisonCase']['protocolConfig']
    return SimpleNamespace(datasets=[view['dataset']],partitions=[view['partition']],seeds=[view['seed']],
        families=[family],environments=['clean'],nodes=inner['nodes'],rounds=inner['rounds'],
        local_steps=inner['localSteps'],alpha=inner['dirichletAlpha'],sigma=inner['sigma'],
        attack_start=inner['attackStart'],attack_type=inner['attackType'],gaussian_std=inner.get('gaussianAttackStd',1.))


def decorate(reference):
    view=copy.deepcopy(reference)
    view['plainCleanControl']=dict(revision=REVISION,referenceRunId=reference['runId'],
        selection='immediate_predecessor',proximalMu=0.,screening=False,ranking=False,
        averaging=False,noiseMitigation=False,start='initialization')
    view.update(proximalMu=0.,window=1,byzantineBudget=0)
    digest=hashlib.sha256(json.dumps(view,sort_keys=True).encode()).hexdigest()[:12]
    view['runId']=f"{view['dataset']}-{view['partition']}-{view['seed']}-basil-plain-clean-{digest}"
    return view


def selected(args):return [decorate(v) for v in references(args)]


def effective_config(view):
    reference=references(options_for(view,'basil'))[0]
    if view!=decorate(reference):raise ValueError('BASIL clean identity or declared transfer/objective differs.')
    with patch.object(base,'REVISION',BASE_REVISION):cfg=BASE_EFFECTIVE(reference)
    cfg.update(method='basil_plain_clean_ring',proximalMu=0.,window=1,byzantineBudget=0,
               receiverRule='immediate_predecessor',plainCleanRevision=REVISION,
               startup='first_activation_initial_model_then_immediate_predecessor')
    cfg['switches']={k:False for k in cfg['switches']}
    if cfg['sigma']!=0 or cfg['actualAttackers']!=0 or cfg['cart']:
        raise ValueError('BASIL clean control cannot use attacks, noise or sharing.')
    return cfg


# Both controls execute the same validated direct predecessor transfer.
select_input=sharp.select_input


def configure_worker(view,device='GPU'):
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    return CompiledTransferWorker(effective_config(view),device)


@contextmanager
def intervention(view,worker):
    import basil_core.sharp_cart_v2_protocol as protocol
    cfg=effective_config(view)
    def validate(actual):
        if actual!=cfg:raise ValueError('Runtime differs from the declared BASIL clean control.')
    with patch.object(protocol,'validate_config',validate),patch.object(protocol,'select_input',select_input):
        yield cfg


@contextmanager
def installed():
    with patch.object(base,'REVISION',REVISION),patch.object(base,'selected',selected),\
            patch.object(base,'source',source),patch.object(base,'check_source',check_source),\
            patch.object(base,'effective_config',effective_config),patch.object(base,'configure_worker',configure_worker),\
            patch.object(base,'intervention',intervention),BASE_INSTALLED():
        yield
