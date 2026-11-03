"""Fresh clean FedSharp controls: IID mu=0, nonIID mu=1; no FedSharp+."""
from contextlib import contextmanager
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

from basil_core import basil_sharp_thesis as base

ROOT=base.ROOT
REVISION='fedsharp_clean_mu_by_partition_20261103_v1'
BASE_REVISION=base.REVISION
BASE_SELECTED=base.selected
BASE_EFFECTIVE=base.effective_config
BASE_SOURCE=base.source
BASE_INSTALLED=base.installed


def source():
    names=('basil_core/fedsharp_clean_controls.py','scripts/run_fedsharp_clean_controls.py',
        'scripts/verify_fedsharp_clean_controls_gpu.py','configs/research/hal9k_fedsharp_clean_500r.json')
    return {**BASE_SOURCE(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def check_source(expected):
    if source()!=expected:raise ValueError('Clean-control implementation changed; preserve this plan and use a new output.')


def decorate(reference):
    view=copy.deepcopy(reference)
    view['cleanControl']=dict(revision=REVISION,referenceRunId=reference['runId'],
        proximalMu=0. if reference['partition']=='iid' else 1.,start='initialization')
    view['proximalMu']=view['cleanControl']['proximalMu']
    digest=hashlib.sha256(json.dumps(view,sort_keys=True).encode()).hexdigest()[:12]
    view['runId']=f"{view['dataset']}-{view['partition']}-{view['seed']}-fedsharp-clean-mu{view['cleanControl']['proximalMu']:g}-{digest}"
    return view


def references(args):
    if tuple(args.families)!=('sharp',) or tuple(args.environments)!=('clean',):
        raise ValueError('Only base FedSharp clean controls are supported; no FedSharp+ or attacked cases.')
    with patch.object(base,'REVISION',BASE_REVISION):
        return BASE_SELECTED(args)


def selected(args):return [decorate(v) for v in references(args)]


def effective_config(view):
    inner=view['comparisonCase']['protocolConfig']
    args=SimpleNamespace(datasets=[view['dataset']],partitions=[view['partition']],seeds=[view['seed']],
        families=['sharp'],environments=['clean'],nodes=inner['nodes'],rounds=inner['rounds'],
        local_steps=inner['localSteps'],alpha=inner['dirichletAlpha'],sigma=inner['sigma'],
        attack_start=inner['attackStart'],attack_type=inner['attackType'],gaussian_std=inner.get('gaussianAttackStd',1.))
    reference=references(args)[0]
    if decorate(reference)!=view:raise ValueError('Clean-control identity or declared mu differs.')
    with patch.object(base,'REVISION',BASE_REVISION):cfg=BASE_EFFECTIVE(reference)
    cfg['proximalMu']=view['cleanControl']['proximalMu']
    cfg['cleanControlRevision']=REVISION
    if cfg['sigma']!=0 or cfg['actualAttackers']!=0 or cfg['cart']:
        raise ValueError('Clean controls must have no noise, attackers or class-aware extension.')
    return cfg


@contextmanager
def installed():
    with patch.object(base,'REVISION',REVISION),patch.object(base,'selected',selected),\
            patch.object(base,'source',source),patch.object(base,'check_source',check_source),\
            patch.object(base,'effective_config',effective_config),BASE_INSTALLED():
        yield
