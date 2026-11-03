import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from basil_core import fedsharp_clean_controls as clean,basil_sharp_thesis as base
from scripts import run_fedsharp_clean_controls as launcher
from tests.test_basil_sharp_thesis import options


def opts(**kwargs):
    return options(families=['sharp'],environments=['clean'],rounds=500,**kwargs)


def test_exact_12_fresh_base_controls_and_old_source_unchanged():
    before=base.source();views=clean.selected(opts())
    assert len(views)==12 and len({v['runId'] for v in views})==12
    for v in views:
        cfg=clean.effective_config(v)
        assert cfg['proximalMu']==v['proximalMu']==(0 if v['partition']=='iid' else 1)
        assert cfg['sigma']==0 and cfg['actualAttackers']==0 and not cfg['cart'] and not cfg['switches']['cart']
        assert cfg['window']==66 and cfg['nodes']==100 and cfg['rounds']==500 and cfg['localSteps']==5
        assert v['comparisonCase']['addon'] is None and v['comparisonCase']['arm']=='sharp_clean'
        assert v['cleanControl']['start']=='initialization'
    with clean.installed():
        assert base.source()==clean.source() and len(clean.source())==len(before)+4
        assert base.selected(opts())==views
        for v in views:assert base.effective_config(v)==clean.effective_config(v)
    assert base.source()==before and base.REVISION==clean.BASE_REVISION


def test_changed_mu_or_cart_refused():
    view=clean.selected(opts())[0];view['cleanControl']['proximalMu']=3
    with pytest.raises(ValueError):clean.effective_config(view)
    with pytest.raises(ValueError):clean.selected(options(families=['sharp'],environments=['joint']))


def test_description_never_prepares_data_or_devices(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(launcher.driver,'prepare',lambda *a:pytest.fail('must not prepare'))
    monkeypatch.setattr(launcher,'execute',lambda *a:pytest.fail('must not execute'))
    target=tmp_path/'configs.json'
    assert launcher.main(['--output',str(tmp_path/'out'),'--describe','--write-configs',str(target)])==0
    report=json.loads(capsys.readouterr().out)
    assert report['configurations']==12 and report['activations']==600000 and report['optimizerSteps']==3000000
    assert not report['trainingStarted'] and not report['fedSharpPlus']
    assert not (tmp_path/'out').exists() and len(json.loads(target.read_text())['configs'])==12


@pytest.fixture
def cpu():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()


@pytest.mark.parametrize('split',['iid','dirichlet'])
def test_full_model_resume_and_non_iid_original_equivalence(cpu,tmp_path,split):
    from tests.test_cart_comparison_runtime import data
    import basil_core.sharp_cart_v2_protocol as protocol
    from basil_core.sharp_protocol import save_state,load_state
    args=opts(datasets=['mnist'],partitions=[split],seeds=[2025],nodes=4,local_steps=5)
    args.rounds=3;view=clean.selected(args)[0];x,y,chunks=data()
    with clean.installed():
        worker=base.configure_worker(view,'CPU')
        with base.intervention(view,worker) as cfg:expected=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20])
        interrupted=base.configure_worker(view,'CPU')
        with base.intervention(view,interrupted) as cfg:
            partial=protocol.run(cfg,interrupted,x,y,chunks,x[:20],y[:20],max_activations=5)
        contract=dict(config=cfg,source=clean.source());path=tmp_path/'checkpoint.npz';save_state(path,partial,contract)
        resumed=base.configure_worker(view,'CPU');state=load_state(path,contract,resumed.dimension)
        with base.intervention(view,resumed) as cfg:actual=protocol.run(cfg,resumed,x,y,chunks,x[:20],y[:20],state=state)
    np.testing.assert_array_equal(actual['nodeParams'],expected['nodeParams'])
    np.testing.assert_array_equal(actual['outbound'],expected['outbound'])
    assert actual['telemetry']==expected['telemetry'] and actual['roundMetrics']==expected['roundMetrics']
    assert sum(t['training']['optimizerSteps'] for t in actual['telemetry'])==60
    penalties=[step['proximalPenalty'] for t in actual['telemetry'] for step in t['training']['steps']]
    if split=='iid':assert all(p==0 for p in penalties)
    else:
        assert any(p>0 for p in penalties)
        reference=clean.references(args)[0];worker=base.configure_worker(reference,'CPU')
        with base.intervention(reference,worker) as cfg:original=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20])
        np.testing.assert_array_equal(original['nodeParams'],expected['nodeParams'])
        assert original['telemetry']==expected['telemetry'] and original['roundMetrics']==expected['roundMetrics']


def test_gate_compares_selection_and_mu_objectives_on_both_datasets(cpu,tmp_path):
    from scripts.verify_fedsharp_clean_controls_gpu import verify
    args=opts(seeds=[2025]);views=clean.selected(args)
    with clean.installed():
        source=clean.source();(tmp_path/'plan.json').write_text(json.dumps(dict(configs=views,source=source)))
        receipt=verify(tmp_path,0,None,'CPU')
    assert receipt['status']=='PASSED' and len(receipt['checks'])==4
    assert {c['mu'] for c in receipt['checks']}=={0,1}
    assert all(c['exactSelection'] and c['exactFiveStepUpdate'] and not c['cart'] for c in receipt['checks'])


def test_execution_refuses_other_server(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher.socket,'gethostname',lambda:'themachine')
    with pytest.raises(ValueError,match='restricted to hal9k'):
        launcher.main(['--output',str(tmp_path/'out'),'--execute'])
    assert not (tmp_path/'out').exists()


def test_gate_to_scheduler_handoff_has_matching_plan_and_worker_entrypoint(tmp_path,monkeypatch):
    from dataclasses import asdict
    from basil_core import live_gpu_scheduler as live,execution_profile,cart_worker_admission
    from basil_core.execution_profile import ExecutionProfile
    from basil_core.cart_campaign_io import read_json,write_json
    root=tmp_path/'run';root.mkdir()
    views=clean.selected(opts())
    hardware={g:ExecutionProfile(name='hpc',gpu=g,gpu_uuid=f'GPU-{g:08x}',memory_mib=3072,
        headroom_mib=8192,ram_mib=8192,output_root=str(root)) for g in (0,1)}
    inventory=[dict(index=g,uuid=p.gpu_uuid,**{'memory.free':90000}) for g,p in hardware.items()]
    monkeypatch.setattr(execution_profile,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(launcher.adaptive.scaled,'available_ram_mib',lambda:100000)
    monkeypatch.setattr(launcher.adaptive.scaled,'acquire_leases',lambda profiles:{g:(tmp_path/f'lease{g}').open('a') for g in hardware})
    def gate(command,**kwargs):
        g=int(command[command.index('--gpu')+1]);assert kwargs['pass_fds']
        plan=read_json(root/'plan.json')
        write_json(root/f'revision_gpu_{g}.json',dict(status='PASSED',source=plan['source'],
            planSha256=launcher.adaptive.sha(root/'plan.json'),runtime=dict(host=launcher.socket.gethostname(),profile=asdict(hardware[g]))))
    monkeypatch.setattr(launcher.subprocess,'run',gate)
    def scheduling(args):
        assert live.WORKER==Path(launcher.__file__).resolve() and live.WORKER.name in live.OWNERS
        assert live.controller(root) is None
        actual,devices,assigned,digest=live.verified(root)
        assert len(actual['configs'])==12 and len(devices)==2 and len(assigned)==12
        return 0
    monkeypatch.setattr(live,'execute',scheduling)
    args=SimpleNamespace(output=root,wait_for_resources=False,poll_seconds=3,ntfy_topic=None)
    with clean.installed():
        plan=dict(revision=clean.REVISION,source=clean.source(),configs=views,
            profiles=[asdict(hardware[i%2]) for i in range(12)],host=launcher.socket.gethostname(),maximumWorkersPerGpu=24,
            resourceBudget=dict(context_overhead_mib=1024,ram_headroom_mib=16384,ram_mib_per_worker=8192))
        write_json(root/'plan.json',plan)
        assert launcher.execute(args,plan)==0
