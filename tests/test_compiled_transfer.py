import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('dataset',['mnist','cifar10'])
def test_compiled_outputs_features_losses_and_updates_match_original(dataset):
    import numpy as np
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    from basil_core.sharp_study import configuration
    cfg=configuration(dataset,'iid',2025,'attack_noise_defense_rla_cart',.01,nodes=4,local_steps=2)
    old=SharpWorker(cfg);fast=CompiledTransferWorker(cfg)
    initial=old.export();np.testing.assert_array_equal(initial,fast.export())
    shape=(28,28,1) if dataset=='mnist' else (32,32,3)
    rng=np.random.default_rng(3);x=rng.uniform(size=(9,*shape)).astype(np.float32);y=np.arange(9,dtype=np.int32)%10
    for offset in (0.,.001):
        flat=initial+np.float32(offset)
        for features in (False,True):
            np.testing.assert_array_equal(old.outputs(flat,x,features=features,batch=4),
                fast.outputs(flat,x,features=features,batch=4))
        assert old.probe(flat,x,y)==fast.probe(flat,x,y)
    a,ar=old.train(initial,x,y,0,0,.001,np.arange(9))
    b,br=fast.train(initial,x,y,0,0,.001,np.arange(9))
    np.testing.assert_array_equal(a,b);assert ar==br
    with pytest.raises(ValueError):fast.load(initial[:-1])
    with pytest.raises(ValueError):fast.load(initial*np.nan)


def test_transfer_requires_new_verified_receipt_and_preserves_plan(tmp_path,monkeypatch):
    from dataclasses import asdict
    from basil_core.execution_profile import ExecutionProfile
    import scripts.run_cart_v2_transfer as transfer
    p=ExecutionProfile(name='hpc',gpu_uuid='GPU-aaaa',output_root=str(tmp_path))
    plan=dict(source={});monkeypatch.setattr(transfer,'frozen_prepare',lambda _:(plan,[p],[p]))
    args=SimpleNamespace(output=tmp_path)
    with pytest.raises(FileNotFoundError):transfer.prepare(args)
    receipt=dict(status='PASSED',source={},programSha256=transfer.program_hashes(),runtime={'profile':asdict(p)})
    path=tmp_path/'compiled_transfer_verification_0.json';path.write_text(json.dumps(receipt))
    assert transfer.prepare(args)[0]==plan
    receipt['programSha256']={};path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='verification'):transfer.prepare(args)


# Reuse the real-child failure fixture against the separate optimized launcher.
from tests.test_cart_v2_resilient import queue,children


def test_transfer_preserves_independent_failure_and_retry_behavior(queue,monkeypatch):
    base,args,plan=queue
    import scripts.run_cart_v2_transfer as transfer
    monkeypatch.setattr(transfer,'prepare',base.prepare)
    started,_=children(monkeypatch,transfer,args.output,bad=(1,))
    assert transfer.execute(args)==2 and sorted(started)==list(range(8))
    assert json.loads((args.output/'campaign.json').read_text())['completed']==7
    started,_=children(monkeypatch,transfer,args.output,bad=())
    assert transfer.execute(args)==0 and started==[1]


def test_transfer_cli_does_not_silently_start_a_new_study(tmp_path):
    import subprocess,sys
    path=Path(__file__).resolve().parents[1]/'scripts/run_cart_v2_transfer.py'
    result=subprocess.run([sys.executable,str(path),'--execute','--output',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==2 and 'existing frozen plan' in result.stderr
    assert not list(tmp_path.iterdir())
    code=(f'import runpy,sys;sys.argv=[{str(path)!r},"--help"];'
        f'\ntry:runpy.run_path({str(path)!r},run_name="__main__")'
        '\nexcept SystemExit as e:assert e.code==0'
        '\nassert not {"numpy","tensorflow","matplotlib"}.intersection(sys.modules)')
    assert subprocess.run([sys.executable,'-I','-c',code],capture_output=True).returncode==0


def test_eta_uses_only_work_done_since_adapter_switch(tmp_path):
    from tests.test_sharp_monitor import setup,running
    from basil_core.sharp_monitor import snapshot
    configs=setup(tmp_path,[(2025,'clean','mnist',dict(running(60,110),runtimeAdapter=dict(
        name='compiled_transfer_v1',startedCursor=50,startedElapsedSeconds=100))),
        (2026,'clean','mnist',None)])
    campaign=dict(status='RUNNING',runtimeAdapter='compiled_transfer_v1')
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    result=snapshot(tmp_path)
    assert result['entries'][0]['rate']==1. and result['entries'][0]['eta']==40
    assert result['entries'][1]['eta']==100
