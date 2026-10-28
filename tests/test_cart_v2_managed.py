import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_eta_pools_derived_run_ids_and_separates_shared_gpu_slots(tmp_path):
    from tests.test_sharp_monitor import setup,running
    from basil_core.sharp_monitor import snapshot
    configs=setup(tmp_path,[(2025,'clean','mnist',running(20,40)),(2026,'clean','mnist',None)])
    for i,cfg in enumerate(configs):cfg.update(baseRunId=f'base-{i}',guardedRunId=f'guard-{i}')
    plan=json.loads((tmp_path/'plan.json').read_text());plan['configs']=configs;plan['profiles']=[{'gpu':0},{'gpu':0}]
    (tmp_path/'plan.json').write_text(json.dumps(plan));state=snapshot(tmp_path)
    assert state['untimedJobs']==0 and state['eta']==200
    assert state['remainingByGpu']=={'0/slot0':160,'0/slot1':200}


def test_matching_campaign_history_replaces_a_prior(tmp_path):
    from tests.test_sharp_monitor import setup,running
    from basil_core.sharp_monitor import snapshot
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    setup(old,[(2025,'noise_rla','cifar10',dict(running(100,200),status='COMPLETED'))])
    setup(new,[(2026,'noise_rla','cifar10',None)])
    result=snapshot(new,timing_roots=[old])
    assert result['entries'][0]['eta']==200 and result['entries'][0]['etaSource']=='history'


def test_ntfy_utf8_and_retry_are_observable_without_exposing_token(tmp_path,monkeypatch):
    import basil_core.ntfy_notifications as module
    token=tmp_path/'token';token.write_text('private-fixture-token');calls=[]
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def read(self):return b'{"event":"message","id":"fixture"}'
    def send(request,timeout):
        calls.append(request)
        if len(calls)==1:raise OSError('offline')
        return Response()
    monkeypatch.setattr(module,'urlopen',send);monkeypatch.setattr(module.time,'sleep',lambda _:None)
    result=module.publish('fixture-topic','Completed','mean ± SD',token_file=token)
    assert result==dict(deliveredToServer=True,messageId='fixture',attempts=2)
    assert json.loads(calls[-1].data)['message']=='mean ± SD'
    assert 'private-fixture-token' not in json.dumps(result)


def test_multiworker_gpu_memory_budget_includes_all_caps_and_headroom(tmp_path,monkeypatch):
    import basil_core.execution_profile as resources
    from scripts.run_cart_v2_parallel import profiles
    monkeypatch.setattr(resources,'gpu_inventory',lambda:[dict(index=0,uuid='GPU-aaaa',**{'memory.free':18000})])
    args=SimpleNamespace(gpus='0',profile='hpc',gpu_memory_mib=8192,jobs_per_gpu=2,output=tmp_path)
    with pytest.raises(RuntimeError,match='18432'):profiles(args)
    args.gpu_memory_mib=6144
    assert profiles(args)[0].memory_mib==6144


def test_charts_use_seed_means_and_preserve_all_training_inputs(tmp_path,monkeypatch):
    from tests.test_cart_v2_completed_report import fixture
    from reporting.cart_v2_campaign import generate
    campaign=tmp_path/'campaign';campaign.mkdir();fixture(campaign,monkeypatch)
    hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in campaign.rglob('*') if p.is_file()}
    before=hashes();target=tmp_path/'figures';result=generate(campaign,target)
    assert hashes()==before and result['status']=='FIGURES_COMPLETE'
    assert len(list(target.glob('*learning_curves.png')))==2
    assert len(list(target.glob('*final_bars.png')))==2 and len(list(target.glob('*primary_bars.png')))==2
    summary=json.loads((target/'summary.json').read_text())
    assert summary['pairedCartEffects'][0]['effects']['final']['mean']==pytest.approx(-.02)
    with pytest.raises(FileExistsError):generate(campaign,target)


def test_failed_notification_is_recorded_after_all_retries(monkeypatch):
    import basil_core.ntfy_notifications as module
    def offline(*_,**__):raise OSError('offline')
    monkeypatch.setattr(module,'urlopen',offline);monkeypatch.setattr(module.time,'sleep',lambda _:None)
    result=module.publish('fixture-topic','Test','offline')
    assert not result['deliveredToServer'] and result['attempts']==3


def test_invalid_ntfy_credentials_are_rejected_before_training(tmp_path):
    from basil_core.ntfy_notifications import validate_destination
    with pytest.raises(ValueError,match='topic'):validate_destination('bad/topic','https://ntfy.sh')
    token=tmp_path/'empty-token';token.write_text('')
    with pytest.raises(ValueError,match='nonempty token'):validate_destination('fixture','https://ntfy.sh',token)


def test_manager_exports_then_notifies_and_reports_failure_separately(tmp_path,monkeypatch):
    from tests.test_cart_v2_completed_report import fixture
    import basil_core.ntfy_notifications as notifications
    import basil_core.sharp_monitor as tracking
    import scripts.run_cart_v2_managed as managed
    campaign=tmp_path/'campaign';campaign.mkdir();fixture(campaign,monkeypatch)
    (campaign/'campaign.json').write_text(json.dumps(dict(status='COMPLETED')))
    sent=[]
    def notify(topic,title,message,**kwargs):
        sent.append((title,message));return dict(deliveredToServer=True,messageId='fixture')
    monkeypatch.setattr(notifications,'publish',notify)
    monkeypatch.setattr(managed.signal,'signal',lambda *_:None)
    args=SimpleNamespace(output=campaign,figures=None,ntfy_topic='fixture',ntfy_server='https://ntfy.sh',
        ntfy_token_file=None,watch=False,postprocess_only=True,timing_from=[])
    assert managed.run(args,[])==0
    assert (campaign/'figures'/'artifacts_manifest.json').exists()
    assert json.loads((campaign/'managed_status.json').read_text())['status']=='COMPLETED'
    assert 'complete' in sent[-1][0].lower()
    (campaign/'campaign.json').write_text(json.dumps(dict(status='FAILED')))
    monkeypatch.setattr(tracking,'snapshot',lambda *_,**__:dict(status='FAILED'))
    args.watch=True;args.postprocess_only=False
    assert managed.run(args,[])==1
    assert 'failed' in sent[-1][0].lower()


def test_parallel_scheduler_uses_two_slots_and_skips_finished_runs(tmp_path,monkeypatch):
    from dataclasses import asdict
    from basil_core.execution_profile import ExecutionProfile
    import scripts.run_cart_v2_parallel as parallel
    p=ExecutionProfile(name='hpc',gpu=0,gpu_uuid='GPU-aaaa',memory_mib=3072,output_root=str(tmp_path))
    configs=[dict(runId=f'case-{i}',dataset='mnist',model='paper') for i in range(5)]
    complete=tmp_path/'case-0';complete.mkdir();(complete/'run.json').write_text(json.dumps(dict(status='COMPLETED')))
    receipt=dict(status='PASSED',source={},runtime={'profile':asdict(p)},checks=[{'dataset':'mnist','model':'paper'}])
    (tmp_path/'gpu_verification_0.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(parallel,'profiles',lambda _:[p]);monkeypatch.setattr(parallel.core,'selected',lambda _:configs)
    monkeypatch.setattr(parallel.core,'fingerprint',lambda:{})
    monkeypatch.setattr(parallel.core,'check_source',lambda _:None)
    monkeypatch.setattr(parallel.core,'data_contract',lambda _:({},{}))
    monkeypatch.setattr(parallel.core,'worker_environment',lambda _:{})
    monkeypatch.setattr(parallel.signal,'signal',lambda *_:None);monkeypatch.setattr(parallel.time,'sleep',lambda _:None)
    state=dict(active=0,peak=0,launched=[])
    class Process:
        def __init__(self,command,**kwargs):
            self.returncode=None;self.polls=0;state['active']+=1;state['peak']=max(state['peak'],state['active'])
            state['launched'].append(int(command[command.index('--worker')+1]))
        def poll(self):
            self.polls+=1
            if self.polls>=2 and self.returncode is None:self.returncode=0;state['active']-=1
            return self.returncode
    monkeypatch.setattr(parallel.subprocess,'Popen',Process)
    args=SimpleNamespace(output=tmp_path,jobs_per_gpu=2,recover=False,profile='hpc')
    parallel.execute(args)
    assert state['peak']==2 and sorted(state['launched'])==[1,2,3,4]
    assert json.loads((tmp_path/'campaign.json').read_text())['status']=='COMPLETED'


def test_inherited_gpu_lease_still_excludes_unrelated_workers(tmp_path,monkeypatch):
    import subprocess
    import sys
    import basil_core.execution_profile as resources
    import basil_core.iid_runtime as runtime
    from scripts.run_cart_v2_parallel import acquire_leases
    from scripts.cart_v2_slot_worker import adopt_lease
    profile=resources.ExecutionProfile(name='hpc',gpu_uuid='GPU-aaaa',memory_mib=3072,output_root=str(tmp_path))
    monkeypatch.setattr(resources,'gpu_inventory',lambda:[dict(index=0,uuid='GPU-aaaa')])
    monkeypatch.setattr(runtime,'_gpu_lease',None)
    handles=acquire_leases([profile]);fd=handles[0].fileno()
    try:
        with resources.use_profile(profile):
            wrong=(tmp_path/'wrong').open('a')
            try:
                with pytest.raises(RuntimeError,match='not for this physical GPU'):adopt_lease(wrong.fileno())
            finally:wrong.close()
            adopt_lease(fd)
        path=handles[0].name
        code='import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
        other=subprocess.run([sys.executable,'-c',code,path],capture_output=True)
        assert other.returncode!=0 and b'BlockingIOError' in other.stderr
    finally:
        for handle in runtime._gpu_lease or []:handle.close()
        runtime._gpu_lease=None
        for handle in handles.values():handle.close()


def test_orchestration_help_does_not_import_training_or_plot_libraries():
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    for script in ('run_cart_v2_managed.py','run_cart_v2_parallel.py','monitor_sharp_study.py'):
        path=root/'scripts'/script
        code=(f'import runpy,sys;sys.argv=[{str(path)!r},"--help"];'
            f'\ntry:runpy.run_path({str(path)!r},run_name="__main__")'
            '\nexcept SystemExit as e:assert e.code==0'
            '\nassert not {"numpy","tensorflow","matplotlib"}.intersection(sys.modules)')
        result=subprocess.run([sys.executable,'-I','-c',code],cwd='/tmp',capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_gpu_precheck_prefers_the_second_order_objective_when_selected(tmp_path,monkeypatch):
    import scripts.run_cart_v2_parallel as parallel
    from basil_core.execution_profile import ExecutionProfile
    profile=ExecutionProfile(name='hpc',gpu_uuid='GPU-aaaa',output_root=str(tmp_path))
    clean=dict(dataset='mnist',switches={'noise_mitigation':False})
    robust=dict(dataset='mnist',switches={'noise_mitigation':True})
    monkeypatch.setattr(parallel,'profiles',lambda _:[profile])
    monkeypatch.setattr(parallel.core,'selected',lambda _:[clean,robust])
    monkeypatch.setattr(parallel.core,'fingerprint',lambda:{})
    monkeypatch.setattr(parallel.core,'worker_environment',lambda _:{})
    monkeypatch.setattr(parallel.subprocess,'run',lambda *_,**__:None)
    parallel.verify(SimpleNamespace(output=tmp_path))
    assert json.loads((tmp_path/'verification_plan.json').read_text())['configs']==[robust]
