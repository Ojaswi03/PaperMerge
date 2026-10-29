import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from subprocess import Popen as RealPopen

import pytest


@pytest.fixture
def queue(tmp_path,monkeypatch):
    from dataclasses import asdict
    from basil_core.execution_profile import ExecutionProfile
    import scripts.run_cart_v2_resilient as module
    profile=ExecutionProfile(name='hpc',gpu_uuid='GPU-aaaa',output_root=str(tmp_path))
    configs=[dict(runId=f'job-{i}',dataset='mnist',model='paper') for i in range(8)]
    plan=dict(configs=configs,source={},data={},partitions={},profiles=[asdict(profile)]*2,jobsPerGpu=2)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(module,'prepare',lambda _:(plan,[profile],[profile]*2))
    monkeypatch.setattr(module.core,'check_source',lambda _:None)
    monkeypatch.setattr(module.core,'worker_environment',lambda _:None)
    monkeypatch.setattr(module.scaled,'can_admit',lambda *_:True)
    monkeypatch.setattr(module.scaled,'acquire_leases',lambda _:{0:(tmp_path/'lease').open('a')})
    monkeypatch.setattr(module.signal,'signal',lambda *_:None)
    # Keep real child processes/exit codes, but avoid two-second poll intervals.
    import time
    sleep=time.sleep
    monkeypatch.setattr(module.time,'sleep',lambda _:sleep(.01))
    args=SimpleNamespace(output=tmp_path,profile='hpc',resume_plan=True)
    return module,args,plan


def children(monkeypatch,module,root,*,bad=(0,),launch_error=(),missing=()):
    original=RealPopen;started=[];live=[]
    class Child:
        def __init__(self,command,**kwargs):
            index=int(command[command.index('--worker')+1]);started.append(index)
            if index in launch_error:raise OSError('fixture launch refusal')
            code=("import json,pathlib,time,sys;time.sleep(.03);"
                f"p=pathlib.Path({str(root)!r})/('job-'+str({index}));p.mkdir(exist_ok=True);"
                f"(p/'run.json').write_text(json.dumps(dict(status={'FAILED' if index in bad else 'RUNNING' if index in missing else 'COMPLETED'!r},"
                f"completedActivations=17,error={'fixture numerical failure' if index in bad else None!r})));"
                f"sys.exit({7 if index in bad else 0})")
            self.process=original([__import__('sys').executable,'-c',code],**kwargs)
            self.pid=self.process.pid;live.append(self)
        def poll(self):return self.process.poll()
        def terminate(self):raise AssertionError('A peer worker was terminated by an individual failure.')
        def wait(self):return self.process.wait()
    monkeypatch.setattr(module.subprocess,'Popen',Child)
    return started,live


def test_real_worker_exit_does_not_cancel_peers_or_pending_jobs(queue,monkeypatch):
    module,args,plan=queue;root=args.output
    complete=root/'job-7';complete.mkdir();(complete/'run.json').write_text('{"status":"COMPLETED"}')
    before=(root/'plan.json').read_bytes();started,live=children(monkeypatch,module,root,bad=(0,4))
    assert module.execute(args)==2
    campaign=json.loads((root/'campaign.json').read_text())
    assert campaign['status']=='COMPLETED_WITH_FAILURES'
    assert campaign['completed']==6 and campaign['failed']==2 and campaign['pending']==0
    assert not campaign['activeSlots'] and sorted(started)==list(range(7))
    assert all(p.poll() is not None for p in live)
    assert (root/'plan.json').read_bytes()==before
    assert {f['runId'] for f in campaign['failures']}=={'job-0','job-4'}
    assert all(f['exitCode']==7 and f['error']=='fixture numerical failure' for f in campaign['failures'])
    assert len((root/'worker_failure_events.jsonl').read_text().splitlines())==2
    assert len(json.loads((root/'failed_runs.json').read_text()))==2
    # A later explicit recovery retries only failed configurations, once each.
    started,_=children(monkeypatch,module,root,bad=())
    assert module.execute(args)==0
    assert sorted(started)==[0,4]
    assert json.loads((root/'campaign.json').read_text())['completed']==8
    assert len((root/'worker_failure_events.jsonl').read_text().splitlines())==2
    assert json.loads((root/'failed_runs.json').read_text())==[]


@pytest.mark.parametrize('launch_error,missing',[((),(0,)),((0,),())])
def test_launch_error_and_incomplete_zero_exit_are_isolated(queue,monkeypatch,launch_error,missing):
    module,args,_=queue
    started,_=children(monkeypatch,module,args.output,bad=(),launch_error=launch_error,missing=missing)
    assert module.execute(args)==2
    campaign=json.loads((args.output/'campaign.json').read_text())
    assert campaign['completed']==7 and campaign['failed']==1
    assert sorted(started)==list(range(8))


def test_recovery_preserves_original_plan_and_source_contract(tmp_path,monkeypatch):
    from dataclasses import asdict
    from basil_core.execution_profile import ExecutionProfile
    import basil_core.execution_profile as resources
    import scripts.run_cart_v2_resilient as module
    p=ExecutionProfile(name='hpc',gpu_uuid='GPU-aaaa',output_root=str(tmp_path),headroom_mib=2048)
    original='scripts/run_cart_v2_parallel.py'
    execution={original:hashlib.sha256((module.ROOT/original).read_bytes()).hexdigest()}
    cfg=dict(runId='case',dataset='mnist',model='paper')
    plan=dict(configs=[cfg],source={},data={'mnist':{}},partitions={'case':{}},
        profiles=[asdict(p)]*2,jobsPerGpu=2,executionProgram=execution)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));before=path.read_bytes()
    (tmp_path/'gpu_verification_0.json').write_text(json.dumps(dict(status='PASSED',source={},
        runtime=dict(profile=asdict(p)),checks=[cfg])))
    monkeypatch.setattr(module.core,'check_source',lambda _:None)
    monkeypatch.setattr(module.core,'data_contract',lambda _:(plan['data'],plan['partitions']))
    monkeypatch.setattr(module.scaled,'available_ram_mib',lambda:2_000_000)
    monkeypatch.setattr(resources,'gpu_inventory',lambda:[dict(index=0,uuid='GPU-aaaa',**{'memory.free':97887})])
    args=SimpleNamespace(output=tmp_path,resume_plan=True)
    recovered,_,slots=module.prepare(args)
    assert recovered==plan and path.read_bytes()==before
    assert len(slots)==2 and slots[0]==p and args.jobs_per_gpu==2
    plan['executionProgram'][original]='changed';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='Original execution program changed'):module.prepare(args)


def test_resilient_help_and_default_horizon_are_lazy(tmp_path):
    import subprocess,sys
    path=Path(__file__).resolve().parents[1]/'scripts/run_cart_v2_resilient.py'
    code=(f'import runpy,sys;sys.argv=[{str(path)!r},"--help"];'
        f'\ntry:runpy.run_path({str(path)!r},run_name="__main__")'
        '\nexcept SystemExit as e:assert e.code==0'
        '\nassert not {"numpy","tensorflow","matplotlib"}.intersection(sys.modules)')
    result=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([sys.executable,str(path),'--output',str(tmp_path),'--conditions','clean'],
        capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['rounds']==100
    assert json.loads(result.stdout)['workerFailurePolicy']=='continue'


def test_resume_plan_rejects_new_horizon_before_touching_outputs(tmp_path):
    import subprocess,sys
    result=subprocess.run([sys.executable,'scripts/run_cart_v2_resilient.py','--execute','--resume-plan',
        '--output',str(tmp_path),'--rounds','100'],capture_output=True,text=True)
    assert result.returncode!=0 and 'do not supply new training/resource settings' in result.stderr
    assert not list(tmp_path.iterdir())


def test_monitor_keeps_eta_running_after_individual_failure(tmp_path):
    from tests.test_sharp_monitor import setup,running
    from basil_core.sharp_monitor import snapshot,render
    configs=setup(tmp_path,[(2025,'clean','mnist',dict(running(20,40),status='FAILED')),
        (2026,'clean','mnist',running(20,40)),(2027,'clean','mnist',None)])
    campaign=dict(status='RUNNING',workerFailurePolicy='continue',pendingRunIds=[configs[2]['runId']],
        activeRunIds=[configs[1]['runId']])
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    state=snapshot(tmp_path)
    assert state['failed']==1 and state['eta']>0
    assert state['entries'][0]['eta'] is None
    assert 'Campaign stopped' not in render(state)
    # Old failed metadata awaiting retry is pending for the current invocation.
    campaign['pendingRunIds'].append(configs[0]['runId'])
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    assert snapshot(tmp_path)['entries'][0]['status']=='PENDING'
    campaign.update(status='COMPLETED_WITH_FAILURES',pendingRunIds=[],activeRunIds=[])
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    assert snapshot(tmp_path)['eta']==0
    assert 'Queue finished with failures' in render(snapshot(tmp_path))


def test_manager_reports_queue_exhaustion_without_pretending_all_succeeded(tmp_path,monkeypatch):
    import scripts.run_cart_v2_managed as managed
    import basil_core.ntfy_notifications as notifications
    payload=dict(status='COMPLETED_WITH_FAILURES',completed=7,total=8,failed=1,failures=[dict(runId='job-0')])
    (tmp_path/'campaign.json').write_text(json.dumps(payload));sent=[]
    monkeypatch.setattr(notifications,'publish',lambda topic,title,message,**_:
        sent.append((title,message)) or dict(deliveredToServer=True,messageId='fixture'))
    monkeypatch.setattr(managed.signal,'signal',lambda *_:None)
    monkeypatch.setattr(managed,'export_figures',lambda *_:pytest.fail('Incomplete three-seed figures must wait.'))
    args=SimpleNamespace(output=tmp_path,figures=None,watch=False,postprocess_only=True,
        ntfy_topic='fixture',ntfy_server='https://ntfy.sh',ntfy_token_file=None,timing_from=[])
    assert managed.run(args,[])==2
    saved=json.loads((tmp_path/'managed_status.json').read_text())
    assert saved['status']=='COMPLETED_WITH_FAILURES' and saved['failed']==1
    assert 'Completed: 7 / 8\nFailed: 1' in sent[-1][1]
    assert '\n\n' in sent[-1][1]


def test_io_retries_transient_eio_but_not_full_disk(monkeypatch,tmp_path):
    import errno
    import basil_core.cart_campaign_io as io
    calls=[];monkeypatch.setattr(io.time,'sleep',lambda seconds:calls.append(seconds))
    count=0
    def flaky():
        nonlocal count
        count+=1
        if count<3:raise OSError(errno.EIO,'fixture transient I/O')
        return 'ok'
    assert io.retry_io(flaky)=='ok' and count==3 and calls==[.25,.5]
    def persistent():raise OSError(errno.EIO,'fixture persistent I/O')
    with pytest.raises(OSError):io.retry_io(persistent)
    calls.clear()
    def full_disk():raise OSError(errno.ENOSPC,'fixture full disk')
    with pytest.raises(OSError):io.retry_io(full_disk)
    assert not calls
    io.write_json(tmp_path/'fixture.json',dict(complete=True))
    assert io.read_json(tmp_path/'fixture.json')==dict(complete=True)


def test_manager_monitor_eio_does_not_cancel_training(tmp_path,monkeypatch,capsys):
    import errno
    import scripts.run_cart_v2_managed as managed
    import basil_core.sharp_monitor as monitor
    monkeypatch.setattr(managed.signal,'signal',lambda *_:None)
    monkeypatch.setattr(managed.time,'sleep',lambda _:None)
    monkeypatch.setattr(managed,'export_figures',lambda *_:dict(files=[],status='FIGURES_COMPLETE'))
    checks=[]
    def unavailable(*_,**__):
        checks.append(True);raise OSError(errno.EIO,'fixture shared filesystem read failure')
    monkeypatch.setattr(monitor,'snapshot',unavailable)
    (tmp_path/'plan.json').write_text('{"configs":[]}')
    (tmp_path/'campaign.json').write_text('{"status":"COMPLETED"}')
    class Process:
        def __init__(self,*_,**__):self.returncode=None;self.polls=0
        def poll(self):
            self.polls+=1
            if self.polls==3:self.returncode=0
            return self.returncode
        def send_signal(self,*_):pytest.fail('Monitoring I/O failure cancelled training.')
    monkeypatch.setattr(managed.subprocess,'Popen',Process)
    args=SimpleNamespace(output=tmp_path,figures=None,watch=False,postprocess_only=False,
        ntfy_topic=None,timing_from=[])
    assert managed.run(args,[])==0 and len(checks)==8
    assert 'Monitor filesystem error; training continues' in capsys.readouterr().err
