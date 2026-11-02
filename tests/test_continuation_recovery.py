import json
from pathlib import Path
import time

import pytest

from basil_core.continuation_recovery import requeue,validate
from basil_core.cart_campaign_io import write_json,read_json
from basil_core.round_continuation import sha


def fixture(tmp_path):
    root=tmp_path/'output';root.mkdir();queue=tmp_path/'queue';queue.mkdir()
    seed=queue/'seed.npz';seed.write_bytes(b'verified seed')
    rows={}
    for run,status in [('failed','FAILED'),('healthy','RUNNING'),('done','COMPLETED')]:
        folder=root/run;folder.mkdir();write_json(folder/'run.json',dict(status=status,
            completedActivations=10000,error='earlier failure',workerExitCode=1))
        rows[run]=(folder/'run.json').read_bytes()
    plan=dict(configs=[dict(runId=name) for name in rows],sharedQueue=str(queue),
        continuations={'failed':dict(sourceCursor=10000,seedRelativePath='seed.npz',seedSha256=sha(seed))})
    write_json(root/'plan.json',plan);digest=sha(root/'plan.json')
    request=dict(id='request',output=str(root),planSha256=digest,verifiedAt=time.time(),
        datasetChecksPassed=True,runs=[dict(runId='failed',failedRowSha256=sha(root/'failed/run.json'))])
    path=tmp_path/'request.json';write_json(path,request)
    return root,plan,digest,path,rows


def test_boundary_retry_preserves_failed_attempt_and_never_changes_healthy_rows(tmp_path):
    root,plan,digest,path,rows=fixture(tmp_path)
    assert requeue(root,plan,digest,path,{1:object()})==['failed']
    assert read_json(root/'failed/run.json')['status']=='PENDING'
    assert (root/'operational_adapters/continuation_retries/failed/failed-run.json').read_bytes()==rows['failed']
    assert all((root/name/'run.json').read_bytes()==rows[name] for name in ('healthy','done'))
    assert sha(root/'plan.json')==digest


@pytest.mark.parametrize('failure',['active','changed','advanced','checkpoint','consumed','expired','seed','dataset'])
def test_retry_refuses_unsafe_or_unverified_target_before_any_reset(tmp_path,failure):
    root,plan,digest,path,rows=fixture(tmp_path);active={}
    if failure=='active':active[0]=object()
    elif failure in ('changed','advanced'):
        row=read_json(root/'failed/run.json');row['completedActivations']=10001 if failure=='advanced' else 10000
        row['extra']='changed';write_json(root/'failed/run.json',row)
    elif failure=='checkpoint':(root/'failed/checkpoint.npz').touch()
    elif failure=='consumed':(root/'operational_adapters/continuation_retries/failed').mkdir(parents=True)
    elif failure=='seed':(Path(plan['sharedQueue'])/'seed.npz').write_bytes(b'changed')
    else:
        request=read_json(path)
        if failure=='expired':request['verifiedAt']-=1801
        else:request['datasetChecksPassed']=False
        write_json(path,request)
    before=(root/'failed/run.json').read_bytes()
    with pytest.raises(ValueError):validate(root,plan,digest,path,active)
    assert (root/'failed/run.json').read_bytes()==before


def test_cooperative_live_dispatch_stop_does_not_signal_training_worker(tmp_path,monkeypatch):
    import os
    import signal
    from basil_core import continuation_recovery as recovery,live_gpu_scheduler as live
    root,plan,digest,path,rows=fixture(tmp_path)
    owner=dict(pid=12345,argv=['python',str(live.ROOT/'scripts/run_fedsharp_500r.py'),'--execute','--output',str(root)])
    monkeypatch.setattr(live,'controller',lambda _:owner)
    monkeypatch.setattr(live,'discover',lambda *a:{1:dict(pid=54321)})
    monkeypatch.setattr(live,'same_process',lambda _:True)
    monkeypatch.setattr(recovery.os,'pidfd_open',lambda pid:os.open('/dev/null',os.O_RDONLY))
    signals=[];monkeypatch.setattr(recovery.signal,'pidfd_send_signal',lambda fd,sig:signals.append(sig))
    monkeypatch.setattr(recovery.select,'select',lambda *a:([a[0][0]],[],[]))
    write_json(root/'campaign.json',dict(scheduler=dict(dispatch='shared_queue_verified_gpus')))
    recovery.stop_live_dispatch(root,plan,digest,path)
    assert signals==[signal.SIGTERM]
    assert all((root/name/'run.json').read_bytes()==data for name,data in rows.items())


def test_retry_releases_live_locks_and_adopts_real_surviving_worker(tmp_path,monkeypatch):
    from dataclasses import asdict
    import os
    import signal
    import socket
    import subprocess
    import sys
    from types import SimpleNamespace
    from basil_core import live_gpu_scheduler as live
    from basil_core.continuation_recovery import stop_live_dispatch
    from tests.test_live_gpu_scheduler import profile
    root,plan,digest,path,rows=fixture(tmp_path)
    project=tmp_path/'project';(project/'scripts').mkdir(parents=True);(project/'basil_core').mkdir()
    for name in ('basil_core/live_gpu_scheduler.py','scripts/rebalance_basil_sharp_live.py'):
        (project/name).write_bytes((live.ROOT/name).read_bytes())
    worker=project/'scripts/basil_sharp_thesis_worker.py';worker.write_text('import time\ntime.sleep(60)\n')
    dispatcher=project/'scripts/run_fedsharp_500r.py';ready=tmp_path/'ready'
    dispatcher.write_text('''import fcntl,json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[2]);locks=[]
for name in ('supervisor.lock','live_scheduler.lock'):
    handle=(root/name).open('a');fcntl.flock(handle,fcntl.LOCK_EX);locks.append(handle)
child=subprocess.Popen([sys.executable,sys.argv[1],'--worker','1','--output',str(root)],env=os.environ.copy())
stopped=False
def stop(*args):
    global stopped
    stopped=True
signal.signal(signal.SIGTERM,stop)
(root/'campaign.json').write_text(json.dumps(dict(host=__import__('socket').gethostname(),ownerPid=os.getpid(),status='RUNNING',elapsedSeconds=0,scheduler=dict(dispatch='shared_queue_verified_gpus'))))
pathlib.Path(sys.argv[3]).write_text(str(child.pid))
while not stopped:time.sleep(.01)
''')
    p=profile(root,0)
    parent=subprocess.Popen([sys.executable,str(dispatcher),str(worker),str(root),str(ready),
        '--execute','--output',str(root)],cwd=project,
        env={**os.environ,'PAPERMERGE_EXECUTION_PROFILE':json.dumps(asdict(p))})
    child=None
    try:
        deadline=time.monotonic()+5
        while not ready.exists():
            assert parent.poll() is None and time.monotonic()<deadline
            time.sleep(.01)
        child=int(ready.read_text());child_identity=live.identity(child)
        monkeypatch.setattr(live,'ROOT',project);monkeypatch.setattr(live,'WORKER',worker)
        stop_live_dispatch(root,plan,digest,path)
        assert parent.wait(timeout=5)==0 and live.same_process(child_identity)
        monkeypatch.setattr(live,'verified',lambda _:(plan,{0:p},[p,p,p],digest))
        def schedule(args,actual_plan,hardware,assigned,actual_digest,active,leases):
            assert active[1][0].pid==child and active[1][0].poll() is None
            assert read_json(root/'failed/run.json')['status']=='PENDING'
            assert all((root/name/'run.json').read_bytes()==rows[name] for name in ('healthy','done'))
            return 0
        monkeypatch.setattr(live,'schedule',schedule)
        assert live.execute(SimpleNamespace(output=root,retry_request=path))==0
        assert live.same_process(child_identity) and sha(root/'plan.json')==digest
    finally:
        if child:
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
        if parent.poll() is None:parent.kill()
        parent.wait(timeout=5)


def test_selected_import_retry_does_not_include_other_failed_runs(tmp_path,monkeypatch):
    from scripts import retry_fedsharp_500r as retry
    root=tmp_path/'output';root.mkdir()
    for run in ('boundary','later'):
        (root/run).mkdir()
        write_json(root/run/'run.json',dict(status='FAILED',error='original failure'))
    before=(root/'later/run.json').read_bytes()
    monkeypatch.setattr(retry.live,'verified',lambda _:(dict(sharedQueue=str(tmp_path)),{},[], 'plan-sha'))
    check=dict(path=str(tmp_path/'seed'),matchesFrozenSeed=True,metadataReadable=True,
        originalContractMatches=True,boundaryMatches=True,stableChecksum=True)
    monkeypatch.setattr(retry,'diagnose',lambda *a,**kw:dict(producerUnchanged=True,
        failures=[dict(runId=run,checkpoints={'shared':check}) for run in ('boundary','later')],
        datasets={'mnist':dict(reads=[dict(verified=True),dict(verified=True)])}))
    plan,request=retry.prepare(tmp_path,root,run_ids=['boundary'])
    assert [item['runId'] for item in request['runs']]==['boundary']
    assert (root/'later/run.json').read_bytes()==before
