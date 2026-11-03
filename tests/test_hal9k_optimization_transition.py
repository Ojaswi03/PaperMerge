import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from basil_core import fleet_queue as fleet
from basil_core.continuation_campaign import atomic
from basil_core.cart_campaign_io import read_json
from scripts import optimize_fedsharp_hal9k as rollout
from tests.test_fleet_queue import setup_queue,join


def test_rollout_rejects_themachine_before_reading_or_signalling(monkeypatch,tmp_path):
    monkeypatch.setattr(rollout.socket,'gethostname',lambda:'themachine')
    monkeypatch.setattr(rollout.runner,'prepare',lambda *a:pytest.fail('must not prepare'))
    monkeypatch.setattr(rollout.signal,'pidfd_send_signal',lambda *a:pytest.fail('must not signal'))
    with pytest.raises(ValueError,match='other than hal9k'):rollout.transition(tmp_path,'')


def test_reset_preserves_other_host_claims_and_exact_checkpoint(tmp_path):
    queue,master=setup_queue(tmp_path,count=2);original,output=join(queue,'hal9k');remote,_=join(queue,'themachine')
    a=fleet.claim(queue,'hal9k',1);b=fleet.claim(queue,'themachine',0)
    fleet.update(queue,a['runId'],'hal9k',a['claim'],'RUNNING')
    fleet.update(queue,b['runId'],'themachine',b['claim'],'RUNNING')
    old_remote=copy.deepcopy(fleet.state(queue)['runs'][b['runId']]);old_claim=(queue/'fleet/claims'/f'{b["runId"]}.json').read_bytes()
    contract=dict(example='unchanged');folder=original/a['runId']
    atomic(folder/'run.json',dict(status='INTERRUPTED',contract=contract,completedActivations=12340,elapsedSeconds=100))
    meta=dict(contract=contract,cursor=12345,roundMetrics=[{}]*123,elapsedSeconds=101)
    np.savez(folder/'checkpoint.npz',metadata=np.frombuffer(json.dumps(meta).encode(),np.uint8))
    before=(folder/'checkpoint.npz').read_bytes()
    receipt=rollout.resume_claims(queue,[str(original),str(output)])
    after=fleet.state(queue)
    assert after['runs'][b['runId']]==old_remote
    assert (queue/'fleet/claims'/f'{b["runId"]}.json').read_bytes()==old_claim
    assert after['runs'][a['runId']]['state']=='WAITING'
    assert after['runs'][a['runId']]['boundGpu']==1 and after['runs'][a['runId']]['claim']==a['claim']
    assert read_json(folder/'run.json')['completedActivations']==12345
    assert Path(receipt[0]['checkpointBackup']).read_bytes()==before
    assert (folder/'checkpoint.npz').read_bytes()==before


def test_live_claim_cannot_be_reset(tmp_path,monkeypatch):
    queue,master=setup_queue(tmp_path,count=2);original,output=join(queue,'hal9k')
    claim=fleet.claim(queue,'hal9k',0)
    fleet.update(queue,claim['runId'],'hal9k',claim['claim'],'RUNNING',process=dict(pid=42))
    monkeypatch.setattr(rollout.live,'same_process',lambda item:True)
    before=(queue/'fleet/state.json').read_bytes()
    with pytest.raises(ValueError,match='still running'):rollout.resume_claims(queue,[str(original),str(output)])
    assert (queue/'fleet/state.json').read_bytes()==before


def test_eta_uses_observed_post_optimization_work_and_keeps_total_elapsed(tmp_path):
    queue,master=setup_queue(tmp_path,count=2);item=master['items'][0]
    row=dict(status='RUNNING',completedActivations=30200,elapsedSeconds=1050,
        runtimeAdapter=dict(startedElapsedSeconds=10,executionOptimization=dict(startedCursor=30000,startedElapsedSeconds=1000)))
    result=fleet.public_row(item,row)
    assert result['etaSeconds']==4950 and result['etaSource']=='optimized_session'
    assert result['elapsedSeconds']==1040
    row['elapsedSeconds']=None
    assert fleet.public_row(item,row)['etaSeconds'] is None


def test_transition_signals_only_local_workers_and_keeps_remote_process_running(tmp_path,monkeypatch):
    from dataclasses import asdict
    from tests.test_live_gpu_scheduler import profile
    queue,master=setup_queue(tmp_path,count=2);original,output=join(queue,'hal9k');output.mkdir()
    remote,_=join(queue,'themachine')
    a=fleet.claim(queue,'hal9k',0);b=fleet.claim(queue,'themachine',0)
    old_remote=copy.deepcopy(fleet.state(queue)['runs'][b['runId']])
    folder=original/a['runId'];ready=tmp_path/'ready'
    code='''import json,pathlib,signal,sys,time,numpy as np
folder=pathlib.Path(sys.argv[1]); contract={'fixture':'transition'}
def stop(*args):
 meta=dict(contract=contract,cursor=10005,elapsedSeconds=20,roundMetrics=[{}]*100)
 np.savez(folder/'checkpoint.npz',metadata=np.frombuffer(json.dumps(meta).encode(),np.uint8))
 (folder/'run.json').write_text(json.dumps(dict(status='INTERRUPTED',contract=contract,completedActivations=10000,elapsedSeconds=19)))
 sys.exit(0)
signal.signal(signal.SIGTERM,stop)
pathlib.Path(sys.argv[2]).touch()
while True:time.sleep(.02)
'''
    worker=subprocess.Popen([sys.executable,'-c',code,str(folder),str(ready)])
    owner=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    try:
        deadline=time.monotonic()+10
        while not ready.exists():
            assert time.monotonic()<deadline and worker.poll() is None
            time.sleep(.02)
        identity=rollout.live.identity(worker.pid)
        fleet.update(queue,a['runId'],'hal9k',a['claim'],'RUNNING',process=identity)
        plan=dict(profiles=[asdict(profile(output,0))],source={})
        monkeypatch.setattr(rollout.socket,'gethostname',lambda:'hal9k')
        monkeypatch.setattr(rollout.runner,'prepare',lambda q:(original,output,{},plan,master))
        monkeypatch.setattr(rollout.runner,'discover_gates',lambda *a:{})
        monkeypatch.setattr(rollout,'agent_identity',lambda *a:rollout.live.identity(owner.pid))
        monkeypatch.setattr(rollout.runner,'discover',lambda *a:{'run0':identity} if worker.poll() is None else {})
        real_signal=rollout.signal.pidfd_send_signal;signalled=[]
        def send(fd,sig):
            fields=dict(line.split(':',1) for line in Path(f'/proc/self/fdinfo/{fd}').read_text().splitlines())
            pid=int(fields['Pid']);assert pid in (owner.pid,worker.pid)
            signalled.append(pid);real_signal(fd,sig)
        monkeypatch.setattr(rollout.signal,'pidfd_send_signal',send)
        launched=[]
        monkeypatch.setattr(rollout.subprocess,'run',lambda command,**kw:launched.append(command))
        rollout.transition(queue,'')
        assert set(signalled)=={owner.pid,worker.pid} and other.poll() is None
        assert fleet.state(queue)['runs'][b['runId']]==old_remote
        assert fleet.state(queue)['runs'][a['runId']]['state']=='WAITING'
        assert read_json(output/'optimization_policy.json')['state']=='ACTIVE'
        assert launched and '--background' in launched[0]
        assert read_json(folder/'run.json')['completedActivations']==10005
    finally:
        for process in (worker,owner,other):
            if process.poll() is None:process.terminate()
            process.wait(timeout=5)
