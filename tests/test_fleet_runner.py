from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from basil_core import fleet_runner as runner,fleet_queue as fleet
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic
from tests.test_fleet_queue import setup_queue,join
from tests.test_live_gpu_scheduler import profile


def test_execution_plan_preserves_every_config_and_original_plan_bytes(tmp_path,monkeypatch):
    queue,master=setup_queue(tmp_path)
    original=Path(read_json(queue/'host_bindings/hal9k/owner.json')['output'])
    legacy=dict(configs=[i['config'] for i in master['items'] if i['assignedHost']=='hal9k'],source={},
        resourceBudget=dict(context_overhead_mib=1024,ram_headroom_mib=16384),sharedQueue=str(queue),
        continuationQueueId=master['queueId'],continuationPlanSha256=runner.sha(queue/'continuation_plan.json'))
    atomic(original/'plan.json',legacy);before=(original/'plan.json').read_bytes()
    monkeypatch.setattr(runner,'host_name',lambda:'hal9k')
    monkeypatch.setattr(runner.live,'verified',lambda _:(legacy,{0:profile(original,0)},[],runner.sha(original/'plan.json')))
    monkeypatch.setattr(runner,'source_record',lambda i:dict(data=dict(dataset='same'),partition=dict(seed=i['config']['seed'])))
    old,output,actual,plan,frozen=runner.prepare(queue)
    assert plan['configs']==[i['config'] for i in master['items']]
    assert plan['source']==legacy['source'] and plan['continuationPlanSha256']==runner.sha(queue/'continuation_plan.json')
    assert (original/'plan.json').read_bytes()==before
    assert all(item['output_root']==str(output) for item in plan['profiles'])
    runner.prepare(queue) # repeated read/prepare preserves the same immutable plan


def test_old_fixed_server_dispatcher_is_fenced_after_join(tmp_path):
    atomic(tmp_path/'fleet_owner.json',dict(host='hal9k'))
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='cross-server fleet'):
        runner.live.execute(SimpleNamespace(output=tmp_path))


def test_live_worker_is_preserved_while_waiting_job_moves_to_other_host(tmp_path,monkeypatch):
    from scripts import run_cart_v2_study,run_cart_v2_scaled
    queue,master=setup_queue(tmp_path,count=2);join(queue,'themachine')
    original=Path(read_json(queue/'host_bindings/hal9k/owner.json')['output']);output=tmp_path/'hal9k_fleet';output.mkdir()
    legacy=dict(configs=[master['items'][0]['config']],source={},resourceBudget=dict(context_overhead_mib=1024,ram_headroom_mib=16384))
    plan=dict(configs=[i['config'] for i in master['items']],source={},profiles=[asdict(profile(output,0))],
        continuationPlanSha256=runner.sha(queue/'continuation_plan.json'),
        continuations={i['config']['runId']:i for i in master['items']},resourceBudget=legacy['resourceBudget'],maximumWorkersPerGpu=24)
    atomic(original/'plan.json',legacy);atomic(output/'plan.json',plan)
    before_master=(queue/'continuation_plan.json').read_bytes();before_plan=(original/'plan.json').read_bytes()
    atomic(original/'run0/run.json',dict(status='RUNNING',completedActivations=10000,completedRounds=100,elapsedSeconds=10.))
    ready=tmp_path/'ready';release=tmp_path/'release'
    program='''import json,pathlib,sys,time
pathlib.Path(sys.argv[2]).touch()
while not pathlib.Path(sys.argv[3]).exists():time.sleep(.01)
pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(status='COMPLETED',completedActivations=50000,completedRounds=500,elapsedSeconds=100.,meanAccuracy=.9)))
'''
    child=subprocess.Popen([sys.executable,'-c',program,str(original/'run0/run.json'),str(ready),str(release)])
    deadline=time.monotonic()+5
    while not ready.exists():
        assert child.poll() is None and time.monotonic()<deadline
        time.sleep(.01)
    identity=runner.live.identity(child.pid)
    survivor=dict(identity,profile=asdict(profile(original,0)))
    real_discover=runner.live.discover
    def discover(root,count):
        if root==original and child.poll() is None:return {0:survivor}
        return {}
    monkeypatch.setattr(runner.live,'discover',discover)
    monkeypatch.setattr(runner,'discover_gates',lambda *a:{})
    monkeypatch.setattr(runner,'prepare',lambda q:(original,output,legacy,plan,master))
    monkeypatch.setattr(runner,'host_name',lambda:'hal9k')
    monkeypatch.setattr(runner,'stop_legacy',lambda root:None)
    monkeypatch.setattr(runner.study,'check_source',lambda source:None)
    monkeypatch.setattr(runner,'gate_passed',lambda *a:True)
    hardware=profile(output,0)
    inventory=[dict(index=0,uuid=hardware.gpu_uuid,**{'memory.free':90000})]
    monkeypatch.setattr(runner.live,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(runner.live,'gpu_processes',lambda:[])
    monkeypatch.setattr(run_cart_v2_scaled,'available_ram_mib',lambda:90000)
    monkeypatch.setattr(run_cart_v2_study,'worker_environment',lambda p:{'test':'only'})
    def lease(p,leases):
        if p.gpu not in leases:leases[p.gpu]=(tmp_path/'lease').open('a')
        release.touch()
        return True
    monkeypatch.setattr(runner.live,'lease_available',lease)
    launches=[]
    class FinishedWorker:
        pid=99999999
        def __init__(self,command,**kwargs):
            root=Path(command[command.index('--output')+1]);index=int(command[command.index('--worker')+1])
            cfg=plan['configs'][index];launches.append((cfg['runId'],root))
            atomic(root/cfg['runId']/'run.json',dict(status='COMPLETED',completedActivations=50000,
                completedRounds=500,elapsedSeconds=100.,meanAccuracy=.9))
        def poll(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',FinishedWorker)
    try:
        assert runner.run(queue,ntfy_topic=None)==0
        assert child.wait(timeout=5)==0 # no termination signal interrupted the existing worker
        assert launches==[('run1',output)]
        assert all(e['state']=='DONE' for e in fleet.state(queue)['runs'].values())
        assert read_json(original/'run0/run.json')['completedActivations']==50000
        assert (queue/'continuation_plan.json').read_bytes()==before_master
        assert (original/'plan.json').read_bytes()==before_plan
        donor=Path(read_json(queue/'host_bindings/themachine/owner.json')['output'])
        assert read_json(donor/'run1/run.json')['status']=='PENDING'
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)
