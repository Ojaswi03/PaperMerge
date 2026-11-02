from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from basil_core import fleet_queue as fleet
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic


def setup_queue(tmp_path,count=4):
    queue=tmp_path/'queue';queue.mkdir()
    items=[]
    for index in range(count):
        host='hal9k' if index<count//2 else 'themachine'
        cfg=dict(runId=f'run{index}',nodes=100,rounds=500,localSteps=5,dataset='mnist',partition='iid',
            seed=2025+index,sigma=.01,attackStart=20,attackType='hidden',
            comparisonCase=dict(family='basil',addon=None,environment='noise_only',effectiveAttackers=0))
        items.append(dict(config=cfg,assignedHost=host,sourceRounds=100,sourceCursor=10000,
            sourceElapsedSeconds=10.,initialAccuracy=.8))
    master=dict(queueId='test-queue',source={},items=items,probes={'hal9k':{},'themachine':{}},
        resources=dict(poll_seconds=.01,maximum_workers_per_gpu=24))
    atomic(queue/'continuation_plan.json',master)
    for host in master['probes']:
        root=tmp_path/(host+'_original');root.mkdir()
        atomic(queue/'host_bindings'/host/'owner.json',dict(output=str(root)))
        for item in items:
            if item['assignedHost']==host:
                atomic(root/item['config']['runId']/'run.json',dict(status='PENDING',completedActivations=10000,
                    completedRounds=100,elapsedSeconds=10.,meanAccuracy=.8))
    return queue,master


def join(queue,host,active=None):
    master=read_json(queue/'continuation_plan.json')
    root=Path(read_json(queue/'host_bindings'/host/'owner.json')['output']);output=root.parent/(host+'_fleet')
    rows={i['config']['runId']:read_json(root/i['config']['runId']/'run.json') for i in master['items'] if i['assignedHost']==host}
    fleet.join(queue,host,root,output,rows,active or {})
    return root,output


def test_only_joined_untouched_jobs_can_cross_servers(tmp_path):
    queue,master=setup_queue(tmp_path)
    original=(queue/'continuation_plan.json').read_bytes()
    join(queue,'hal9k')
    assert fleet.claim(queue,'hal9k',0)['runId']=='run0'
    assert fleet.claim(queue,'hal9k',0)['runId']=='run1'
    assert fleet.claim(queue,'hal9k',0) is None # donor still uses its original dispatcher
    donor,receiver=join(queue,'themachine')
    claim=fleet.claim(queue,'hal9k',0)
    assert claim['runId']=='run2' and claim['owner']=='hal9k'
    assert claim['location']['output'].endswith('hal9k_fleet')
    assert claim['location']['index']==2
    assert read_json(donor/'run2/run.json')['status']=='PENDING'
    assert (queue/'continuation_plan.json').read_bytes()==original


def test_partial_checkpoints_and_active_models_never_migrate(tmp_path):
    queue,master=setup_queue(tmp_path)
    root=Path(read_json(queue/'host_bindings/themachine/owner.json')['output'])
    atomic(root/'run2/run.json',dict(status='INTERRUPTED',completedActivations=20000,
        contract=dict(runtime=dict(profile=dict(gpu=1)))))
    (root/'run2/checkpoint.npz').write_bytes(b'partial checkpoint')
    join(queue,'themachine',{'run3':dict(pid=123,profile=dict(gpu=0))});join(queue,'hal9k')
    fleet.claim(queue,'hal9k',0);fleet.claim(queue,'hal9k',0)
    assert fleet.claim(queue,'hal9k',0) is None
    assert fleet.claim(queue,'themachine',0) is None
    assert fleet.claim(queue,'themachine',1)['runId']=='run2'
    assert (root/'run2/checkpoint.npz').read_bytes()==b'partial checkpoint'


def test_concurrent_processes_never_claim_the_same_configuration(tmp_path):
    queue,master=setup_queue(tmp_path,count=24);join(queue,'hal9k');join(queue,'themachine')
    program='''import json,sys
from basil_core.fleet_queue import claim
found=[]
while True:
    item=claim(sys.argv[1],sys.argv[2],0)
    if item is None:break
    found.append(item['runId'])
print(json.dumps(found))
'''
    processes=[subprocess.Popen([sys.executable,'-c',program,str(queue),host],stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True) for host in ('hal9k','themachine','hal9k')]
    claims=[]
    for process in processes:
        stdout,stderr=process.communicate(timeout=15)
        assert process.returncode==0,stderr
        claims+=json.loads(stdout)
    assert len(claims)==len(set(claims))==24
    assert len(list((queue/'fleet/claims').glob('*.json')))==24


def test_crash_after_exclusive_claim_cannot_release_job_to_other_host(tmp_path):
    queue,master=setup_queue(tmp_path,count=2);join(queue,'hal9k');join(queue,'themachine')
    current=fleet.state(queue);entry=current['runs']['run1']
    location=dict(output=current['hosts']['hal9k']['fleet'],index=entry['index'])
    token=fleet.reserve(queue,'run1','hal9k',location,1)
    assert fleet.claim(queue,'themachine',0)['runId']=='run0'
    saved=fleet.state(queue)['runs']['run1']
    assert saved['owner']=='hal9k' and saved['claim']==token and saved['boundGpu']==1


def test_terminal_and_stale_claims_are_not_reassigned(tmp_path):
    queue,master=setup_queue(tmp_path,count=2);join(queue,'hal9k');join(queue,'themachine')
    claim=fleet.claim(queue,'hal9k',0)
    fleet.update(queue,claim['runId'],'hal9k',claim['claim'],'RUNNING')
    other=fleet.claim(queue,'themachine',0)
    fleet.update(queue,other['runId'],'themachine',other['claim'],'FAILED')
    assert fleet.claim(queue,'themachine',0) is None
    public=fleet.public_snapshot(queue)
    assert public['failed']==1 and public['status']=='SUPERVISOR_RECOVERY_NEEDED'
    with pytest.raises(ValueError,match='collision'):
        fleet.update(queue,claim['runId'],'themachine',claim['claim'],'DONE')


def test_transferred_progress_has_one_public_row_and_no_private_fields(tmp_path):
    queue,master=setup_queue(tmp_path,count=2);join(queue,'hal9k');join(queue,'themachine')
    fleet.claim(queue,'hal9k',0);claim=fleet.claim(queue,'hal9k',1)
    item=master['items'][1]
    row=fleet.public_row(item,dict(status='RUNNING',completedActivations=25000,completedRounds=250,
        elapsedSeconds=110.,runtimeAdapter=dict(startedElapsedSeconds=10.),meanAccuracy=.9),gpu=1)
    fleet.update(queue,claim['runId'],'hal9k',claim['claim'],'RUNNING')
    atomic(queue/'fleet/reports/hal9k.json',dict(startedAt=time.time()-100,updatedAt=time.time(),
        runIds=['run1'],snapshot=dict(runs=[row])))
    public=fleet.public_snapshot(queue);raw=json.dumps(public)
    assert len(public['runs'])==2 and public['runs'][1]['completedActivations']==25000
    assert public['runs'][1]['phase'].startswith('shared:hal9k:TAKEN:')
    assert '/tmp/' not in raw and 'output' not in public and 'pid' not in raw and 'claim.json' not in raw


def test_only_original_host_can_resume_an_abandoned_started_claim(tmp_path):
    queue,master=setup_queue(tmp_path,count=2);join(queue,'hal9k');join(queue,'themachine')
    claim=fleet.claim(queue,'hal9k',0);fleet.update(queue,'run0','hal9k',claim['claim'],'RUNNING')
    fleet.resume_waiting(queue,'run0','hal9k',claim['claim'])
    other=fleet.claim(queue,'themachine',0)
    assert other['runId']=='run1'
    assert fleet.claim(queue,'themachine',0) is None
    resumed=fleet.claim(queue,'hal9k',0)
    assert resumed['claim']==claim['claim'] and resumed['location']==claim['location']
