"""Atomic cross-host ownership overlay; frozen experiments and plans stay intact."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic
from basil_core.round_continuation import sha


def enabled(queue):return (Path(queue)/'fleet/state.json').is_file()


@contextmanager
def locked(queue):
    queue=Path(queue);folder=queue/'fleet';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'claims.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        master=read_json(queue/'continuation_plan.json');path=folder/'state.json'
        if path.exists():
            state=read_json(path)
            if state['queueId']!=master['queueId'] or state['masterSha256']!=sha(queue/'continuation_plan.json'):
                raise ValueError('Fleet ownership does not match the immutable master plan.')
        else:
            runs={};indices=Counter()
            for index,item in enumerate(master['items']):
                host=item['assignedHost'];output=read_json(queue/'host_bindings'/host/'owner.json')['output']
                runs[item['config']['runId']]=dict(state='LEGACY',owner=host,originHost=host,index=index,
                    origin=dict(output=output,index=indices[host]),location=dict(output=output,index=indices[host]),
                    eligibleHosts=[host],everStarted=False,claim=None,boundGpu=None)
                indices[host]+=1
            state=dict(schema=1,queueId=master['queueId'],masterSha256=sha(queue/'continuation_plan.json'),
                createdAt=time.time(),hosts={},runs=runs)
        if set(state['runs'])!={item['config']['runId'] for item in master['items']}:
            raise ValueError('The fleet does not cover the exact original configuration catalog.')
        yield state,master
        atomic(path,state)


def state(queue):return read_json(Path(queue)/'fleet/state.json')


def reserve(queue,run,host,location,gpu):
    """An atomic hard-link claim also fences execution independently of flock."""
    folder=Path(queue)/'fleet/claims';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(run+'.json')
    value=dict(host=host,token=uuid.uuid4().hex,location=location,gpu=gpu)
    temporary=folder/('.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value))
        try:os.link(temporary,path)
        except FileExistsError:
            value=read_json(path)
            if value['host']!=host:raise ValueError('This configuration has an exclusive claim on another server.')
            if value['location']!=location:raise ValueError('A claimed output namespace cannot change.')
            if value['gpu']!=gpu:raise ValueError('A claimed GPU cannot change during local recovery.')
    finally:temporary.unlink(missing_ok=True)
    return value['token']


def join(queue,host,original,fleet,rows,active):
    """The host owns original supervisor locks; its old dispatcher has exited."""
    with locked(queue) as (current,master):
        previous=current['hosts'].get(host)
        if previous and (previous['original']!=str(original) or previous['fleet']!=str(fleet)):
            raise ValueError('This host is already fenced to another fleet namespace.')
        for item in master['items']:
            run=item['config']['runId'];entry=current['runs'][run]
            if entry['originHost']!=host or entry['state']!='LEGACY':continue
            row=rows[run];status=row.get('status','PENDING');cursor=row.get('completedActivations',item['sourceCursor'])
            path=Path(original)/run
            untouched=(status=='PENDING' and cursor==item['sourceCursor'] and not row.get('contract')
                and not (path/'checkpoint.npz').exists() and not (path/'continuation.json').exists())
            entry['everStarted']=not untouched
            profile=row.get('contract',{}).get('runtime',{}).get('profile')
            entry['boundGpu']=profile['gpu'] if profile else None
            entry['public']=public_row(item,row)
            if run in active:
                entry['boundGpu']=active[run]['profile']['gpu']
                token=reserve(queue,run,host,entry['location'],entry['boundGpu'])
                entry.update(state='RUNNING',claim=token,owner=host,process=active[run],everStarted=True)
            elif status=='COMPLETED':entry['state']='DONE'
            elif status=='FAILED':entry['state']='FAILED'
            else:
                if status not in ('PENDING','INTERRUPTED','RUNNING','STARTING'):
                    raise ValueError('Unknown legacy state; no ownership changed.')
                entry.update(state='WAITING',owner=None if untouched else host,
                    eligibleHosts=list(master['probes']) if untouched else [host])
        current['hosts'][host]=dict(original=str(original),fleet=str(fleet),joinedAt=(previous or {}).get('joinedAt',time.time()))


def candidates(current,host,gpu,foreign=True):
    values=[]
    for run,entry in current['runs'].items():
        if entry['state']!='WAITING' or host not in entry['eligibleHosts']:continue
        if entry.get('boundGpu') is not None and entry['boundGpu']!=gpu:continue
        local=entry['owner']==host or entry['originHost']==host
        if not local and not foreign:continue
        values.append((not local,entry['index'],run))
    return [run for _,_,run in sorted(values)]


def claim(queue,host,gpu,foreign=True):
    with locked(queue) as (current,master):
        if host not in current['hosts']:raise ValueError('Join the fleet before claiming work.')
        available=candidates(current,host,gpu,foreign)
        while available:
            run=available.pop(0);entry=current['runs'][run]
            claim_path=Path(queue)/'fleet/claims'/(run+'.json')
            if claim_path.exists() and entry['claim'] is None:
                saved=read_json(claim_path)
                # A crash after linking the claim but before saving state never
                # makes that task available to a second server.
                entry.update(state='CLAIMED',owner=saved['host'],claim=saved['token'],
                    location=saved['location'],boundGpu=saved['gpu'],eligibleHosts=[saved['host']])
                continue
            break
        else:return None
        if entry['owner']==host:
            location=entry['location']
        elif entry['originHost']==host:
            location=entry['origin']
        else:
            if entry['everStarted']:raise ValueError('A started run cannot migrate between servers.')
            location=dict(output=current['hosts'][host]['fleet'],index=entry['index'])
        token=reserve(queue,run,host,location,gpu)
        entry.update(state='CLAIMED',owner=host,claim=token,claimedAt=time.time(),
            location=location,boundGpu=gpu,eligibleHosts=[host])
        return json.loads(json.dumps(dict(runId=run,**entry)))


def update(queue,run,host,token,status,**fields):
    with locked(queue) as (current,master):
        entry=current['runs'][run]
        if entry['owner']!=host or entry['claim']!=token:raise ValueError('Cross-host claim collision.')
        saved=read_json(Path(queue)/'fleet/claims'/(run+'.json'))
        if saved['host']!=host or saved['token']!=token:raise ValueError('Exclusive execution claim differs.')
        if entry['state'] not in ('CLAIMED','RUNNING'):
            raise ValueError('A terminal or unclaimed configuration cannot be updated by this worker.')
        if status not in ('RUNNING','DONE','FAILED'):raise ValueError('Invalid claim transition.')
        entry.update(state=status,everStarted=True,**fields)


def resume_waiting(queue,run,host,token,public=None):
    """Only the same host may recover a dead worker's incomplete checkpoint."""
    with locked(queue) as (current,master):
        entry=current['runs'][run]
        if entry['owner']!=host or entry['claim']!=token or entry['state'] not in ('CLAIMED','RUNNING'):
            raise ValueError('Invalid local recovery claim.')
        entry.update(state='WAITING',eligibleHosts=[host])
        if public is not None:entry['public']=public


def public_row(item,row,status=None,gpu=None):
    from reporting.fedsharp_publication import label
    cfg=item['config'];case=cfg['comparisonCase'];adapter=row.get('runtimeAdapter',{})
    cursor=row.get('completedActivations',item['sourceCursor']);total=cfg['nodes']*cfg['rounds']
    elapsed=row.get('elapsedSeconds')
    if elapsed is not None:elapsed=max(0,elapsed-adapter.get('startedElapsedSeconds',item['sourceElapsedSeconds']))
    new=max(0,cursor-item['sourceCursor']);speed=new/elapsed if elapsed and new else None
    status=status or row.get('status','PENDING')
    env=dict(clean='clean',joint='attack+noise',attack_only='attack only',noise_only='noise only')
    return dict(gpu=row.get('physicalGpuId',gpu),status=status,dataset=cfg['dataset'],partition=cfg['partition'],
        seed=cfg['seed'],sigma=cfg['sigma'],condition=label(case['family'],case['addon'])+' / '+env[case['environment']],
        nodes=cfg['nodes'],rounds=cfg['rounds'],localSteps=cfg['localSteps'],attackStart=cfg['attackStart'],
        completedRounds=row.get('completedRounds',item['sourceRounds']),completedActivations=cursor,totalActivations=total,
        accuracy=row.get('meanAccuracy',item['initialAccuracy']),elapsedSeconds=elapsed,
        etaSeconds=0. if status=='COMPLETED' else (total-cursor)/speed if speed and status=='RUNNING' else None,
        etaSource='measured' if speed else None,phase=row.get('phase','CONTINUATION_PENDING'))


def public_snapshot(queue):
    queue=Path(queue);current=state(queue);master=read_json(queue/'continuation_plan.json')
    reports={};starts=[];ends=[]
    for host in master['probes']:
        path=queue/('fleet/reports' if host in current['hosts'] else 'hosts')/(host+'.json')
        if not path.exists():continue
        report=read_json(path);starts.append(report['startedAt']);ends.append(report['updatedAt'])
        if len(report['runIds'])!=len(report['snapshot']['runs']) or len(set(report['runIds']))!=len(report['runIds']):
            raise ValueError('Invalid fleet report coverage.')
        reports[host]=dict(report,rows=dict(zip(report['runIds'],report['snapshot']['runs'])))
    rows=[];unhealthy=False
    for item in master['items']:
        run=item['config']['runId'];entry=current['runs'][run];host=entry['owner']
        row=entry.get('public',public_row(item,{})).copy()
        if entry['state']=='WAITING':
            row['status']='PENDING';row['gpu']=entry.get('boundGpu')
        elif entry['state']=='LEGACY':
            row=reports.get(host,{}).get('rows',{}).get(run,row).copy()
            if row['status']=='RECOVERY_NEEDED':unhealthy=True
        else:
            row=reports.get(host,{}).get('rows',{}).get(run,entry.get('public',row)).copy()
            if entry['state']=='DONE':row['status']='COMPLETED'
            elif entry['state']=='FAILED':row['status']='FAILED'
            elif entry['state']=='CLAIMED':row['status']='STARTING'
            else:row['status']='RUNNING' if row['status'] not in ('COMPLETED','FAILED') else row['status']
            if entry['state'] in ('CLAIMED','RUNNING') and time.time()-reports.get(host,{}).get('updatedAt',0)>60:
                row['status']='RECOVERY_NEEDED';unhealthy=True
        claim_label='DONE' if row['status']=='COMPLETED' else 'WAITING' if row['status']=='PENDING' else 'TAKEN'
        attack=item['config']['attackType'] if item['config']['comparisonCase']['effectiveAttackers'] else 'none'
        phase=row.get('phase') or 'UNKNOWN'
        if phase.startswith('shared:'):phase=phase.split(':',4)[3]
        row['phase']=f"shared:{host or 'pool'}:{claim_label}:{phase}:{attack}"
        rows.append(row)
    done=sum(r['status']=='COMPLETED' for r in rows);failed=sum(r['status']=='FAILED' for r in rows)
    running=any(r['status'] in ('RUNNING','STARTING') for r in rows)
    status='RUNNING' if running else 'WAITING_FOR_RESOURCES'
    if unhealthy:status='SUPERVISOR_RECOVERY_NEEDED'
    if done+failed==len(rows):status='COMPLETED_WITH_FAILURES' if failed else 'COMPLETED'
    elapsed=max(0,(max(ends) if done+failed==len(rows) and ends else time.time())-min(starts)) if starts else None
    return dict(schema=1,generatedAt=datetime.now(timezone.utc).isoformat(),status=status,total=len(rows),
        completed=done,failed=failed,pending=sum(r['status']=='PENDING' for r in rows),
        completedActivations=sum(r['completedActivations'] for r in rows),totalActivations=sum(r['totalActivations'] for r in rows),
        elapsedSeconds=elapsed,etaSeconds=None,etaLowSeconds=None,etaHighSeconds=None,
        activationsPerSecond=None,etaMode='activation_throughput',untimedJobs=0,runs=rows)
