"""Server-owned continuations with dynamic GPU queues and one shared status view."""
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import socket
import time
import uuid

from basil_core import basil_sharp_thesis as study
from basil_core import round_continuation as continuation
from basil_core.cart_campaign_io import read_json,write_json


def atomic(path,value):
    """Unique temporary names also make concurrent read-only reporters harmless."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def host_name():return socket.gethostname().split('.')[0]


def probe(spec):
    from basil_core.execution_profile import gpu_inventory
    from scripts.run_cart_v2_scaled import available_ram_mib
    host=host_name()
    if host not in spec['hosts']:raise ValueError('This host is not authorized by the continuation preset.')
    inventory=gpu_inventory();resources=spec['resources'];available=available_ram_mib()
    devices=[]
    for gpu in spec['hosts'][host]['gpus']:
        row=next((r for r in inventory if r['index']==gpu),None)
        if row is None:continue
        slots=max(0,(row['memory.free']-resources['gpu_headroom_mib'])//
            (resources['gpu_memory_mib']+resources['context_overhead_mib']))
        devices.append(dict(gpu=gpu,uuid=row['uuid'],freeMiB=row['memory.free'],
            slots=min(resources['maximum_workers_per_gpu'],slots)))
    ram_slots=max(0,(available-resources['ram_headroom_mib'])//resources['ram_mib_per_worker'])
    return dict(host=host,observedAt=time.time(),devices=devices,availableRamMiB=available,
        maximumWorkersNow=min(sum(r['slots'] for r in devices),ram_slots),resources=resources)


def assignments(items,capacities):
    """Keep three-seed conditions together and balance by probed worker capacity."""
    if not capacities or any(type(v) is not int or v<=0 for v in capacities.values()):
        raise ValueError('All selected servers require positive verified capacity.')
    groups=defaultdict(list)
    for item in items:
        c=item['config'];groups[item['sourceArchive'],c['dataset'],c['partition'],c['comparisonCase']['arm']].append(item)
    load=Counter();result={}
    for key,members in sorted(groups.items()):
        if sorted(i['config']['seed'] for i in members)!=[2025,2026,2027]:
            raise ValueError('Continuation condition lacks the three matching seeds.')
        host=min(capacities,key=lambda h:(load[h]/capacities[h],h))
        for item in members:result[item['config']['runId']]=host
        load[host]+=len(members)
    return result


def initialize(queue,spec,items,hosts):
    queue=Path(queue).resolve();queue.mkdir(parents=True,exist_ok=True)
    with (queue/'initialization.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (queue/'continuation_plan.json').exists():raise FileExistsError('Continuation assignment is already frozen.')
        probes={host:read_json(queue/'probes'/f'{host}.json') for host in hosts}
        for host,p in probes.items():
            if p['host']!=host or p['resources']!=spec['resources'] or abs(time.time()-p['observedAt'])>1800:
                raise ValueError('Refresh the resource probe on '+host)
        allocation=assignments(items,{h:p['maximumWorkersNow'] for h,p in probes.items()})
        if any(i['sourceHost'].split('.')[0]!=host_name() for i in items):
            raise ValueError('Initialize on the original producing server so its local checkpoints are verified.')
        # Verify every required original path before copying a single checkpoint.
        checkpoints=[Path(i['originalCheckpoint']) for i in items]
        for path in checkpoints:
            if not path.is_file() or path.is_symlink():raise ValueError('Missing source checkpoint: '+str(path))
        needed=sum(p.stat().st_size for p in checkpoints)+2*1024**3
        if shutil.disk_usage(queue).free<needed:raise RuntimeError('Insufficient shared storage for checkpoint seed copies and headroom.')
        prepared=[]
        for index,item in enumerate(items):
            item=dict(item);target=item['config']['runId']
            relative=Path('seeds')/Path(item['sourceArchive']).name/item['sourceRunId']/'checkpoint.npz'
            item.update(seedRelativePath=str(relative),assignedHost=allocation[target])
            item['seedSha256']=continuation.copy_seed(item['originalCheckpoint'],queue/relative)
            record=continuation.source_record(item);item['initialAccuracy']=record['meanAccuracy']
            prepared.append(item)
            print(f'Seed {index+1}/{len(items)} verified and copied: {target}',flush=True)
        source=study.source()
        for name,expected in source.items():
            path=queue/'source_snapshots'/name;path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and continuation.sha(path)!=expected:raise ValueError('Conflicting shared source snapshot.')
            if not path.exists():shutil.copyfile(study.ROOT/name,path)
        master=dict(schema=1,queueId=uuid.uuid4().hex,createdAt=time.time(),revision=study.REVISION,
            specification=spec,resources=spec['resources'],probes=probes,source=source,items=prepared,
            assignmentPolicy='Fixed server ownership; one exclusive shared GPU dispatcher on each server.',
            totalConfigurations=len(prepared))
        atomic(queue/'continuation_plan.json',master)
        refresh_csv(queue,public_snapshot(queue))
        return master


def prepare_host(queue,output):
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    queue,output=Path(queue).resolve(),Path(output).resolve();master=read_json(queue/'continuation_plan.json')
    study.check_source(master['source']);host=host_name()
    own=[i for i in master['items'] if i['assignedHost']==host]
    if not own:raise ValueError('No configurations assigned to this server.')
    if output==queue or queue in output.parents or output in queue.parents:
        raise ValueError('Use a separate host-local output directory.')
    # A second output namespace on the same server must never duplicate its tasks.
    binding=queue/'host_bindings'/host;binding.parent.mkdir(exist_ok=True)
    try:
        binding.mkdir()
        atomic(binding/'owner.json',dict(host=host,output=str(output)))
    except FileExistsError:
        if not (binding/'owner.json').is_file() or read_json(binding/'owner.json')!={'host':host,'output':str(output)}:
            raise ValueError('This server is already bound to another output; no duplicate execution allowed.')
    resources=master['resources'];inventory=gpu_inventory()
    devices=[d['gpu'] for d in master['probes'][host]['devices'] if d['slots']>0]
    hardware=[ExecutionProfile(name='hpc',gpu=g,memory_mib=resources['gpu_memory_mib'],
        headroom_mib=resources['gpu_headroom_mib'],ram_mib=resources['ram_mib_per_worker'],
        threads=resources['threads'],output_root=str(output)).bind(inventory) for g in devices]
    original_by_gpu={d['gpu']:d['uuid'] for d in master['probes'][host]['devices']}
    if any(p.gpu_uuid!=original_by_gpu[p.gpu] for p in hardware):raise ValueError('Probed physical GPU identity changed.')
    if not hardware:raise ValueError('No verified GPU with probed capacity.')
    if (output/'plan.json').exists():
        plan=read_json(output/'plan.json')
        if (plan.get('sharedQueue')!=str(queue) or plan.get('continuationQueueId')!=master['queueId']
                or plan.get('continuationPlanSha256')!=continuation.sha(queue/'continuation_plan.json')
                or plan['host']!=socket.gethostname() or plan['source']!=master['source']
                or plan['configs']!=[i['config'] for i in own]):raise ValueError('Existing host plan differs.')
        return plan
    output.mkdir(parents=True,exist_ok=True)
    configs=[i['config'] for i in own];partitions={};data={};parents={}
    for item in own:
        record=continuation.source_record(item);run=item['config']['runId'];dataset=item['config']['dataset']
        if dataset in data and data[dataset]!=record['data']:raise ValueError('Source datasets do not match.')
        data[dataset]=record['data'];partitions[run]=record['partition'];parents[run]=record
    plan=dict(revision=study.REVISION,variant='thesis',configs=configs,source=master['source'],
        data=data,partitions=partitions,profiles=[asdict(hardware[i%len(hardware)]) for i in range(len(configs))],
        host=socket.gethostname(),jobsPerGpu=1,scheduler='shared',maximumWorkersPerGpu=resources['maximum_workers_per_gpu'],
        inputLoading='original_training_archives_in_memory',etaMode='activation_throughput',
        sharedQueue=str(queue),continuationQueueId=master['queueId'],
        continuationPlanSha256=continuation.sha(queue/'continuation_plan.json'),
        continuations={i['config']['runId']:i for i in own},
        resourceBudget=dict(context_overhead_mib=resources['context_overhead_mib'],
            ram_mib_per_worker=resources['ram_mib_per_worker'],ram_headroom_mib=resources['ram_headroom_mib'],
            allow_unattributed_process_memory=False))
    for name,expected in plan['source'].items():
        target=output/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(queue/'source_snapshots'/name,target)
        if continuation.sha(target)!=expected:raise ValueError('Local source snapshot differs.')
    for cfg in configs:
        run=cfg['runId'];folder=output/run;folder.mkdir(exist_ok=True);parent=parents[run]
        pending=dict(status='PENDING',phase='CONTINUATION_PENDING',config=cfg,
            completedActivations=parent['completedActivations'],completedRounds=parent['completedRounds'],
            totalActivations=cfg['nodes']*cfg['rounds'],meanAccuracy=parent['meanAccuracy'],
            elapsedSeconds=parent['elapsedSeconds'],runtimeAdapter=dict(name=study.REVISION,
                startedCursor=parent['completedActivations'],startedElapsedSeconds=parent['elapsedSeconds']))
        if (folder/'run.json').exists() and read_json(folder/'run.json')!=pending:
            raise FileExistsError('Unfrozen host output already contains different run records.')
        write_json(folder/'run.json',pending)
    write_json(output/'plan.json',plan)
    publish_host(queue,output)
    return plan


def publish_host(queue,output):
    """Only the exclusive owner on this server writes its progress snapshot."""
    from basil_core.cart_public_monitor import public_snapshot as local_snapshot
    queue,output=Path(queue),Path(output);plan=read_json(output/'plan.json')
    report=dict(host=host_name(),updatedAt=time.time(),output=str(output),
        runIds=[c['runId'] for c in plan['configs']],snapshot=local_snapshot(output))
    if any(i['assignedHost']!=host_name() for i in plan['continuations'].values()):
        raise ValueError('Attempted to publish another server\'s configurations.')
    path=queue/'hosts'/f'{host_name()}.json'
    if path.exists():report['startedAt']=read_json(path).get('startedAt',report['updatedAt'])
    else:report['startedAt']=report['updatedAt']
    atomic(path,report);refresh_csv(queue,public_snapshot(queue))


def public_snapshot(queue):
    """Use existing public schema; placement is encoded in allowlisted phase."""
    from reporting.fedsharp_publication import label
    queue=Path(queue);master=read_json(queue/'continuation_plan.json');rows={};items={}
    environments=dict(clean='clean',attack_only='attack only',joint='attack+noise',noise_only='noise only')
    for item in master['items']:
        cfg=item['config'];case=cfg['comparisonCase'];run=cfg['runId'];items[run]=item
        rows[run]=dict(gpu=None,status='PENDING',dataset=cfg['dataset'],partition=cfg['partition'],seed=cfg['seed'],
            sigma=cfg['sigma'],condition=label(case['family'],case['addon'])+' / '+environments[case['environment']],
            nodes=cfg['nodes'],rounds=cfg['rounds'],localSteps=cfg['localSteps'],attackStart=cfg['attackStart'],
            completedRounds=item['sourceRounds'],completedActivations=item['sourceCursor'],
            totalActivations=cfg['nodes']*cfg['rounds'],accuracy=item['initialAccuracy'],
            elapsedSeconds=None,etaSeconds=None,etaSource=None,
            phase=f"shared:{item['assignedHost']}:WAITING:CONTINUATION_PENDING:"+
                (cfg['attackType'] if case['effectiveAttackers'] else 'none'))
    reports=[];unhealthy=False
    for host in master['probes']:
        path=queue/'hosts'/f'{host}.json'
        if not path.exists():continue
        report=read_json(path);reports.append(report);snap=report['snapshot']
        if len(report['runIds'])!=len(snap['runs']):raise ValueError('Host status coverage differs.')
        heartbeat=queue/'heartbeats'/f'{host}.json'
        last_seen=read_json(heartbeat)['updatedAt'] if heartbeat.exists() else report['updatedAt']
        stale=time.time()-last_seen>60 and snap['status'] not in ('COMPLETED','COMPLETED_WITH_FAILURES','COMPLETED_REPORT_FAILED')
        recovery=stale or snap['status'] in ('SUPERVISOR_RECOVERY_NEEDED','FAILED','INTERRUPTED')
        unhealthy|=recovery
        for run,row in zip(report['runIds'],snap['runs']):
            if run not in items or items[run]['assignedHost']!=host:raise ValueError('Cross-server configuration ownership collision.')
            row=dict(row);state=row['status'];claim=('DONE' if state=='COMPLETED' else 'WAITING' if state=='PENDING' else 'TAKEN')
            if recovery and state not in ('COMPLETED','FAILED','PENDING'):
                state='RECOVERY_NEEDED';claim='RECOVERY_NEEDED'
            item=items[run];case=item['config']['comparisonCase']
            attack=item['config']['attackType'] if case['effectiveAttackers'] else 'none'
            row.update(status=state,condition=rows[run]['condition'],
                phase=f"shared:{host}:{claim}:{row.get('phase') or 'UNKNOWN'}:{attack}")
            rows[run]=row
    values=list(rows.values());done=sum(r['status']=='COMPLETED' for r in values);failed=sum(r['status']=='FAILED' for r in values)
    pending=sum(r['status']=='PENDING' for r in values);running=any(r['status'] in ('RUNNING','STARTING') for r in values)
    total=sum(r['totalActivations'] for r in values);completed=sum(r['completedActivations'] for r in values)
    inherited=sum(i['sourceCursor'] for i in items.values());elapsed=None;speed=None;eta=None
    if reports:
        elapsed=max(0,time.time()-min(r['startedAt'] for r in reports))
        if running and not failed and not unhealthy and elapsed>0 and completed>inherited:
            speed=(completed-inherited)/elapsed;eta=(total-completed)/speed
    status='RUNNING' if running else 'PREPARED'
    if not running:
        for phase in ('VERIFYING_GPUS','WAITING_FOR_RESOURCES','STARTING'):
            if any(r['snapshot']['status']==phase for r in reports):status=phase;break
    if unhealthy:status='SUPERVISOR_RECOVERY_NEEDED'
    if done+failed==len(values):status='COMPLETED_WITH_FAILURES' if failed else 'COMPLETED';eta=0.
    return dict(schema=1,generatedAt=datetime.now(timezone.utc).isoformat(),status=status,
        total=len(values),completed=done,failed=failed,pending=pending,completedActivations=completed,
        totalActivations=total,elapsedSeconds=elapsed,etaSeconds=eta,etaLowSeconds=eta,etaHighSeconds=eta,
        activationsPerSecond=speed,etaMode='activation_throughput',untimedJobs=0,runs=values)


def refresh_csv(queue,snapshot):
    path=Path(queue)/'configuration_status.csv';stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['SERVER','GPU','CLAIM','STATUS','DATASET','SPLIT','SEED','CONDITION','ROUND','ACTIVATIONS','ACCURACY','ETA_SECONDS'])
    for row in snapshot['runs']:
        placement=row['phase'].split(':')
        writer.writerow([placement[1],row['gpu'],placement[2],row['status'],row['dataset'],row['partition'],row['seed'],
            row['condition'],row['completedRounds'],row['completedActivations'],row['accuracy'],row['etaSeconds']])
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:temporary.write_text(stream.getvalue());temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def render_status(queue):
    from basil_core.sharp_monitor import duration
    state=public_snapshot(queue)
    lines=[f"500-round continuation | {state['status']} | {state['completed']}/{state['total']} complete | failed {state['failed']}",
        f"Progress {state['completedActivations']}/{state['totalActivations']} activations | ETA {duration(state['etaSeconds'])}",
        'SERVER      GPU CLAIM           STATUS          DATA/SPLIT       SEED CONDITION                         ROUND    ACTIVATIONS    ETA']
    for row in state['runs']:
        _,host,claim,_,attack=row['phase'].split(':',4)
        split='IID' if row['partition']=='iid' else 'nonIID'
        lines.append(f"{host:11} {str(row['gpu']):3} {claim:15} {row['status']:15} {row['dataset']+'/'+split:16} "
            f"{row['seed']} {row['condition']+' '+attack:33} {row['completedRounds']:3}/{row['rounds']:<3} "
            f"{row['completedActivations']:5}/{row['totalActivations']:<5} {duration(row['etaSeconds'])}")
    return '\n'.join(lines)
