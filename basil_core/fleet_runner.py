"""Join live hosts without restarting workers, then dispatch atomic fleet claims."""
from collections import Counter
from dataclasses import asdict,replace
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from basil_core import basil_sharp_thesis as study, fleet_queue as fleet
from basil_core import live_gpu_scheduler as live
from basil_core.cart_campaign_io import read_json,write_json
from basil_core.cart_worker_admission import admission
from basil_core.continuation_campaign import atomic,host_name,refresh_csv
from basil_core.execution_profile import ExecutionProfile
from basil_core.round_continuation import source_record,sha

ROOT=Path(__file__).resolve().parents[1]
GATE=ROOT/'scripts/verify_basil_sharp_thesis_gpu.py'


def prepare(queue):
    queue=Path(queue).resolve();host=host_name();master=read_json(queue/'continuation_plan.json')
    original=Path(read_json(queue/'host_bindings'/host/'owner.json')['output']).resolve()
    legacy,hardware,assigned,digest=live.verified(original)
    expected=[i['config'] for i in master['items'] if i['assignedHost']==host]
    if (legacy['configs']!=expected or legacy['source']!=master['source']
            or Path(legacy['sharedQueue']).resolve()!=queue or legacy['continuationQueueId']!=master['queueId']
            or legacy['continuationPlanSha256']!=sha(queue/'continuation_plan.json')):
        raise ValueError('Original host plan does not match the frozen continuation catalog.')
    output=original.parent/f'{host}_fedsharp-fleet-500r-{master["queueId"][:12]}'
    marker=original/'fleet_owner.json'
    if marker.exists() and read_json(marker)!=dict(queue=str(queue),output=str(output),host=host,
            masterSha256=sha(queue/'continuation_plan.json'),originalPlanSha256=digest):
        raise ValueError('Original output is fenced to a different fleet identity.')
    profiles=[asdict(replace(p,output_root=str(output))) for p in hardware.values()]
    data={};partitions={}
    for item in master['items']:
        record=source_record(item);dataset=item['config']['dataset']
        if dataset in data and data[dataset]!=record['data']:raise ValueError('Parent dataset contracts differ.')
        data[dataset]=record['data'];partitions[item['config']['runId']]=record['partition']
    plan=dict(revision=study.REVISION,variant='thesis',configs=[i['config'] for i in master['items']],
        source=master['source'],data=data,partitions=partitions,profiles=profiles,host=socket.gethostname(),
        jobsPerGpu=1,scheduler='fleet',maximumWorkersPerGpu=master['resources']['maximum_workers_per_gpu'],
        sharedQueue=str(queue),continuationQueueId=master['queueId'],
        continuationPlanSha256=sha(queue/'continuation_plan.json'),
        continuations={i['config']['runId']:i for i in master['items']},resourceBudget=legacy['resourceBudget'],
        originalPlanSha256=digest,fleetSchema=1)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():
        if read_json(output/'plan.json')!=plan:raise ValueError('Existing fleet execution plan differs.')
    else:atomic(output/'plan.json',plan)
    for name,expected_hash in plan['source'].items():
        source=queue/'source_snapshots'/name;target=output/'source_snapshots'/name
        if sha(source)!=expected_hash:raise ValueError('Shared numerical source changed.')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=expected_hash:raise ValueError('Fleet numerical source snapshot changed.')
        else:target.write_bytes(source.read_bytes())
    return original,output,legacy,plan,master


def discover(original,output,legacy,plan):
    found={}
    for root,local in ((original,legacy),(output,plan)):
        for index,item in live.discover(root,len(local['configs'])).items():
            run=local['configs'][index]['runId']
            if run in found:raise ValueError('Two live workers own the same global configuration.')
            found[run]=dict(item,output=str(root),index=index)
    return found


def discover_gates(output,hardware):
    found={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            item=live.identity(int(proc.name));script,args=live.script_command(item)
            if script!=GATE or not live.output_matches(args,item,output):continue
            gpu=int(live.argv_value(args,'--gpu'))
            env=dict(value.split(b'=',1) for value in (proc/'environ').read_bytes().split(b'\0') if b'=' in value)
            profile=ExecutionProfile(**json.loads(env[b'PAPERMERGE_EXECUTION_PROFILE']))
            if profile!=hardware[gpu] or gpu in found:raise ValueError('An existing fleet gate has an incompatible or duplicate profile.')
            found[gpu]=dict(item,profile=profile)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return found


def stop_legacy(original):
    owner=live.controller(original)
    if owner is None:return
    script,_=live.script_command(owner)
    row=read_json(original/'campaign.json')
    if script.name not in ('run_fedsharp_500r.py','retry_fedsharp_500r.py','rebalance_basil_sharp_live.py') or row.get('scheduler',{}).get('dispatch')!='shared_queue_verified_gpus':
        raise ValueError('The existing dispatcher cannot be safely joined while its workers run.')
    descriptor=os.pidfd_open(owner['pid'])
    try:
        if not live.same_process(owner):raise ProcessLookupError('Dispatcher identity changed.')
        signal.pidfd_send_signal(descriptor,signal.SIGTERM)
        if not live.select.select([descriptor],[],[],20)[0]:raise RuntimeError('Waiting for old dispatch to exit; workers remain running.')
    finally:os.close(descriptor)


def gate_passed(root,profile,plan):
    path=root/f'revision_gpu_{profile.gpu}.json'
    if not path.exists():return False
    receipt=read_json(path)
    return (receipt.get('status')=='PASSED' and receipt.get('source')==plan['source']
        and receipt.get('planSha256')==sha(root/'plan.json')
        and receipt.get('runtime',{}).get('profile')==asdict(profile))


def _row(location,run):
    path=Path(location['output'])/run/'run.json'
    return read_json(path) if path.exists() else {}


def publish(queue,host,original,active,blocked):
    current=fleet.state(queue);master=read_json(queue/'continuation_plan.json');rows=[];runs=[]
    for item in master['items']:
        run=item['config']['runId'];entry=current['runs'][run]
        if entry['owner']!=host or entry['state']=='LEGACY':continue
        row=_row(entry['location'],run)
        status=dict(DONE='COMPLETED',FAILED='FAILED',WAITING='PENDING',CLAIMED='STARTING',RUNNING='RUNNING')[entry['state']]
        value=fleet.public_row(item,row,status,gpu=entry.get('boundGpu'))
        if status=='STARTING' and f'gate-{entry.get("boundGpu")}' in active:
            value['phase']='VERIFYING_DESTINATION_GPU'
        runs.append(run);rows.append(value)
    old=queue/'hosts'/f'{host}.json'
    started=read_json(old).get('startedAt',time.time()) if old.exists() else time.time()
    report_path=queue/'fleet/reports'/f'{host}.json'
    if report_path.exists():started=read_json(report_path)['startedAt']
    atomic(report_path,dict(host=host,ownerPid=os.getpid(),updatedAt=time.time(),startedAt=started,
        runIds=runs,snapshot=dict(runs=rows),blocked=blocked,
        activeWorkers=sum(job['kind']=='worker' for job in active.values())))
    # Only the operational projection reads transferred ownership.
    from basil_core.continuation_monitor import snapshot
    refresh_csv(queue,snapshot(queue))


def run(queue,retry_request=None,ntfy_topic='papermerge-ojaswi',inherited_leases=(),handover_id=None):
    from scripts.run_cart_v2_study import worker_environment
    from scripts.run_cart_v2_scaled import available_ram_mib
    queue=Path(queue).resolve();original,output,legacy,plan,master=prepare(queue);host=host_name()
    policy_path=output/'optimization_policy.json';optimization=None
    if policy_path.exists():
        from basil_core import fedsharp_runtime_optimization
        optimization=read_json(policy_path);fedsharp_runtime_optimization.validate(optimization)
        if host!='hal9k' or optimization.get('host')!=host or optimization.get('state')!='ACTIVE':
            raise ValueError('Optimization policy is not active for this host.')
    active={};assigned={};leases={};stopped=False;blocked={}
    if inherited_leases:
        from basil_core.fleet_lease_transfer import inherited_handles
        leases=inherited_handles(inherited_leases,{p['gpu']:ExecutionProfile(**p) for p in plan['profiles']})
        for spec in inherited_leases:os.close(int(spec.split(':')[1]))
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    # One host agent fences both old launchers and concurrent fleet launchers.
    with (output/'agent.lock').open('a') as agent_lock:
        fcntl.flock(agent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=discover(original,output,legacy,plan)
        already=(original/'fleet_owner.json').exists()
        if not already:stop_legacy(original)
        with (original/'supervisor.lock').open('a') as supervision,(original/'live_scheduler.lock').open('a') as dispatch:
            fcntl.flock(supervision,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(dispatch,fcntl.LOCK_EX|fcntl.LOCK_NB)
            found=discover(original,output,legacy,plan)
            for run,item in before.items():
                if live.same_process(item) and (run not in found or found[run]['pid']!=item['pid'] or found[run]['start']!=item['start']):
                    raise ValueError('A surviving worker identity changed during activation.')
            if retry_request:
                from basil_core.continuation_recovery import requeue
                if already:raise ValueError('Apply import retries during initial fleet activation only.')
                active_indices={item['index']:item for item in found.values() if item['output']==str(original)}
                requeue(original,legacy,sha(original/'plan.json'),retry_request,active_indices)
            hardware={p['gpu']:ExecutionProfile(**p) for p in plan['profiles']}
            # Prove every survivor still uses its exact original host/GPU profile.
            for run,item in found.items():
                profile=ExecutionProfile(**item['profile']);expected=replace(hardware[profile.gpu],output_root=item['output'])
                if profile!=expected:raise ValueError('A surviving worker has an incompatible numerical/resource profile.')
            rows={cfg['runId']:_row(dict(output=str(original)),cfg['runId']) for cfg in legacy['configs']}
            atomic(original/'fleet_owner.json',dict(queue=str(queue),output=str(output),host=host,
                masterSha256=sha(queue/'continuation_plan.json'),originalPlanSha256=sha(original/'plan.json')))
            fleet.join(queue,host,original,output,rows,found)
            atomic(output/'agent.json',dict(pid=os.getpid(),host=host,source=plan['source'],planSha256=sha(output/'plan.json')))
            receipt=dict(host=host,activatedAt=time.time(),originalPlanSha256=sha(original/'plan.json'),
                fleetPlanSha256=sha(output/'plan.json'),masterSha256=sha(queue/'continuation_plan.json'),
                workersBefore=before,workersAfter=found,workerSignalsSent=False,operationalSource={})
            for name in ('basil_core/fleet_runner.py','basil_core/fleet_queue.py','scripts/run_fedsharp_fleet.py',
                    'basil_core/fleet_lease_transfer.py','scripts/refill_fedsharp_gpus.py'):
                source=ROOT/name;digest=sha(source);target=output/'operational_snapshots'/digest[:12]/name
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha(target)!=digest:raise ValueError('Operational snapshot differs.')
                if not target.exists():target.write_bytes(source.read_bytes())
                receipt['operationalSource'][name]=digest
            atomic(output/'activations'/f'{time.time_ns()}.json',receipt)
            current=fleet.state(queue)
            for run,item in found.items():
                entry=current['runs'][run]
                if entry['owner']!=host or entry['location']['output']!=item['output']:
                    raise ValueError('A survivor differs from the exclusive global claim.')
                try:process=live.Adopted(item)
                except ProcessLookupError:continue
                active[run]=dict(process=process,kind='worker',location=entry['location'],token=entry['claim'],log=None)
                assigned[run]=ExecutionProfile(**item['profile'])
                if entry['state']=='CLAIMED':fleet.update(queue,run,host,entry['claim'],'RUNNING',process=item)
            for gpu,item in discover_gates(output,hardware).items():
                key=f'gate-{gpu}'
                try:process=live.Adopted(item)
                except ProcessLookupError:continue
                active[key]=dict(process=process,kind='gate',log=None);assigned[key]=item['profile']
            # Recover only this host's stranded claims; never expire another host's claim.
            for run,entry in current['runs'].items():
                if entry['owner']!=host or entry['state'] not in ('CLAIMED','RUNNING') or run in active:continue
                row=_row(entry['location'],run)
                if row.get('status') in ('COMPLETED','FAILED'):
                    item=master['items'][entry['index']]
                    fleet.update(queue,run,host,entry['claim'],'DONE' if row['status']=='COMPLETED' else 'FAILED',public=fleet.public_row(item,row))
                elif entry['state']=='RUNNING':
                    item=master['items'][entry['index']]
                    fleet.resume_waiting(queue,run,host,entry['claim'],fleet.public_row(item,row))
            print(f'{host}: adopted {len(active)} existing workers; original plans/checkpoints preserved.',flush=True)
            if handover_id:
                atomic(output/'lease_handover_ready.json',dict(handoverId=handover_id,
                    agent=live.identity(os.getpid()),gpus=sorted(leases),workerSignalsSent=False))
            (output/'operational_adapters').mkdir(exist_ok=True)
            live.notify(output,ntfy_topic,'PaperMerge cross-server queue active',
                f'{host}: adopted {len(active)} existing processes without restarting training. Waiting jobs can move after both servers join.')
            quarantined={}
            for path in (original/'operational_adapters/gpu-quarantine.json',output/'quarantined_gpus.json'):
                if path.exists():quarantined.update({int(gpu):reason for gpu,reason in read_json(path).items()})
            try:
                while not stopped:
                    study.check_source(plan['source'])
                    if optimization:fedsharp_runtime_optimization.validate(optimization)
                    if sha(queue/'continuation_plan.json')!=plan['continuationPlanSha256']:
                        raise ValueError('Immutable global continuation catalog changed.')
                    for key,job in list(active.items()):
                        code=job['process'].poll()
                        if code is None:continue
                        if job['log']:job['log'].close()
                        if isinstance(job['process'],live.Adopted):job['process'].close()
                        del active[key]
                        if job['kind']=='gate':
                            if code!=0 or not gate_passed(output,assigned[key],plan):
                                raise RuntimeError('Destination GPU gate failed; active workers remain untouched.')
                            print(f'GPU {assigned[key].gpu}: full fleet gate passed.',flush=True)
                            continue
                        row=_row(job['location'],key);item=plan['continuations'][key]
                        if code==0 and row.get('status')=='COMPLETED':status='DONE'
                        else:
                            status='FAILED'
                            if row.get('status')!='FAILED':
                                row.update(status='FAILED',error=f'Fleet worker exited {code} without a completed result.')
                                atomic(Path(job['location']['output'])/key/'run.json',row)
                            logpath=Path(job['location']['output'])/(key+'.log');tail=''
                            if logpath.exists():
                                with logpath.open('rb') as stream:
                                    stream.seek(max(0,logpath.stat().st_size-16384));tail=stream.read().decode(errors='replace')
                            if live.gpu_failure(str(row.get('error',''))+'\n'+tail):
                                quarantined[assigned[key].gpu]=dict(reason='Worker GPU/memory failure',at=time.time())
                                atomic(output/'quarantined_gpus.json',quarantined)
                        fleet.update(queue,key,host,job['token'],status,public=fleet.public_row(item,row))
                        if status=='FAILED':live.notify(output,ntfy_topic,'PaperMerge fleet worker failed',
                            f'{host} GPU {assigned[key].gpu}: {key} was set aside. Other workers continue.')
                    current=fleet.state(queue)
                    if all(e['state'] in ('DONE','FAILED') for e in current['runs'].values()):
                        publish(queue,host,original,active,blocked)
                        failures=sum(e['state']=='FAILED' for e in current['runs'].values())
                        live.notify(output,ntfy_topic,'PaperMerge fleet queue finished',
                            f'{len(current["runs"])-failures}/{len(current["runs"])} completed; {failures} failed. {host} has finished dispatch.')
                        return 2 if failures else 0
                    try:
                        inventory=live.gpu_inventory();processes=live.gpu_processes();blocked={}
                        counts=Counter(assigned[key].gpu for key in active)
                        for gpu,profile in sorted(hardware.items(),key=lambda pair:(counts[pair[0]],pair[0])):
                            if gpu in quarantined:
                                blocked[gpu]=dict(admit=False,reason='GPU quarantined after a memory/device failure');continue
                            if not live.lease_available(profile,leases):
                                blocked[gpu]=dict(admit=False,reason='waiting for existing GPU lease');continue
                            residents=live.residents_of({key:(job['process'],job['log']) for key,job in active.items()},assigned)
                            decision=admission(profile,residents,inventory,processes,available_ram_mib(),
                                context_mib=plan['resourceBudget']['context_overhead_mib'],
                                ram_headroom_mib=plan['resourceBudget']['ram_headroom_mib'],maximum=plan['maximumWorkersPerGpu'])
                            blocked[gpu]=decision
                            if not decision['admit']:continue
                            current=fleet.state(queue)
                            reserved=[dict(runId=run,**entry) for run,entry in current['runs'].items()
                                if entry['owner']==host and entry['state']=='CLAIMED' and entry['boundGpu']==gpu and run not in active]
                            job=reserved[0] if reserved else fleet.claim(queue,host,gpu)
                            if job is None:continue
                            run=job['runId'];location=job['location'];root=Path(location['output'])
                            selected=replace(profile,output_root=str(root));local_plan=legacy if root==original else plan
                            latest=_row(location,run)
                            if latest.get('status') in ('COMPLETED','FAILED'):
                                # A survivor can finish between activation scans;
                                # never launch over an already terminal record.
                                state='DONE' if latest['status']=='COMPLETED' else 'FAILED'
                                fleet.update(queue,run,host,job['claim'],state,
                                    public=fleet.public_row(plan['continuations'][run],latest))
                                continue
                            if not gate_passed(root,selected,local_plan):
                                if root!=output:raise ValueError('Original GPU gate changed; no new job started.')
                                blocked[gpu]=dict(admit=False,reason='verifying transferred-work GPU gate')
                                if any(value['kind']=='gate' for value in active.values()):continue
                                key=f'gate-{gpu}';log=(output/f'gate-{gpu}.log').open('a')
                                command=[os.sys.executable,str(GATE),'--output',str(output),'--gpu',str(gpu),
                                    '--lease-fd',str(leases[gpu].fileno())]
                                try:
                                    process=subprocess.Popen(command,cwd=ROOT,env=worker_environment(selected),
                                        pass_fds=(leases[gpu].fileno(),),stdout=log,stderr=subprocess.STDOUT)
                                except OSError:log.close();raise
                                active[key]=dict(process=process,kind='gate',log=log);assigned[key]=selected
                                print(f'GPU {gpu}: verifying destination for transferred work.',flush=True)
                                continue
                            log=(root/(run+'.log')).open('a')
                            command=[os.sys.executable,str(live.OPTIMIZED_WORKER if optimization else live.WORKER),'--output',str(root),'--worker',str(location['index']),
                                '--lease-fd',str(leases[gpu].fileno())]
                            if optimization:command.extend(['--optimization-manifest',str(policy_path)])
                            try:
                                process=subprocess.Popen(command,cwd=ROOT,env=worker_environment(selected),
                                    pass_fds=(leases[gpu].fileno(),),stdout=log,stderr=subprocess.STDOUT)
                            except OSError:log.close();raise
                            active[run]=dict(process=process,kind='worker',location=location,token=job['claim'],log=log)
                            assigned[run]=selected
                            try:identity=live.identity(process.pid)
                            except (FileNotFoundError,ProcessLookupError):identity=dict(pid=process.pid,exitedBeforeIdentity=True)
                            fleet.update(queue,run,host,job['claim'],'RUNNING',process=identity)
                            print(f'{host} GPU {gpu}: claimed {run} from {job["originHost"]}.',flush=True)
                            # Sample again after each launch, including workers still starting.
                            inventory=live.gpu_inventory();processes=live.gpu_processes()
                    except (OSError,RuntimeError) as error:
                        blocked={'telemetry':type(error).__name__}
                        print('Fleet dispatch waiting: '+str(error),flush=True)
                    publish(queue,host,original,active,blocked)
                    time.sleep(master['resources']['poll_seconds'])
                publish(queue,host,original,active,blocked)
                return 4
            except BaseException:
                live.notify(output,ntfy_topic,'PaperMerge fleet dispatch needs recovery',
                    f'{host}: dispatch paused. Existing training workers were not signalled; inspect fleet_agent.log and resume this host agent.')
                raise
            finally:
                for job in active.values():
                    if isinstance(job['process'],live.Adopted):job['process'].close()
                    if job.get('log'):job['log'].close()
                for handle in leases.values():handle.close()
