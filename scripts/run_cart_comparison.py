#!/usr/bin/env python3
"""Run isolated CART mechanism, impairment and ACDS comparisons.

Failed workers are isolated. Identical-plan resume skips completed cases and
uses existing checkpoints; no checkpoint starts from zero. Old results remain.
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_cart_v2_scaled as scaled
from basil_core.cart_campaign_io import read_json,write_json,retry_io
from basil_core.cart_comparison_study import source,check_source,selected,revision,ARMS,ENVIRONMENTS,PRESETS,data_contract
core=scaled.core


def program_hashes():return source()


def prepare(args):
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    root=args.output;path=root/'plan.json'
    if args.resume:
        plan=read_json(path);check_source(plan['source'])
        slots=[ExecutionProfile(**p).validate() for p in plan['profiles']]
        hardware=list({p.gpu:p for p in slots}.values());args.jobs_per_gpu=plan['jobsPerGpu']
        args.profile=hardware[0].name
        for name,value in plan['resourceBudget'].items():setattr(args,name,value)
        if any(p.output_root and Path(p.output_root).resolve()!=root for p in slots):raise ValueError('Resume requires original output path.')
        inventory=gpu_inventory()
        for p in hardware:
            p.bind(inventory)
            required=args.jobs_per_gpu*(p.memory_mib+args.context_overhead_mib)+p.headroom_mib
            if next(r['memory.free'] for r in inventory if r['index']==p.gpu)<required:raise RuntimeError('Insufficient GPU headroom for saved slots.')
        if scaled.available_ram_mib()<sum(p.ram_mib for p in slots)+args.ram_headroom_mib:raise RuntimeError('Insufficient saved-plan host RAM.')
        data,partitions=data_contract(plan['configs'])
        if data!=plan['data'] or partitions!=plan['partitions']:raise ValueError('TRAIN contract changed.')
    else:
        noise,configs=selected(args);hardware=scaled.profiles(args);slots=hardware*args.jobs_per_gpu
        data,partitions=data_contract(configs)
        from basil_core.sharp_cart_v2_protocol import fingerprint
        plan=dict(configs=configs,source=source(),baseSource=fingerprint(),variant=args.variant,revision=revision(args.variant),
            data=data,partitions=partitions,profiles=[asdict(p) for p in slots],jobsPerGpu=args.jobs_per_gpu,
            noiseSelection=dict(count=noise.count,values=list(noise.values)),
            reference=str(args.reference.resolve()) if args.reference else None,
            figureDestination=str(args.latest_figures.resolve()) if args.latest_figures else None,
            resourceBudget={name:getattr(args,name) for name in ('context_overhead_mib','ram_headroom_mib',
                'ram_mib_per_worker','allow_unattributed_process_memory')})
        root.mkdir(parents=True,exist_ok=True)
        if path.exists() and read_json(path)!=plan:raise ValueError('Existing plan differs; use a fresh output or --resume.')
        if not path.exists():write_json(path,plan)
    return plan,hardware,slots


def verify(args):
    plan,hardware,_=prepare(args);leases=scaled.acquire_leases(hardware)
    try:
        for p in hardware:
            fd=leases[p.gpu].fileno()
            subprocess.run([sys.executable,str(ROOT/'scripts/verify_cart_comparison_gpu.py'),
                '--output',str(args.output),'--gpu',str(p.gpu),'--lease-fd',str(fd)],cwd=ROOT,env=core.worker_environment(p),pass_fds=(fd,),check=True)
    finally:
        for handle in leases.values():handle.close()


def prepared_for_execution(args):
    plan,hardware,slots=prepare(args)
    for p in hardware:
        row=read_json(args.output/f'revision_gpu_{p.gpu}.json')
        if row['status']!='PASSED' or row['source']!=plan['source'] or row['runtime']['profile']!=asdict(p):
            raise ValueError('Missing matching revision GPU verification.')
    for name in plan['source']:
        target=args.output/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    return plan,hardware,slots

def execute(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,hardware,slots=prepared_for_execution(args);configs=plan['configs']
        # Separate orchestration provenance: never rewrite the original frozen plan.
        execution=dict(programSha256=program_hashes(),planSha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
            workerFailurePolicy='record-and-continue; one attempt per invocation')
        provenance=root/'revision_execution.json'
        if provenance.exists() and read_json(provenance)!=execution:
            raise ValueError('Revision execution contract changed; refusing recovery.')
        write_json(provenance,execution)
        queues={slot:deque() for slot in range(len(slots))};completed=0
        for index,cfg in enumerate(configs):
            row=root/cfg['runId']/'run.json'
            if row.exists() and read_json(row)['status']=='COMPLETED':completed+=1
            else:queues[index%len(slots)].append(index)
        failures=[];active={};stopped=False;status='RUNNING';started=time.monotonic();report_error=None
        def stop(*_):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        def campaign():
            write_json(root/'campaign.json',dict(status=status,total=len(configs),completed=completed,
                failed=len(failures),pending=sum(map(len,queues.values())),activeSlots=list(active),
                pendingRunIds=[configs[i]['runId'] for queue in queues.values() for i in queue],
                activeRunIds=[configs[i]['runId'] for _,_,i in active.values()],
                failures=failures,workerFailurePolicy='continue',runtimeAdapter=plan['revision'],elapsedSeconds=time.monotonic()-started,
                host=socket.gethostname(),profile=args.profile,ownerPid=__import__('os').getpid()))
        def failed(index,slot,code,message):
            cfg=configs[index];path=root/cfg['runId']/'run.json';path.parent.mkdir(exist_ok=True)
            row=read_json(path) if path.exists() else {}
            cause=row.get('error') or message
            row.update(status='FAILED',error=cause,workerExitCode=code)
            write_json(path,row)
            failure=dict(runId=cfg['runId'],index=index,gpu=slots[slot].gpu,exitCode=code,
                error=cause,log=cfg['runId']+'.log')
            failures.append(failure)
            # Append-only across recoveries; a later retry cannot erase the evidence.
            with (root/'worker_failure_events.jsonl').open('a') as log:log.write(json.dumps(failure)+'\n')
            write_json(root/'failed_runs.json',failures)
            print(f'FAILED {cfg["runId"]}: {cause}; continuing other configurations.',flush=True)
        leases=scaled.acquire_leases(hardware)
        try:
            write_json(root/'failed_runs.json',failures)
            campaign()
            while any(queues.values()) or active:
                for slot,(process,handle,index) in list(active.items()):
                    code=process.poll()
                    if code is None:continue
                    handle.close();del active[slot]
                    row=root/configs[index]['runId']/'run.json'
                    saved=read_json(row) if row.exists() else {}
                    if code or saved.get('status')!='COMPLETED':
                        failed(index,slot,code,f'Worker exited with code {code}; inspect its log.')
                    else:completed+=1
                if stopped:raise InterruptedError('Supervisor stop requested.')
                for slot,p in enumerate(slots):
                    if not queues[slot] or slot in active:continue
                    owned=[proc for other,(proc,_,_) in active.items() if slots[other].gpu==p.gpu]
                    if not scaled.can_admit(p,owned,args):continue
                    retry_io(check_source,plan['source']);index=queues[slot].popleft();cfg=configs[index]
                    log=(root/(cfg['runId']+'.log')).open('a');fd=leases[p.gpu].fileno()
                    command=[sys.executable,str(ROOT/'scripts/cart_comparison_worker.py'),'--worker',str(index),
                        '--output',str(root),'--lease-fd',str(fd)]
                    try:
                        process=subprocess.Popen(command,cwd=ROOT,env=core.worker_environment(p),
                            pass_fds=(fd,),stdout=log,stderr=subprocess.STDOUT)
                    except OSError as error:
                        log.close();failed(index,slot,None,f'Worker launch failed: {error}');continue
                    active[slot]=(process,log,index)
                    print(f'GPU {p.gpu}, slot {slot}: {cfg["runId"]}',flush=True)
                campaign()
                if active or any(queues.values()):time.sleep(2)
            status='COMPLETED_WITH_FAILURES' if failures else 'COMPLETED'
            print(f'{status}: {completed}/{len(configs)} completed; {len(failures)} failed.',flush=True)
            if not failures:
                from reporting.cart_comparison import generate
                try:
                    generate(root,Path(plan.get('figureDestination') or root/'figures'),reference=plan.get('reference'))
                except Exception as error:
                    report_error=f'{type(error).__name__}: {error}';status='COMPLETED_REPORT_FAILED'
                    write_json(root/'report_status.json',dict(trainingComplete=True,status=status,error=report_error))
                    print('Training completed; report needs correction: '+report_error,flush=True)
            if args.ntfy_topic:
                from basil_core.ntfy_notifications import publish
                from basil_core.cart_notification_text import event as notice
                kind='partial' if failures else ('report_failed' if report_error else 'complete')
                cfg=configs[0]
                title,message=notice(kind,root,completed=completed,total=len(configs),failed=len(failures),nodes=cfg['nodes'],rounds=cfg['rounds'])
                receipt=publish(args.ntfy_topic,title,message)
                write_json(root/'notification.json',receipt)
            return 2 if failures else (3 if report_error else 0)
        except BaseException as error:
            status='INTERRUPTED' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'FAILED'
            for process,_,_ in active.values():
                if process.poll() is None:process.terminate()
            for process,log,_ in active.values():process.wait();log.close()
            raise
        finally:
            campaign()
            for handle in leases.values():handle.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);action=p.add_mutually_exclusive_group()
    for name in ('preflight','verify-gpus','execute','monitor'):action.add_argument('--'+name,action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preset',choices=tuple(PRESETS),default='screen');p.add_argument('--arms',nargs='+',choices=ARMS)
    p.add_argument('--environments',nargs='+',choices=ENVIRONMENTS)
    p.add_argument('--share-fraction',type=float,default=.05);p.add_argument('--acds-batches',type=int,default=5)
    p.add_argument('--reference',type=Path);p.add_argument('--latest-figures',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--ntfy-topic')
    p.add_argument('--tmux');p.add_argument('--gpus',default='0');p.add_argument('--profile',choices=('local','hpc'),default='hpc')
    p.add_argument('--jobs-per-gpu',type=int,choices=range(1,9),default=2);p.add_argument('--gpu-memory-mib',type=int)
    p.add_argument('--gpu-headroom-mib',type=int,default=8192);p.add_argument('--context-overhead-mib',type=int,default=1024)
    p.add_argument('--ram-mib-per-worker',type=int,default=32768);p.add_argument('--ram-headroom-mib',type=int,default=32768)
    p.add_argument('--allow-unattributed-process-memory',action='store_true')
    p.add_argument('--datasets',nargs='+',choices=('cifar10','mnist','cifar100'),default=['cifar10'])
    p.add_argument('--partitions',nargs='+',choices=('iid','dirichlet'),default=['iid','dirichlet'])
    p.add_argument('--seeds',nargs='+',type=int,choices=(2025,2026,2027),default=None)
    p.add_argument('--noise-levels',type=int,choices=(1,2,3),default=1);p.add_argument('--noise-values',nargs='+',type=float)
    p.add_argument('--nodes',type=int,default=100);p.add_argument('--rounds',type=int)
    p.add_argument('--local-steps',type=int,default=5);p.add_argument('--attack-start',type=int)
    p.add_argument('--alpha',type=float,default=.5);p.add_argument('--model',choices=('paper','resnet20'),default='paper')
    p.add_argument('--train-per-class',type=int);p.add_argument('--validation-per-class',type=int)
    args=p.parse_args();args.output=args.output.resolve()
    for name,value in PRESETS[args.preset].items():
        if getattr(args,name) is None:setattr(args,name,list(value) if isinstance(value,(list,tuple)) else value)
    args.variant='comparison'
    if args.gpu_headroom_mib<256 or args.context_overhead_mib<512 or args.ram_mib_per_worker<4096 or args.ram_headroom_mib<1024:
        p.error('Resource budgets below supported minimums.')
    if args.resume and not args.execute:p.error('--resume requires --execute.')
    if args.resume:
        supplied={v.split('=')[0] for v in sys.argv[1:] if v.startswith('--')}
        if supplied-{'--resume','--execute','--output','--tmux','--ntfy-topic'}:p.error('Resume uses the frozen plan; do not supply new settings.')
    if args.ntfy_topic:
        from basil_core.ntfy_notifications import validate_destination
        validate_destination(args.ntfy_topic,'https://ntfy.sh')
    if args.tmux:
        if not args.execute:p.error('--tmux requires --execute.')
        import shlex
        forwarded=sys.argv[1:];pos=forwarded.index('--tmux');forwarded=forwarded[:pos]+forwarded[pos+2:]
        args.output.mkdir(parents=True,exist_ok=True)
        command=shlex.join([sys.executable,str(Path(__file__).resolve()),*forwarded])
        command+=' > '+shlex.quote(str(args.output/'supervisor.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',args.tmux,'bash','-c',command],check=True)
        print(f'Started {args.tmux}. Training continues after SSH disconnect.');return 0
    if args.monitor:
        from basil_core.sharp_monitor import snapshot,render
        import shutil
        try:
            while True:
                state=snapshot(args.output);size=shutil.get_terminal_size()
                if sys.stdout.isatty():print('\033[2J\033[H',end='')
                print(render(state,width=size.columns,height=size.lines,table=True).replace('CART admission revision','CART mechanism comparison'),flush=True)
                if state['status'] in ('COMPLETED','COMPLETED_WITH_FAILURES','FAILED','INTERRUPTED','COMPLETED_REPORT_FAILED'):break
                time.sleep(5)
        except KeyboardInterrupt:print('Monitor closed; workers continue.')
    elif args.verify_gpus:verify(args)
    elif args.execute:return execute(args)
    elif args.preflight:
        plan,_,slots=prepare(args)
        print(json.dumps(dict(configs=len(plan['configs']),nodes=args.nodes,rounds=args.rounds,localSteps=args.local_steps,
            revision=plan['revision'],arms=args.arms,environments=args.environments,acdsShareFraction=args.share_fraction,maximumConcurrent=min(len(slots),len(plan['configs'])),trainingStarted=False),indent=2))
    else:
        noise,configs=selected(args)
        print(json.dumps(dict(configs=len(configs),noiseLevels=noise.count,noiseValues=noise.values,nodes=args.nodes,
            rounds=args.rounds,arms=args.arms,environments=args.environments,acdsShareFraction=args.share_fraction,trainingStarted=False),indent=2))


if __name__=='__main__':raise SystemExit(main())
