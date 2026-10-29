#!/usr/bin/env python3
"""Run independent CART configurations without cancelling peers on worker failure.

--resume-plan preserves an existing frozen plan, source, slots and checkpoints.
Worker errors are attempted once per invocation; global contract errors still stop.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_cart_v2_scaled as scaled
from basil_core.cart_campaign_io import read_json,write_json,retry_io
core=scaled.core


def program_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('scripts/run_cart_v2_resilient.py','scripts/run_cart_v2_scaled.py','scripts/cart_v2_slot_worker.py',
         'basil_core/cart_campaign_io.py')}


def prepare(args):
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    root=args.output;path=root/'plan.json'
    if args.resume_plan:
        plan=read_json(path);retry_io(core.check_source,plan['source'])
        for name,value in plan['executionProgram'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:
                raise ValueError('Original execution program changed; refusing frozen-plan recovery: '+name)
        slots=[ExecutionProfile(**p).validate() for p in plan['profiles']]
        if not slots:raise ValueError('The saved plan has no GPU slots.')
        args.jobs_per_gpu=plan['jobsPerGpu'];args.profile=slots[0].name
        budget=plan.get('resourceBudget',{})
        args.context_overhead_mib=budget.get('contextOverheadMiB',1024)
        args.ram_headroom_mib=budget.get('hostHeadroomMiB',32768)
        args.allow_unattributed_process_memory=budget.get('allowUnattributedProcessMemory',False)
        args.ram_mib_per_worker=max(p.ram_mib for p in slots)
        if any(p.output_root and Path(p.output_root).resolve()!=root for p in slots):
            raise ValueError('Recovery must use the original absolute output directory.')
        hardware=list({p.gpu:p for p in slots}.values())
        inventory=gpu_inventory()
        for p in hardware:
            p.bind(inventory)
            required=args.jobs_per_gpu*(p.memory_mib+args.context_overhead_mib)+p.headroom_mib
            free=next(r['memory.free'] for r in inventory if r['index']==p.gpu)
            if free<required:raise RuntimeError(f'GPU {p.gpu}: {required} MiB free required; found {free}.')
        ram=sum(p.ram_mib for p in slots)+args.ram_headroom_mib
        if scaled.available_ram_mib()<ram:raise RuntimeError(f'Host RAM admission requires {ram} MiB available.')
        data,partitions=core.data_contract(plan['configs'])
        if data!=plan['data'] or partitions!=plan['partitions']:
            raise ValueError('Saved TRAIN data or partitions differ from the frozen plan.')
    else:
        configs=core.selected(args);source=core.fingerprint();hardware=scaled.profiles(args)
        slots=hardware*args.jobs_per_gpu;data,partitions=core.data_contract(configs)
        plan=dict(configs=configs,source=source,data=data,partitions=partitions,
            profiles=[asdict(p) for p in slots],jobsPerGpu=args.jobs_per_gpu,
            resourceBudget=dict(contextOverheadMiB=args.context_overhead_mib,hostHeadroomMiB=args.ram_headroom_mib,
                allowUnattributedProcessMemory=args.allow_unattributed_process_memory),
            executionProgram=program_hashes())
        if path.exists():
            if not args.recover:raise FileExistsError('Use a fresh output, --recover, or --resume-plan.')
            if read_json(path)!=plan:raise ValueError('Recovery plan or execution source differs.')
        elif args.recover:raise ValueError('No existing recovery plan.')
    for p in hardware:
        receipt=read_json(root/f'gpu_verification_{p.gpu}.json')
        checks={(c['dataset'],c['model']) for c in receipt['checks']}
        if (receipt['status']!='PASSED' or receipt['source']!=plan['source']
                or receipt['runtime']['profile']!=asdict(p)
                or not {(c['dataset'],c['model']) for c in plan['configs']}.issubset(checks)):
            raise ValueError('GPU verification does not match this frozen plan/profile.')
    if not path.exists():write_json(path,plan)
    for name in {**plan['source'],**plan['executionProgram']}:
        target=root/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    return plan,hardware,slots


def execute(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,hardware,slots=prepare(args);configs=plan['configs']
        # Separate orchestration provenance: never rewrite the original frozen plan.
        execution=dict(programSha256=program_hashes(),planSha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
            workerFailurePolicy='record-and-continue; one attempt per invocation')
        provenance=root/'resilient_execution.json'
        if provenance.exists() and read_json(provenance)!=execution:
            raise ValueError('Resilient execution contract changed; refusing recovery.')
        write_json(provenance,execution)
        queues={slot:deque() for slot in range(len(slots))};completed=0
        for index,cfg in enumerate(configs):
            row=root/cfg['runId']/'run.json'
            if row.exists() and read_json(row)['status']=='COMPLETED':completed+=1
            else:queues[index%len(slots)].append(index)
        failures=[];active={};stopped=False;status='RUNNING';started=time.monotonic()
        def stop(*_):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        def campaign():
            write_json(root/'campaign.json',dict(status=status,total=len(configs),completed=completed,
                failed=len(failures),pending=sum(map(len,queues.values())),activeSlots=list(active),
                pendingRunIds=[configs[i]['runId'] for queue in queues.values() for i in queue],
                activeRunIds=[configs[i]['runId'] for _,_,i in active.values()],
                failures=failures,workerFailurePolicy='continue',elapsedSeconds=time.monotonic()-started,
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
                    retry_io(core.check_source,plan['source']);index=queues[slot].popleft();cfg=configs[index]
                    log=(root/(cfg['runId']+'.log')).open('a');fd=leases[p.gpu].fileno()
                    command=[sys.executable,str(ROOT/'scripts/cart_v2_slot_worker.py'),'--worker',str(index),
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
            return 2 if failures else 0
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
    from basil_core.sharp_study import DATASETS,SEEDS,SIGMAS,CONDITION_NAMES
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group()
    for name in ('preflight','verify-gpus','execute','monitor','report'):action.add_argument('--'+name,action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='0')
    p.add_argument('--profile',choices=('local','hpc'),default='hpc')
    p.add_argument('--jobs-per-gpu',type=int,choices=range(1,9),default=2);p.add_argument('--gpu-memory-mib',type=int)
    p.add_argument('--gpu-headroom-mib',type=int,default=8192);p.add_argument('--context-overhead-mib',type=int,default=1024)
    p.add_argument('--ram-mib-per-worker',type=int,default=32768);p.add_argument('--ram-headroom-mib',type=int,default=32768)
    p.add_argument('--allow-unattributed-process-memory',action='store_true')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=['mnist','cifar10'])
    p.add_argument('--partitions',nargs='+',choices=('iid','dirichlet'),default=['iid','dirichlet'])
    p.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    p.add_argument('--conditions',nargs='+',choices=CONDITION_NAMES,default=list(CONDITION_NAMES))
    p.add_argument('--sigmas',nargs=3,type=float,default=list(SIGMAS));p.add_argument('--only-sigma',type=float)
    p.add_argument('--nodes',type=int,default=100);p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--local-steps',type=int,default=5);p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--model',choices=('paper','resnet20'),default='paper');p.add_argument('--attack-start',type=int,default=20)
    p.add_argument('--train-per-class',type=int);p.add_argument('--validation-per-class',type=int)
    recovery=p.add_mutually_exclusive_group();recovery.add_argument('--recover',action='store_true')
    recovery.add_argument('--resume-plan',action='store_true',help='Use the saved slots, configurations and checkpoints verbatim.')
    p.set_defaults(methods=['sharp'],hop_denominator='actual_selected',curvature=1.,mu=1.,learning_rate=.03,
        attack_type='hidden',budget=None,basil_window=None)
    args=p.parse_args();args.output=args.output.resolve()
    if args.gpu_headroom_mib<256 or args.context_overhead_mib<512 or args.ram_mib_per_worker<4096 or args.ram_headroom_mib<1024:
        p.error('Resource headroom/context/RAM budgets are below supported minimums.')
    if args.resume_plan and not args.execute:p.error('--resume-plan requires --execute.')
    if args.resume_plan:
        supplied={s.split('=')[0] for s in sys.argv[1:] if s.startswith('--')}
        if supplied-{'--execute','--output','--resume-plan'}:
            p.error('--resume-plan reads the frozen plan; do not supply new training/resource settings.')
    if args.verify_gpus:scaled.verify(args)
    elif args.execute:return execute(args)
    elif args.monitor:core.monitor(args)
    elif args.report:core.report(args)
    else:
        configs=core.selected(args)
        if args.preflight:
            data,partitions=core.data_contract(configs);args.output.mkdir(parents=True,exist_ok=True)
            core.write_json(args.output/'preflight.json',dict(source=core.fingerprint(),configs=configs,data=data,partitions=partitions))
        print(json.dumps(dict(configs=len(configs),nodes=args.nodes,rounds=args.rounds,
            jobsPerGpu=args.jobs_per_gpu,workerFailurePolicy='continue',trainingStarted=False),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
