#!/usr/bin/env python3
"""Resume a frozen CART plan with a verified compiled-transfer runtime adapter."""
from __future__ import annotations
import argparse
from collections import deque
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
from scripts.run_cart_v2_resilient import prepare as frozen_prepare
from basil_core.cart_campaign_io import read_json,write_json,retry_io
core=scaled.core

def program_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('scripts/run_cart_v2_transfer.py','scripts/cart_v2_transfer_worker.py',
         'scripts/verify_cart_v2_transfer.py','basil_core/sharp_compiled_transfer.py',
         'scripts/run_cart_v2_resilient.py','scripts/run_cart_v2_scaled.py',
         'scripts/cart_v2_slot_worker.py','basil_core/cart_campaign_io.py')}

def prepare(args):
    from dataclasses import asdict
    plan,hardware,slots=frozen_prepare(args)
    for p in hardware:
        receipt=read_json(args.output/f'compiled_transfer_verification_{p.gpu}.json')
        if (receipt['status']!='PASSED' or receipt['source']!=plan['source']
                or receipt['programSha256']!=program_hashes()
                or receipt['runtime']['profile']!=asdict(p)):
            raise ValueError('Compiled-transfer verification does not match runtime/source/profile.')
    for name in program_hashes():
        target=args.output/'runtime_sources'/'compiled_transfer_v1'/name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    return plan,hardware,slots

def verify(args):
    from basil_core.execution_profile import ExecutionProfile
    with (args.output/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read_json(args.output/'plan.json');core.check_source(plan['source'])
        hardware={p['gpu']:ExecutionProfile(**p) for p in plan['profiles']}
        for gpu,p in hardware.items():
            command=[sys.executable,str(ROOT/'scripts/verify_cart_v2_transfer.py'),
                '--output',str(args.output),'--gpu',str(gpu)]
            subprocess.run(command,cwd=ROOT,env=core.worker_environment(p),check=True)

def execute(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,hardware,slots=prepare(args);configs=plan['configs']
        # Separate orchestration provenance: never rewrite the original frozen plan.
        execution=dict(programSha256=program_hashes(),planSha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
            workerFailurePolicy='record-and-continue; one attempt per invocation')
        provenance=root/'compiled_transfer_execution.json'
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
                runtimeAdapter='compiled_transfer_v1',
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
                    retry_io(core.check_source,plan['source'])
                    if program_hashes()!=execution['programSha256']:raise ValueError('Compiled-transfer runtime source changed.')
                    index=queues[slot].popleft();cfg=configs[index]
                    log=(root/(cfg['runId']+'.log')).open('a');fd=leases[p.gpu].fileno()
                    command=[sys.executable,str(ROOT/'scripts/cart_v2_transfer_worker.py'),'--worker',str(index),
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
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-gpus',action='store_true')
    action.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-plan',action='store_true')
    args=p.parse_args();args.output=args.output.resolve()
    if args.execute and not args.resume_plan:p.error('Compiled transfer resumes an existing frozen plan; use --resume-plan.')
    if args.verify_gpus:verify(args);return 0
    return execute(args)

if __name__=='__main__':raise SystemExit(main())
