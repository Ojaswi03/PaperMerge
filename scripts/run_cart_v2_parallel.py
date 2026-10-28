#!/usr/bin/env python3
"""Schedule independent frozen CART workers in bounded slots per physical GPU."""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_cart_v2_study as core


def acquire_leases(hardware):
    handles={}
    try:
        for p in hardware:
            path=Path('/tmp')/f'papermerge-{os.getuid()}-{p.gpu_uuid}.lock'
            handle=path.open('a');handles[p.gpu]=handle
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return handles
    except BaseException:
        for handle in handles.values():handle.close()
        raise RuntimeError('A physical GPU is held by another PaperMerge supervisor/worker.')


def profiles(args):
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    inventory=gpu_inventory();ids=[int(v) for v in args.gpus.split(',')]
    if len(ids)!=len(set(ids)):raise ValueError('GPU IDs must be distinct.')
    memory=args.gpu_memory_mib or (8192 if args.profile=='hpc' else 6144)
    result=[]
    for gpu in ids:
        profile=ExecutionProfile(name=args.profile,gpu=gpu,memory_mib=memory,
            headroom_mib=2048 if args.profile=='hpc' else 1024,
            output_root=str(args.output.resolve()) if args.profile=='hpc' else '').bind(inventory)
        available=next(r['memory.free'] for r in inventory if r['index']==gpu)
        required=args.jobs_per_gpu*memory+profile.headroom_mib
        if available<required:raise RuntimeError(f'GPU {gpu}: {required} MiB free required for {args.jobs_per_gpu} slots; found {available}.')
        result.append(profile)
    return result


def verify(args):
    hardware=profiles(args);configs=core.selected(args);args.output.mkdir(parents=True,exist_ok=True)
    chosen=[]
    for dataset in dict.fromkeys(c['dataset'] for c in configs):
        candidates=[c for c in configs if c['dataset']==dataset]
        chosen.append(next((c for c in candidates if c['switches']['noise_mitigation']),candidates[0]))
    core.write_json(args.output/'verification_plan.json',dict(source=core.fingerprint(),configs=chosen))
    for p in hardware:
        command=[sys.executable,str(ROOT/'scripts/run_cart_v2_study.py'),'--verify-worker',str(p.gpu),'--output',str(args.output)]
        subprocess.run(command,env=core.worker_environment(p),cwd=ROOT,check=True)


def execute(args):
    configs=core.selected(args);source=core.fingerprint();hardware=profiles(args)
    slots=hardware*args.jobs_per_gpu;args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data,partitions=core.data_contract(configs)
        plan=dict(configs=configs,source=source,data=data,partitions=partitions,
            profiles=[asdict(p) for p in slots],jobsPerGpu=args.jobs_per_gpu,
            executionProgram={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('scripts/run_cart_v2_parallel.py','scripts/cart_v2_slot_worker.py')})
        path=args.output/'plan.json'
        if path.exists():
            if not args.recover:raise FileExistsError('Use a fresh output or identical arguments with --recover.')
            if json.loads(path.read_text())!=plan:raise ValueError('Recovery plan or execution source differs.')
        elif args.recover:raise ValueError('No existing recovery plan.')
        for p in hardware:
            receipt=json.loads((args.output/f'gpu_verification_{p.gpu}.json').read_text())
            if (receipt['status']!='PASSED' or receipt['source']!=source or receipt['runtime']['profile']!=asdict(p)
                    or not {(c['dataset'],c['model']) for c in configs}.issubset({(c['dataset'],c['model']) for c in receipt['checks']})):
                raise ValueError('GPU verification does not match this plan/profile.')
        if not path.exists():core.write_json(path,plan)
        for name in {**source,**plan['executionProgram']}:
            target=args.output/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        queues={slot:deque() for slot in range(len(slots))}
        for index,cfg in enumerate(configs):
            row=args.output/cfg['runId']/'run.json'
            if row.exists() and json.loads(row.read_text())['status']=='COMPLETED':continue
            queues[index%len(slots)].append(index)
        active={};stopped=False;status='RUNNING';started=time.monotonic()
        def stop(*_):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        def campaign():
            core.write_json(args.output/'campaign.json',dict(status=status,total=len(configs),
                pending=sum(map(len,queues.values())),activeSlots=list(active),
                elapsedSeconds=time.monotonic()-started,host=socket.gethostname(),profile=args.profile,ownerPid=os.getpid()))
        leases=acquire_leases(hardware)
        try:
            campaign()
            while any(queues.values()) or active:
                for slot,(process,handle) in list(active.items()):
                    if process.poll() is not None:
                        handle.close();del active[slot]
                        if process.returncode:raise RuntimeError(f'Worker failed on GPU {slots[slot].gpu}; inspect its log.')
                if stopped:raise InterruptedError('Supervisor stop requested.')
                for slot,p in enumerate(slots):
                    if queues[slot] and slot not in active:
                        core.check_source(source);index=queues[slot].popleft();cfg=configs[index]
                        log=(args.output/(cfg['runId']+'.log')).open('a')
                        fd=leases[p.gpu].fileno()
                        command=[sys.executable,str(ROOT/'scripts/cart_v2_slot_worker.py'),'--worker',str(index),
                            '--output',str(args.output),'--lease-fd',str(fd)]
                        active[slot]=(subprocess.Popen(command,cwd=ROOT,env=core.worker_environment(p),
                            pass_fds=(fd,),stdout=log,stderr=subprocess.STDOUT),log)
                        print(f'GPU {p.gpu}, slot {slot}: {cfg["runId"]}',flush=True)
                campaign()
                if active:time.sleep(2)
            status='COMPLETED';print('ALL SELECTED EXPERIMENTS COMPLETE.',flush=True)
        except BaseException as error:
            status='INTERRUPTED' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'FAILED'
            for process,_ in active.values():
                if process.poll() is None:process.terminate()
            for process,log in active.values():process.wait();log.close()
            raise
        finally:
            campaign()
            for handle in leases.values():handle.close()


def main():
    # Reuse the frozen launcher's argument definitions without changing its file.
    from basil_core.sharp_study import DATASETS,SEEDS,SIGMAS,CONDITION_NAMES
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group()
    for name in ('preflight','verify-gpus','execute','monitor','report'):action.add_argument('--'+name,action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='0')
    p.add_argument('--profile',choices=('local','hpc'),default='hpc')
    p.add_argument('--jobs-per-gpu',type=int,choices=(1,2),default=1);p.add_argument('--gpu-memory-mib',type=int)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=['mnist','cifar10'])
    p.add_argument('--partitions',nargs='+',choices=('iid','dirichlet'),default=['iid','dirichlet'])
    p.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    p.add_argument('--conditions',nargs='+',choices=CONDITION_NAMES,default=list(CONDITION_NAMES))
    p.add_argument('--sigmas',nargs=3,type=float,default=list(SIGMAS));p.add_argument('--only-sigma',type=float)
    p.add_argument('--nodes',type=int,default=100);p.add_argument('--rounds',type=int,default=35)
    p.add_argument('--local-steps',type=int,default=5);p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--model',choices=('paper','resnet20'),default='paper');p.add_argument('--attack-start',type=int,default=20)
    p.add_argument('--train-per-class',type=int);p.add_argument('--validation-per-class',type=int)
    p.add_argument('--recover',action='store_true')
    p.set_defaults(methods=['sharp'],hop_denominator='actual_selected',curvature=1.,mu=1.,learning_rate=.03,
        attack_type='hidden',budget=None,basil_window=None)
    args=p.parse_args();args.output=args.output.resolve()
    if args.verify_gpus:verify(args)
    elif args.execute:execute(args)
    elif args.monitor:core.monitor(args)
    elif args.report:core.report(args)
    else:
        configs=core.selected(args)
        if args.preflight:
            data,partitions=core.data_contract(configs);args.output.mkdir(parents=True,exist_ok=True)
            core.write_json(args.output/'preflight.json',dict(source=core.fingerprint(),configs=configs,data=data,partitions=partitions))
        print(json.dumps(dict(configs=len(configs),nodes=args.nodes,rounds=args.rounds,
            jobsPerGpu=args.jobs_per_gpu,trainingStarted=False),indent=2))


if __name__=='__main__':main()
