#!/usr/bin/env python3
"""Prepare a future study with one, two or three explicit noise levels.

Default: CART-only, both datasets/splits, three seeds, sigma 0.01, 100 rounds.
No training starts here; existing saved studies cannot change their selection.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def program_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('scripts/prepare_cart_study.py','basil_core/cart_noise_selection.py')}


def summary(args,selection,configs):
    return dict(configs=len(configs),datasets=args.datasets,partitions=args.partitions,seeds=args.seeds,
        noiseLevels=selection.count,noiseValues=list(selection.values),
        actualNoisySigmas=sorted({c['sigma'] for c in configs if c['sigma']>0}),
        conditions=args.conditions,nodes=args.nodes,rounds=args.rounds,jobsPerGpu=args.jobs_per_gpu,
        maximumConcurrentConfigurations=min(len(configs),len(args.gpus.split(','))*args.jobs_per_gpu),
        trainingStarted=False,
        controlReferenceNeededForComparison=any(c['cart'] and c['condition'][:-5] not in args.conditions for c in configs))


def prepare(args,selection,configs):
    from scripts import run_cart_v2_scaled as scaled
    from basil_core.cart_campaign_io import read_json,write_json
    root=args.output;path=root/'plan.json'
    # Check before writing anything into a pre-existing experiment.
    existing=read_json(path) if path.exists() else None
    if existing and existing.get('executionProgram')!=program_hashes():
        raise FileExistsError('Existing frozen plan belongs to another version; use a fresh output directory.')
    data,partitions=scaled.core.data_contract(configs)
    contract=dict(configs=configs,source=scaled.core.fingerprint(),data=data,partitions=partitions,
        noiseSelection=dict(count=selection.count,values=list(selection.values)),executionProgram=program_hashes())
    if existing and any(existing.get(k)!=v for k,v in contract.items()):
        raise ValueError('Saved configuration/noise/source differs; use a fresh output directory.')
    hardware=scaled.profiles(args)
    plan=dict(contract,profiles=[asdict(p) for p in hardware*args.jobs_per_gpu],jobsPerGpu=args.jobs_per_gpu,
        resourceBudget=dict(contextOverheadMiB=args.context_overhead_mib,hostHeadroomMiB=args.ram_headroom_mib,
            allowUnattributedProcessMemory=args.allow_unattributed_process_memory))
    if existing and existing!=plan:raise ValueError('Saved runtime plan differs; use a fresh output directory.')
    if not existing:write_json(path,plan)
    return plan,hardware


def verify(args,selection,configs):
    from scripts import run_cart_v2_scaled as scaled
    from basil_core.cart_campaign_io import write_json
    root=args.output
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,hardware=prepare(args,selection,configs);chosen=[]
        for dataset in dict.fromkeys(c['dataset'] for c in configs):
            candidates=[c for c in configs if c['dataset']==dataset]
            chosen.append(next((c for c in candidates if c['switches']['noise_mitigation']),candidates[0]))
        write_json(root/'verification_plan.json',dict(source=plan['source'],configs=chosen))
        for p in hardware:
            subprocess.run([sys.executable,str(ROOT/'scripts/run_cart_v2_study.py'),
                '--verify-worker',str(p.gpu),'--output',str(root)],env=scaled.core.worker_environment(p),cwd=ROOT,check=True)
    from scripts.run_cart_v2_transfer import verify as verify_transfer
    verify_transfer(args)


def main():
    from basil_core.sharp_study import DATASETS,SEEDS,CONDITION_NAMES
    from basil_core.cart_noise_selection import configurations
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group();action.add_argument('--preflight',action='store_true')
    action.add_argument('--verify-gpus',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--noise-levels',type=int,choices=(1,2,3),default=1)
    p.add_argument('--noise-values',type=float,nargs='+')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=['mnist','cifar10'])
    p.add_argument('--partitions',nargs='+',choices=('iid','dirichlet'),default=['iid','dirichlet'])
    p.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    p.add_argument('--conditions',nargs='+',choices=CONDITION_NAMES,default=['attack_noise_defense_rla_cart'])
    p.add_argument('--nodes',type=int,default=100);p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--local-steps',type=int,default=5);p.add_argument('--attack-start',type=int,default=20)
    p.add_argument('--alpha',type=float,default=.5);p.add_argument('--model',choices=('paper','resnet20'),default='paper')
    p.add_argument('--train-per-class',type=int);p.add_argument('--validation-per-class',type=int)
    p.add_argument('--profile',choices=('local','hpc'),default='hpc');p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--jobs-per-gpu',type=int,choices=range(1,9),default=6)
    p.add_argument('--gpu-memory-mib',type=int);p.add_argument('--gpu-headroom-mib',type=int,default=8192)
    p.add_argument('--context-overhead-mib',type=int,default=1024);p.add_argument('--ram-mib-per-worker',type=int,default=32768)
    p.add_argument('--ram-headroom-mib',type=int,default=32768);p.add_argument('--allow-unattributed-process-memory',action='store_true')
    args=p.parse_args()
    if args.gpu_headroom_mib<256 or args.context_overhead_mib<512 or args.ram_mib_per_worker<4096 or args.ram_headroom_mib<1024:
        p.error('Resource budgets are below supported minimums.')
    try:selection,configs=configurations(args)
    except ValueError as error:p.error(str(error))
    if args.preflight or args.verify_gpus:
        if not args.output:p.error('--output is required for preflight/verification.')
        args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
        if args.verify_gpus:verify(args,selection,configs)
        else:
            from scripts import run_cart_v2_study as core
            from basil_core.cart_campaign_io import read_json,write_json
            path=args.output/'plan.json'
            if path.exists():
                old=read_json(path)
                if old.get('executionProgram')!=program_hashes() or old['configs']!=configs or old['source']!=core.fingerprint():
                    raise ValueError('An existing frozen study cannot change noise/config selection.')
            data,partitions=core.data_contract(configs)
            write_json(args.output/'preflight.json',dict(configs=configs,source=core.fingerprint(),data=data,partitions=partitions,
                noiseSelection=dict(count=selection.count,values=list(selection.values))))
    print(json.dumps(summary(args,selection,configs),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
