#!/usr/bin/env python3
"""The requested six IID and eight non-IID curves, with safe parallel workers."""
import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import shlex
import socket
import os
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import basil_sharp_thesis as study
from basil_core.cart_campaign_io import read_json,write_json
from basil_core.cart_worker_admission import placement
from scripts import run_cart_contribution as controller
from scripts import run_cart_adaptive as adaptive


def prepare(args):
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    if args.resume:
        with study.installed(),patch.object(controller,'study',study):
            return controller.prepare(args)
    root=args.output
    if (root/'plan.json').exists():raise FileExistsError('Use --resume or a fresh output directory.')
    configs=study.selected(args);inventory=gpu_inventory()
    ids=[int(v) for v in args.gpus.split(',')] if args.gpus else list(range(args.gpu_count))
    if not ids or len(set(ids))!=len(ids):raise ValueError('Distinct physical GPU IDs required.')
    hardware=[ExecutionProfile(name='hpc',gpu=g,memory_mib=args.gpu_memory_mib,headroom_mib=8192,
        ram_mib=args.ram_mib_per_worker,output_root=str(root)).bind(inventory) for g in ids]
    print('Loading original training archives into memory and verifying their fingerprints.',flush=True)
    data,partitions=study.data_contract(configs)
    with patch.object(controller,'study',study):profiles=controller.assignment(configs,hardware)
    plan=dict(revision=study.REVISION,variant='thesis',configs=configs,source=study.source(),data=data,
        partitions=partitions,profiles=profiles,jobsPerGpu=1,host=socket.gethostname(),
        inputLoading='original_training_archives_in_memory',etaMode='activation_throughput',
        resourceBudget=dict(context_overhead_mib=1024,ram_headroom_mib=args.ram_headroom_mib,
            ram_mib_per_worker=args.ram_mib_per_worker,allow_unattributed_process_memory=False))
    plan.update(attackType=getattr(args,'attack_type','hidden'),
        environments=list(getattr(args,'environments',study.ENVIRONMENTS)),
        scheduler=getattr(args,'scheduler','adaptive'),
        maximumWorkersPerGpu=getattr(args,'max_workers_per_gpu',24))
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'plan.json',plan);return plan


def readiness(root,plan,maximum=24):
    from basil_core.execution_profile import gpu_inventory
    hardware,assigned=placement(plan);counts=Counter(p.gpu for p in assigned);inventory=gpu_inventory()
    budget=plan['resourceBudget'];available=adaptive.scaled.available_ram_mib()
    capacity={g:min(counts[g],maximum,max(0,(next(r['memory.free'] for r in inventory if r['index']==g)
        -p.headroom_mib)//(p.memory_mib+budget['context_overhead_mib']))) for g,p in hardware.items()}
    host_capacity=max(0,(available-budget['ram_headroom_mib'])//assigned[0].ram_mib)
    return dict(configurations=len(assigned),physicalGpus=list(hardware),casesPerGpu=dict(counts),
        gpuCapacityNow=capacity,hostAvailableMiB=available,hostWorkerReservationMiB=assigned[0].ram_mib,
        hostHeadroomMiB=budget['ram_headroom_mib'],hostWorkerCapacityNow=host_capacity,
        maximumConcurrentNow=min(len(assigned),sum(capacity.values()),host_capacity),
        hostRequiredForAllMiB=budget['ram_headroom_mib']+len(assigned)*assigned[0].ram_mib,
        output=str(root),trainingStarted=False)


def execute(args, plan):
    """Reuse the verified scheduler; keep historical producer files unchanged."""
    from basil_core.execution_profile import gpu_inventory
    import time
    root=args.output;hardware,_=placement(plan)
    with adaptive.acquire_supervisor(root,False,180):
        adaptive.ensure_source_snapshots(root,plan)
        completed=sum((root/c['runId']/'run.json').exists() and
            read_json(root/c['runId']/'run.json').get('status')=='COMPLETED' for c in plan['configs'])
        def startup(status,reason):
            write_json(root/'campaign.json',dict(status=status,total=len(plan['configs']),completed=completed,
                failed=0,pending=len(plan['configs'])-completed,host=socket.gethostname(),ownerPid=os.getpid(),reason=reason))
        while True:
            try:
                for g,p in hardware.items():
                    p.bind(gpu_inventory())
                    free=next(r['memory.free'] for r in gpu_inventory() if r['index']==g)
                    if free<p.memory_mib+p.headroom_mib+1024:raise RuntimeError(f'GPU {g}: waiting for memory.')
                if adaptive.scaled.available_ram_mib()<plan['resourceBudget']['ram_headroom_mib']+next(iter(hardware.values())).ram_mib:
                    raise RuntimeError('Waiting for host RAM.')
                leases=adaptive.scaled.acquire_leases(list(hardware.values()));break
            except RuntimeError as error:
                if not args.wait_for_resources:raise
                startup('WAITING_FOR_RESOURCES',str(error))
                print(str(error),flush=True);time.sleep(5)
        try:
            digest=adaptive.sha(root/'plan.json')
            for g,p in hardware.items():
                receipt=root/f'revision_gpu_{g}.json'
                saved=read_json(receipt) if receipt.exists() else {}
                if saved.get('status')!='PASSED' or saved.get('planSha256')!=digest:
                    fd=leases[g].fileno()
                    startup('VERIFYING_GPUS',f'Checking physical GPU {g}.')
                    print(f'Verifying BASIL/SHARP on GPU {g}.',flush=True)
                    subprocess.run([sys.executable,str(ROOT/'scripts/verify_basil_sharp_thesis_gpu.py'),
                        '--output',str(root),'--gpu',str(g),'--lease-fd',str(fd)],cwd=ROOT,
                        env=adaptive.scaled.core.worker_environment(p),pass_fds=(fd,),check=True)
            def report(root, plan):
                from reporting.basil_sharp_thesis import generate
                generate(root,root/'thesis_figures')
            with patch.object(adaptive,'check_plan_source',lambda p:study.check_source(p['source'])), \
                    patch.object(adaptive,'worker_script',lambda p:'scripts/basil_sharp_thesis_worker.py'), \
                    patch.object(adaptive,'completed_report',report):
                checked,hardware,assigned=adaptive.validate_plan(root)
                baseline=sum(read_json(root/c['runId']/'run.json').get('completedActivations',0)
                    for c in plan['configs'] if (root/c['runId']/'run.json').exists())
                write_json(root/'throughput_baseline.json',dict(completedActivations=baseline,
                    planSha256=digest))
                if plan.get('scheduler')!='shared':
                    return adaptive.schedule(args,checked,hardware,assigned,digest,leases)
        finally:
            for handle in leases.values():handle.close()
    # Release verification leases and the old supervisor lock before the shared
    # scheduler acquires them. This starts balanced dispatch directly, without a
    # second shell or a live handover racing the first worker launches.
    from basil_core import live_gpu_scheduler
    return live_gpu_scheduler.execute(SimpleNamespace(output=root,
        maximum=plan.get('maximumWorkersPerGpu',24),poll_seconds=args.poll_seconds,
        ntfy_topic=args.ntfy_topic))


def partition_name(value):
    names={'iid':'iid','noniid':'dirichlet','non-iid':'dirichlet','dirichlet':'dirichlet'}
    try:return names[value.lower()]
    except KeyError:raise argparse.ArgumentTypeError('Choose IID or nonIID.')


def parse_options(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,help='JSON settings preset; explicit terminal flags override it.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-count',type=int,choices=(1,2,3,4),default=4);p.add_argument('--gpus')
    p.add_argument('--gpu-memory-mib',type=int,choices=(3072,4096,6144,8192),default=3072)
    p.add_argument('--ram-mib-per-worker',type=int,choices=(8192,12288,16384,32768),default=16384)
    p.add_argument('--ram-headroom-mib',type=int,choices=(8192,16384,32768),default=32768)
    p.add_argument('--datasets',nargs='+',choices=('mnist','cifar10'),default=['mnist','cifar10'])
    p.add_argument('--partitions',nargs='+',type=partition_name,choices=('iid','dirichlet'),default=['iid','dirichlet'],metavar='IID/nonIID')
    p.add_argument('--seeds',nargs='+',type=int,choices=(2025,2026,2027),default=[2025,2026,2027])
    p.add_argument('--rounds',type=int,default=100);p.add_argument('--nodes',type=int,default=100)
    p.add_argument('--local-steps',type=int,default=5);p.add_argument('--attack-start',type=int,default=20)
    p.add_argument('--sigma',type=float,default=.01);p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--attack-type',choices=study.ATTACK_TYPES,default='hidden',
        help='Byzantine outgoing-model attack; separate from Gaussian channel noise (--sigma).')
    p.add_argument('--gaussian-std',type=float,default=1.,
        help='Pure Gaussian replacement attack standard deviation (BASIL default: 1.0); not channel sigma.')
    p.add_argument('--environments',nargs='+',choices=study.ENVIRONMENTS,default=list(study.ENVIRONMENTS),
        help='Use attack_only joint for 60 attacked cases, reusing existing clean controls.')
    p.add_argument('--scheduler',choices=('shared','adaptive'),default='shared',
        help='Shared queue balances new workers across healthy GPUs under live RAM/VRAM checks.')
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--describe',action='store_true');p.add_argument('--wait-for-resources',action='store_true')
    p.add_argument('--tmux');p.add_argument('--max-workers-per-gpu',type=int,choices=range(1,25),default=24)
    p.add_argument('--poll-seconds',type=int,choices=range(1,31),default=3);p.add_argument('--ntfy-topic')
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--config',type=Path)
    config=pre.parse_known_args(argv)[0].config;defaults=[]
    if config:
        try:
            document=json.loads(config.read_text())
            if set(document)-{'schema','description','settings'} or document.get('schema')!=1:
                raise ValueError('Expected schema 1 with a settings object.')
            settings=document['settings']
            if not isinstance(settings,dict):raise ValueError('Settings must be an object.')
            actions={a.dest:a for a in p._actions}
            prohibited={'help','config','output','execute','resume','describe','tmux','wait_for_resources'}
            for key,value in settings.items():
                if key not in actions or key in prohibited:raise ValueError('Unsupported preset setting: '+key)
                values=value if isinstance(value,list) else [value]
                if not values or any(type(v) not in (str,int,float) for v in values):
                    raise ValueError('Preset settings require nonempty scalar values/lists: '+key)
                defaults.extend([actions[key].option_strings[0],*[str(v) for v in values]])
        except (OSError,ValueError,KeyError,TypeError) as error:p.error('Invalid config: '+str(error))
    args=p.parse_args(defaults+argv)
    args.output=args.output.resolve();args.variants=None;args.preset='thesis'
    args.supplied_options={a.split('=')[0] for a in argv if a.startswith('--')}
    args.forwarded_arguments=argv
    return args


def main(argv=None):
    args=parse_options(argv)
    if args.ntfy_topic:
        from basil_core.ntfy_notifications import validate_destination
        validate_destination(args.ntfy_topic,'https://ntfy.sh')
    if args.resume:
        supplied=args.supplied_options
        if supplied-{'--output','--execute','--resume','--tmux','--wait-for-resources','--ntfy-topic'}:
            raise SystemExit('Resume uses the frozen plan and resource profile; omit config/attack/resource overrides.')
    if args.describe:
        views=study.selected(args)
        counts={('IID' if split=='iid' else 'nonIID'):len({study.variant(v) for v in views if v['partition']==split})
            for split in args.partitions}
        print(json.dumps(dict(configurations=len(views),attackType=args.attack_type,
            environments=args.environments,curvesPerDataset=counts,
            datasets=args.datasets,seeds=args.seeds,nodes=args.nodes,rounds=args.rounds,
            localSteps=args.local_steps,channelSigma=args.sigma,attackStartsAtDisplayedRound=args.attack_start+1,
            gaussianAttackStd=args.gaussian_std if args.attack_type=='gaussian' else None,
            gaussianAttackMode='pure_model_replacement' if args.attack_type=='gaussian' else None,
            requestedGpuCount=args.gpu_count,scheduler=args.scheduler,
            activations=len(views)*args.nodes*args.rounds,trainingStarted=False),indent=2));return 0
    if args.tmux:
        if not args.execute:raise SystemExit('--tmux requires --execute.')
        forwarded=args.forwarded_arguments;pos=forwarded.index('--tmux');forwarded=forwarded[:pos]+forwarded[pos+2:]
        args.output.mkdir(parents=True,exist_ok=True)
        command=shlex.join([sys.executable,str(Path(__file__).resolve()),*forwarded])
        command+=' >> '+shlex.quote(str(args.output/'adaptive_supervisor.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',args.tmux,'bash','-c',command],check=True)
        print(f'Started {args.tmux}. Log: {args.output}/adaptive_supervisor.log');return 0
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(args)
    print(json.dumps(readiness(args.output,plan,args.max_workers_per_gpu),indent=2),flush=True)
    return execute(args,plan) if args.execute else 0


if __name__=='__main__':raise SystemExit(main())
