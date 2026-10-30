#!/usr/bin/env python3
"""The requested six IID and eight non-IID curves, with safe parallel workers."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import basil_sharp_thesis as study
from basil_core.cart_campaign_io import read_json
from basil_core.cart_worker_admission import placement
from scripts import run_cart_contribution as controller
from scripts import run_cart_adaptive as adaptive


def execute(args, plan):
    """Reuse the verified scheduler; keep historical producer files unchanged."""
    from basil_core.execution_profile import gpu_inventory
    import time
    root=args.output;hardware,_=placement(plan)
    with adaptive.acquire_supervisor(root,False,180):
        adaptive.ensure_source_snapshots(root,plan)
        while True:
            try:
                for g,p in hardware.items():
                    p.bind(gpu_inventory())
                    free=next(r['memory.free'] for r in gpu_inventory() if r['index']==g)
                    if free<p.memory_mib+p.headroom_mib+1024:raise RuntimeError(f'GPU {g}: waiting for memory.')
                if adaptive.scaled.available_ram_mib()<32768+16384:raise RuntimeError('Waiting for host RAM.')
                leases=adaptive.scaled.acquire_leases(list(hardware.values()));break
            except RuntimeError as error:
                if not args.wait_for_resources:raise
                print(str(error),flush=True);time.sleep(5)
        try:
            digest=adaptive.sha(root/'plan.json')
            for g,p in hardware.items():
                receipt=root/f'revision_gpu_{g}.json'
                saved=read_json(receipt) if receipt.exists() else {}
                if saved.get('status')!='PASSED' or saved.get('planSha256')!=digest:
                    fd=leases[g].fileno()
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
                return adaptive.schedule(args,checked,hardware,assigned,digest,leases)
        finally:
            for handle in leases.values():handle.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-count',type=int,choices=(1,2,3,4),default=4);p.add_argument('--gpus')
    p.add_argument('--gpu-memory-mib',type=int,choices=(3072,4096,6144,8192),default=3072)
    p.add_argument('--datasets',nargs='+',choices=('mnist','cifar10'),default=['mnist','cifar10'])
    p.add_argument('--partitions',nargs='+',choices=('iid','dirichlet'),default=['iid','dirichlet'])
    p.add_argument('--seeds',nargs='+',type=int,choices=(2025,2026,2027),default=[2025,2026,2027])
    p.add_argument('--rounds',type=int,default=100);p.add_argument('--nodes',type=int,default=100)
    p.add_argument('--local-steps',type=int,default=5);p.add_argument('--attack-start',type=int,default=20)
    p.add_argument('--sigma',type=float,default=.01);p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--describe',action='store_true');p.add_argument('--wait-for-resources',action='store_true')
    p.add_argument('--tmux');p.add_argument('--max-workers-per-gpu',type=int,choices=range(1,25),default=24)
    p.add_argument('--poll-seconds',type=int,choices=range(1,31),default=3);p.add_argument('--ntfy-topic')
    args=p.parse_args();args.output=args.output.resolve();args.variants=None;args.preset='thesis'
    if args.ntfy_topic:
        from basil_core.ntfy_notifications import validate_destination
        validate_destination(args.ntfy_topic,'https://ntfy.sh')
    if args.resume:
        supplied={a.split('=')[0] for a in sys.argv[1:] if a.startswith('--')}
        if supplied-{'--output','--execute','--resume','--tmux','--wait-for-resources','--ntfy-topic'}:
            p.error('Resume uses the frozen plan and resource profile.')
    if args.describe:
        views=study.selected(args)
        print(json.dumps(dict(configurations=len(views),curvesPerDataset=dict(IID=6,nonIID=8),
            nodes=args.nodes,rounds=args.rounds,trainingStarted=False),indent=2));return 0
    if args.tmux:
        if not args.execute:p.error('--tmux requires --execute.')
        forwarded=sys.argv[1:];pos=forwarded.index('--tmux');forwarded=forwarded[:pos]+forwarded[pos+2:]
        args.output.mkdir(parents=True,exist_ok=True)
        command=shlex.join([sys.executable,str(Path(__file__).resolve()),*forwarded])
        command+=' >> '+shlex.quote(str(args.output/'adaptive_supervisor.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',args.tmux,'bash','-c',command],check=True)
        print(f'Started {args.tmux}. Log: {args.output}/adaptive_supervisor.log');return 0
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with study.installed(),patch.object(controller,'study',study):
            plan=controller.prepare(args)
    print(json.dumps(controller.readiness(args.output,plan),indent=2),flush=True)
    return execute(args,plan) if args.execute else 0


if __name__=='__main__':raise SystemExit(main())
