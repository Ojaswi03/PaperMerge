#!/usr/bin/env python3
"""Prepare, verify and run 12 fresh themachine plain clean FedSharp ring controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import fedsharp_plain_clean as study
from scripts import run_basil_sharp_thesis as driver,run_cart_adaptive as adaptive

PRESET=ROOT/'configs/research/themachine_fedsharp_plain_clean_500r.json'


def execute(args,plan):
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core.cart_worker_admission import placement
    from basil_core.execution_profile import gpu_inventory
    from basil_core import live_gpu_scheduler as live
    root=args.output;hardware,_=placement(plan)
    with adaptive.acquire_supervisor(root,False,180):
        adaptive.ensure_source_snapshots(root,plan)
        write_json(root/'campaign.json',dict(status='WAITING_FOR_RESOURCES',host=socket.gethostname(),
            ownerPid=os.getpid(),total=len(plan['configs']),completed=0,elapsedSeconds=0.))
        while True:
            try:
                for gpu,profile in hardware.items():
                    profile.bind(gpu_inventory())
                    free=next(g['memory.free'] for g in gpu_inventory() if g['index']==gpu)
                    if free<profile.memory_mib+profile.headroom_mib+1024:raise RuntimeError(f'GPU {gpu}: waiting for memory.')
                if adaptive.scaled.available_ram_mib()<plan['resourceBudget']['ram_headroom_mib']+next(iter(hardware.values())).ram_mib:
                    raise RuntimeError('Waiting for host RAM.')
                leases=adaptive.scaled.acquire_leases(list(hardware.values()));break
            except RuntimeError as error:
                if not args.wait_for_resources:raise
                print(str(error),flush=True);time.sleep(5)
        try:
            digest=adaptive.sha(root/'plan.json')
            for gpu,profile in hardware.items():
                path=root/f'revision_gpu_{gpu}.json';saved=read_json(path) if path.exists() else {}
                if saved.get('status')=='PASSED' and saved.get('source')==plan['source'] and saved.get('planSha256')==digest:continue
                fd=leases[gpu].fileno()
                write_json(root/'campaign.json',dict(status='VERIFYING_GPUS',host=socket.gethostname(),
                    ownerPid=os.getpid(),total=len(plan['configs']),completed=0,elapsedSeconds=0.))
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--role','gate','--output',str(root),
                    '--gpu',str(gpu),'--lease-fd',str(fd)],cwd=ROOT,env=adaptive.scaled.core.worker_environment(profile),pass_fds=(fd,),check=True)
        finally:
            for handle in leases.values():handle.close()
    with patch.object(live,'WORKER',Path(__file__).resolve()),patch.object(live,'OWNERS',live.OWNERS|{Path(__file__).name}):
        return live.execute(SimpleNamespace(output=root,maximum=plan['maximumWorkersPerGpu'],
            poll_seconds=args.poll_seconds,ntfy_topic=args.ntfy_topic))


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--role',choices=('driver','gate','verify'),default='driver')
    pre.add_argument('--worker',type=int);pre.add_argument('--gpu',type=int);pre.add_argument('--lease-fd',type=int)
    pre.add_argument('--background',action='store_true');pre.add_argument('--monitor',action='store_true')
    pre.add_argument('--write-configs',type=Path)
    control,remaining=pre.parse_known_args(argv)
    if '--config' not in remaining and '--resume' not in remaining:remaining=['--config',str(PRESET),*remaining]
    args=driver.parse_options(remaining)
    if control.monitor:
        from basil_core.sharp_monitor import snapshot,render
        try:
            while True:
                state=snapshot(args.output);text=render(state,width=160,height=40,table=True)
                if state['status']!='WAITING_FOR_PLAN':text='FedSharp plain clean ring |'+text.split('|',1)[1]
                print('\033[2J\033[H'+text.replace('SHARP / clean','FedSharp plain ring / clean'),flush=True);time.sleep(5)
        except KeyboardInterrupt:return 0
    if control.worker is not None or control.role=='gate' or args.execute:
        if socket.gethostname().split('.')[0]!='themachine':raise ValueError('Execution is restricted to themachine.')
    with study.installed():
        if control.worker is not None:
            from scripts.cart_comparison_worker import run
            if control.lease_fd is None:raise ValueError('Worker requires its inherited GPU lease.')
            return run(args.output,control.worker,control.lease_fd)
        if control.role=='gate':
            from scripts.verify_fedsharp_plain_clean_gpu import verify
            verify(args.output,control.gpu,control.lease_fd);return 0
        if control.role=='verify':
            from scripts.verify_cart_comparison_results import verify
            verify(args.output);return 0
        if args.describe or control.write_configs:
            views=study.selected(args);effective=[study.effective_config(v) for v in views]
            report=dict(revision=study.REVISION,configurations=len(views),datasets=args.datasets,
                seeds=args.seeds,rounds=args.rounds,nodes=args.nodes,localSteps=args.local_steps,
                proximalMu={'IID':0.,'nonIID':0.},receiverRule='immediate_predecessor',
                screening=False,ranking=False,averaging=False,noiseRegularization=False,effectiveChannelSigma=0.,effectiveAttackers=0,
                fedSharpPlus=False,requestedGpuCount=args.gpu_count,start='initialization',activations=len(views)*args.rounds*args.nodes,
                optimizerSteps=len(views)*args.rounds*args.nodes*args.local_steps,trainingStarted=False)
            if control.write_configs:
                control.write_configs.parent.mkdir(parents=True,exist_ok=True)
                control.write_configs.write_text(json.dumps(dict(description=report,configs=views),indent=2)+'\n')
            print(json.dumps(report,indent=2));return 0
        if not args.execute:raise ValueError('Use --describe or --execute.')
        if args.resume and args.supplied_options-{'--output','--execute','--resume','--ntfy-topic','--wait-for-resources'}:
            raise ValueError('Resume requires the original frozen settings; omit overrides.')
        if control.background:
            args.output.mkdir(parents=True,exist_ok=True)
            child_argv=[v for v in argv if v!='--background']
            with (args.output/'plain_clean_supervisor.log').open('a') as log:
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*child_argv],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print(f'Plain clean supervisor PID {process.pid}. Log: {args.output}/plain_clean_supervisor.log');return 0
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/'preparation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=driver.prepare(args)
        print(json.dumps(driver.readiness(args.output,plan,args.max_workers_per_gpu),indent=2),flush=True)
        if args.ntfy_topic:
            # The plan now exists, so publisher startup cannot race preparation.
            published=subprocess.run([sys.executable,str(ROOT/'scripts/share_cart_monitor.py'),
                '--output',str(args.output),'--background','--restart-publisher',
                '--interval','1','--ntfy-topic',args.ntfy_topic],cwd=ROOT,check=False)
            if published.returncode:print('Browser publisher startup failed; training will continue.',flush=True)
        return execute(args,plan)


if __name__=='__main__':main()
