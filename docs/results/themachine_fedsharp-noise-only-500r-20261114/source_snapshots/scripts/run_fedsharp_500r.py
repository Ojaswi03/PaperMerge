#!/usr/bin/env python3
"""Prepare all verified 100-to-500-round forks and balance each server's GPUs."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import round_continuation as continuation
from basil_core import continuation_campaign as campaign
from basil_core.cart_campaign_io import read_json


def options(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=continuation.PRESET)
    p.add_argument('--queue',type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    for action in ('describe','probe','initialize','execute','monitor','status'):
        group.add_argument('--'+action,action='store_true')
    p.add_argument('--write-configs',type=Path)
    p.add_argument('--hosts',nargs='+',choices=('hal9k','themachine'),default=['hal9k','themachine'])
    p.add_argument('--output',type=Path)
    p.add_argument('--background',action='store_true')
    p.add_argument('--ntfy-topic',default='papermerge-ojaswi')
    p.add_argument('--no-publish',action='store_true')
    a=p.parse_args(argv)
    if not a.describe and a.queue is None:p.error('--queue is required.')
    if a.background and not a.execute:p.error('--background requires --execute.')
    if len(set(a.hosts))!=len(a.hosts):p.error('Choose distinct hosts.')
    if a.queue:a.queue=a.queue.resolve()
    return a


@contextmanager
def publish_updates(queue,output):
    from scripts import run_basil_sharp_thesis as runner
    from basil_core import live_gpu_scheduler as live
    original=runner.write_json
    stopped=threading.Event()
    def heartbeat():
        try:
            campaign.atomic(queue/'heartbeats'/f'{campaign.host_name()}.json',
                dict(host=campaign.host_name(),ownerPid=os.getpid(),updatedAt=time.time()))
        except OSError as error:
            print('Shared heartbeat delayed: '+type(error).__name__,flush=True)
    def pulse():
        while not stopped.wait(3):heartbeat()
    heartbeat();thread=threading.Thread(target=pulse,daemon=True);thread.start()
    def write(path,value):
        original(path,value)
        if Path(path)==output/'campaign.json':
            try:campaign.publish_host(queue,output)
            except (OSError,ValueError) as error:
                print('Shared status delayed: '+type(error).__name__,flush=True)
    try:
        with patch.object(runner,'write_json',write),patch.object(live,'write_json',write):yield
    finally:
        stopped.set();thread.join(timeout=1)


def execute(a):
    from scripts import run_basil_sharp_thesis as runner
    from basil_core import live_gpu_scheduler as live
    master=read_json(a.queue/'continuation_plan.json');host=campaign.host_name()
    output=(a.output or Path('/var/tmp')/f'papermerge-{os.getuid()}'/
        f'{host}_fedsharp-all-500r-{master["queueId"][:12]}').resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=campaign.prepare_host(a.queue,output)
    if a.background:
        # Reject an already-held supervisor lock in the foreground, before spawning.
        with (output/'supervisor.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--queue',str(a.queue),
            '--output',str(output),'--execute','--ntfy-topic',a.ntfy_topic,'--no-publish']
        with (output/'adaptive_supervisor.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        print(f'{host}: detached supervisor PID {process.pid}; {len(plan["configs"])} assigned configurations.')
        print('Host-local output: '+str(output))
        print('Log: '+str(output/'adaptive_supervisor.log'))
        if host=='hal9k' and not a.no_publish:
            subprocess.run([sys.executable,str(ROOT/'scripts/share_cart_monitor.py'),'--output',str(a.queue),
                '--background','--interval','1','--ntfy-topic',a.ntfy_topic],cwd=ROOT,check=True)
        return 0
    if a.output is None:
        raise ValueError('Foreground execution needs explicit --output; prefer --execute --background.')
    # Re-entry adopts live children and never starts a second original dispatcher.
    # Resume skips GPU warmup if matching gates already exist, but retains source checks.
    args=SimpleNamespace(output=output,wait_for_resources=True,poll_seconds=master['resources']['poll_seconds'],
        ntfy_topic=a.ntfy_topic,max_workers_per_gpu=master['resources']['maximum_workers_per_gpu'])
    with publish_updates(a.queue,output):
        complete_gates=all((output/f'revision_gpu_{p["gpu"]}.json').exists() for p in plan['profiles'])
        if complete_gates:
            try:
                live.verified(output)
            except (OSError,ValueError,KeyError):complete_gates=False
        if complete_gates:
            result=live.execute(SimpleNamespace(output=output,maximum=args.max_workers_per_gpu,
                poll_seconds=args.poll_seconds,ntfy_topic=args.ntfy_topic))
        else:result=runner.execute(args,plan)
        campaign.publish_host(a.queue,output)
        return result


def main(argv=None):
    a=options(argv)
    if a.describe or a.initialize:
        spec,items=continuation.catalog(a.config)
        if a.write_configs:
            campaign.atomic(a.write_configs,dict(description=continuation.description(spec,items),items=items))
        if a.describe:
            print(json.dumps(continuation.description(spec,items),indent=2));return 0
        master=campaign.initialize(a.queue,spec,items,a.hosts)
        print(json.dumps(dict(queue=str(a.queue),configurations=len(items),
            assignedPerServer=dict(Counter(i['assignedHost'] for i in master['items'])),
            checkpointCopiesVerified=True,trainingStarted=False),indent=2));return 0
    if a.probe:
        spec=continuation.read(a.config);result=campaign.probe(spec)
        campaign.atomic(a.queue/'probes'/f'{result["host"]}.json',result)
        print(json.dumps(result,indent=2));return 0
    if a.execute:return execute(a)
    if a.status:print(campaign.render_status(a.queue));return 0
    try:
        while True:
            print('\033[2J\033[H'+campaign.render_status(a.queue),flush=True)
            if campaign.public_snapshot(a.queue)['status'] in ('COMPLETED','COMPLETED_WITH_FAILURES'):return 0
            time.sleep(3)
    except KeyboardInterrupt:return 0


if __name__=='__main__':raise SystemExit(main())
