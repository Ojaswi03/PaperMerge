#!/usr/bin/env python3
"""Checkpoint and resume only hal9k with a versioned execution adapter."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import fleet_runner as runner,fleet_queue as fleet,live_gpu_scheduler as live
from basil_core import fedsharp_runtime_optimization as optimization
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic
from basil_core.round_continuation import sha
from scripts.refill_fedsharp_gpus import agent_identity


def wait_exited(descriptors,seconds):
    remaining=list(descriptors);deadline=time.monotonic()+seconds
    while remaining and time.monotonic()<deadline:
        finished,_,_=live.select.select(remaining,[],[],min(5,max(0,deadline-time.monotonic())))
        remaining=[fd for fd in remaining if fd not in finished]
    if remaining:raise RuntimeError('Processes are still checkpointing. No forced kill was sent. Rerun after they exit.')


def resume_claims(queue,locations):
    """Only stopped, checkpointed hal9k claims; other host entries are untouched."""
    import numpy as np
    receipts=[];changes=[]
    snapshot=fleet.state(queue);master=read_json(Path(queue)/'continuation_plan.json')
    # Hash/parse large local checkpoints outside the global queue lock.
    for run,entry in snapshot['runs'].items():
        if entry['owner']!='hal9k' or entry['state'] not in ('RUNNING','CLAIMED'):continue
        root=Path(entry['location']['output'])
        if str(root) not in locations:raise ValueError('Unknown hal9k output namespace.')
        if entry.get('process') and live.same_process(entry['process']):
            raise ValueError('A hal9k worker is still running; no claim was reset.')
        row_path=root/run/'run.json';row=read_json(row_path) if row_path.exists() else {}
        if row.get('status')=='COMPLETED':
            changes.append((run,entry,'DONE',row,None));continue
        if row.get('status')=='FAILED':
            changes.append((run,entry,'FAILED',row,None));continue
        checkpoint=root/run/'checkpoint.npz'
        if row.get('status') in ('RUNNING','INTERRUPTED') and not checkpoint.exists():
            raise ValueError('Started worker has no checkpoint; inspect before recovery.')
        if checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as archive:
                meta=json.loads(archive['metadata'].tobytes())
            if meta['contract']!=row.get('contract') or meta['cursor']<row.get('completedActivations',0):
                raise ValueError('Stopped checkpoint differs from its last recorded contract/progress.')
            # Hard links preserve the exact pre-transition archive: saves
            # replace checkpoint.npz atomically, never edit the old inode.
            backup=root/'operational_adapters/optimization_boundary'/run/'checkpoint.npz'
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():os.link(checkpoint,backup)
            if sha(backup)!=sha(checkpoint):raise ValueError('An earlier optimization boundary already exists.')
            receipts.append(dict(runId=run,cursor=meta['cursor'],checkpointSha256=sha(backup),
                originalGpu=entry['boundGpu'],checkpointBackup=str(backup)))
            (backup.parent/'run.json').write_bytes(row_path.read_bytes())
            row.update(completedActivations=meta['cursor'],completedRounds=len(meta['roundMetrics']),elapsedSeconds=meta['elapsedSeconds'])
        row.update(status='INTERRUPTED',phase='OPTIMIZATION_RESUME_PENDING')
        changes.append((run,entry,'WAITING',row,row_path))
    with fleet.locked(queue) as (state,master):
        for run,expected,status,row,path in changes:
            if state['runs'][run]!=expected:raise ValueError('A hal9k claim changed during checkpoint verification.')
        for run,expected,status,row,path in changes:
            if path:atomic(path,row)
            state['runs'][run].update(state=status,public=fleet.public_row(master['items'][expected['index']],row))
            if status=='WAITING':state['runs'][run]['eligibleHosts']=['hal9k']
    return receipts


def transition(queue,topic):
    if socket.gethostname().split('.')[0]!='hal9k':raise ValueError('Refusing to modify a host other than hal9k.')
    queue=Path(queue).resolve();original,output,legacy,plan,master=runner.prepare(queue)
    active_policy=output/'optimization_policy.json'
    if active_policy.exists():
        policy=read_json(active_policy);optimization.validate(policy)
        if policy.get('state')=='ACTIVE':raise ValueError('Optimization is already active. Use the fleet monitor; do not repeat the transition.')
    with ExitStack() as stack:
        guard=stack.enter_context((output/'optimization_transition.lock').open('a'))
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        profiles={p['gpu']:runner.ExecutionProfile(**p) for p in plan['profiles']}
        if runner.discover_gates(output,profiles):raise ValueError('A GPU verification is active; let it finish before transitioning.')
        try:owner=agent_identity(output,queue)
        except (FileNotFoundError,ProcessLookupError):owner=None
        if owner:
            fd=os.pidfd_open(owner['pid']);stack.callback(os.close,fd)
            if not live.same_process(owner):raise ProcessLookupError('Dispatcher identity changed.')
            signal.pidfd_send_signal(fd,signal.SIGTERM);wait_exited([fd],40)
        # Fence all old host dispatch paths before checkpointing workers.
        for path in (output/'agent.lock',original/'supervisor.lock',original/'live_scheduler.lock'):
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        workers=runner.discover(original,output,legacy,plan)
        receipt_path=output/'optimization_transition.json'
        transition_id=uuid.uuid4().hex
        atomic(receipt_path,dict(id=transition_id,host='hal9k',stage='CHECKPOINTING',workers=workers,
            scientificSource=plan['source'],otherHostsSignalled=False,startedAt=time.time()))
        descriptors=[]
        for item in workers.values():
            try:fd=os.pidfd_open(item['pid'])
            except ProcessLookupError:continue
            stack.callback(os.close,fd)
            if not live.same_process(item):continue
            signal.pidfd_send_signal(fd,signal.SIGTERM);descriptors.append(fd)
        print(f'Checkpoint requested for {len(descriptors)} hal9k workers; themachine is untouched.',flush=True)
        wait_exited(descriptors,600)
        if runner.discover(original,output,legacy,plan):raise ValueError('Unexpected live hal9k worker remains.')
        boundaries=resume_claims(queue,[str(original),str(output)])
        policy=dict(revision=optimization.REVISION,source=optimization.source(),host='hal9k',state='ACTIVE',
            transitionId=transition_id,outputs=[str(original),str(output)],createdAt=time.time())
        for root in (original,output):
            for name,digest in policy['source'].items():
                target=root/'optimization_snapshots'/digest/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha(target)!=digest:raise ValueError('Optimization snapshot differs.')
                if not target.exists():target.write_bytes((ROOT/name).read_bytes())
            atomic(root/'optimization_policy.json',policy)
        atomic(receipt_path,dict(id=transition_id,host='hal9k',stage='CHECKPOINTED',boundaries=boundaries,
            otherHostsSignalled=False,scientificSource=plan['source'],optimization=policy))
    command=[sys.executable,'-u',str(ROOT/'scripts/run_fedsharp_fleet.py'),'--queue',str(queue),
        '--execute','--background','--ntfy-topic',topic]
    subprocess.run(command,cwd=ROOT,check=True)
    print('hal9k optimization enabled. Workers resume from saved activation cursors on their recorded GPUs.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--ntfy-topic',default='papermerge-ojaswi')
    args=parser.parse_args()
    if socket.gethostname().split('.')[0]!='hal9k':parser.error('This command is restricted to hal9k.')
    if not args.execute:parser.error('--execute is required to checkpoint and resume hal9k.')
    if args.background:
        original,output,*_=runner.prepare(args.queue)
        with (output/'optimization_transition.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--queue',str(args.queue.resolve()),
                '--execute','--ntfy-topic',args.ntfy_topic],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'Transition PID {child.pid}. Log: {output}/optimization_transition.log')
        return 0
    transition(args.queue,args.ntfy_topic)
    return 0


if __name__=='__main__':raise SystemExit(main())
