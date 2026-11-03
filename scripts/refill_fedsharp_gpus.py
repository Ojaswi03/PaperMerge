#!/usr/bin/env python3
"""Refill spare GPU capacity by transferring leases; never signal training workers."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core import fleet_runner as runner, live_gpu_scheduler as live
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic
from basil_core.execution_profile import ExecutionProfile
from basil_core.fleet_lease_transfer import capture


def agent_identity(output, queue):
    identity = live.identity(read_json(output/'agent.json')['pid'])
    script, args = live.script_command(identity)
    value = live.argv_value(args, '--queue')
    if (script != ROOT/'scripts/run_fedsharp_fleet.py' or '--serve' not in args
            or '--execute' not in args or value is None
            or (Path(identity['cwd'])/value).resolve() != queue):
        raise ValueError('The recorded process is not this queue\'s fleet dispatcher.')
    return identity


def handoff(queue, execute=False, ntfy_topic='papermerge-ojaswi'):
    queue = Path(queue).resolve()
    original, output, legacy, plan, master = runner.prepare(queue)
    if not (original/'fleet_owner.json').exists():
        raise ValueError('This host must already belong to the running fleet.')
    with ExitStack() as stack:
        guard = stack.enter_context((output/'lease_transfer.lock').open('a'))
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = agent_identity(output, queue)
        owner_fd = os.pidfd_open(owner['pid'])
        stack.callback(os.close, owner_fd)
        if not live.same_process(owner):
            raise ProcessLookupError('Dispatcher identity changed before handoff.')
        profiles = {p['gpu']: ExecutionProfile(**p) for p in plan['profiles']}
        before = runner.discover(original, output, legacy, plan)
        identities = [owner] + list(before.values()) + list(runner.discover_gates(output, profiles).values())
        leases = {}; details = []
        # All handles must be captured BEFORE the dispatcher receives any signal.
        for gpu, profile in profiles.items():
            handle, detail = capture(profile, identities)
            leases[gpu] = stack.enter_context(handle)
            details.append(detail)
        result = dict(eligible=True,execute=execute,gpus=details,workerSignalsSent=False,
            scientificSourceUnchanged=True,resourceBudgetsUnchanged=True)
        if not execute:
            return result
        if not live.same_process(owner):
            raise ProcessLookupError('Dispatcher changed; no signal sent.')
        transfer = uuid.uuid4().hex
        atomic(output/'lease_transfers'/f'{transfer}.json',dict(result,oldAgent=owner,
            startedAt=time.time(),workersBefore=before,stage='leases captured'))
        # The known fleet handler stops dispatch only. Its worker children survive.
        signal.pidfd_send_signal(owner_fd, signal.SIGTERM)
        if not live.select.select([owner_fd], [], [], 25)[0]:
            raise RuntimeError('Dispatcher has not exited; workers continue. No replacement was launched.')
        command = [sys.executable, '-u', str(ROOT/'scripts/run_fedsharp_fleet.py'),
            '--queue',str(queue),'--execute','--serve','--ntfy-topic',ntfy_topic,'--handover-id',transfer]
        for gpu, handle in leases.items():
            command.extend(['--inherited-gpu-lease',f'{gpu}:{handle.fileno()}'])
        with (output/'fleet_agent.log').open('a') as log:
            child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,
                pass_fds=tuple(handle.fileno() for handle in leases.values()))
        deadline = time.monotonic()+30
        ready_path = output/'lease_handover_ready.json'
        while time.monotonic()<deadline:
            if ready_path.exists():
                ready = read_json(ready_path)
                if (ready.get('handoverId')==transfer and ready.get('agent',{}).get('pid')==child.pid
                        and ready.get('gpus')==sorted(profiles) and live.same_process(ready['agent'])):
                    after = runner.discover(original,output,legacy,plan)
                    missing = [run for run,item in before.items() if live.same_process(item)
                        and (run not in after or after[run]['pid']!=item['pid'] or after[run]['start']!=item['start'])]
                    if missing:raise RuntimeError('A surviving worker was not rediscovered; inspect the handoff receipt.')
                    result.update(agentPid=child.pid,workersPreserved=sum(live.same_process(item) for item in before.values()),
                        observedWorkersAfter=len(after),handoverId=transfer)
                    atomic(output/'lease_transfers'/f'{transfer}.json',dict(result,
                        oldAgent=owner,workersBefore=before,workersAfter=after,stage='ready',completedAt=time.time()))
                    return result
            if child.poll() is not None:
                raise RuntimeError(f'Replacement dispatcher exited {child.returncode}; workers continue. Inspect {output}/fleet_agent.log.')
            time.sleep(.2)
        raise RuntimeError(f'Replacement dispatcher readiness is unconfirmed (PID {child.pid}); do not launch another. Inspect {output}/fleet_agent.log.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='Replace only the dispatcher after every existing GPU lease is safely captured.')
    parser.add_argument('--ntfy-topic',default='papermerge-ojaswi')
    args=parser.parse_args()
    try:
        print(json.dumps(handoff(args.queue,args.execute,args.ntfy_topic),indent=2))
    except (OSError,ValueError,RuntimeError) as error:
        print(str(error),file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
