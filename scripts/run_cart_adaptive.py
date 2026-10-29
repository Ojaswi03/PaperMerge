#!/usr/bin/env python3
"""Fill verified GPUs as memory permits, preserving the frozen plan/checkpoints.

Default is read-only preflight. --execute starts scheduling. --take-over asks
the validated old supervisor to checkpoint and release its leases first.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core.cart_campaign_io import read_json, write_json, retry_io
from basil_core.cart_worker_admission import Resident, admission, placement
from basil_core.cart_comparison_study import check_source
from basil_core.execution_profile import gpu_inventory, gpu_processes
from scripts import run_cart_v2_scaled as scaled


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan(root):
    plan = read_json(root / 'plan.json')
    check_source(plan['source'])
    hardware, assigned = placement(plan)
    inventory = gpu_inventory()
    for p in hardware.values():
        p.bind(inventory)
        if p.name != 'hpc' or Path(p.output_root).resolve() != root:
            raise ValueError('Adaptive scheduling requires the original HPC output path.')
        receipt = read_json(root / f'revision_gpu_{p.gpu}.json')
        if (receipt['status'] != 'PASSED' or receipt['source'] != plan['source'] or
                receipt['runtime']['profile'] != asdict(p) or
                receipt['runtime']['host'] != socket.gethostname()):
            raise ValueError('GPU verification differs from this host/source/profile.')
    campaign = read_json(root / 'campaign.json') if (root / 'campaign.json').exists() else {}
    if campaign.get('host', socket.gethostname()) != socket.gethostname():
        raise ValueError('Run belongs to a different server; no process will be stopped.')
    return plan, hardware, assigned


def validated_owner(root, campaign):
    """Refuse PID reuse, other users, another host, or an unrelated command."""
    if campaign.get('host') != socket.gethostname():
        raise ValueError('Supervisor belongs to another host.')
    pid = campaign.get('ownerPid')
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        raise ValueError('Invalid supervisor PID.')
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return None
    if proc.stat().st_uid != os.getuid():
        raise ValueError('Supervisor belongs to another user.')
    argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    allowed = {str(ROOT / 'scripts/run_cart_comparison.py'), str(Path(__file__).resolve())}
    if len(argv) < 2 or argv[1] not in allowed or '--execute' not in argv:
        raise ValueError('PID is not a recognized comparison supervisor.')
    output = next((a.split('=', 1)[1] for a in argv if a.startswith('--output=')), None)
    if '--output' in argv:
        output = argv[argv.index('--output') + 1]
    if output is None or Path(output).resolve() != root:
        raise ValueError('Supervisor PID refers to another output.')
    return pid


def acquire_supervisor(root, take_over, timeout):
    """Take the same exclusive lock only after graceful checkpoint completion."""
    handle = (root / 'supervisor.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        if not take_over:
            handle.close()
            raise RuntimeError('Supervisor is running. Use --take-over to preserve and resume its checkpoints.')
    try:
        campaign = read_json(root / 'campaign.json')
        pid = validated_owner(root, campaign)
        if pid is None:
            raise RuntimeError('Supervisor lock is held but its recorded owner is absent.')
        print(f'Asking supervisor {pid} to checkpoint; waiting for its workers.', flush=True)
        # pidfd targets this exact process even if the PID exits and is reused.
        fd = os.pidfd_open(pid)
        try:
            if validated_owner(root, campaign) != pid:
                raise RuntimeError('Supervisor exited during handoff.')
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                time.sleep(1)
        raise RuntimeError('Checkpoint handoff timed out. No workers were force-killed; inspect their logs.')
    except BaseException:
        handle.close()
        raise


def archive(root):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    history = root / 'scheduling_history' / stamp
    history.mkdir(parents=True)
    hashes = {}
    for name in ('plan.json', 'revision_execution.json', 'campaign.json',
                 'supervisor.log', 'adaptive_execution.json', 'failed_runs.json'):
        path = root / name
        if path.exists():
            shutil.copyfile(path, history / name)
            hashes[name] = sha(history / name)
    write_json(history / 'manifest.json', dict(files=hashes, checkpointsUnmodified=True))
    return str(history.relative_to(root))


def residents_of(active, assigned):
    result = []
    for index, (process, _) in active.items():
        if process.poll() is not None:
            continue
        # Workers are single processes with threads, not loader child processes.
        lines = Path(f'/proc/{process.pid}/status').read_text().splitlines()
        values = {v.split(':', 1)[0]: v.split(':', 1)[1].strip() for v in lines if ':' in v}
        rss = (int(values['VmRSS'].split()[0]) + int(values.get('VmSwap', '0 kB').split()[0])) // 1024
        result.append(Resident(process.pid, assigned[index].gpu, rss))
    return result


def preflight(root, maximum):
    plan, hardware, assigned = validate_plan(root)
    counts = Counter(p.gpu for p in assigned)
    budget = plan['resourceBudget']
    inventory = gpu_inventory()
    ram = scaled.available_ram_mib()
    # Current free memory is conservative before a handoff: existing workers
    # still occupy their allocations. Recheck live after they release them.
    capacity = {}
    for gpu, p in hardware.items():
        free = next(r['memory.free'] for r in inventory if r['index'] == gpu)
        capacity[gpu] = min(counts[gpu], maximum, max(0, (free - p.headroom_mib) //
                            (p.memory_mib + budget['context_overhead_mib'])))
    return dict(configurations=len(assigned), gpuCapacityFromCurrentFreeMemory=capacity,
                hostAvailableMiB=ram, hostWorkerCapacity=max(0, (ram - budget['ram_headroom_mib']) // assigned[0].ram_mib),
                maximumPerGpu=maximum, planPreserved=True, trainingStarted=False)


def execute(args):
    root = args.output
    # Serialize handoffs as well as scheduling; never stop a second controller.
    with (root / 'adaptive.lock').open('a') as operation:
        fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, hardware, assigned = validate_plan(root)
        original_plan = sha(root / 'plan.json')
        with acquire_supervisor(root, args.take_over, args.stop_timeout):
            leases = scaled.acquire_leases(list(hardware.values()))
            try:
                return schedule(args, plan, hardware, assigned, original_plan, leases)
            finally:
                for lease in leases.values():
                    lease.close()


def schedule(args, plan, hardware, assigned, original_plan, leases):
    root = args.output
    history = archive(root)
    script_names = ('scripts/run_cart_adaptive.py', 'basil_core/cart_worker_admission.py')
    write_json(root / 'adaptive_execution.json', dict(
        planSha256=original_plan, source=plan['source'], previousScheduling=history,
        schedulerSha256={n: sha(ROOT / n) for n in script_names},
        maximumPerGpu=args.max_workers_per_gpu, memoryPolicy='reserve-unallocated-worker-budgets',
        checkpointPolicy='resume original worker contracts; never reassign GPUs'))
    for name in script_names:
        target = root / 'scheduler_snapshots' / Path(history).name / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    queues = {gpu: deque() for gpu in hardware}
    completed = 0
    for i, cfg in enumerate(plan['configs']):
        path = root / cfg['runId'] / 'run.json'
        if path.exists() and read_json(path)['status'] == 'COMPLETED':
            completed += 1
        else:
            queues[assigned[i].gpu].append(i)
    active = {}
    failures = []
    write_json(root / 'failed_runs.json', failures)
    blocked = {}
    status = 'RUNNING'
    started = time.monotonic()
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def campaign():
        write_json(root / 'campaign.json', dict(
            status=status, total=len(assigned), completed=completed, failed=len(failures),
            pending=sum(map(len, queues.values())), activeSlots=list(active),
            pendingRunIds=[plan['configs'][i]['runId'] for q in queues.values() for i in q],
            activeRunIds=[plan['configs'][i]['runId'] for i in active], failures=failures,
            workerFailurePolicy='continue', runtimeAdapter=plan['revision'],
            elapsedSeconds=time.monotonic() - started, host=socket.gethostname(),
            profile=assigned[0].name, ownerPid=os.getpid(),
            scheduler=dict(mode='adaptive', maximumPerGpu=args.max_workers_per_gpu,
                activePerGpu=dict(Counter(assigned[i].gpu for i in active)), blocked=blocked,
                lanesByRunId={c['runId']: f'{assigned[i].gpu}/adaptive{i}' for i, c in enumerate(plan['configs'])})))

    def failure(i, code, message):
        cfg = plan['configs'][i]
        path = root / cfg['runId'] / 'run.json'
        path.parent.mkdir(exist_ok=True)
        row = read_json(path) if path.exists() else {}
        row.update(status='FAILED', error=row.get('error') or message, workerExitCode=code)
        write_json(path, row)
        event = dict(runId=cfg['runId'], index=i, gpu=assigned[i].gpu,
                     exitCode=code, error=row['error'], log=cfg['runId'] + '.log')
        failures.append(event)
        with (root / 'worker_failure_events.jsonl').open('a') as log:
            log.write(json.dumps(event) + '\n')
        write_json(root / 'failed_runs.json', failures)
        print(f'FAILED {cfg["runId"]}: {row["error"]}', flush=True)

    campaign()
    try:
        while active or any(queues.values()):
            for i, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[i]
                path = root / plan['configs'][i]['runId'] / 'run.json'
                row = read_json(path) if path.exists() else {}
                if code or row.get('status') != 'COMPLETED':
                    failure(i, code, f'Worker exited {code}; inspect its log.')
                else:
                    completed += 1
            if stopped:
                raise InterruptedError('Adaptive supervisor stop requested.')
            retry_io(check_source, plan['source'])
            if sha(root / 'plan.json') != original_plan:
                raise ValueError('Frozen plan changed during execution.')
            # One launch per GPU per pass, round-robin. Every pass samples live
            # memory anew; newly spawned workers reserve their entire budgets.
            for gpu, p in hardware.items():
                if not queues[gpu]:
                    continue
                try:
                    decision = admission(p, residents_of(active, assigned), gpu_inventory(),
                        gpu_processes(), scaled.available_ram_mib(),
                        context_mib=plan['resourceBudget']['context_overhead_mib'],
                        ram_headroom_mib=plan['resourceBudget']['ram_headroom_mib'],
                        maximum=args.max_workers_per_gpu)
                except (OSError, RuntimeError) as error:
                    decision = dict(admit=False, reason=str(error))
                blocked[gpu] = decision
                if not decision['admit']:
                    continue
                i = queues[gpu].popleft()
                cfg = plan['configs'][i]
                log = (root / (cfg['runId'] + '.log')).open('a')
                fd = leases[gpu].fileno()
                command = [sys.executable, str(ROOT / 'scripts/cart_comparison_worker.py'),
                           '--worker', str(i), '--output', str(root), '--lease-fd', str(fd)]
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=scaled.core.worker_environment(p),
                        pass_fds=(fd,), stdout=log, stderr=subprocess.STDOUT)
                except OSError as error:
                    log.close()
                    failure(i, None, str(error))
                    continue
                active[i] = (process, log)
                print(f'GPU {gpu}: started {cfg["runId"]}; {len(active)} total active', flush=True)
            campaign()
            if active or any(queues.values()):
                time.sleep(args.poll_seconds)
        status = 'COMPLETED_WITH_FAILURES' if failures else 'COMPLETED'
        if not failures:
            try:
                from reporting.cart_comparison import generate
                generate(root, Path(plan.get('figureDestination') or root / 'figures'), reference=plan.get('reference'))
            except Exception as error:
                status = 'COMPLETED_REPORT_FAILED'
                write_json(root / 'report_status.json', dict(trainingComplete=True, status=status, error=str(error)))
        print(f'{status}: {completed}/{len(assigned)} complete.', flush=True)
        if args.ntfy_topic:
            from basil_core.ntfy_notifications import publish
            from basil_core.cart_notification_text import event
            kind = 'partial' if failures else ('report_failed' if status == 'COMPLETED_REPORT_FAILED' else 'complete')
            title, message = event(kind, root, completed=completed, total=len(assigned), failed=len(failures),
                nodes=plan['configs'][0]['nodes'], rounds=plan['configs'][0]['rounds'])
            write_json(root / 'notification.json', publish(args.ntfy_topic, title, message))
        return 2 if failures else (3 if status == 'COMPLETED_REPORT_FAILED' else 0)
    except BaseException as error:
        status = 'INTERRUPTED' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED'
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, log in active.values():
            process.wait()
            log.close()
        raise
    finally:
        campaign()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--take-over', action='store_true')
    parser.add_argument('--max-workers-per-gpu', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--poll-seconds', type=int, choices=range(1, 31), default=3, metavar='SECONDS')
    parser.add_argument('--stop-timeout', type=int, choices=range(10, 601), default=180, metavar='SECONDS')
    parser.add_argument('--tmux')
    parser.add_argument('--ntfy-topic')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.ntfy_topic:
        from basil_core.ntfy_notifications import validate_destination
        validate_destination(args.ntfy_topic, 'https://ntfy.sh')
    print(json.dumps(preflight(args.output, args.max_workers_per_gpu), indent=2), flush=True)
    if not args.execute:
        return 0
    if args.tmux:
        forwarded = sys.argv[1:]
        pos = forwarded.index('--tmux')
        forwarded = forwarded[:pos] + forwarded[pos + 2:]
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), *forwarded])
        command += ' >> ' + shlex.quote(str(args.output / 'adaptive_supervisor.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', args.tmux, 'bash', '-c', command], check=True)
        print(f'Started {args.tmux}; inspect adaptive_supervisor.log for handoff and admission.', flush=True)
        return 0
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
