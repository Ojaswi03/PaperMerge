#!/usr/bin/env python3
"""Headless A/B/C/D queue using the GUI's serial, checkpoint-aware runner.

Default operation is read-only. --execute is required to start training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRESET = ROOT / 'gui/queues/iid_abcd_500r_gpu_fresh.json'


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError('Invalid queue owner PID.')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def queue_lease(root):
    """Prevent duplicate terminal queues and refuse active GUI/worker owners."""
    worker_root = root / 'gui/worker_state/iid'
    worker_root.mkdir(parents=True, exist_ok=True)
    marker = root / 'gui/.queue_active.json'
    with (worker_root / 'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another terminal queue is already running.') from error
        if marker.exists():
            owner = json.loads(marker.read_text())
            if process_alive(owner.get('pid')):
                raise ValueError('A GUI or terminal queue is already active. Stop it first.')
        for path in worker_root.glob('*.pid.json'):
            if process_alive(json.loads(path.read_text()).get('pid')):
                raise ValueError(f'An IID worker is still active: {path.name}. Wait for graceful exit.')
        temporary = marker.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(dict(pid=os.getpid(),executionMode='terminal_queue'))+'\n')
        temporary.replace(marker)
        try:
            yield
        finally:
            if marker.exists() and json.loads(marker.read_text()).get('pid') == os.getpid():
                marker.unlink()


def load_prepared_queue(path, root=ROOT):
    from basil_core.iid_campaign import verify_campaign
    from basil_core.protocol_checkpoint import recovery_plan
    from gui.services.queue_service import QueueService
    from gui.state.experiment_state import to_persisted
    state = QueueService(path).load()
    configs = [to_persisted(entry.config) for entry in state.entries]
    verify_campaign(configs)
    if any(config['nRounds'] != 500 or config.get('executionDevice') != 'GPU' for config in configs):
        raise ValueError('This command requires the prepared four-condition 500-round GPU preset.')
    plans = [recovery_plan(root / config['resultDirectory'], config) for config in configs]
    return state, configs, plans


def run_queue(state, configs, *, root=ROOT, handle_factory=None, wait=time.sleep):
    from basil_core.protocol_checkpoint import recovery_plan
    from gui.services.queue_service import QueueService
    from gui.state.queue_state import QueueStatus
    if handle_factory is None:
        from gui.services.iid_execution_service import IidExecutionHandle
        handle_factory = IidExecutionHandle
    persistence = QueueService(root / 'gui/queue_state.json')
    stopped = False
    active = None

    def request_stop(_signum, _frame):
        nonlocal stopped
        if not stopped:
            print('Graceful stop requested; waiting for checkpoint and worker exit. Do not kill the process.', flush=True)
        stopped = True
        if active is not None:
            active.stop()

    original = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for entry, config in zip(state.entries, configs):
            if stopped:
                return 130
            plan = recovery_plan(root / config['resultDirectory'], config)
            if plan['mode'] == 'completed':
                entry.status = QueueStatus.COMPLETED
                persistence.save(state)
                print(f"{config['conditionId']}: already complete; skipped.", flush=True)
                continue
            terminal = None
            last_round = -1

            def event_sink(event):
                nonlocal terminal, last_round
                payload = event.payload
                if event.kind in {'execution_completed','execution_failed','execution_stopped'}:
                    terminal = event
                elif event.kind == 'progress_updated' and 'completedRounds' in payload:
                    done = payload['completedRounds']
                    if done != last_round:
                        last_round = done
                        print(f"{config['conditionId']}: {done}/{config['nRounds']} rounds complete", flush=True)
                elif event.kind == 'device_status_changed':
                    print(f"Device: {payload['device']} — {payload['deviceName']}", flush=True)
                elif event.kind == 'execution_restored':
                    print(f"Restored activation {payload['recoveryCompleted']}; preserved {payload['round']} completed rounds", flush=True)
                elif event.kind == 'status_updated':
                    print(payload.get('message',''), flush=True)

            print(f"{config['conditionId']}: {config['experimentName']} — {plan['mode']} at activation {plan['startActivation']}", flush=True)
            entry.status = QueueStatus.RUNNING
            entry.error = ''
            persistence.save(state)
            try:
                active = handle_factory(config, event_sink)
                if stopped:
                    active.stop()
                while active.active:
                    active.poll()
                    if active.active:
                        wait(.2)
                active = None
                if terminal is None:
                    raise RuntimeError('Worker exited without a terminal execution event.')
                if terminal.kind == 'execution_completed':
                    # An exit code alone is not evidence of 500 measured rounds.
                    if recovery_plan(root / config['resultDirectory'], config)['mode'] != 'completed':
                        raise RuntimeError('Worker reported completion but no completed result manifest exists.')
                    entry.status = QueueStatus.COMPLETED
                    persistence.save(state)
                    print(f"{config['conditionId']}: completed {config['nRounds']} rounds.", flush=True)
                    if stopped:
                        return 130
                else:
                    interrupted = stopped or terminal.kind == 'execution_stopped'
                    entry.status = QueueStatus.STOPPED if interrupted else QueueStatus.FAILED
                    entry.error = f"Worker return code {terminal.payload.get('returnCode')}; log: {terminal.payload.get('logPath')}"
                    persistence.save(state)
                    print(f"Queue {'stopped' if interrupted else 'failed'}: {entry.error}", flush=True)
                    return 130 if interrupted else 1
            except Exception as error:
                entry.status = QueueStatus.STOPPED if stopped else QueueStatus.FAILED
                entry.error = str(error)
                persistence.save(state)
                raise
        print('All four conditions are complete.', flush=True)
        return 0
    finally:
        if active is not None:
            active.stop()
            while active.active:
                active.poll()
                if active.active:
                    wait(.2)
        for sig, handler in original.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,default=PRESET)
    parser.add_argument('--execute',action='store_true',help='Manually start or resume the prepared queue')
    args = parser.parse_args(argv)
    state, configs, plans = load_prepared_queue(args.queue)
    for config, plan in zip(configs, plans):
        print(f"{config['conditionId']}: {config['nRounds']} rounds, GPU, {plan['mode']}, {plan['completedRounds']} rounds saved", flush=True)
    if not args.execute:
        print('Validation only — research runs not started. Add --execute to start/resume.', flush=True)
        return 0
    with queue_lease(ROOT):
        return run_queue(state, configs)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'Queue error: {error}',file=sys.stderr,flush=True)
        raise SystemExit(1)
