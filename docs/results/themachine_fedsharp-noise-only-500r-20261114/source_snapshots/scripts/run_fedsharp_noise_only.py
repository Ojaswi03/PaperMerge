#!/usr/bin/env python3
"""Prepare and run 18 fresh noise-only controls on hal9k's two GPUs."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core import fedsharp_noise_only as study
from scripts import run_basil_sharp_thesis as driver, run_cart_adaptive as adaptive

PRESET = ROOT / 'configs/research/hal9k_fedsharp_noise_only_500r.json'


def execute(args, plan):
    from basil_core.cart_campaign_io import read_json, write_json
    from basil_core.cart_worker_admission import placement
    from basil_core.execution_profile import gpu_inventory
    from basil_core import live_gpu_scheduler as live
    root = args.output
    hardware, _ = placement(plan)
    with adaptive.acquire_supervisor(root, False, 180):
        adaptive.ensure_source_snapshots(root, plan)
        while True:
            try:
                inventory = gpu_inventory()
                for gpu, profile in hardware.items():
                    profile.bind(inventory)
                    free = next(g['memory.free'] for g in inventory if g['index'] == gpu)
                    if free < profile.memory_mib + profile.headroom_mib + 1024:
                        raise RuntimeError(f'GPU {gpu}: waiting for memory.')
                if adaptive.scaled.available_ram_mib() < plan['resourceBudget']['ram_headroom_mib'] + next(iter(hardware.values())).ram_mib:
                    raise RuntimeError('Waiting for host RAM.')
                leases = adaptive.scaled.acquire_leases(list(hardware.values()))
                break
            except RuntimeError as error:
                write_json(root / 'campaign.json', dict(status='WAITING_FOR_RESOURCES',
                    host=socket.gethostname(), ownerPid=os.getpid(), total=len(plan['configs']),
                    completed=0, reason=str(error), elapsedSeconds=0.))
                if not args.wait_for_resources:
                    raise
                print(str(error), flush=True)
                time.sleep(5)
        try:
            digest = adaptive.sha(root / 'plan.json')
            for gpu, profile in hardware.items():
                path = root / f'revision_gpu_{gpu}.json'
                saved = read_json(path) if path.exists() else {}
                if saved.get('status') == 'PASSED' and saved.get('source') == plan['source'] and saved.get('planSha256') == digest:
                    continue
                write_json(root / 'campaign.json', dict(status='VERIFYING_GPUS',
                    host=socket.gethostname(), ownerPid=os.getpid(), total=len(plan['configs']),
                    completed=0, elapsedSeconds=0.))
                fd = leases[gpu].fileno()
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--role', 'gate', '--output', str(root), '--gpu', str(gpu), '--lease-fd', str(fd)],
                    cwd=ROOT, env=adaptive.scaled.core.worker_environment(profile), pass_fds=(fd,), check=True)
        finally:
            for handle in leases.values():
                handle.close()
    with patch.object(live, 'WORKER', Path(__file__).resolve()), \
         patch.object(live, 'OWNERS', live.OWNERS | {Path(__file__).name}):
        return live.execute(SimpleNamespace(output=root, maximum=plan['maximumWorkersPerGpu'],
            poll_seconds=args.poll_seconds, ntfy_topic=args.ntfy_topic))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--role', choices=('driver', 'gate', 'verify'), default='driver')
    pre.add_argument('--worker', type=int)
    pre.add_argument('--gpu', type=int)
    pre.add_argument('--lease-fd', type=int)
    pre.add_argument('--background', action='store_true')
    pre.add_argument('--monitor', action='store_true')
    pre.add_argument('--official-test', action='store_true')
    pre.add_argument('--write-configs', type=Path)
    control, remaining = pre.parse_known_args(argv)
    if '--config' not in remaining and '--resume' not in remaining:
        remaining = ['--config', str(PRESET), *remaining]
    args = driver.parse_options(remaining)
    if control.official_test and control.role != 'verify':
        raise ValueError('--official-test is a post-training verification option only.')
    if control.monitor:
        from basil_core.sharp_monitor import snapshot, render
        try:
            while True:
                text = render(snapshot(args.output), width=160, height=45, table=True)
                print('\033[2J\033[HFedSharp noise-only controls |' + text.split('|', 1)[-1], flush=True)
                time.sleep(5)
        except KeyboardInterrupt:
            return 0
    if control.worker is not None or control.role == 'gate' or args.execute:
        if socket.gethostname().split('.')[0] != 'hal9k':
            raise ValueError('Execution is restricted to hal9k.')
    with study.installed():
        if control.worker is not None:
            from scripts.cart_comparison_worker import run
            from basil_core import fedsharp_runtime_optimization as execution, round_timing
            if control.lease_fd is None:
                raise ValueError('A worker requires its inherited GPU lease.')
            plan = json.loads((args.output / 'plan.json').read_text())
            view = plan['configs'][control.worker]
            if plan.get('executionOptimization') != study.runtime_policy():
                raise ValueError('The execution policy differs from the frozen plan.')
            with execution.installed(args.output, view, plan['executionOptimization']), \
                 round_timing.recording(args.output, control.worker):
                return run(args.output, control.worker, control.lease_fd)
        if control.role == 'gate':
            from scripts.verify_fedsharp_noise_only_gpu import verify
            return verify(args.output, control.gpu, control.lease_fd)
        if control.role == 'verify':
            from scripts.verify_cart_comparison_results import verify
            return verify(args.output, official_test=control.official_test)
        if args.describe or control.write_configs:
            views = study.selected(args)
            for view in views:
                study.effective_config(view)
            report = dict(revision=study.REVISION, configurations=len(views),
                fedSharp=sum(not v['comparisonCase']['addon'] for v in views),
                fedSharpPlus=sum(v['comparisonCase']['addon'] == 'cart' for v in views),
                datasets=args.datasets, seeds=args.seeds, rounds=args.rounds, nodes=args.nodes,
                localSteps=args.local_steps, proximalMu=1., effectiveChannelSigma=args.sigma,
                effectiveAttackers=0, requestedGpuCount=args.gpu_count, start='initialization',
                executionOptimization=study.runtime_policy(),
                activations=len(views) * args.rounds * args.nodes,
                optimizerSteps=len(views) * args.rounds * args.nodes * args.local_steps,
                officialTestUsed=False, trainingStarted=False)
            if control.write_configs:
                control.write_configs.parent.mkdir(parents=True, exist_ok=True)
                control.write_configs.write_text(json.dumps(dict(description=report, configs=views), indent=2) + '\n')
            print(json.dumps(report, indent=2))
            return 0
        if not args.execute:
            raise ValueError('Use --describe or --execute.')
        if args.resume and args.supplied_options - {'--output', '--execute', '--resume', '--ntfy-topic', '--wait-for-resources'}:
            raise ValueError('Resume the original output without scientific overrides.')
        args.output.mkdir(parents=True, exist_ok=True)
        if control.background:
            child = [v for v in argv if v != '--background']
            with (args.output / 'noise_only_supervisor.log').open('a') as log:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), *child],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(f'Noise-only supervisor PID {process.pid}. Log: {args.output}/noise_only_supervisor.log')
            return 0
        with (args.output / 'preparation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = driver.prepare(args)
            policy = study.runtime_policy()
            if args.resume and plan.get('executionOptimization') != policy:
                raise ValueError('Frozen execution policy changed.')
            if not args.resume:
                plan['executionOptimization'] = policy
                driver.write_json(args.output / 'plan.json', plan)
        print(json.dumps(driver.readiness(args.output, plan, args.max_workers_per_gpu), indent=2), flush=True)
        if args.ntfy_topic:
            published = subprocess.run([sys.executable, str(ROOT / 'scripts/share_cart_monitor.py'),
                '--output', str(args.output), '--background', '--restart-publisher',
                '--interval', '1', '--ntfy-topic', args.ntfy_topic], cwd=ROOT, check=False)
            if published.returncode:
                print('Browser publisher startup failed; training continues.', flush=True)
        return execute(args, plan)


if __name__ == '__main__':
    main()
