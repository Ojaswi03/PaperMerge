#!/usr/bin/env python3
"""Run the eighteen noise-only controls on themachine with per-device admission."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core import fedsharp_noise_only as study, noise_only_machine as machine
from scripts import run_basil_sharp_thesis as driver, run_cart_adaptive as adaptive

PRESET = ROOT / 'configs/research/themachine_fedsharp_noise_only_500r.json'


def execute(args, plan):
    from basil_core.cart_campaign_io import write_json
    from basil_core import live_gpu_scheduler as live
    with adaptive.acquire_supervisor(args.output, False, 180):
        adaptive.ensure_source_snapshots(args.output, plan)
        if not (args.output / 'campaign.json').exists():
            write_json(args.output / 'campaign.json', dict(status='WAITING_FOR_RESOURCES',
                host=socket.gethostname(), ownerPid=os.getpid(), total=len(plan['configs']),
                completed=0, failed=0, pending=len(plan['configs']), elapsedSeconds=0.))
    with machine.scheduler(args.output, plan):
        return live.execute(SimpleNamespace(output=args.output, maximum=plan['maximumWorkersPerGpu'],
            poll_seconds=args.poll_seconds, ntfy_topic=args.ntfy_topic))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--role', choices=('driver', 'gate', 'verify', 'export'), default='driver')
    pre.add_argument('--worker', type=int)
    pre.add_argument('--gpu', type=int)
    pre.add_argument('--lease-fd', type=int)
    pre.add_argument('--background', action='store_true')
    pre.add_argument('--monitor', action='store_true')
    pre.add_argument('--official-test', action='store_true')
    pre.add_argument('--write-configs', type=Path)
    pre.add_argument('--destination', type=Path)
    control, remaining = pre.parse_known_args(argv)
    if '--config' not in remaining and '--resume' not in remaining:
        remaining = ['--config', str(PRESET), *remaining]
    args = driver.parse_options(remaining)
    if control.official_test and control.role != 'verify':
        raise ValueError('--official-test is a post-training verification option only.')
    if control.monitor:
        from scripts.run_fedsharp_noise_only import main as monitor
        return monitor(['--output', str(args.output), '--monitor'])
    if control.worker is not None or control.role == 'gate' or args.execute:
        if socket.gethostname().split('.')[0] != 'themachine':
            raise ValueError('This launcher is restricted to themachine.')
    if args.ntfy_topic:
        from basil_core.ntfy_notifications import validate_destination
        validate_destination(args.ntfy_topic, 'https://ntfy.sh')
    with machine.installed():
        if control.role == 'export':
            from scripts.export_fedsharp_noise_only import export
            if control.destination is None:
                raise ValueError('--destination is required for export.')
            print(json.dumps(export(args.output, control.destination, verifier_script=Path(__file__).resolve()), indent=2))
            return 0
        if control.role == 'verify':
            from scripts.verify_cart_comparison_results import verify
            return verify(args.output, official_test=control.official_test)
        if control.role == 'gate':
            from scripts.verify_fedsharp_noise_only_gpu import verify
            return verify(args.output, control.gpu, control.lease_fd)
        if control.worker is not None:
            from scripts.cart_comparison_worker import run
            from basil_core import fedsharp_runtime_optimization as execution, round_timing
            plan = json.loads((args.output / 'plan.json').read_text())
            machine.verified(args.output)
            if control.lease_fd is None:
                raise ValueError('A worker requires its inherited GPU lease.')
            view = plan['configs'][control.worker]
            with execution.installed(args.output, view, plan['executionOptimization']), \
                 round_timing.recording(args.output, control.worker):
                return run(args.output, control.worker, control.lease_fd)
        if args.describe or control.write_configs:
            # Reuse the pure scientific description, then include placement.
            views = study.selected(args)
            for view in views:
                study.effective_config(view)
            report = dict(revision=study.REVISION, configurations=len(views),
                fedSharp=sum(not v['comparisonCase']['addon'] for v in views),
                fedSharpPlus=sum(v['comparisonCase']['addon'] == 'cart' for v in views),
                nodes=args.nodes, rounds=args.rounds, localSteps=args.local_steps,
                effectiveChannelSigma=args.sigma, effectiveAttackers=0, proximalMu=1.,
                start='initialization', machineGpuPolicy=machine.POLICY,
                activations=len(views) * args.nodes * args.rounds,
                optimizerSteps=len(views) * args.nodes * args.rounds * args.local_steps,
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
            with (args.output / 'noise_only_supervisor.log').open('a') as log:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    *[v for v in argv if v != '--background']], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(f'Noise-only supervisor PID {process.pid}. Log: {args.output}/noise_only_supervisor.log')
            return 0
        with (args.output / 'preparation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = driver.prepare(args)
            if not args.resume:
                plan.update(executionOptimization=study.runtime_policy(), machineGpuPolicy=machine.POLICY)
                driver.write_json(args.output / 'plan.json', plan)
            machine.verified(args.output)
        print(json.dumps(driver.readiness(args.output, plan, args.max_workers_per_gpu), indent=2), flush=True)
        if args.ntfy_topic:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/share_cart_monitor.py'),
                '--output', str(args.output), '--background', '--restart-publisher',
                '--interval', '1', '--ntfy-topic', args.ntfy_topic], cwd=ROOT, check=False)
            if result.returncode:
                print('Browser publisher startup failed; training continues.', flush=True)
        return execute(args, plan)


if __name__ == '__main__':
    main()
