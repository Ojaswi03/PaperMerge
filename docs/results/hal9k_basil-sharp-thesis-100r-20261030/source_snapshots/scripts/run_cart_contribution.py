#!/usr/bin/env python3
"""Prepare, verify and adaptively run matched CART/ACDS contribution cases."""
import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core import cart_contribution_study as study
from basil_core.cart_campaign_io import read_json, write_json
from basil_core.execution_profile import ExecutionProfile, gpu_inventory
from basil_core.cart_worker_admission import placement
from scripts import run_cart_adaptive as adaptive


def assignment(configs, hardware):
    """Balance total work and spread each method across physical devices."""
    total, per_mode = Counter(), Counter()
    limit = math.ceil(len(configs) / len(hardware))
    result = []
    for cfg in configs:
        mode = study.variant(cfg)
        p = min((p for p in hardware if total[p.gpu] < limit),
                key=lambda p: (per_mode[mode, p.gpu], total[p.gpu], p.gpu))
        result.append(asdict(p))
        total[p.gpu] += 1
        per_mode[mode, p.gpu] += 1
    return result


def prepare(args):
    root = args.output
    if args.resume:
        plan = read_json(root / 'plan.json')
        study.check_source(plan['source'])
        hardware, _ = placement(plan)
        for p in hardware.values():
            if Path(p.output_root).resolve() != root:
                raise ValueError('Resume requires original output path.')
            p.bind(gpu_inventory())
        if plan['host'] != socket.gethostname():
            raise ValueError('Resume requires original host and GPU UUIDs.')
        return plan
    if (root / 'plan.json').exists():
        raise FileExistsError('Preserve existing plan; use --resume or a new output.')
    configs = study.selected(args)
    inventory = gpu_inventory()
    ids = [int(v) for v in args.gpus.split(',')] if args.gpus else list(range(args.gpu_count))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('GPU selection must contain distinct physical IDs.')
    hardware = [ExecutionProfile(name='hpc', gpu=g, memory_mib=args.gpu_memory_mib,
        headroom_mib=8192, ram_mib=16384, output_root=str(root)).bind(inventory) for g in ids]
    from basil_core.cart_comparison_study import data_contract
    data, partitions = data_contract(configs)
    plan = dict(revision=study.REVISION, variant='contribution', configs=configs,
        source=study.source(), data=data, partitions=partitions, profiles=assignment(configs, hardware),
        jobsPerGpu=1, host=socket.gethostname(),
        resourceBudget=dict(context_overhead_mib=1024, ram_headroom_mib=32768,
            ram_mib_per_worker=16384, allow_unattributed_process_memory=False))
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'plan.json', plan)
    return plan


def readiness(root, plan):
    hardware, assigned = placement(plan)
    inventory = gpu_inventory()
    counts = Counter(p.gpu for p in assigned)
    capacity = {g: min(counts[g], max(0, (next(r['memory.free'] for r in inventory
        if r['index'] == g) - p.headroom_mib) // (p.memory_mib + 1024))) for g, p in hardware.items()}
    return dict(configurations=len(assigned), physicalGpus=list(hardware), casesPerGpu=dict(counts),
        capacityNow=capacity, hostRequiredForAllMiB=32768 + len(assigned) * assigned[0].ram_mib,
        hostAvailableMiB=adaptive.scaled.available_ram_mib(), gpuWorkerCapMiB=assigned[0].memory_mib,
        startupAndContextReserved=True, trainingStarted=False, output=str(root))


def execute(args, plan):
    root = args.output
    hardware, _ = placement(plan)
    with adaptive.acquire_supervisor(root, False, 180):
        adaptive.ensure_source_snapshots(root, plan)
        leases = None
        while leases is None:
            try:
                # Verification also needs free memory; never evict an existing process.
                for g, p in hardware.items():
                    item = next(r for r in gpu_inventory() if r['index'] == g)
                    if item['memory.free'] < p.memory_mib + p.headroom_mib + 1024:
                        raise RuntimeError(f'GPU {g}: waiting for verification memory.')
                if adaptive.scaled.available_ram_mib() < 32768 + 16384:
                    raise RuntimeError('Waiting for host RAM.')
                leases = adaptive.scaled.acquire_leases(list(hardware.values()))
            except RuntimeError as error:
                if not args.wait_for_resources:
                    raise
                write_json(root / 'campaign.json', dict(status='WAITING_FOR_RESOURCES',
                    total=len(plan['configs']), completed=0, failed=0, pending=len(plan['configs']),
                    host=socket.gethostname(), ownerPid=os.getpid(), reason=str(error)))
                print(str(error), flush=True)
                time.sleep(5)
        try:
            for g, p in hardware.items():
                receipt = root / f'revision_gpu_{g}.json'
                if not receipt.exists() or read_json(receipt).get('interventionGate') is not True:
                    print(f'Verifying physical GPU {g} under its saved memory cap.', flush=True)
                    fd = leases[g].fileno()
                    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_cart_contribution_gpu.py'),
                        '--output', str(root), '--gpu', str(g), '--lease-fd', str(fd)],
                        cwd=ROOT, env=adaptive.scaled.core.worker_environment(p), pass_fds=(fd,), check=True)
            checked, hardware, assigned = adaptive.validate_plan(root)
            return adaptive.schedule(args, checked, hardware, assigned,
                                     adaptive.sha(root / 'plan.json'), leases)
        finally:
            for handle in leases.values():
                handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preset', choices=('comparison', 'contribution', 'screen'), default='contribution')
    p.add_argument('--gpu-count', type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument('--gpus', help='Explicit physical IDs, e.g. 0,2; overrides --gpu-count.')
    p.add_argument('--gpu-memory-mib', type=int, choices=(4096, 6144, 8192), default=4096)
    p.add_argument('--variants', nargs='+', choices=tuple(study.VARIANTS))
    p.add_argument('--datasets', nargs='+', choices=('mnist', 'cifar10'), default=['mnist', 'cifar10'])
    p.add_argument('--partitions', nargs='+', choices=('iid', 'dirichlet'), default=['iid', 'dirichlet'])
    p.add_argument('--seeds', nargs='+', type=int, choices=(2025, 2026, 2027))
    p.add_argument('--rounds', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--describe', action='store_true', help='Print cases without data/GPU access or writing a plan.')
    p.add_argument('--wait-for-resources', action='store_true')
    p.add_argument('--tmux')
    p.add_argument('--max-workers-per-gpu', type=int, choices=range(1, 25), default=24)
    p.add_argument('--poll-seconds', type=int, default=3, choices=range(1, 31))
    p.add_argument('--ntfy-topic')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.ntfy_topic:
        from basil_core.ntfy_notifications import validate_destination
        validate_destination(args.ntfy_topic, 'https://ntfy.sh')
    if args.resume:
        supplied = {a.split('=')[0] for a in sys.argv[1:] if a.startswith('--')}
        allowed = {'--resume', '--execute', '--output', '--tmux', '--wait-for-resources', '--ntfy-topic'}
        if supplied - allowed:
            p.error('Resume uses saved scientific and resource settings; do not override them.')
    if args.rounds is None:
        args.rounds = 10 if args.preset == 'screen' else 100
    if args.rounds < 1:
        p.error('Positive round count required.')
    args.seeds = args.seeds or ([2025] if args.preset == 'screen' else [2025, 2026, 2027])
    # Match the completed full comparison, apart from explicitly declared variants.
    args.noise_levels, args.noise_values = 1, [.01]
    args.share_fraction, args.acds_batches = .05, 5
    args.environments, args.nodes, args.local_steps = ['joint'], 100, 5
    args.model, args.alpha = 'paper', .5
    args.attack_start = 3 if args.preset == 'screen' else 20
    args.train_per_class = args.validation_per_class = None
    if args.describe:
        configs = study.selected(args)
        print(json.dumps(dict(configurations=len(configs), rounds=args.rounds, nodes=100,
            methods=sorted({study.variant(c) for c in configs}), trainingStarted=False), indent=2))
        return 0
    if args.tmux:
        if not args.execute:
            p.error('--tmux requires --execute.')
        forwarded = sys.argv[1:]
        pos = forwarded.index('--tmux')
        forwarded = forwarded[:pos] + forwarded[pos + 2:]
        args.output.mkdir(parents=True, exist_ok=True)
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), *forwarded])
        command += ' >> ' + shlex.quote(str(args.output / 'adaptive_supervisor.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', args.tmux, 'bash', '-c', command], check=True)
        print(f'Started {args.tmux}. Log: {args.output}/adaptive_supervisor.log')
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'preparation.lock').open('a') as operation:
        fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args)
    print(json.dumps(readiness(args.output, plan), indent=2), flush=True)
    return execute(args, plan) if args.execute else 0


if __name__ == '__main__':
    raise SystemExit(main())
