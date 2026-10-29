#!/usr/bin/env python3
"""Local matched ACDS selection diagnostics, separate from active HPC runs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def diagnostic_source():
    names = ('scripts/diagnose_acds_pool.py', 'basil_core/sharp_checked_pool.py')
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}


def cases(args):
    from basil_core.cart_comparison_study import selected
    from types import SimpleNamespace
    _, views = selected(SimpleNamespace(noise_levels=1, noise_values=[.01], share_fraction=.05,
        acds_batches=5, arms=['acds'], environments=['joint'], datasets=['cifar10'],
        partitions=[args.partition], seeds=[2025], nodes=100, rounds=args.rounds, local_steps=5,
        model='paper', alpha=.5, attack_start=args.attack_start, train_per_class=None, validation_per_class=None))
    return [dict(mode=mode, view=views[0]) for mode in args.modes]


def worker(args):
    from scripts.cart_v2_slot_worker import adopt_lease
    adopt_lease(args.lease_fd)
    from scripts.run_cart_v2_study import setup_gpu, runtime
    tf = setup_gpu()
    import numpy as np
    from basil_core.cart_campaign_io import read_json, write_json
    from basil_core.cart_comparison_study import source, configure_worker, partitions_for, intervention
    from basil_core.sharp_data import training_data, digest
    from basil_core.sharp_protocol import save_state, load_state, vector_hash
    from basil_core.sharp_checked_pool import selector
    import basil_core.sharp_cart_v2_protocol as protocol
    plan = read_json(args.output / 'plan.json')
    if plan['producerSource'] != source() or plan['diagnosticSource'] != diagnostic_source():
        raise ValueError('Diagnostic or producer source changed.')
    case = plan['cases'][args.worker]
    view, mode = case['view'], case['mode']
    folder = args.output / mode
    folder.mkdir(exist_ok=True)
    model = configure_worker(view, 'GPU')
    initial = vector_hash(model.export())
    x, y = training_data('cifar10')
    chunks, held, audit = partitions_for(view, y)
    contract = dict(case=case, producerSource=plan['producerSource'], diagnosticSource=diagnostic_source(),
        runtime=runtime(tf), initialHash=initial, partition=audit,
        data=dict(imagesSha256=digest(x), labelsSha256=digest(y)))
    checkpoint = folder / 'checkpoint.npz'
    state = load_state(checkpoint, contract, model.dimension) if checkpoint.exists() else None
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    summaries = []

    def round_done(metric, current):
        records = current['telemetry'][-100:]
        checks = [r['selection']['poolQuality'] for r in records if r['selection']['poolQuality']['checked']]
        multiple = [r for r in checks if r['originalSelectedCount'] > 1]
        summary = dict(round=metric['round'] + 1, meanAccuracy=metric['meanAccuracy'],
            worstAccuracy=metric['worstNodeAccuracy'],
            classRecall=np.asarray(metric['perClass']).mean(axis=0).tolist(),
            meanSelectedCount=float(np.mean([r['selection']['selectedCount'] for r in records])),
            meanOriginalSelectedCount=float(np.mean([r['originalSelectedCount'] for r in checks])) if checks else None,
            poolComparisons=len(multiple), worsePoolFraction=float(np.mean([r['poolExcessCE'] > 1e-12 for r in multiple])) if multiple else None,
            meanPoolExcessCE=float(np.mean([r['poolExcessCE'] for r in multiple])) if multiple else None,
            poolFallbacks=sum(r['fallback'] for r in checks),
            trainingCE=float(np.mean([r['training']['localTrainingLoss'] for r in records])),
            attackActivations=sum(r['attackActive'] for r in records),
            elapsedSeconds=current['elapsedSeconds'])
        summaries.append(summary)
        write_json(folder / 'progress.json', dict(status='RUNNING', mode=mode, rounds=summaries))
        print(f'{mode} R{summary["round"]}: acc={100*metric["meanAccuracy"]:.3f}%, '
              f'q={summary["meanSelectedCount"]:.2f}, pool excess CE={summary["meanPoolExcessCE"]}, '
              f'fallbacks={summary["poolFallbacks"]}', flush=True)

    with intervention(view, model) as cfg:
        original = protocol.select_input
        protocol.select_input = selector(original, mode)
        try:
            result = protocol.run(cfg, model, x, y, chunks, x[held], y[held], state=state,
                checkpoint=lambda s: save_state(checkpoint, s, contract),
                on_round=round_done, should_stop=lambda: stopped)
        finally:
            protocol.select_input = original
    save_state(checkpoint, result, contract)
    assert all(r['training']['optimizerSteps'] == 5 for r in result['telemetry'])
    np.savez_compressed(folder / 'metrics.npz',
        roundNodeAccuracy=np.asarray([r['nodeAccuracy'] for r in result['roundMetrics']]),
        roundNodePerClassAccuracy=np.asarray([r['perClass'] for r in result['roundMetrics']]))
    write_json(folder / 'result.json', dict(status='COMPLETED', mode=mode, contract=contract,
        rounds=summaries, finalAccuracy=result['roundMetrics'][-1]['meanAccuracy'],
        optimizerSteps=sum(r['training']['optimizerSteps'] for r in result['telemetry']),
        interpretation='Matched local selection diagnostic. ACDS data sharing is unchanged; selection variants are not the original baseline.'))


def execute(args):
    from basil_core.execution_profile import ExecutionProfile, gpu_inventory
    from scripts.run_cart_v2_scaled import acquire_leases, available_ram_mib
    from scripts.run_cart_v2_study import worker_environment
    from basil_core.cart_comparison_study import source
    from basil_core.cart_campaign_io import read_json, write_json
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = gpu_inventory()
    profile = ExecutionProfile(name='hpc', gpu=0, memory_mib=4096, headroom_mib=1024,
        ram_mib=8192, output_root=str(args.output)).bind(inventory)
    if inventory[0]['memory.free'] < 6144 or available_ram_mib() < 10240:
        raise RuntimeError('Insufficient local GPU/RAM headroom.')
    plan = dict(cases=cases(args), profile=asdict(profile), producerSource=source(),
                diagnosticSource=diagnostic_source())
    if (args.output / 'plan.json').exists() and read_json(args.output / 'plan.json') != plan:
        raise ValueError('Existing diagnostic plan differs; preserve it and use another output.')
    write_json(args.output / 'plan.json', plan)
    for name in {**plan['producerSource'], **plan['diagnosticSource']}:
        target = args.output / 'source_snapshots' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    leases = acquire_leases([profile])
    try:
        for index, case in enumerate(plan['cases']):
            path = args.output / case['mode'] / 'result.json'
            if path.exists() and read_json(path)['status'] == 'COMPLETED':
                continue
            fd = leases[0].fileno()
            with (args.output / f'{case["mode"]}.log').open('a') as log:
                command = [sys.executable, str(Path(__file__).resolve()), '--output', str(args.output),
                           '--worker', str(index), '--lease-fd', str(fd)]
                process = subprocess.Popen(command, cwd=ROOT, env=worker_environment(profile), pass_fds=(fd,),
                    stdout=log, stderr=subprocess.STDOUT)
                try:
                    if process.wait():
                        raise RuntimeError(f'Diagnostic {case["mode"]} failed; see {case["mode"]}.log')
                finally:
                    if process.poll() is None:
                        process.terminate(); process.wait()
            print(f'Completed {case["mode"]}', flush=True)
        results = [read_json(args.output / c['mode'] / 'result.json') for c in plan['cases']]
        write_json(args.output / 'summary.json', dict(status='COMPLETED', results=results))
    finally:
        for lease in leases.values():
            lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--attack-start', type=int, default=20)
    parser.add_argument('--partition', choices=('iid', 'dirichlet'), default='dirichlet')
    parser.add_argument('--modes', nargs='+', choices=('raw', 'checked', 'single'), default=['raw', 'checked', 'single'])
    parser.add_argument('--worker', type=int)
    parser.add_argument('--lease-fd', type=int)
    args = parser.parse_args(); args.output = args.output.resolve()
    if args.rounds < 1 or args.attack_start < 0 or len(args.modes) != len(set(args.modes)):
        parser.error('Positive rounds, nonnegative attack start and distinct modes required.')
    if args.worker is not None:
        worker(args)
    elif args.execute:
        execute(args)
    else:
        print(json.dumps(dict(cases=cases(args), trainingStarted=False), indent=2))


if __name__ == '__main__':
    main()
