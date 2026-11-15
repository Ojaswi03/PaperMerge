#!/usr/bin/env python3
"""Audited recovery of one themachine noise-only CUDA timeout, without retraining completed runs."""
import argparse
from contextlib import ExitStack
import copy
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core import fedsharp_noise_only as study, noise_only_machine as machine
from basil_core import live_gpu_scheduler as live
from basil_core.cart_campaign_io import read_json, write_json
from basil_core.cart_worker_admission import admission
from basil_core.execution_profile import ExecutionProfile
from scripts.check_fedsharp_checkpoint_integrity import digest, inspect


def measurements(root, views):
    """Hash completed public measurements, without rereading all seventeen large models."""
    result = {}
    for view in views:
        for name in ('run.json', 'metrics.npz', 'mechanisms.json', 'optimization.json', 'round_timings.jsonl'):
            path = root / view['runId'] / name
            result[str(path.relative_to(root))] = digest(path)
    return result


def validate_checkpoint(root, plan, view, row):
    import numpy as np
    from basil_core.sharp_protocol import load_state, vector_hash
    path = root / view['runId'] / 'checkpoint.npz'
    before = digest(path)
    archive_check = inspect(path)
    if not archive_check['ok']:
        raise ValueError('Checkpoint CRC/integrity check failed: ' + json.dumps(archive_check))
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata'].tobytes())
        dimension = archive['nodeParams'].shape[1]
    contract = metadata['contract']
    cfg = study.effective_config(view)
    if (contract != row.get('contract') or contract['source'] != plan['source']
            or contract['config'] != cfg or contract.get('comparisonCase') != view['comparisonCase']
            or contract.get('methodRevision') != plan['revision']
            or contract.get('data') != plan['data'][view['dataset']]
            or contract.get('partition') != plan['partitions'][view['runId']]):
        raise ValueError('Checkpoint source/config/data/runtime contract differs from the failed run.')
    policy = plan['executionOptimization']
    optimized = metadata.get('executionOptimization', {})
    if any(optimized.get(k) != policy[k] for k in ('revision', 'source', 'transitionId')):
        raise ValueError('Checkpoint execution optimization differs from the frozen plan.')
    state = load_state(path, contract, dimension)
    cursor, nodes = state['cursor'], cfg['nodes']
    if not nodes <= cursor < nodes * cfg['rounds']:
        raise ValueError('An incomplete checkpoint with committed client updates is required.')
    if not math.isfinite(state['elapsedSeconds']) or state['elapsedSeconds'] < 0:
        raise ValueError('Invalid retained elapsed time.')
    for index, event in enumerate(state['telemetry']):
        if ((event['round'], event['node'], event['activation']) != (index // nodes, index % nodes, index + 1)
                or event['training']['optimizerSteps'] != cfg['localSteps']
                or event['attackActive'] or event['configuredByzantine']
                or event['trainedHash'] != event['outboundHash']):
            raise ValueError('Committed activation history or zero-attack evidence differs.')
    for node in range(nodes):
        latest = cursor - 1 - ((cursor - 1 - node) % nodes)
        event = state['telemetry'][latest]
        if (vector_hash(state['nodeParams'][node]) != event['trainedHash']
                or vector_hash(state['outbound'][node]) != event['outboundHash']):
            raise ValueError('Retained client model differs from its last committed activation.')
    for index, metrics in enumerate(state['roundMetrics']):
        accuracy = np.asarray(metrics['nodeAccuracy'])
        if (metrics['round'] != index or accuracy.shape != (nodes,)
                or not np.isfinite(accuracy).all() or np.any((accuracy < 0) | (accuracy > 1))
                or abs(float(accuracy.mean()) - metrics['meanAccuracy']) > 1e-12):
            raise ValueError('Retained validation metrics differ from committed rounds.')
    if digest(path) != before:
        raise ValueError('Checkpoint changed during verification.')
    return state, before


def snapshot(root, run):
    if socket.gethostname().split('.')[0] != 'themachine':
        raise ValueError('This recovery is restricted to the original themachine server.')
    plan, hardware, _, plan_sha = machine.verified(root)
    expected = study.selected(SimpleNamespace(datasets=['mnist', 'cifar10'], partitions=['iid', 'dirichlet'],
        seeds=[2025, 2026, 2027], families=['sharp'], environments=['noise_only'], nodes=100, rounds=500,
        local_steps=5, alpha=.5, sigma=.01, attack_start=20, attack_type='hidden', gaussian_std=1.))
    if plan['configs'] != expected:
        raise ValueError('Expected the original eighteen-case noise-only matrix.')
    views = {view['runId']: view for view in plan['configs']}
    if run not in views:
        raise ValueError('Requested run is absent from the frozen plan.')
    campaign = read_json(root / 'campaign.json')
    if (campaign['status'] != 'COMPLETED_WITH_FAILURES' or campaign.get('activeSlots')
            or campaign.get('activeRunIds')):
        raise ValueError('This recovery requires a finished queue with no active workers.')
    if live.discover(root, len(views)):
        raise ValueError('An experiment worker is still alive; no changes applied.')
    row = read_json(root / run / 'run.json')
    if (row.get('status') != 'FAILED' or row.get('workerExitCode') != 1
            or 'CUDA_ERROR_LAUNCH_TIMEOUT' not in row.get('error', '')):
        raise ValueError('Only the reported CUDA launch-timeout failure is supported.')
    others = [view for view in plan['configs'] if view['runId'] != run]
    for view in others:
        saved = read_json(root / view['runId'] / 'run.json')
        if (saved.get('status') != 'COMPLETED' or saved.get('completedActivations') != 50000
                or saved.get('config') != view or saved.get('source') != plan['source']):
            raise ValueError('All seventeen other configurations must remain completed.')
    profile = ExecutionProfile(**row['contract']['runtime']['profile']).validate()
    if profile.gpu not in hardware or profile != hardware[profile.gpu]:
        raise ValueError('Retained GPU/runtime profile differs from the verified plan.')
    state, checkpoint_sha = validate_checkpoint(root, plan, views[run], row)
    return dict(plan=plan, planSha256=plan_sha, row=row, state=state, profile=profile,
        checkpointSha256=checkpoint_sha, completedMeasurements=measurements(root, others))


def recover(output, run, execute=False):
    root = Path(output).resolve()
    with machine.installed(), machine.scheduler(root, read_json(root / 'plan.json')), ExitStack() as locks:
        if execute:
            # These locks fence compatible controllers. Never stop a worker or
            # bypass a lock held by another recovery/supervisor.
            for name in ('live_scheduler.lock', 'supervisor.lock'):
                handle = locks.enter_context((root / name).open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = snapshot(root, run)
        state, profile, plan = checked['state'], checked['profile'], checked['plan']
        result = dict(runId=run, checkpointVerified=True, checkpointSha256=checked['checkpointSha256'],
            resumeActivations=state['cursor'], remainingActivations=50000-state['cursor'],
            replayActivations=max(0, checked['row'].get('completedActivations', state['cursor'])-state['cursor']),
            gpu=profile.gpu, completedRunsPreserved=17, workerSignalsSent=False,
            execute=execute, gpuHealthVerified=False)
        if not execute:
            return result
        from scripts.run_cart_v2_scaled import available_ram_mib
        inventory, processes = live.gpu_inventory(), live.gpu_processes()
        decision = admission(profile, [], inventory, processes, available_ram_mib(),
            context_mib=plan['resourceBudget']['context_overhead_mib'],
            ram_headroom_mib=plan['resourceBudget']['ram_headroom_mib'], maximum=plan['maximumWorkersPerGpu'])
        if not decision['admit']:
            raise RuntimeError('Waiting for original GPU/RAM headroom: ' + json.dumps(decision))
        leases = {}
        if not live.lease_available(profile, leases):
            raise RuntimeError('Original GPU lease is still held; no changes applied.')
        locks.enter_context(leases[profile.gpu])
        folder = root / run
        quarantine = root / 'operational_adapters/gpu-quarantine.json'
        control_paths = [root/'plan.json', root/'campaign.json', folder/'run.json', quarantine,
                         root/f'revision_gpu_{profile.gpu}.json']
        before = {p: p.read_bytes() if p.exists() else None for p in control_paths}
        audit = root / 'operational_adapters/noise_timeout_retries' / run / str(time.time_ns())
        audit.mkdir(parents=True)
        for path in control_paths:
            if before[path] is not None:
                (audit / path.name).write_bytes(before[path])
        os.link(folder/'checkpoint.npz', audit/'checkpoint.npz')
        for name in ('mechanisms.json', 'optimization.json', 'round_timings.jsonl'):
            if (folder/name).exists():
                shutil.copyfile(folder/name, audit/name)
        if (root/(run+'.log')).exists():
            shutil.copyfile(root/(run+'.log'), audit/'failed-worker.log')
        result.update(status='CHECKING_GPU', audit=str(audit), source=plan['source'],
                      recoverySourceSha256=digest(Path(__file__)))
        shutil.copyfile(Path(__file__), audit/Path(__file__).name)
        write_json(audit/'receipt.json', result)
        health = audit / 'gpu_check'
        health.mkdir()
        (health/'plan.json').write_bytes(before[root/'plan.json'])
        print(f'Checkpoint verified at activation {state["cursor"]}. Checking GPU {profile.gpu}; log: {audit}/gpu-check.log', flush=True)
        try:
            with (audit/'gpu-check.log').open('a') as log:
                subprocess.run([sys.executable, '-u', str(machine.ENTRYPOINT), '--role', 'gate',
                    '--output', str(health), '--gpu', str(profile.gpu), '--lease-fd', str(leases[profile.gpu].fileno())],
                    cwd=ROOT, env=machine.live_worker_environment(profile), pass_fds=(leases[profile.gpu].fileno(),),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            if not machine.valid_gate(health, profile.gpu, profile, plan, checked['planSha256']):
                raise ValueError('Fresh GPU check did not produce the required matching receipt.')
            gate = read_json(health/f'revision_gpu_{profile.gpu}.json')
            if gate['runtime'] != checked['row']['contract']['runtime']:
                raise ValueError('Fresh GPU runtime differs from the checkpoint; migration is refused.')
            study.check_source(plan['source'])
            if (any((p.read_bytes() if p.exists() else None) != value for p, value in before.items())
                    or digest(folder/'checkpoint.npz') != checked['checkpointSha256']
                    or measurements(root, [v for v in plan['configs'] if v['runId'] != run]) != checked['completedMeasurements']):
                raise ValueError('Campaign/checkpoint/completed measurements changed during the GPU check.')
        except BaseException as error:
            result.update(status='GPU_CHECK_OR_INPUT_VALIDATION_FAILED', error=f'{type(error).__name__}: {error}')
            write_json(audit/'receipt.json', result)
            raise
        # Only the device that passed this fresh check may leave quarantine.
        quarantined = read_json(quarantine) if quarantine.exists() else {}
        removed = quarantined.pop(str(profile.gpu), None)
        restored = copy.deepcopy(checked['row'])
        restored.update(status='INTERRUPTED', phase='VERIFIED_TIMEOUT_CHECKPOINT_RETRY',
            completedActivations=state['cursor'], completedRounds=len(state['roundMetrics']),
            elapsedSeconds=state['elapsedSeconds'])
        restored.pop('error', None)
        restored.pop('workerExitCode', None)
        result.update(status='PREPARED', gpuHealthVerified=True, clearedQuarantine=removed,
                      completedMeasurementHashes=checked['completedMeasurements'])
        write_json(audit/'receipt.json', result)
        write_json(root/f'revision_gpu_{profile.gpu}.json', gate)
        if quarantine.exists():
            write_json(quarantine, quarantined)
        write_json(folder/'run.json', restored)
        result['status'] = 'REQUEUED'
        write_json(audit/'receipt.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--background', action='store_true', help='After requeue, start the original supervisor in the background.')
    parser.add_argument('--ntfy-topic', default='papermerge-ojaswi')
    args = parser.parse_args()
    if args.background and not args.execute:
        parser.error('--background requires --execute.')
    result = recover(args.output, args.run_id, args.execute)
    print(json.dumps(result, indent=2), flush=True)
    if args.background:
        subprocess.run([sys.executable, str(ROOT/'scripts/run_fedsharp_noise_only_auto.py'),
            '--output', str(args.output.resolve()), '--resume', '--execute', '--background',
            '--wait-for-resources', '--ntfy-topic', args.ntfy_topic], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
