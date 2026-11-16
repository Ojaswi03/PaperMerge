"""Operational placement for the unchanged eighteen noise-only experiments."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

from basil_core import fedsharp_noise_only as study, live_gpu_scheduler as live
from basil_core.cart_campaign_io import read_json, write_json
from basil_core.cart_worker_admission import admission, placement

ROOT = study.ROOT
ORIGINAL_SOURCE = study.source
ORIGINAL_CHOOSE = live.choose
ENTRYPOINT = ROOT / 'scripts/run_fedsharp_noise_only_themachine.py'
POLICY = dict(host='themachine', preferred=[1, 3], opportunistic=[0, 2],
              opportunisticMaximumUtilization=35, opportunisticQuietSeconds=30)


def source():
    names = ('basil_core/noise_only_machine.py', 'scripts/run_fedsharp_noise_only_themachine.py',
             'configs/research/themachine_fedsharp_noise_only_500r.json')
    return {**ORIGINAL_SOURCE(), **{n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}}


@contextmanager
def installed():
    with patch.object(study, 'source', source), study.installed():
        yield


def valid_gate(root, gpu, profile, plan, digest):
    path = root / f'revision_gpu_{gpu}.json'
    if not path.exists():
        return False
    row = read_json(path)
    return (row.get('status') == 'PASSED' and row.get('noiseOnly') is True
        and row.get('actualAttackers') == 0 and row.get('emptyAttackerSetsVerified') is True
        and row.get('source') == plan['source'] and row.get('planSha256') == digest
        and row.get('runtime', {}).get('host') == socket.gethostname()
        and row.get('runtime', {}).get('profile') == asdict(profile))


def verified(root):
    plan = read_json(root / 'plan.json')
    if (plan['host'] != socket.gethostname() or socket.gethostname().split('.')[0] != POLICY['host']
            or plan['revision'] != study.REVISION or plan.get('machineGpuPolicy') != POLICY
            or plan.get('executionOptimization') != study.runtime_policy()):
        raise ValueError('Expected the frozen themachine noise-only plan and scheduling policy.')
    study.check_source(plan['source'])
    hardware, assigned = placement(plan)
    if set(hardware) != {0, 1, 2, 3}:
        raise ValueError('This policy requires physical GPUs 0, 1, 2 and 3.')
    digest = live.sha(root / 'plan.json')
    for view in plan['configs']:
        path = root / view['runId'] / 'run.json'
        if path.exists():
            row = read_json(path)
            if row.get('contract'):
                gpu = row['contract']['runtime']['profile']['gpu']
                if gpu not in hardware or not valid_gate(root, gpu, hardware[gpu], plan, digest):
                    raise ValueError('A retained worker requires its original passed GPU gate.')
    return plan, hardware, assigned, digest


class DeviceAdmission:
    """Gate a device when it can join; never wait for all four devices at startup.

    Quiet utilization is an operational courtesy for the optional devices, not
    a scientific hyperparameter or a guarantee against other users' allocations.
    Device verification is synchronous, but existing training workers continue.
    """
    def __init__(self, root, plan):
        self.root, self.plan = Path(root), plan
        self.digest = live.sha(self.root / 'plan.json')
        self.quiet_since, self.failed, self.ready = {}, {}, set()

    def choose(self, pending, configs, allowed, hardware, residents, inventory, processes,
               available_ram, budget, maximum, quarantined, leases):
        now = time.monotonic()
        by_gpu = {r['index']: r for r in inventory}
        eligible, blocked = {}, {}
        counts = Counter(r.gpu for r in residents)
        order = POLICY['preferred'] + POLICY['opportunistic']
        for gpu in order:
            if gpu not in hardware or gpu in quarantined or gpu not in leases or gpu not in by_gpu:
                continue
            profile = hardware[gpu]
            if gpu in self.failed:
                blocked[gpu] = dict(admit=False, reason='device verification failed; inspect GPU gate log')
                continue
            if gpu in POLICY['opportunistic']:
                utilization = by_gpu[gpu].get('utilization.gpu')
                if utilization is None or utilization > POLICY['opportunisticMaximumUtilization']:
                    self.quiet_since.pop(gpu, None)
                    blocked[gpu] = dict(admit=False, reason='waiting for low GPU utilization', utilization=utilization)
                    continue
                elapsed = now - self.quiet_since.setdefault(gpu, now)
                if elapsed < POLICY['opportunisticQuietSeconds']:
                    blocked[gpu] = dict(admit=False, reason='waiting for quiet GPU samples', quietSeconds=elapsed)
                    continue
            decision = admission(profile, residents, inventory, processes, available_ram,
                context_mib=budget['context_overhead_mib'], ram_headroom_mib=budget['ram_headroom_mib'],
                maximum=maximum)
            if not decision['admit']:
                blocked[gpu] = decision
                continue
            if any(gpu in allowed[i] for i in pending):
                eligible[gpu] = profile
        candidates = sorted(eligible, key=lambda g: (counts[g], order.index(g)))
        for gpu in candidates:
            profile = eligible[gpu]
            if gpu in self.ready:
                continue
            if not valid_gate(self.root, gpu, profile, self.plan, self.digest):
                print(f'GPU {gpu}: verifying this device before admitting training.', flush=True)
                try:
                    with (self.root / f'gpu_{gpu}_gate.log').open('a') as log:
                        subprocess.run([sys.executable, '-u', str(ENTRYPOINT), '--role', 'gate',
                            '--output', str(self.root), '--gpu', str(gpu), '--lease-fd', str(leases[gpu].fileno())],
                            cwd=ROOT, env=live_worker_environment(profile), pass_fds=(leases[gpu].fileno(),),
                            stdout=log, stderr=subprocess.STDOUT, check=True)
                    if not valid_gate(self.root, gpu, profile, self.plan, self.digest):
                        raise ValueError('Device gate did not produce its required receipt.')
                except (subprocess.CalledProcessError, ValueError) as error:
                    self.failed[gpu] = type(error).__name__
                    write_json(self.root / 'operational_adapters/machine-gate-failures.json', self.failed)
                    blocked[gpu] = dict(admit=False, reason='device verification failed; inspect GPU gate log')
                    continue
                # Verification can take time. Refresh every admission input before
                # permitting a worker; never spend the old free-memory sample.
                from scripts.run_cart_v2_scaled import available_ram_mib
                inventory, processes, available_ram = live.gpu_inventory(), live.gpu_processes(), available_ram_mib()
                by_gpu = {r['index']: r for r in inventory}
            self.ready.add(gpu)
            # Only one new device gate per choose call. The shared loop can launch
            # its first worker before checking another device.
            break
        ready = {g: p for g, p in eligible.items() if g in self.ready and g not in self.failed}
        # Refresh optional-device load after a potentially long gate as well.
        ready = {g: p for g, p in ready.items() if g in by_gpu and (g in POLICY['preferred'] or
            (by_gpu[g].get('utilization.gpu') is not None and
             by_gpu[g]['utilization.gpu'] <= POLICY['opportunisticMaximumUtilization']))}
        choice, decisions = ORIGINAL_CHOOSE(pending, configs, allowed, ready, residents, inventory, processes,
            available_ram, budget, maximum, quarantined, leases)
        decisions.update(blocked)
        return choice, decisions


def live_worker_environment(profile):
    from scripts.run_cart_v2_study import worker_environment
    return worker_environment(profile)


def completed_report(root, destination):
    """Summarize the new matrix without invoking the older joint-only plotter."""
    import numpy as np
    plan = read_json(root / 'plan.json')
    rows = []
    for view in plan['configs']:
        row = read_json(root / view['runId'] / 'run.json')
        if row['status'] != 'COMPLETED':
            raise ValueError('Cannot summarize an incomplete run.')
        with np.load(root / view['runId'] / 'metrics.npz', allow_pickle=False) as arrays:
            accuracy = arrays['roundNodeAccuracy']
            if accuracy.shape != (500, 100) or not np.isfinite(accuracy).all():
                raise ValueError('Expected complete finite 500-round metrics.')
            rows.append(dict(runId=view['runId'], dataset=view['dataset'], partition=view['partition'],
                seed=view['seed'], method='FedSharp+' if view['comparisonCase']['addon'] else 'FedSharp',
                finalRoundValidationMean=float(accuracy[-1].mean()),
                rounds491To500ValidationMean=float(accuracy[-10:].mean()),
                recordedProtocolSeconds=row['elapsedSeconds']))
    write_json(root / 'noise_only_training_summary.json', dict(source=plan['source'], runs=rows,
        officialTestUsed=False, scope='Development holdout; separate fresh inference verification is required.'))


@contextmanager
def scheduler(root, plan):
    from reporting import basil_sharp_thesis as reporting
    chooser = DeviceAdmission(root, plan)
    with patch.object(live, 'WORKER', ENTRYPOINT), patch.object(live, 'OWNERS', live.OWNERS | {ENTRYPOINT.name}), \
         patch.object(live, 'verified', verified), patch.object(live, 'choose', chooser.choose), \
         patch.object(reporting, 'generate', completed_report):
        yield
