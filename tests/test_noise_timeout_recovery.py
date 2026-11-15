import copy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from basil_core import fedsharp_noise_only as study, noise_only_machine as machine
from basil_core import fedsharp_runtime_optimization as optimization
from basil_core.cart_campaign_io import read_json, write_json
from basil_core.execution_profile import ExecutionProfile
from basil_core.sharp_protocol import initial_state, vector_hash
from scripts import retry_noise_only_timeout as recovery
from tests.test_fedsharp_noise_only import opts


@pytest.fixture
def failed(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    root.mkdir()
    (root/'operational_adapters').mkdir()
    for name in ('live_scheduler.lock', 'supervisor.lock'):
        (root/name).touch()
    monkeypatch.setattr(recovery.socket, 'gethostname', lambda: 'themachine')
    views = study.selected(opts())
    target = next(v for v in views if v['dataset'] == 'cifar10' and v['seed'] == 2025
                  and v['comparisonCase']['addon'] == 'cart')
    profiles = {g: ExecutionProfile(name='hpc', gpu=g, gpu_uuid=f'GPU-{g:08x}', memory_mib=3072,
        headroom_mib=8192, ram_mib=8192, output_root=str(root)) for g in range(4)}
    profile = profiles[3]
    source = machine.source()
    plan = dict(host='themachine', revision=study.REVISION, source=source, configs=views,
        profiles=[asdict(profiles[i % 4]) for i in range(18)], machineGpuPolicy=machine.POLICY,
        executionOptimization=study.runtime_policy(), maximumWorkersPerGpu=24,
        data={'mnist': {}, 'cifar10': {}}, partitions={v['runId']: {} for v in views},
        resourceBudget=dict(context_overhead_mib=1024, ram_headroom_mib=16384, ram_mib_per_worker=8192))
    write_json(root/'plan.json', plan)
    write_json(root/'campaign.json', dict(status='COMPLETED_WITH_FAILURES', completed=17, failed=1,
        host='themachine', activeSlots=[], activeRunIds=[], elapsedSeconds=500.))
    runtime = dict(host='themachine', profile=asdict(profile), device='/GPU:0',
        tensorflow='fixture', numpy=np.__version__, tf32=False)
    original_gate = dict(status='PASSED', source=source, planSha256=recovery.live.sha(root/'plan.json'),
        runtime=runtime, noiseOnly=True, actualAttackers=0, emptyAttackerSetsVerified=True)
    write_json(root/'revision_gpu_3.json', original_gate)
    quarantine = {'3': dict(reason='CUDA memory/device failure detected in a worker', at=1),
                  '0': dict(reason='unrelated device quarantine', at=2)}
    write_json(root/'operational_adapters/gpu-quarantine.json', quarantine)
    for view in views:
        folder = root/view['runId']
        folder.mkdir()
        write_json(folder/'run.json', dict(status='COMPLETED', config=view, source=source,
            completedActivations=50000, elapsedSeconds=500.))
        np.savez_compressed(folder/'metrics.npz', roundNodeAccuracy=np.full((500, 100), .5))
        for name in ('mechanisms.json', 'optimization.json', 'round_timings.jsonl'):
            (folder/name).write_text('{}\n')
    cfg = study.effective_config(target)
    contract = dict(config=cfg, comparisonCase=target['comparisonCase'], methodRevision=study.REVISION,
        source=source, runtime=runtime, data={}, partition={})
    # Synthetic state for recovery bookkeeping tests, not a trained classifier.
    state = initial_state(cfg, np.arange(7, dtype=np.float32))
    cursor = 142
    state.update(cursor=cursor, elapsedSeconds=100.,
        senderRounds=np.asarray([max(-1, (cursor-1-j)//100) for j in range(100)], np.int32),
        telemetry=[dict(round=i//100, node=i%100, activation=i+1,
            training=dict(optimizerSteps=5), attackActive=False, configuredByzantine=False,
            trainedHash=vector_hash(state['nodeParams'][i%100]),
            outboundHash=vector_hash(state['outbound'][i%100])) for i in range(cursor)],
        roundMetrics=[dict(round=0, nodeAccuracy=[.5]*100, meanAccuracy=.5)])
    folder = root/target['runId']
    optimization.CheckpointWriter(plan['executionOptimization'])(folder/'checkpoint.npz', state, contract)
    row = dict(status='FAILED', error='InternalError: CUDA_ERROR_LAUNCH_TIMEOUT: the launch timed out',
        workerExitCode=1, config=target, source=source, contract=contract, completedActivations=162,
        completedRounds=1, elapsedSeconds=120.)
    write_json(folder/'run.json', row)
    (root/(target['runId']+'.log')).write_text('original CUDA timeout traceback\n')
    monkeypatch.setattr(recovery.live, 'discover', lambda *a: {})
    inventory = [dict(index=g, uuid=p.gpu_uuid, **{'memory.free':90000, 'utilization.gpu':0})
                 for g, p in profiles.items()]
    monkeypatch.setattr(recovery.live, 'gpu_inventory', lambda: inventory)
    monkeypatch.setattr(recovery.live, 'gpu_processes', lambda: [])
    from scripts import run_cart_v2_scaled
    monkeypatch.setattr(run_cart_v2_scaled, 'available_ram_mib', lambda: 1000000)
    monkeypatch.setattr(machine, 'live_worker_environment', lambda *a: {})
    def lease(p, leases):
        leases[p.gpu] = (tmp_path/'gpu-lease').open('a')
        return True
    monkeypatch.setattr(recovery.live, 'lease_available', lease)
    calls = []
    def gate(command, **kwargs):
        calls.append(command)
        assert command[command.index('--role')+1] == 'gate'
        assert command[command.index('--gpu')+1] == '3' and kwargs['pass_fds']
        health = Path(command[command.index('--output')+1])
        assert (health/'plan.json').read_bytes() == (root/'plan.json').read_bytes()
        write_json(health/'revision_gpu_3.json', dict(original_gate, freshCheck=True))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(recovery.subprocess, 'run', gate)
    return root, target['runId'], state, calls


def bytes_under(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_dry_run_validates_retained_models_and_changes_nothing(failed):
    root, run, state, calls = failed
    before = bytes_under(root)
    result = recovery.recover(root, run)
    assert result['resumeActivations'] == 142 and result['replayActivations'] == 20
    assert result['remainingActivations'] == 49858 and result['checkpointVerified']
    assert not result['gpuHealthVerified'] and not calls
    assert bytes_under(root) == before


def test_requeue_preserves_17_results_checkpoint_and_failure_evidence(failed):
    root, run, state, calls = failed
    before = bytes_under(root)
    result = recovery.recover(root, run, True)
    assert result['status'] == 'REQUEUED' and result['gpuHealthVerified'] and len(calls) == 1
    assert result['gpu'] == 3 and not result['workerSignalsSent']
    row = read_json(root/run/'run.json')
    assert row['status'] == 'INTERRUPTED' and row['completedActivations'] == 142
    assert row['elapsedSeconds'] == 100 and row['completedRounds'] == 1
    assert 'error' not in row and 'workerExitCode' not in row
    audit = Path(result['audit'])
    assert (audit/'run.json').read_bytes() == before[run+'/run.json']
    assert (audit/'checkpoint.npz').read_bytes() == before[run+'/checkpoint.npz']
    assert (root/run/'checkpoint.npz').read_bytes() == before[run+'/checkpoint.npz']
    assert (audit/'failed-worker.log').read_bytes() == before[run+'.log']
    assert read_json(root/'revision_gpu_3.json')['freshCheck'] is True
    assert read_json(root/'operational_adapters/gpu-quarantine.json') == {
        '0': dict(reason='unrelated device quarantine', at=2)}
    for name, value in before.items():
        if any(name.startswith(view['runId']+'/') for view in read_json(root/'plan.json')['configs'] if view['runId'] != run):
            assert (root/name).read_bytes() == value
    with machine.installed(), machine.scheduler(root, read_json(root/'plan.json')):
        plan, hardware, _, _ = recovery.live.verified(root)
        assert recovery.live.eligible_profile(root, next(v for v in plan['configs'] if v['runId']==run), hardware) == {3}
        pending = [v['runId'] for v in plan['configs'] if read_json(root/v['runId']/'run.json')['status'] not in ('FAILED','COMPLETED')]
        assert pending == [run]
    with pytest.raises(ValueError, match='Only the reported'):
        recovery.recover(root, run, True)


def test_failed_gpu_check_keeps_target_and_quarantine_unchanged(failed, monkeypatch):
    root, run, state, calls = failed
    before = bytes_under(root)
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(recovery.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        recovery.recover(root, run, True)
    assert all((root/name).read_bytes() == value for name, value in before.items())
    assert any(root.glob('operational_adapters/noise_timeout_retries/*/*/receipt.json'))


@pytest.mark.parametrize('change', ['runtime', 'completed_metrics', 'target_row', 'quarantine'])
def test_changed_runtime_or_inputs_refuse_requeue(failed, monkeypatch, change):
    root, run, state, calls = failed
    before_row = (root/run/'run.json').read_bytes()
    gate = recovery.subprocess.run
    def alter(command, **kwargs):
        gate(command, **kwargs)
        if change == 'runtime':
            health = Path(command[command.index('--output')+1])
            path = health/'revision_gpu_3.json'
            row = read_json(path)
            row['runtime']['tensorflow'] = 'changed'
            write_json(path, row)
        elif change == 'completed_metrics':
            path = next(p for p in root.glob('*/metrics.npz') if p.parent.name != run)
            path.write_bytes(path.read_bytes()+b'changed')
        elif change == 'target_row':
            path = root/run/'run.json'
            path.write_bytes(path.read_bytes()+b' ')
        else:
            write_json(root/'operational_adapters/gpu-quarantine.json', {'3': dict(reason='changed')})
    monkeypatch.setattr(recovery.subprocess, 'run', alter)
    with pytest.raises(ValueError):
        recovery.recover(root, run, True)
    assert read_json(root/run/'run.json')['status'] == 'FAILED'
    if change != 'target_row':
        assert (root/run/'run.json').read_bytes() == before_row


@pytest.mark.parametrize('change', ['model', 'history', 'optimizer', 'contract', 'error', 'completed_run'])
def test_invalid_checkpoint_history_or_failure_is_refused(failed, change):
    root, run, state, calls = failed
    path = root/run/'checkpoint.npz'
    row_path = root/run/'run.json'
    row = read_json(row_path)
    plan = read_json(root/'plan.json')
    if change == 'model':
        state['nodeParams'][0,0] += 1
    elif change == 'history':
        state['telemetry'][-1]['attackActive'] = True
    elif change == 'optimizer':
        state['telemetry'][-1]['training']['optimizerSteps'] = 6
    elif change == 'contract':
        row['contract']['config']['sigma'] = .02
        write_json(row_path, row)
    elif change == 'error':
        row['error'] = 'unrelated error'
        write_json(row_path, row)
    else:
        other = next(v for v in plan['configs'] if v['runId'] != run)
        write_json(root/other['runId']/'run.json', dict(status='FAILED'))
    if change in ('model', 'history', 'optimizer'):
        optimization.CheckpointWriter(plan['executionOptimization'])(path, state, row['contract'])
    before = bytes_under(root)
    with pytest.raises(ValueError):
        recovery.recover(root, run, True)
    assert not calls and bytes_under(root) == before


def test_active_worker_and_controller_lock_are_respected(failed, monkeypatch):
    root, run, state, calls = failed
    monkeypatch.setattr(recovery.live, 'discover', lambda *a: {0: dict(pid=123)})
    with pytest.raises(ValueError, match='still alive'):
        recovery.recover(root, run, True)
    monkeypatch.setattr(recovery.live, 'discover', lambda *a: {})
    with (root/'supervisor.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            recovery.recover(root, run, True)
    assert not calls


def test_original_gpu_memory_headroom_is_required(failed, monkeypatch):
    root, run, state, calls = failed
    inventory = recovery.live.gpu_inventory()
    inventory[3]['memory.free'] = 4000
    monkeypatch.setattr(recovery.live, 'gpu_inventory', lambda: inventory)
    with pytest.raises(RuntimeError, match='headroom'):
        recovery.recover(root, run, True)
    assert not calls and read_json(root/run/'run.json')['status'] == 'FAILED'
