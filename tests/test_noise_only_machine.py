from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from basil_core import fedsharp_noise_only as study, noise_only_machine as machine
from basil_core import live_gpu_scheduler as live
from basil_core.cart_campaign_io import read_json, write_json
from basil_core.cart_worker_admission import Resident
from basil_core.execution_profile import ExecutionProfile
from scripts import run_fedsharp_noise_only_themachine as launcher
from tests.test_fedsharp_noise_only import opts
from tests.test_noise_only_export import completed


def test_same_scientific_matrix_extends_source_pins_only():
    before = study.source()
    views = study.selected(opts())
    with machine.installed():
        assert study.selected(opts()) == views
        assert len(study.source()) == len(before) + 3
        assert all(study.source()[name] == digest for name, digest in before.items())
        for view in views:
            cfg = study.effective_config(view)
            assert cfg['actualAttackers'] == 0 and cfg['sigma'] == .01 and cfg['proximalMu'] == 1
    assert study.source() == before


def test_description_is_pure_and_host_guard_is_machine_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher.driver, 'prepare', lambda *a: pytest.fail('must not load data/devices'))
    target = tmp_path / 'configs.json'
    assert launcher.main(['--output', str(tmp_path / 'run'), '--describe', '--write-configs', str(target)]) == 0
    row = json.loads(capsys.readouterr().out)
    assert (row['configurations'], row['fedSharp'], row['fedSharpPlus']) == (18, 12, 6)
    assert row['machineGpuPolicy']['preferred'] == [1, 3]
    assert row['machineGpuPolicy']['opportunistic'] == [0, 2]
    assert row['activations'] == 900000 and row['optimizerSteps'] == 4500000
    monkeypatch.setattr(launcher.socket, 'gethostname', lambda: 'hal9k')
    with pytest.raises(ValueError, match='restricted to themachine'):
        launcher.main(['--output', str(tmp_path / 'run'), '--execute'])
    with pytest.raises(ValueError, match='post-training'):
        launcher.main(['--output', str(tmp_path / 'run'), '--official-test', '--execute'])
    assert not (tmp_path / 'run').exists()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'operational_adapters').mkdir()
    monkeypatch.setattr(machine.socket, 'gethostname', lambda: 'themachine')
    profiles = {g: ExecutionProfile(name='hpc', gpu=g, gpu_uuid=f'GPU-{g:08x}', memory_mib=3072,
        headroom_mib=8192, ram_mib=8192, output_root=str(root)) for g in range(4)}
    plan = dict(host='themachine', revision=study.REVISION, configs=study.selected(opts()),
        source=machine.source(), machineGpuPolicy=machine.POLICY, executionOptimization=study.runtime_policy(),
        profiles=[asdict(profiles[i % 4]) for i in range(18)], maximumWorkersPerGpu=24,
        resourceBudget=dict(context_overhead_mib=1024, ram_headroom_mib=16384, ram_mib_per_worker=8192))
    write_json(root / 'plan.json', plan)
    write_json(root / 'campaign.json', dict(status='WAITING_FOR_RESOURCES', host='themachine'))
    inventory = [dict(index=g, uuid=profiles[g].gpu_uuid,
        **{'memory.free': free, 'utilization.gpu': use})
        for g, free, use in [(0, 4601, 81), (1, 90143, 31), (2, 52555, 93), (3, 81443, 0)]]
    leases = {g: (tmp_path / f'lease{g}').open('a') for g in profiles}
    checked = []
    def gate(command, **kwargs):
        gpu = int(command[command.index('--gpu') + 1])
        checked.append(gpu)
        assert '--role' in command and command[command.index('--role') + 1] == 'gate'
        assert kwargs['pass_fds'] == (leases[gpu].fileno(),)
        write_json(root / f'revision_gpu_{gpu}.json', dict(status='PASSED', noiseOnly=True,
            actualAttackers=0, emptyAttackerSetsVerified=True, source=plan['source'],
            planSha256=live.sha(root / 'plan.json'), runtime=dict(host='themachine', profile=asdict(profiles[gpu]))))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(machine.subprocess, 'run', gate)
    monkeypatch.setattr(machine, 'live_worker_environment', lambda p: {})
    monkeypatch.setattr(live, 'gpu_inventory', lambda: inventory)
    monkeypatch.setattr(live, 'gpu_processes', lambda: [])
    from scripts import run_cart_v2_scaled
    monkeypatch.setattr(run_cart_v2_scaled, 'available_ram_mib', lambda: 1000000)
    clock = [0.]
    monkeypatch.setattr(machine.time, 'monotonic', lambda: clock[0])
    chooser = machine.DeviceAdmission(root, plan)
    def choose(residents=()):
        return chooser.choose(list(range(18)), plan['configs'], {i: set(profiles) for i in range(18)},
            profiles, list(residents), inventory, [], 1000000, plan['resourceBudget'], 24, {}, leases)
    yield root, plan, profiles, inventory, checked, clock, chooser, choose
    for lease in leases.values():
        lease.close()


def test_busy_optional_devices_do_not_block_gpus_1_and_3(setup):
    root, plan, profiles, inventory, checked, clock, chooser, choose = setup
    choice, blocked = choose()
    assert choice[1].gpu == 1 and checked == [1]
    assert blocked[0]['reason'] == blocked[2]['reason'] == 'waiting for low GPU utilization'
    choice, _ = choose([Resident(101, 1, 8192)])
    assert choice[1].gpu == 3 and checked == [1, 3]
    choose([Resident(101, 1, 8192), Resident(103, 3, 8192)])
    assert checked == [1, 3]
    with machine.installed():
        assert len(machine.verified(root)[1]) == 4  # Unused devices need not pass a gate yet.


def test_optional_devices_join_after_30_seconds_of_quiet_samples(setup):
    root, plan, profiles, inventory, checked, clock, chooser, choose = setup
    choose()
    residents = [Resident(101, 1, 8192)]
    choose(residents)
    residents.append(Resident(103, 3, 8192))
    inventory[0].update({'memory.free': 90000, 'utilization.gpu': 0})
    choice, blocked = choose(residents)
    assert blocked[0]['reason'] == 'waiting for quiet GPU samples' and 0 not in checked
    clock[0] = 31
    choice, _ = choose(residents)
    assert choice[1].gpu == 0 and checked == [1, 3, 0]
    inventory[2]['utilization.gpu'] = 0
    choose(residents)
    clock[0] = 62
    choice, _ = choose([*residents, Resident(100, 0, 8192)])
    assert choice[1].gpu == 2 and checked == [1, 3, 0, 2]


def test_busy_or_unknown_sample_resets_optional_quiet_period(setup):
    root, plan, profiles, inventory, checked, clock, chooser, choose = setup
    inventory[0].update({'memory.free': 90000, 'utilization.gpu': 0})
    choose()
    clock[0] = 25
    inventory[0]['utilization.gpu'] = None
    choose()
    assert 0 not in chooser.quiet_since
    inventory[0]['utilization.gpu'] = 0
    clock[0] = 40
    _, blocked = choose()
    assert blocked[0]['reason'] == 'waiting for quiet GPU samples'
    assert 0 not in checked


def test_free_memory_is_resampled_after_device_gate(setup, monkeypatch):
    root, plan, profiles, inventory, checked, clock, chooser, choose = setup
    gate = machine.subprocess.run
    def memory_consumed(*args, **kwargs):
        result = gate(*args, **kwargs)
        inventory[1]['memory.free'] = 4000
        return result
    monkeypatch.setattr(machine.subprocess, 'run', memory_consumed)
    choice, blocked = choose()
    assert checked == [1] and choice is None
    assert blocked[1]['reason'] == 'waiting for memory'


def test_failed_device_gate_does_not_disable_other_devices(setup, monkeypatch):
    root, plan, profiles, inventory, checked, clock, chooser, choose = setup
    gate = machine.subprocess.run
    def fail_one(command, **kwargs):
        if command[command.index('--gpu') + 1] == '1':
            raise subprocess.CalledProcessError(1, command)
        return gate(command, **kwargs)
    monkeypatch.setattr(machine.subprocess, 'run', fail_one)
    choice, blocked = choose()
    assert choice[1].gpu == 3 and 1 in chooser.failed
    assert blocked[1]['reason'].startswith('device verification failed')
    assert (root / 'operational_adapters/machine-gate-failures.json').is_file()


def test_checkpoint_without_matching_device_gate_is_refused(setup):
    root, plan, profiles, inventory, checked, clock, chooser, choose = setup
    folder = root / plan['configs'][0]['runId']
    folder.mkdir()
    write_json(folder / 'run.json', dict(contract=dict(runtime=dict(profile=asdict(profiles[1])))))
    with machine.installed(), pytest.raises(ValueError, match='original passed GPU gate'):
        machine.verified(root)


def test_scheduler_installs_and_restores_scoped_device_hooks(setup):
    root, plan, profiles, inventory, checked, clock, chooser, choose = setup
    original = live.choose, live.verified, live.WORKER
    from reporting import basil_sharp_thesis as reporting
    report = reporting.generate
    with machine.installed(), machine.scheduler(root, plan):
        assert live.WORKER == machine.ENTRYPOINT and live.WORKER.name in live.OWNERS
        assert live.verified(root)[0] == plan
        assert reporting.generate == machine.completed_report
    assert (live.choose, live.verified, live.WORKER) == original
    assert reporting.generate == report


def test_real_preparation_and_scheduler_handoff_do_not_wait_for_busy_devices(setup, tmp_path, monkeypatch):
    root, plan, profiles, inventory, checked, clock, chooser, choose = setup
    from basil_core import execution_profile
    monkeypatch.setattr(execution_profile, 'gpu_inventory', lambda: inventory)
    monkeypatch.setattr(launcher.driver.study, 'data_contract',
        lambda views: ({'mnist': {}, 'cifar10': {}}, {v['runId']: {} for v in views}))
    target = tmp_path / 'fresh'
    def schedule(args):
        saved, hardware, assigned, digest = live.verified(target)
        assert saved['machineGpuPolicy'] == machine.POLICY and len(hardware) == 4
        assert len(assigned) == 18 and len(saved['source']) == 90
        assert all((target / 'source_snapshots' / name).is_file() for name in saved['source'])
        assert not list(target.glob('revision_gpu_*.json'))
        assert live.WORKER == machine.ENTRYPOINT
        return 0
    monkeypatch.setattr(live, 'execute', schedule)
    assert launcher.main(['--output', str(target), '--execute', '--wait-for-resources']) == 0
    assert checked == []  # Device checks happen at admission, not before handoff.


def test_completion_summary_handles_all_18_cases_without_old_joint_plotter(setup):
    root, plan, profiles, inventory, checked, clock, chooser, choose = setup
    for view in plan['configs']:
        folder = root / view['runId']
        folder.mkdir()
        write_json(folder / 'run.json', dict(status='COMPLETED', elapsedSeconds=500.))
        np.savez_compressed(folder / 'metrics.npz', roundNodeAccuracy=np.full((500, 100), .75))
    machine.completed_report(root, root / 'unused')
    row = read_json(root / 'noise_only_training_summary.json')
    assert len(row['runs']) == 18 and row['officialTestUsed'] is False
    assert all(r['rounds491To500ValidationMean'] == .75 for r in row['runs'])


def test_machine_export_invokes_matching_verifier_and_preserves_all_source_pins(completed, monkeypatch):
    from scripts import export_fedsharp_noise_only as exporter
    root, dest = completed
    with machine.installed():
        plan = read_json(root / 'plan.json')
        plan.update(source=machine.source(), machineGpuPolicy=machine.POLICY)
        write_json(root / 'plan.json', plan)
        for view in plan['configs']:
            path = root / view['runId'] / 'run.json'
            row = read_json(path)
            row['source'] = plan['source']
            write_json(path, row)
        for name in plan['source']:
            path = root / 'source_snapshots' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((machine.ROOT / name).read_bytes())
        def verify(command, **kwargs):
            assert command[1:4] == [str(machine.ENTRYPOINT), '--role', 'verify']
            assert kwargs['check'] is True and '--official-test' not in command
            write_json(root / 'independent_validation.json', dict(status='ALL_COMPARISON_MODELS_VERIFIED',
                source=plan['source'], officialTestUsed=False,
                verified=[dict(runId=v['runId']) for v in plan['configs']]))
        monkeypatch.setattr(exporter.subprocess, 'run', verify)
        assert exporter.export(root, dest, verifier_script=machine.ENTRYPOINT)['configurations'] == 18
    manifest = read_json(dest / 'SHA256.json')
    assert len([p for p in manifest if p.startswith('source_snapshots/')]) == 90
