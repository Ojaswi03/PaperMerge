"""Matched interventions, checkpoint identity and informative recognition gates."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest

from basil_core import cart_contribution_study as study
from tests.test_acds_comparison import arguments


def options(**changes):
    return SimpleNamespace(**(vars(arguments(datasets=['mnist', 'cifar10'])) |
        dict(preset='contribution', variants=None) | changes))


def test_paired_matrix_is_unique_and_retains_physical_contracts():
    from basil_core.cart_comparison_study import effective_config
    views = study.selected(options())
    assert len(views) == len({v['runId'] for v in views}) == 66
    assert Counter(v['dataset'] for v in views) == dict(mnist=12, cifar10=54)
    for view in views:
        cfg = effective_config(view)
        assert cfg['nodes'] == cfg['rounds'] == 100 and cfg['localSteps'] == 5
        assert cfg['sigma'] == .01 and cfg['attackStart'] == 20 and cfg['actualAttackers'] == 33
        assert view['comparisonCase']['shareFraction'] == (.05 if study.variant(view) == 'acds_checked' else 0)


@pytest.mark.parametrize('gpu_count', [1, 2, 3, 4])
def test_assignment_spreads_methods_and_caps_full_cohort(gpu_count, tmp_path):
    from scripts.run_cart_contribution import assignment
    from basil_core.cart_worker_admission import placement
    from basil_core.execution_profile import ExecutionProfile
    views = study.selected(options())
    hardware = [ExecutionProfile(name='hpc', gpu=g, gpu_uuid=f'GPU-{g:08x}',
        memory_mib=4096, headroom_mib=8192, ram_mib=16384, output_root=str(tmp_path))
        for g in range(gpu_count)]
    profiles = assignment(views, hardware)
    devices, assigned = placement(dict(configs=views, profiles=profiles))
    counts = Counter(p.gpu for p in assigned)
    assert max(counts.values()) - min(counts.values()) <= 1
    if gpu_count == 4:
        assert max(counts.values()) == 17
        assert max(counts.values()) * (4096 + 1024) + 8192 < 97887
    assert len(devices) == gpu_count
    for name in study.MAIN:
        chosen = {p['gpu'] for v, p in zip(views, profiles) if study.variant(v) == name}
        assert len(chosen) == gpu_count


def test_variant_is_part_of_checkpoint_identity_and_cannot_be_relabelled():
    view = study.selected(options(datasets=['mnist'], variants=['cart_full'], seeds=[2025]))[0]
    view['comparisonCase']['studyVariant']['name'] = 'acds_checked'
    with pytest.raises(ValueError, match='inconsistent'):
        study.variant(view)


def test_snapshot_creation_preserves_existing_bytes_and_refuses_conflicts(tmp_path, monkeypatch):
    from scripts import run_cart_adaptive as runner
    source = tmp_path / 'repo'
    source.mkdir()
    (source / 'producer.py').write_bytes(b'producer\n')
    monkeypatch.setattr(runner, 'ROOT', source)
    plan = dict(source={'producer.py': hashlib.sha256(b'producer\n').hexdigest()})
    root = tmp_path / 'output'
    runner.ensure_source_snapshots(root, plan)
    saved = root / 'source_snapshots/producer.py'
    stamp = saved.stat().st_mtime_ns
    runner.ensure_source_snapshots(root, plan)
    assert saved.stat().st_mtime_ns == stamp
    saved.write_bytes(b'conflicting evidence')
    with pytest.raises(ValueError, match='Saved producer'):
        runner.ensure_source_snapshots(root, plan)
    assert saved.read_bytes() == b'conflicting evidence'


@pytest.fixture(scope='module', autouse=True)
def cpu():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


def tiny(name):
    return study.selected(options(datasets=['mnist'], partitions=['iid'], seeds=[2025],
        nodes=4, rounds=2, local_steps=1, attack_start=1, variants=[name]))[0]


def test_full_cart_reproduces_completed_producer_exactly():
    from tests.test_cart_comparison_runtime import data
    from basil_core.cart_comparison_study import configure_worker, intervention
    import basil_core.sharp_cart_v2_protocol as protocol
    view = tiny('cart_full')
    x, y, chunks = data()
    old = configure_worker(view, 'CPU')
    with intervention(view, old) as cfg:
        expected = protocol.run(cfg, old, x, y, chunks, x[:20], y[:20])
    new = study.configure_worker(view, 'CPU')
    with study.intervention(view, new) as cfg:
        actual = protocol.run(cfg, new, x, y, chunks, x[:20], y[:20])
    np.testing.assert_array_equal(expected['nodeParams'], actual['nodeParams'])
    np.testing.assert_array_equal(expected['outbound'], actual['outbound'])
    assert expected['roundMetrics'] == actual['roundMetrics']


@pytest.mark.parametrize('name', study.MAIN)
def test_all_interventions_resume_identically(tmp_path, name):
    from tests.test_cart_comparison_runtime import data
    from basil_core.sharp_protocol import save_state, load_state
    import basil_core.sharp_cart_v2_protocol as protocol
    x, y, chunks = data()
    view = tiny(name)
    worker = study.configure_worker(view, 'CPU')
    with study.intervention(view, worker) as cfg:
        expected = protocol.run(cfg, worker, x, y, chunks, x[:20], y[:20])
    worker = study.configure_worker(view, 'CPU')
    with study.intervention(view, worker) as cfg:
        partial = protocol.run(cfg, worker, x, y, chunks, x[:20], y[:20], max_activations=3)
    contract = dict(config=cfg, comparisonCase=view['comparisonCase'])
    path = tmp_path / 'checkpoint.npz'
    save_state(path, partial, contract)
    worker = study.configure_worker(view, 'CPU')
    state = load_state(path, contract, worker.dimension)
    with study.intervention(view, worker) as cfg:
        actual = protocol.run(cfg, worker, x, y, chunks, x[:20], y[:20], state=state)
    np.testing.assert_array_equal(expected['nodeParams'], actual['nodeParams'])
    np.testing.assert_array_equal(expected['outbound'], actual['outbound'])
    assert expected['roundMetrics'] == actual['roundMetrics']
    assert sum(r['training']['optimizerSteps'] for r in actual['telemetry']) == 8
    wrong = dict(contract, comparisonCase=dict(view['comparisonCase'], studyVariant=dict(name='different')))
    with pytest.raises(ValueError, match='identity'):
        load_state(path, wrong, worker.dimension)


@pytest.mark.parametrize('name', study.MAIN)
def test_informative_gates_reach_declared_branches(name):
    from scripts.verify_cart_contribution_gpu import warm
    view = tiny(name)
    row = warm(view, 'CPU')
    assert row['optimizerSteps'] == 1
    if name == 'cart_full':
        assert row['calibrationDraws'] == 19 and row['recognitionContext']


def test_installed_adapter_restores_every_original_function_after_failure():
    from basil_core import cart_comparison_study as base
    original = (base.source, base.check_source, base.configure_worker, base.intervention)
    with pytest.raises(RuntimeError):
        with study.installed():
            assert base.configure_worker is study.configure_worker
            raise RuntimeError('intentional interruption')
    assert (base.source, base.check_source, base.configure_worker, base.intervention) == original


def test_adaptive_scheduler_admits_all_66_cases_under_four_gpu_startup_reservations(tmp_path, monkeypatch):
    import io
    from scripts import run_cart_adaptive as runner
    from scripts.run_cart_contribution import assignment
    from basil_core.execution_profile import ExecutionProfile
    from basil_core.cart_worker_admission import placement, Resident
    views = study.selected(options())
    hardware = [ExecutionProfile(name='hpc', gpu=g, gpu_uuid=f'GPU-{g:08x}',
        memory_mib=4096, headroom_mib=8192, ram_mib=16384, output_root=str(tmp_path)) for g in range(4)]
    plan = dict(configs=views, profiles=assignment(views, hardware), revision=study.REVISION,
        source={}, resourceBudget=dict(context_overhead_mib=1024, ram_headroom_mib=32768))
    runner.write_json(tmp_path / 'plan.json', plan)
    original = (tmp_path / 'plan.json').read_bytes()
    devices, assigned = placement(plan)
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 1000 + len(calls)
            self.index = int(command[command.index('--worker') + 1])
            calls.append((command, self))
        def poll(self):
            if len(calls) < 66:
                return None
            folder = tmp_path / views[self.index]['runId']
            folder.mkdir(exist_ok=True)
            runner.write_json(folder / 'run.json', dict(status='COMPLETED'))
            return 0
    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    monkeypatch.setattr(runner, 'gpu_inventory', lambda: [dict(index=p.gpu, uuid=p.gpu_uuid,
        **{'memory.free': 97887}) for p in hardware])
    monkeypatch.setattr(runner, 'gpu_processes', lambda: [])
    monkeypatch.setattr(runner.scaled, 'available_ram_mib', lambda: 1400000)
    monkeypatch.setattr(runner, 'residents_of', lambda active, assigned: [
        Resident(p.pid, assigned[i].gpu, 0) for i, (p, _) in active.items()])
    monkeypatch.setattr(runner, 'check_plan_source', lambda _: None)
    monkeypatch.setattr(runner.scaled.core, 'worker_environment', lambda p: {})
    monkeypatch.setattr(runner.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    monkeypatch.setattr(runner, 'completed_report', lambda *args: None)
    leases = {g: io.StringIO() for g in devices}
    for lease in leases.values():
        lease.fileno = lambda: 123
    args = SimpleNamespace(output=tmp_path, max_workers_per_gpu=24, poll_seconds=1, ntfy_topic=None)
    assert runner.schedule(args, plan, devices, assigned, runner.sha(tmp_path / 'plan.json'), leases) == 0
    assert len(calls) == 66
    assert all('cart_contribution_worker.py' in command[1] for command, _ in calls)
    assert (tmp_path / 'plan.json').read_bytes() == original
    assert runner.read_json(tmp_path / 'campaign.json')['completed'] == 66


@pytest.mark.parametrize('scale', [0., .5, 1.])
def test_reference_metadata_follows_the_committed_full_model(monkeypatch, scale):
    from basil_core import cart_prior_neutral_commit as module
    previous = dict(reference='previous')
    proposed_reference = dict(reference='unscaled proposal')
    proposal = np.array([4., 5., 6., 7.], np.float32)
    ingress = np.array([0., 1., 2., 3.], np.float32)
    chosen = ingress + scale * (proposal - ingress)
    class Worker:
        dimension = 4
        sizes = [2, 1, 1]
        quality_context = dict(ingress=ingress)
        def load(self, vector):
            self.loaded = vector.copy()
    worker = Worker()
    monkeypatch.setattr(module, 'commit', lambda *args: (chosen, dict(applied=True, scale=scale)))
    original = lambda *args: (proposal, proposed_reference, dict(referenceAdmitted=True))
    actual, memory, evidence = module.transport(original)(worker, proposal, previous, None, None, 0, 100)
    np.testing.assert_array_equal(actual, worker.loaded)
    assert memory is (proposed_reference if scale == 1. else previous)
    assert evidence['referenceAdmitted'] == (scale == 1.)
