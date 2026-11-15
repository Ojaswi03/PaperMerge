import copy
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from basil_core import fedsharp_noise_only as study, basil_sharp_thesis as base
from scripts import run_fedsharp_noise_only as launcher
from tests.test_basil_sharp_thesis import options


def opts(**changes):
    return options(families=['sharp'], environments=['noise_only'], rounds=500, **changes)


def test_all_18_cases_change_only_the_declared_physical_environment():
    before = base.source()
    views = study.selected(opts())
    assert len(views) == len({v['runId'] for v in views}) == 18
    assert sum(v['comparisonCase']['addon'] == 'cart' for v in views) == 6
    for view in views:
        cfg = study.effective_config(view)
        ref = study.reference_for(view)
        original = base.effective_config(ref)
        assert cfg['sigma'] == .01 and cfg['actualAttackers'] == view['actualAttackers'] == 0
        assert not cfg['switches']['attack'] and not view['switches']['attack']
        assert cfg['byzantineBudget'] == 33 and cfg['window'] == 66 and cfg['proximalMu'] == 1
        assert cfg['localSteps'] == 5 and cfg['learningRate'] == .03
        assert cfg['cart'] == (view['comparisonCase']['addon'] == 'cart')
        assert not cfg['cart'] or cfg['partition'] == 'dirichlet'
        assert view['comparisonCase']['environment'] == 'noise_only'
        expected = dict(original, actualAttackers=0, condition=view['condition'], noiseOnlyRevision=study.REVISION)
        expected['switches'] = dict(original['switches'], attack=False)
        assert cfg == expected
    with study.installed():
        assert base.selected(opts()) == views and base.source() == study.source()
        for view in views:
            assert base.effective_config(view) == study.effective_config(view)
    assert base.source() == before


@pytest.mark.parametrize('field,value', [('sigma', .02), ('actualAttackers', 33), ('window', 1)])
def test_view_tampering_is_refused(field, value):
    view = study.selected(opts())[0]
    view[field] = value
    with pytest.raises(ValueError):
        study.effective_config(view)


def test_description_is_pure_and_targets_two_hal9k_gpus(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher.driver, 'prepare', lambda *a: pytest.fail('must not load data/devices'))
    target = tmp_path / 'configs.json'
    assert launcher.main(['--output', str(tmp_path / 'run'), '--describe', '--write-configs', str(target)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result['configurations'], result['fedSharp'], result['fedSharpPlus']) == (18, 12, 6)
    assert result['requestedGpuCount'] == 2 and result['effectiveAttackers'] == 0
    assert result['activations'] == 900000 and result['optimizerSteps'] == 4500000
    assert result['start'] == 'initialization' and not result['trainingStarted']
    assert len(json.loads(target.read_text())['configs']) == 18
    assert not (tmp_path / 'run').exists()


def test_execution_refuses_other_hosts_and_test_data_during_training(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.socket, 'gethostname', lambda: 'themachine')
    with pytest.raises(ValueError, match='restricted to hal9k'):
        launcher.main(['--output', str(tmp_path / 'run'), '--execute'])
    with pytest.raises(ValueError, match='post-training'):
        launcher.main(['--output', str(tmp_path / 'run'), '--execute', '--official-test'])
    assert not (tmp_path / 'run').exists()


def test_two_gpu_gates_then_scheduler_handoff(tmp_path, monkeypatch):
    from dataclasses import asdict
    from pathlib import Path
    from basil_core import live_gpu_scheduler as live, execution_profile
    from basil_core.execution_profile import ExecutionProfile
    from basil_core.cart_campaign_io import read_json, write_json
    root = tmp_path / 'run'
    root.mkdir()
    views = study.selected(opts())
    hardware = {g: ExecutionProfile(name='hpc', gpu=g, gpu_uuid=f'GPU-{g:08x}',
        memory_mib=3072, headroom_mib=8192, ram_mib=8192, output_root=str(root)) for g in range(2)}
    inventory = [dict(index=g, uuid=p.gpu_uuid, **{'memory.free': 90000}) for g, p in hardware.items()]
    monkeypatch.setattr(execution_profile, 'gpu_inventory', lambda: inventory)
    monkeypatch.setattr(launcher.adaptive.scaled, 'available_ram_mib', lambda: 1000000)
    monkeypatch.setattr(launcher.adaptive.scaled, 'acquire_leases',
        lambda profiles: {g: (tmp_path / f'lease{g}').open('a') for g in hardware})
    checked = []
    def gate(command, **kwargs):
        gpu = int(command[command.index('--gpu') + 1])
        checked.append(gpu)
        assert kwargs['pass_fds']
        plan = read_json(root / 'plan.json')
        write_json(root / f'revision_gpu_{gpu}.json', dict(status='PASSED', source=plan['source'],
            planSha256=launcher.adaptive.sha(root / 'plan.json'),
            runtime=dict(host=launcher.socket.gethostname(), profile=asdict(hardware[gpu]))))
    monkeypatch.setattr(launcher.subprocess, 'run', gate)
    def schedule(args):
        assert checked == [0, 1]
        assert live.WORKER == Path(launcher.__file__).resolve() and live.WORKER.name in live.OWNERS
        assert live.controller(root) is None
        actual, devices, assigned, digest = live.verified(root)
        assert len(actual['configs']) == 18 and len(devices) == 2 and len(assigned) == 18
        return 0
    monkeypatch.setattr(live, 'execute', schedule)
    args = SimpleNamespace(output=root, wait_for_resources=False, poll_seconds=3, ntfy_topic=None)
    with study.installed():
        plan = dict(revision=study.REVISION, source=study.source(), configs=views,
            executionOptimization=study.runtime_policy(),
            profiles=[asdict(hardware[i % 2]) for i in range(18)],
            host=launcher.socket.gethostname(), maximumWorkersPerGpu=24,
            resourceBudget=dict(context_overhead_mib=1024, ram_headroom_mib=16384, ram_mib_per_worker=8192))
        write_json(root / 'plan.json', plan)
        assert launcher.execute(args, plan) == 0


@pytest.mark.parametrize('addon', [None, 'cart'])
def test_worker_checkpoint_and_fresh_inference_with_timing_adapter(cpu, tmp_path, monkeypatch, addon):
    import signal
    from tests.test_cart_comparison_runtime import data
    from basil_core import cart_comparison_study as comparison, sharp_data
    from basil_core import fedsharp_runtime_optimization as execution, round_timing
    from basil_core.sharp_protocol import load_state
    from scripts.cart_comparison_worker import run
    from scripts.verify_cart_comparison_results import verify
    args = opts(datasets=['mnist'], partitions=['dirichlet'], seeds=[2025], nodes=4, local_steps=5)
    args.rounds = 2
    view = next(v for v in study.selected(args) if v['comparisonCase']['addon'] == addon)
    x, y, chunks = data()
    held, audit = np.arange(20), dict(fixture='synthetic')
    source, policy = study.source(), study.runtime_policy()
    identity = dict(imagesSha256=sharp_data.digest(x), labelsSha256=sharp_data.digest(y),
        samples=len(y), officialTestLoaded=False)
    plan = dict(revision=study.REVISION, configs=[view], source=source, executionOptimization=policy,
        data=dict(mnist=identity), partitions={view['runId']: audit})
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    for name in source:
        target = tmp_path / 'source_snapshots' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((study.ROOT / name).read_bytes())
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with study.installed():
            monkeypatch.setattr(sharp_data, 'training_data', lambda dataset: (x, y))
            monkeypatch.setattr(comparison, 'partitions_for', lambda v, labels: (chunks, held, audit))
            with execution.installed(tmp_path, view, policy), round_timing.recording(tmp_path, 0):
                run(tmp_path, 0, cpu=True)
            folder = tmp_path / view['runId']
            row = json.loads((folder / 'run.json').read_text())
            assert row['status'] == 'COMPLETED' and row['optimizerSteps'] == 40
            assert row['contract']['config']['actualAttackers'] == 0
            worker = base.configure_worker(view, 'CPU')
            state = load_state(folder / 'checkpoint.npz', row['contract'], worker.dimension)
            assert state['cursor'] == 8
            assert all(not r['attackActive'] and not r['configuredByzantine'] for r in state['telemetry'])
            events = [json.loads(line) for line in (folder / 'round_timings.jsonl').read_text().splitlines()]
            assert sum(e['kind'] == 'round' and e['fullRound'] for e in events) == 2
            assert (folder / 'optimization.json').is_file()
            receipt = verify(tmp_path)
            assert receipt['status'] == 'ALL_COMPARISON_MODELS_VERIFIED'
            assert receipt['verified'][0]['maximumValidationAccuracyDifference'] == 0
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


@pytest.fixture(scope='module')
def cpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


@pytest.mark.parametrize('dataset,split,addon', [
    ('mnist', 'iid', None), ('mnist', 'dirichlet', None), ('mnist', 'dirichlet', 'cart'),
    ('cifar10', 'iid', None), ('cifar10', 'dirichlet', None), ('cifar10', 'dirichlet', 'cart')])
def test_real_model_noise_only_matches_attack_inactive_reference_and_resumes(cpu, tmp_path, dataset, split, addon):
    import basil_core.sharp_cart_v2_protocol as protocol
    from basil_core.sharp_protocol import save_state, load_state
    args = opts(datasets=[dataset], partitions=[split], seeds=[2025], nodes=4, local_steps=5)
    args.rounds = 2
    args.attack_start = 20
    view = next(v for v in study.selected(args) if v['comparisonCase']['addon'] == addon)
    reference = study.reference_for(view)
    shape = (28, 28, 1) if dataset == 'mnist' else (32, 32, 3)
    y = np.tile(np.array([0, 1], np.int32), 40)
    x = np.random.default_rng(421).random((80, *shape), dtype=np.float32)
    chunks = np.array_split(np.arange(len(y)), 4)
    old = base.configure_worker(reference, 'CPU')
    with base.intervention(reference, old) as ref_cfg:
        expected = protocol.run(ref_cfg, old, x, y, chunks, x[:20], y[:20])
    with study.installed():
        worker = base.configure_worker(view, 'CPU')
        with base.intervention(view, worker) as cfg:
            result = protocol.run(cfg, worker, x, y, chunks, x[:20], y[:20])
        interrupted = base.configure_worker(view, 'CPU')
        with base.intervention(view, interrupted) as cfg:
            partial = protocol.run(cfg, interrupted, x, y, chunks, x[:20], y[:20], max_activations=5)
        contract = dict(config=cfg, source=study.source())
        path = tmp_path / 'checkpoint.npz'
        save_state(path, partial, contract)
        resumed = base.configure_worker(view, 'CPU')
        state = load_state(path, contract, resumed.dimension)
        with base.intervention(view, resumed) as cfg:
            actual = protocol.run(cfg, resumed, x, y, chunks, x[:20], y[:20], state=state)
    np.testing.assert_array_equal(result['nodeParams'], expected['nodeParams'])
    np.testing.assert_array_equal(actual['nodeParams'], result['nodeParams'])
    np.testing.assert_array_equal(actual['outbound'], actual['nodeParams'])
    assert actual['telemetry'] == result['telemetry'] and actual['roundMetrics'] == result['roundMetrics']
    assert all(not r['attackActive'] and not r['configuredByzantine'] and r['trainedHash'] == r['outboundHash']
               for r in actual['telemetry'])
    assert sum(r['training']['optimizerSteps'] for r in actual['telemetry']) == 40


def test_timing_records_full_and_partial_rounds_without_modifying_callbacks(tmp_path):
    import basil_core.sharp_cart_v2_protocol as protocol
    from basil_core.round_timing import recording
    root = tmp_path
    (root / 'plan.json').write_text(json.dumps(dict(configs=[dict(runId='case')])))
    received = []
    state = dict(cursor=5)
    def run(cfg, **kwargs):
        kwargs['on_round'](dict(round=1), dict(cursor=8))
        kwargs['on_round'](dict(round=2), dict(cursor=12))
        return 'same result'
    with patch.object(protocol, 'run', run):
        with recording(root, 0):
            assert protocol.run(dict(nodes=4), state=state, on_round=lambda r, s: received.append(s['cursor'])) == 'same result'
        assert protocol.run is run
    rows = [json.loads(line) for line in (root / 'case/round_timings.jsonl').read_text().splitlines()]
    rounds = [row for row in rows if row['kind'] == 'round']
    assert received == [8, 12]
    assert [(r['activations'], r['fullRound']) for r in rounds] == [(3, False), (4, True)]
    assert rows[0]['kind'] == 'segment_start' and rows[-1]['kind'] == 'segment_end'
    assert all(r['elapsedSeconds'] >= 0 for r in rounds)


@pytest.mark.parametrize('dataset', ['mnist', 'cifar10'])
def test_informative_noise_only_gate_runs_19_draws_and_full_commit(cpu, dataset):
    from basil_core import cart_contribution_study as previous
    from scripts.verify_cart_contribution_gpu import warm
    view = next(v for v in study.selected(opts(datasets=[dataset], partitions=['dirichlet'], seeds=[2025]))
                if v['comparisonCase']['addon'] == 'cart')
    with study.installed(), patch.object(previous, 'variant', lambda v: 'cart_full'), \
         patch.object(previous, 'configure_worker', base.configure_worker), \
         patch.object(previous, 'intervention', base.intervention):
        receipt = warm(view, 'CPU')
    assert receipt['recognitionContext'] and receipt['calibrationDraws'] == 19
    assert receipt['composedEmpiricalBoundVerified'] and receipt['optimizerSteps'] == 5


@pytest.mark.parametrize('tamper', ['attack', 'outgoing', 'steps'])
def test_export_attack_chain_audit_refuses_hidden_changes(tmp_path, tamper):
    from scripts.export_fedsharp_noise_only import audit_no_attacks
    view = study.selected(opts(datasets=['mnist'], partitions=['iid'], seeds=[2025], nodes=4))[0]
    cfg = study.effective_config(view)
    folder = tmp_path / view['runId']
    folder.mkdir()
    records = [dict(round=i // 4, node=i % 4, activation=i + 1, attackActive=False,
        configuredByzantine=False, trainedHash='same', outboundHash='same',
        training=dict(optimizerSteps=5)) for i in range(2000)]
    metadata = dict(contract=dict(config=cfg), cursor=2000, telemetry=records)
    def save():
        np.savez(folder / 'checkpoint.npz', metadata=np.frombuffer(json.dumps(metadata).encode(), np.uint8))
    save()
    plan = dict(configs=[view], source=study.source())
    receipt = audit_no_attacks(tmp_path, plan)
    assert receipt['verified'][0]['activationsChecked'] == 2000
    if tamper == 'attack':
        records[-1]['attackActive'] = True
    elif tamper == 'outgoing':
        records[-1]['outboundHash'] = 'different'
    else:
        records[-1]['training']['optimizerSteps'] = 6
    save()
    with pytest.raises(ValueError, match='evidence differs'):
        audit_no_attacks(tmp_path, plan)
