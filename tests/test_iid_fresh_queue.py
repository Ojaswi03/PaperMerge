"""Fresh GPU queue preparation never starts research execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from basil_core.iid_campaign import ROOT, CONFIG_ROOT, campaign_configs, gui_approved, verify_campaign
from gui.services import QueueService
from gui.state import ApplicationState, ExecutionState, QueueStatus
from gui.state.experiment_state import to_persisted, validate_experiment


PRESET = ROOT / 'gui/queues/iid_bcd_100r_gpu_fresh.json'


@pytest.fixture
def fresh_configs():
    return json.loads(PRESET.read_text())


def test_fresh_files_keep_protocol_and_exact_pairing(fresh_configs):
    assert [c['conditionId'] for c in fresh_configs] == list('BCD')
    first = fresh_configs[0]
    canonical = campaign_configs(first['expectedPartitionHash'], first['expectedInitialModelHash'], device='GPU')
    assert verify_campaign([canonical[0], *fresh_configs])['C/D'] == ['ebmMode', 'localObjective']
    for c, expected in zip(fresh_configs, canonical[1:]):
        assert c == expected
        assert gui_approved(c)
        assert 'recoverySourceDirectory' not in c and 'deviceTransition' not in c
        assert json.loads((CONFIG_ROOT / 'gpu/fresh' / f"{c['experimentName']}.json").read_text()) == c


def test_loading_fresh_queue_is_idle_and_round_zero(tmp_path, fresh_configs, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Queue preparation must not launch a process')
    monkeypatch.setattr('subprocess.Popen', forbidden)
    queue = QueueService(PRESET).load()
    QueueService.describe_recovery(queue, tmp_path)
    assert len(queue.entries) == 3
    for entry in queue.entries:
        assert entry.status is QueueStatus.PENDING
        assert entry.completed_rounds == 0
        assert entry.execution_hint == 'Fresh R0→R99 — GPU'
        assert not validate_experiment(entry.config)
        state = ApplicationState(experiment=entry.config)
        assert state.execution is ExecutionState.IDLE and state.worker_count == 0
    saved = tmp_path / 'queue.json'
    QueueService(saved).save(queue)
    loaded = QueueService(saved).load()
    assert [to_persisted(e.config) for e in loaded.entries] == fresh_configs


def test_manual_fresh_launch_omits_recovery(tmp_path, fresh_configs, monkeypatch):
    import gui.services.iid_execution_service as service
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    monkeypatch.setattr(service, 'WORKER_ROOT', tmp_path / 'workers')
    launches = []
    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(pid=1234)
    monkeypatch.setattr(service.subprocess, 'Popen', launch)
    monkeypatch.setattr(service.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda: None))
    for c in fresh_configs:
        service.IidExecutionHandle(c, lambda event: None)
    assert len(launches) == 3
    assert all('--recover' not in command for command, _ in launches)
    assert all(options['env']['PAPERMERGE_IID_DEVICE'] == 'GPU' for _, options in launches)


def test_builder_gpu_presets_point_to_fresh_files():
    # Check routing without constructing a Tk root or starting a worker.
    source = (ROOT / 'gui/views/experiment_builder_view.py').read_text()
    assert "for letter in 'ABCD':" in source
    assert "f'gpu/fresh/{item[0]}_gpu'" in source


def test_fresh_a_only_queue_is_canonical_gpu_and_idle(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A preparation must not launch research execution')
    monkeypatch.setattr('subprocess.Popen', forbidden)
    preset = ROOT / 'gui/queues/iid_a_100r_gpu_fresh.json'
    raw = json.loads(preset.read_text())
    assert len(raw) == 1 and raw[0]['conditionId'] == 'A'
    config = raw[0]
    expected = campaign_configs(config['expectedPartitionHash'], config['expectedInitialModelHash'], device='GPU')[0]
    assert config == expected and gui_approved(config)
    assert 'recoverySourceDirectory' not in config and 'deviceTransition' not in config
    assert config['resolvedAttackerIds'] == [] and config['attackerCount'] == 0
    assert not config['useChannelNoise'] and config['localObjective'] == 'cross_entropy'
    assert config['ebmMode'] == 'none' and config['basilMemorySize'] == 5
    queue = QueueService(preset).load()
    QueueService.describe_recovery(queue, tmp_path)
    entry = queue.entries[0]
    assert entry.status is QueueStatus.PENDING and entry.completed_rounds == 0
    assert entry.execution_hint == 'Fresh R0→R99 — GPU'
    assert json.loads((CONFIG_ROOT / 'gpu/fresh' / f"{config['experimentName']}.json").read_text()) == config


def test_all_fresh_gpu_configs_preserve_abcd_contrasts():
    configs = json.loads((ROOT / 'gui/queues/iid_abcd_100r_gpu_fresh.json').read_text())
    assert [c['conditionId'] for c in configs] == list('ABCD')
    differences = verify_campaign(configs)
    assert differences['A/B'] == ['actualAttackerCount', 'attackHidden', 'attackerCount', 'attackerIds', 'resolvedAttackerIds']
    assert differences['C/D'] == ['ebmMode', 'localObjective']
    assert all('recoverySourceDirectory' not in c and 'deviceTransition' not in c for c in configs)
