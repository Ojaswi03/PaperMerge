from pathlib import Path
import json
import subprocess

import pytest

from scripts import run_fedsharp_noise_only_auto as auto


@pytest.mark.parametrize('host,script', [
    ('hal9k', 'run_fedsharp_noise_only.py'),
    ('hal9k.cs.ohio.edu', 'run_fedsharp_noise_only.py'),
    ('themachine', 'run_fedsharp_noise_only_themachine.py'),
    ('themachine.cs.ohio.edu', 'run_fedsharp_noise_only_themachine.py'),
])
def test_real_hostname_selects_launcher_and_preserves_arguments(host, script):
    arguments = ['--output', '/var/tmp/experiment with spaces', '--resume', '--execute', '--background']
    command = auto.command_for_host(host, arguments)
    assert Path(command[1]).name == script and command[2:] == arguments


def test_unknown_host_is_refused_before_execution(monkeypatch):
    monkeypatch.setattr(auto.socket, 'gethostname', lambda: 'unrecognized')
    monkeypatch.setattr(auto.os, 'execv', lambda *a: pytest.fail('must not execute'))
    with pytest.raises(ValueError, match='supports hal9k and themachine'):
        auto.main(['--execute'])


def test_process_is_replaced_without_spoofing_hostname_or_launching_a_second_supervisor(monkeypatch):
    monkeypatch.setattr(auto.socket, 'gethostname', lambda: 'themachine')
    calls = []
    monkeypatch.setattr(auto.os, 'execv', lambda executable, command: calls.append((executable, command)))
    auto.main(['--output', '/var/tmp/test', '--execute'])
    assert len(calls) == 1 and calls[0][0] == calls[0][1][0]
    assert Path(calls[0][1][1]).name == 'run_fedsharp_noise_only_themachine.py'
    assert auto.socket.gethostname() == 'themachine'


@pytest.mark.parametrize('host,script', [
    ('hal9k', 'export_fedsharp_noise_only.py'),
    ('themachine', 'run_fedsharp_noise_only_themachine.py'),
])
def test_export_uses_matching_server_verifier(host, script):
    command = auto.command_for_host(host, ['--role', 'export', '--output', '/var/tmp/run',
        '--destination', 'docs/results/export'])
    assert Path(command[1]).name == script
    assert command[command.index('--output') + 1] == '/var/tmp/run'
    assert command[command.index('--destination') + 1] == 'docs/results/export'


@pytest.mark.parametrize('host', ['hal9k', 'themachine'])
def test_selected_launcher_loads_its_actual_preset_without_training(host, tmp_path):
    command = auto.command_for_host(host, ['--output', str(tmp_path / host), '--describe'])
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    row = json.loads(result.stdout)
    assert row['configurations'] == 18 and row['fedSharp'] == 12 and row['fedSharpPlus'] == 6
    assert row['trainingStarted'] is False and row['effectiveAttackers'] == 0
    if host == 'hal9k':
        assert row['requestedGpuCount'] == 2
    else:
        assert row['machineGpuPolicy']['preferred'] == [1, 3]
        assert row['machineGpuPolicy']['opportunistic'] == [0, 2]
    assert not (tmp_path / host).exists()
