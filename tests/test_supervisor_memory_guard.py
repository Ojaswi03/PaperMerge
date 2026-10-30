from pathlib import Path
from types import SimpleNamespace

import pytest

from basil_core.supervisor_memory_guard import residents_of
from basil_core.cart_worker_admission import Resident, admission
from basil_core.execution_profile import ExecutionProfile


def worker():
    return SimpleNamespace(pid=12345, poll=lambda: None)


@pytest.mark.parametrize('status', ['Name: worker\nState: Z (zombie)\n',
    'Name: worker\nVmRSS: unavailable\n', 'Name: worker\nVmRSS:\n'])
def test_missing_or_partial_memory_keeps_full_worker_reservation(monkeypatch, status):
    monkeypatch.setattr(Path, 'read_text', lambda self: status)
    assert residents_of({0: (worker(), None)}, [SimpleNamespace(gpu=1)]) == [Resident(12345, 1, 0)]


@pytest.mark.parametrize('error', [FileNotFoundError(), PermissionError(), ProcessLookupError()])
def test_exit_during_status_read_is_conservative(monkeypatch, error):
    def read(_):
        raise error
    monkeypatch.setattr(Path, 'read_text', read)
    assert residents_of({0: (worker(), None)}, [SimpleNamespace(gpu=1)]) == [Resident(12345, 1, 0)]


def test_resident_and_swapped_memory_are_counted(monkeypatch):
    monkeypatch.setattr(Path, 'read_text', lambda _: 'VmRSS: 2048 kB\nVmSwap: 1024 kB\n')
    assert residents_of({0: (worker(), None)}, [SimpleNamespace(gpu=0)]) == [Resident(12345, 0, 3)]


def test_reaped_process_is_removed_without_reading_proc(monkeypatch):
    monkeypatch.setattr(Path, 'read_text', lambda _: pytest.fail('Process already exited'))
    process = SimpleNamespace(pid=12345, poll=lambda: 0)
    assert residents_of({0: (process, None)}, [SimpleNamespace(gpu=0)]) == []


def test_unknown_memory_does_not_allow_oversubscription(monkeypatch):
    monkeypatch.setattr(Path, 'read_text', lambda _: 'State: Z (zombie)\n')
    profile = ExecutionProfile(name='hpc', gpu=0, gpu_uuid='GPU-00000000', memory_mib=3072,
                               headroom_mib=8192, ram_mib=8192, output_root='/tmp/memory-guard-test')
    residents = residents_of({0: (worker(), None)}, [profile])
    result = admission(profile, residents, [dict(index=0, uuid=profile.gpu_uuid, **{'memory.free': 97000})],
                       [], 32000, context_mib=1024, ram_headroom_mib=16384, maximum=24)
    assert result['ramRequiredMiB'] == 32768 and not result['admit']


def test_resume_wrapper_preserves_plan_and_records_operational_guard(tmp_path, monkeypatch):
    import sys
    from scripts import resume_basil_sharp_resilient as wrapper
    from scripts import run_basil_sharp_thesis as runner
    from basil_core import basil_sharp_thesis as study
    source = study.source()
    plan = tmp_path / 'plan.json';plan.write_bytes(b'original frozen plan\n')
    called = []
    original = runner.adaptive.residents_of
    def schedule(*args):
        called.append(runner.adaptive.residents_of)
        return 0
    monkeypatch.setattr(runner.adaptive, 'schedule', schedule)
    monkeypatch.setattr(runner, 'main', lambda: runner.adaptive.schedule(
        SimpleNamespace(output=tmp_path), {}, {}, [], 'original digest', {}))
    monkeypatch.setattr(sys, 'argv', ['resume_basil_sharp_resilient.py', '--resume', '--execute'])
    assert wrapper.main() == 0 and called == [residents_of]
    assert runner.adaptive.residents_of is original
    assert plan.read_bytes() == b'original frozen plan\n'
    assert study.source() == source
    assert len(list((tmp_path / 'operational_adapters').glob('*.json'))) == 1
