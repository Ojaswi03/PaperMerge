"""SSH diagnostic activity uses measured timing, without GPU or research runs."""
import io
import json
import subprocess
import sys

import pytest

from basil_core.terminal.progress import DiagnosticProgress


def test_first_gpu_reports_activity_without_inventing_eta_or_training_progress():
    stream=io.StringIO();now=[0.];history={}
    with DiagnosticProgress('Physical GPU 0',('Libraries','CNN'),stream=stream,
            estimates=history,clock=lambda:now[0],interval=60.) as progress:
        now[0]=2.;progress.refresh()
        progress.run('CNN',lambda:None)
        now[0]=7.;progress.finish('passed')
    output=stream.getvalue()
    assert 'Physical GPU 0 | Stage 1/2: Libraries | running' in output
    assert 'elapsed 00:00:02' in output and 'ETA unavailable' in output
    assert 'Stage 2/2: CNN' in output and 'PASSED | elapsed 00:00:07' in output
    assert history=={'Libraries':[2.],'CNN':[5.]}
    assert '%' not in output and not progress.thread.is_alive()


def test_next_gpu_eta_uses_measured_matching_stages_and_handles_overrun():
    stream=io.StringIO();now=[0.]
    history={'Libraries':[2.],'CNN':[5.]}
    with DiagnosticProgress('Physical GPU 1',('Libraries','CNN'),stream=stream,
            estimates=history,clock=lambda:now[0],interval=60.) as progress:
        assert 'ETA ~00:00:07 for this GPU' in stream.getvalue()
        now[0]=2.;progress.run('CNN',lambda:None)
        now[0]=3.;progress.refresh()
        assert 'ETA ~00:00:04 for this GPU' in stream.getvalue()
        now[0]=8.;progress.refresh()
        assert 'ETA recalculating' in stream.getvalue()
        progress.finish('passed')


def test_missing_stage_timing_never_extrapolates_from_unrelated_checks():
    stream=io.StringIO()
    with DiagnosticProgress('GPU 2',('Import','EBM'),stream=stream,
            estimates={'Import':[1.]},interval=60.) as progress:
        assert 'ETA unavailable' in stream.getvalue()
        progress.finish('blocked')


@pytest.mark.parametrize('error,status',[(RuntimeError('synthetic failure'),'FAILED'),
                                       (KeyboardInterrupt(),'INTERRUPTED')])
def test_failed_or_interrupted_check_stops_heartbeat_without_learning_success_timing(error,status):
    stream=io.StringIO();history={}
    with pytest.raises(type(error)):
        with DiagnosticProgress('GPU 3',('Import','CNN'),stream=stream,
                estimates=history,interval=60.) as progress:
            def fail():raise error
            progress.run('CNN',fail)
    assert status in stream.getvalue()
    assert 'PASSED' not in stream.getvalue() and history=={}
    assert not progress.thread.is_alive()


def test_heartbeat_continues_during_a_real_blocking_subprocess():
    stream=io.StringIO()
    with DiagnosticProgress('GPU synthetic',('Discovery','Slow CPU fixture'),
            stream=stream,interval=.01) as progress:
        result=progress.run('Slow CPU fixture',lambda:subprocess.run(
            [sys.executable,'-c','import time; time.sleep(.15); print("fixture done")'],
            capture_output=True,text=True,check=True,timeout=5))
        progress.finish('passed')
    assert result.stdout.strip()=='fixture done'
    assert stream.getvalue().count('Stage 2/2: Slow CPU fixture | running')>=2
    assert not progress.thread.is_alive()


def test_activity_stays_on_stderr_and_final_stdout_remains_json(capsys):
    with DiagnosticProgress('GPU 1',('Probe',),interval=60.) as progress:
        progress.refresh();progress.finish('passed')
    print(json.dumps({'status':'passed'}))
    captured=capsys.readouterr()
    assert json.loads(captured.out)=={'status':'passed'}
    assert 'GPU 1' in captured.err and 'PASSED' in captured.err


@pytest.mark.parametrize('term,interactive',[('xterm-256color',True),('dumb',False)])
def test_ssh_rendering_and_plain_terminal_fallback(monkeypatch,term,interactive):
    monkeypatch.setenv('TERM',term)
    class Terminal(io.StringIO):
        def isatty(self):return True
    stream=Terminal()
    with DiagnosticProgress('GPU 0',('Check',),stream=stream,interval=60.) as progress:
        progress.refresh();progress.refresh();progress.finish('passed')
    assert ('\033[2K' in stream.getvalue()) is interactive
    if not interactive:assert '\r' not in stream.getvalue()


@pytest.mark.parametrize('interval',[0.,-1.,float('inf'),float('nan')])
def test_invalid_refresh_interval_is_rejected(interval):
    with pytest.raises(ValueError,match='finite and positive'):
        DiagnosticProgress('GPU 0',('Check',),interval=interval)


def test_success_requires_all_declared_stages():
    stream=io.StringIO()
    with pytest.raises(ValueError,match='final stage'):
        with DiagnosticProgress('GPU 0',('Import','CNN'),stream=stream,interval=60.) as progress:
            progress.finish('passed')
    assert 'PASSED' not in stream.getvalue() and not progress.thread.is_alive()


def test_progress_module_import_is_lazy():
    subprocess.run([sys.executable,'-c',
        'import sys; import basil_core.terminal.progress; '
        'assert not ({"tensorflow","numpy","tkinter","wandb"} & sys.modules.keys())'],
        check=True,capture_output=True,text=True)
