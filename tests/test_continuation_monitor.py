from copy import deepcopy
import json

import pytest

from basil_core import continuation_monitor as monitor


def setup(tmp_path,monkeypatch,status='RUNNING',elapsed=100):
    from basil_core import continuation_campaign
    rows=[dict(status=s,totalActivations=50000,completedActivations=n) for s,n in
        [('RUNNING',20000),('FAILED',10000),('PENDING',10000),('COMPLETED',50000)]]
    state=dict(status=status,elapsedSeconds=elapsed,completedActivations=90000,totalActivations=200000,
        completed=1,total=4,runs=rows)
    (tmp_path/'continuation_plan.json').write_text(json.dumps(dict(items=[dict(sourceCursor=10000)]*4)))
    monkeypatch.setattr(continuation_campaign,'public_snapshot',lambda _:deepcopy(state))
    return state


def test_failure_preserves_measured_speed_and_excludes_only_blocked_remaining_work(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch)
    result=monitor.snapshot(tmp_path)
    assert result['activationsPerSecond']==500 # inherited 40,000 excluded
    assert result['etaSeconds']==140 # 30,000 active + 40,000 pending
    assert result['untimedJobs']==1 and result['etaMode']=='activation_throughput_excludes_blocked'


def test_recovery_status_keeps_rate_but_does_not_promise_finish(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,status='SUPERVISOR_RECOVERY_NEEDED')
    result=monitor.snapshot(tmp_path)
    assert result['activationsPerSecond']==500 and result['etaSeconds'] is None


def test_zero_elapsed_cannot_create_infinite_rate(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,elapsed=0)
    result=monitor.snapshot(tmp_path)
    assert result['activationsPerSecond'] is None and result['etaSeconds'] is None


def test_overall_eta_accounts_for_slow_running_job(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch)
    state['runs'][0]['etaSeconds']=10000
    result=monitor.snapshot(tmp_path)
    assert result['activationsPerSecond']==500 and result['etaSeconds']==10000
    assert result['etaMode']=='throughput_and_running_jobs_excludes_blocked'


def test_failed_job_estimate_does_not_extend_runnable_eta(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch)
    state['runs'][0]['etaSeconds']=10
    state['runs'][1]['etaSeconds']=999999
    result=monitor.snapshot(tmp_path)
    assert result['etaSeconds']==140


def test_invalid_job_etas_never_leak_nonfinite_numbers(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch)
    state['runs'][0]['etaSeconds']=float('nan')
    result=monitor.snapshot(tmp_path)
    assert result['etaSeconds']==140


def test_completed_elapsed_stops_at_last_host_report(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch,status='COMPLETED',elapsed=99999)
    state.update(completed=4,completedActivations=200000)
    for row in state['runs']:row.update(status='COMPLETED',completedActivations=50000)
    folder=tmp_path/'hosts';folder.mkdir()
    (folder/'hal9k.json').write_text(json.dumps(dict(startedAt=1000,updatedAt=1200)))
    result=monitor.snapshot(tmp_path)
    assert result['elapsedSeconds']==200 and result['activationsPerSecond']==800
    assert result['etaSeconds']==0 and result['untimedJobs']==0
