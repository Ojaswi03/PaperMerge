"""Combined queue dependencies; synthetic manifests, no training/cloud jobs."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def stages(monkeypatch,tmp_path):
    import basil_core.dirichlet_horizon_study as study
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'RECEIPT',tmp_path/'receipt.json')
    monkeypatch.setattr(study,'EVIDENCE',tmp_path/'evidence.json')
    def approval(cfg):
        study.validate_configuration(cfg)
        if cfg['nRounds']==500 and not study.RECEIPT.exists():raise ValueError('500 blocked')
    monkeypatch.setattr(study,'validate_evidence',approval)
    one=study.configurations('a'*64,'b'*64,'c'*64,100)
    five=study.configurations('a'*64,'b'*64,'c'*64,500)
    def completed(letter,*,status='completed',rounds=100,activations=1000):
        cfg=one[ord(letter)-ord('A')]
        path=tmp_path/cfg['resultDirectory']/'run.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(config=cfg,status=status,completedRounds=rounds,completedActivations=activations)))
    return study,one,five,completed


def test_loading_all_eight_does_not_unlock_500(stages):
    from gui.services.staged_queue_service import validate_queue_for_loading,dependency_state
    from gui.state.queue_state import QueueEntry
    from gui.state.experiment_state import from_persisted
    study,one,five,_=stages
    entries=[QueueEntry(from_persisted(c)) for c in one+five]
    validate_queue_for_loading(entries)
    assert all(dependency_state(c)['missing']==list('ABCD') for c in five)
    assert all(not study.gui_approved(c) for c in five)
    assert all(e.status.value=='pending' for e in entries)


def test_needs_every_100_round_condition_and_all_activations(stages):
    from gui.services.staged_queue_service import dependency_state
    _,_,five,completed=stages
    for letter in 'ABC':completed(letter)
    assert dependency_state(five[0])==dict(state='waiting',missing=['D'],prerequisite='dirichlet_prior_anchor_100r_v1')
    completed('D',status='running',rounds=99,activations=999)
    assert dependency_state(five[0])['state']=='waiting'
    completed('D')
    assert dependency_state(five[0])['state']=='needs_validation'


@pytest.mark.parametrize('failure',['failed','missing_activations'])
def test_failure_or_fake_completion_cannot_open_the_gate(stages,failure):
    from gui.services.staged_queue_service import dependency_state
    _,_,five,completed=stages
    for letter in 'ABCD':completed(letter)
    completed('C',status='failed' if failure=='failed' else 'completed',activations=999)
    with pytest.raises(ValueError):dependency_state(five[0])


def test_receipt_is_still_checked_by_real_runner_approval(stages,monkeypatch):
    from gui.services.staged_queue_service import dependency_state
    study,_,five,_=stages
    study.RECEIPT.write_text('{}')
    def stale(_):raise ValueError('stale receipt')
    monkeypatch.setattr(study,'validate_evidence',stale)
    with pytest.raises(ValueError,match='stale receipt'):dependency_state(five[0])


def test_duplicate_queue_conditions_rejected(stages):
    from gui.services.staged_queue_service import validate_queue_for_loading
    from gui.state.queue_state import QueueEntry
    from gui.state.experiment_state import from_persisted
    _,one,_,_=stages
    with pytest.raises(ValueError,match='Duplicate'):
        validate_queue_for_loading([QueueEntry(from_persisted(one[0])) for _ in range(2)])


def test_combined_totals_include_waiting_stage(stages):
    from gui.work_progress import queue_work
    from gui.state.queue_state import QueueEntry
    from gui.state.experiment_state import from_persisted
    _,one,five,_=stages
    totals=queue_work([QueueEntry(from_persisted(c)) for c in one+five])
    assert totals['totalRuns']==8 and totals['totalRounds']==2400
    assert totals['totalActivations']==24000 and totals['totalOptimizerSteps']==1224000
    assert totals['completedOptimizerSteps']==0


@pytest.mark.parametrize('outcome',['pass','fail','stopped','changed'])
def test_background_gate_never_starts_a_worker_before_success(monkeypatch,stages,outcome):
    import threading
    from gui.app import PaperMergeApp
    from gui.state.application_state import ApplicationState,ExecutionState
    from gui.state.queue_state import QueueState,QueueEntry
    from gui.state.experiment_state import from_persisted
    import gui.services.staged_queue_service as gate
    _,_,five,_=stages
    entry=QueueEntry(from_persisted(five[0]))
    app=PaperMergeApp.__new__(PaperMergeApp)
    app.state=ApplicationState(experiment=entry.config,execution=ExecutionState.COMPLETED)
    app.queue_state=QueueState([entry]);app._queue_running=True
    app._stage_gate_running=False;app._stage_gate_results=None
    callbacks=[];launches=[];audits=[]
    app.root=SimpleNamespace(after=lambda delay,callback:callbacks.append(callback))
    app.queue_changed=lambda:None
    app._start_entry=lambda pending:launches.append(pending)
    def inspect():
        audits.append(True)
        if outcome=='fail':raise ValueError('synthetic failed protocol audit')
    monkeypatch.setattr(gate,'validate_completed_100',inspect)
    class Thread:
        def __init__(self,target,**kwargs):self.target=target
        def start(self):self.target()
    monkeypatch.setattr(threading,'Thread',Thread)
    app._begin_stage_validation(entry,dict(five[0]))
    assert audits==[True] and not launches
    assert app.state.can_stop and not app.state.can_run
    if outcome=='stopped':app._queue_running=False
    if outcome=='changed':entry.config.experiment_name='edited while auditing'
    app._poll_stage_validation()
    assert not app.state.stage_validation_active
    assert not launches
    if outcome=='pass':
        assert len(callbacks)==1
        callbacks[0]();assert launches==[entry]
    else:
        assert callbacks==[]
        assert not app._queue_running


def test_incomplete_dependency_pauses_before_execution_start(monkeypatch,stages):
    from gui.app import PaperMergeApp
    from gui.state.application_state import ApplicationState,ExecutionState
    from gui.state.queue_state import QueueEntry
    from gui.state.experiment_state import from_persisted
    _,one,five,_=stages
    app=PaperMergeApp.__new__(PaperMergeApp)
    app.state=ApplicationState(experiment=from_persisted(one[0]),execution=ExecutionState.COMPLETED)
    app._queue_running=True
    app._start_entry(QueueEntry(from_persisted(five[0])))
    assert not app._queue_running
    assert 'waiting' in app.state.status.lower()


def test_staged_queue_import_is_lazy():
    subprocess.run([sys.executable,'-c','import sys;import gui.services.staged_queue_service;'
        'assert not ({"tensorflow","numpy","wandb","matplotlib"}&sys.modules.keys())'],check=True)


@pytest.mark.parametrize('failure',[False,True])
def test_audit_receipt_is_written_only_after_success(monkeypatch,stages,failure):
    from gui.services.staged_queue_service import validate_completed_100
    study,_,_,_=stages
    study.EVIDENCE.write_text(json.dumps(dict(partitionHash='a'*64,initialModelHash='b'*64)))
    monkeypatch.setattr(study,'validate_evidence',lambda cfg:None)
    monkeypatch.setattr(study,'execution_fingerprints',lambda cfg:{'science':'same'})
    def inspect(*args,**kwargs):
        assert kwargs['expected_rounds']==100
        if failure:raise ValueError('non-finite telemetry')
        return dict(status='protocol_and_numerical_checks_passed',manifests={c:c*64 for c in 'ABCD'})
    monkeypatch.setattr(study,'inspect_completed',inspect)
    if failure:
        with pytest.raises(ValueError):validate_completed_100()
        assert not study.RECEIPT.exists()
    else:
        validate_completed_100()
        assert json.loads(study.RECEIPT.read_text())['implementationSha256']=={'science':'same'}
