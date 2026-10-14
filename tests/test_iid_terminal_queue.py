"""Terminal orchestration contracts; mocked workers never train."""
import json
import os
import signal

import pytest

from basil_core.iid_campaign import campaign_configs
from gui.services.execution_service import ExecutionEvent
from gui.services.queue_service import QueueService
from gui.state.experiment_state import to_persisted
from gui.state.queue_state import QueueStatus
from scripts import run_iid_queue as runner


@pytest.fixture
def prepared(tmp_path):
    configs = campaign_configs('a'*64,'b'*64,device='GPU',rounds=500)
    path=tmp_path/'preset.json'
    path.write_text(json.dumps(configs))
    state,_,_=runner.load_prepared_queue(path,root=tmp_path)
    return state,configs,path


def manifest(root,config,status):
    path=root/config['resultDirectory']/'run.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(config=config,status=status)))


def test_dry_run_never_launches_or_writes(prepared,tmp_path,monkeypatch,capsys):
    _,_,path=prepared
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    original=runner.load_prepared_queue
    monkeypatch.setattr(runner,'load_prepared_queue',lambda path:original(path,root=tmp_path))
    monkeypatch.setattr('subprocess.Popen',lambda *a,**kw:pytest.fail('Worker launched during dry-run'))
    assert runner.main(['--queue',str(path)])==0
    assert 'research runs not started' in capsys.readouterr().out
    assert list(tmp_path.iterdir())==[path]


def test_completed_condition_skipped_then_serial_order(prepared,tmp_path):
    state,configs,_=prepared
    manifest(tmp_path,configs[0],'completed')
    calls=[]
    class Worker:
        def __init__(self,config,sink):
            self.config=config;self.sink=sink;self.active=True
            calls.append(config['conditionId'])
        def poll(self):
            manifest(tmp_path,self.config,'completed')
            self.sink(ExecutionEvent('execution_completed',{'returnCode':0}))
            self.active=False
        def stop(self):pytest.fail('Completed worker should not be stopped')
    assert runner.run_queue(state,configs,root=tmp_path,handle_factory=Worker)==0
    assert calls==list('BCD')
    assert json.loads((tmp_path/'gui/queue_state.json').read_text())==[]


@pytest.mark.parametrize('kind,code',[('execution_failed',1),('execution_stopped',130)])
def test_failure_or_stop_never_advances_queue(prepared,tmp_path,kind,code):
    state,configs,_=prepared
    calls=[]
    class Worker:
        def __init__(self,config,sink):
            calls.append(config['conditionId']);self.sink=sink;self.active=True
        def poll(self):
            self.sink(ExecutionEvent(kind,dict(returnCode=code,logPath='local.log')))
            self.active=False
        def stop(self):pass
    assert runner.run_queue(state,configs,root=tmp_path,handle_factory=Worker)==code
    assert calls==['A']
    saved=QueueService(tmp_path/'gui/queue_state.json').load()
    assert saved.entries[0].status.value==('failed' if code==1 else 'stopped')
    assert [to_persisted(e.config) for e in saved.entries]==configs


def test_ctrl_c_requests_graceful_stop_and_preserves_remaining_work(prepared,tmp_path):
    state,configs,_=prepared
    stopped=[]
    class Worker:
        def __init__(self,config,sink):self.sink=sink;self.active=True
        def poll(self):
            os.kill(os.getpid(),signal.SIGINT)
            self.sink(ExecutionEvent('execution_stopped',{'returnCode':130}))
            self.active=False
        def stop(self):stopped.append(True)
    original=signal.getsignal(signal.SIGINT)
    assert runner.run_queue(state,configs,root=tmp_path,handle_factory=Worker)==130
    assert stopped==[True]
    assert signal.getsignal(signal.SIGINT)==original
    assert len(QueueService(tmp_path/'gui/queue_state.json').load().entries)==4


def test_false_completion_is_rejected(prepared,tmp_path):
    state,configs,_=prepared
    class Worker:
        def __init__(self,config,sink):self.sink=sink;self.active=True
        def poll(self):
            self.sink(ExecutionEvent('execution_completed',{'returnCode':0}));self.active=False
        def stop(self):pass
    with pytest.raises(RuntimeError,match='no completed result manifest'):
        runner.run_queue(state,configs,root=tmp_path,handle_factory=Worker)


def test_queue_lock_and_active_marker_prevent_duplicates(tmp_path):
    with runner.queue_lease(tmp_path):
        assert (tmp_path/'gui/.queue_active.json').exists()
        with pytest.raises(ValueError,match='already running'):
            with runner.queue_lease(tmp_path):pytest.fail('Duplicate queue permitted')
    assert not (tmp_path/'gui/.queue_active.json').exists()
    marker=tmp_path/'gui/.queue_active.json'
    marker.write_text(json.dumps({'pid':os.getpid()}))
    with pytest.raises(ValueError,match='already active'):
        with runner.queue_lease(tmp_path):pytest.fail('Active GUI ignored')
    assert marker.exists()


def test_live_orphan_worker_is_not_restarted(tmp_path):
    workers=tmp_path/'gui/worker_state/iid';workers.mkdir(parents=True)
    (workers/'A.pid.json').write_text(json.dumps({'pid':os.getpid()}))
    with pytest.raises(ValueError,match='worker is still active'):
        with runner.queue_lease(tmp_path):pytest.fail('Orphan ignored')


def test_terminal_uses_same_checkpoint_recovery_launch_as_gui():
    source=(runner.ROOT/'gui/services/iid_execution_service.py').read_text()
    assert "if plan['mode']!='fresh'" in source
    assert "command.append('--recover')" in source


def test_resume_preparation_preserves_configs_results_and_a_first_order(prepared,tmp_path,monkeypatch):
    from scripts import prepare_iid_queue_resume as preparation
    state,configs,path=prepared
    state.entries[0].status=QueueStatus.STOPPED
    evidence=tmp_path/'saved_result.json';evidence.write_text('unchanged result')
    plans=[dict(mode='checkpoint' if index<2 else 'fresh',completedRounds=5 if index==0 else 2 if index==1 else 0,
        startActivation=55 if index==0 else 29 if index==1 else 0) for index in range(4)]
    monkeypatch.setattr(preparation,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'PRESET',path)
    monkeypatch.setattr(runner,'load_prepared_queue',lambda *args,**kwargs:(state,configs,plans))
    monkeypatch.setattr(QueueService,'describe_recovery',lambda *args:None)
    preparation.main()
    for target in ('gui/queue_state.json','gui/queues/iid_abcd_500r_gpu_resume.json'):
        saved=QueueService(tmp_path/target).load()
        assert [to_persisted(e.config) for e in saved.entries]==configs
        assert all(e.status.value=='pending' and e.error=='' for e in saved.entries)
    assert evidence.read_text()=='unchanged result'


def test_worker_session_isolated_and_stop_targets_worker_only(prepared,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import gui.services.iid_execution_service as execution
    _,configs,_=prepared
    calls=[];signals=[]
    process=SimpleNamespace(pid=12345,poll=lambda:None,send_signal=signals.append)
    monkeypatch.setattr(execution,'WORKER_ROOT',tmp_path/'workers')
    monkeypatch.setattr(execution,'recovery_plan',lambda *args:dict(mode='checkpoint'))
    monkeypatch.setattr(execution.subprocess,'Popen',lambda *args,**kwargs:(calls.append((args,kwargs)) or process))
    monkeypatch.setattr(execution.threading,'Thread',lambda **kwargs:SimpleNamespace(start=lambda:None))
    worker=execution.IidExecutionHandle(configs[0],lambda event:None)
    assert calls[0][1]['start_new_session'] is True
    assert '--recover' in calls[0][0][0]
    worker.stop()
    assert signals==[signal.SIGTERM]


def test_explicit_fresh_reset_preserves_local_backup_and_leaves_queue_idle(prepared,tmp_path):
    from scripts.reset_iid_queue import reset
    _,configs,path=prepared
    file=tmp_path/configs[0]['resultDirectory']/'old_result.txt'
    file.parent.mkdir(parents=True);file.write_text('recoverable partial evidence')
    unrelated=tmp_path/'newResults/nonIID/untouched.txt'
    unrelated.parent.mkdir(parents=True);unrelated.write_text('do not reset')
    evidence=reset(tmp_path,path)
    assert (tmp_path/evidence['backup']/file.relative_to(tmp_path)).read_text()=='recoverable partial evidence'
    assert file.parent.exists() and not list(file.parent.iterdir())
    assert unrelated.read_text()=='do not reset'
    queue=QueueService(tmp_path/'gui/queue_state.json').load()
    assert [to_persisted(e.config) for e in queue.entries]==configs
    assert all(e.status.value=='pending' for e in queue.entries)
    settings=json.loads((tmp_path/'gui/worker_state/live_tracking.json').read_text())
    assert settings['group']==evidence['trackingGroup'] and 'preparedRuns' not in settings


def test_fresh_reset_refuses_symlink_targets(prepared,tmp_path):
    from scripts.reset_iid_queue import reset
    _,configs,path=prepared
    target=tmp_path/configs[0]['resultDirectory'];target.parent.mkdir(parents=True)
    elsewhere=tmp_path/'keep';elsewhere.mkdir();target.symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):reset(tmp_path,path)
    assert target.is_symlink()
