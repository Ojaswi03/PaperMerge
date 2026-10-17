"""No research runs or cloud connections: fixed-horizon configuration contracts."""
import json
import os
import subprocess
import sys

import pytest


def configs(rounds=100):
    from basil_core.dirichlet_horizon_study import configurations
    return configurations('a'*64,'b'*64,'c'*64,rounds)


@pytest.mark.parametrize('rounds',[100,500])
def test_horizon_science_and_pairing_unchanged(rounds):
    from basil_core.dirichlet_horizon_study import validate_configuration,display_label
    from basil_core.dirichlet_study import verify_pairing
    from basil_core.research_protocol import validate_config
    cfgs=configs(rounds)
    assert verify_pairing(cfgs)['C/D']==['ebmMode','localObjective']
    for cfg in cfgs:
        validate_configuration(cfg);validate_config(cfg)
        assert cfg['nRounds']==rounds and cfg['localEpochs']==5 and cfg['batchSize']==512
        assert cfg['researchValid'] and cfg['totalOptimizerSteps']==rounds*510
        assert cfg['liveTrackingEnabled']==(rounds==500)
        assert f'{rounds} rounds GPU' in display_label(cfg)
        assert cfg['snapshotAnchorMu']==.01 and cfg['basilMemorySize']==5
    assert cfgs[0]['resolvedAttackerIds']==[]
    assert all(c['resolvedAttackerIds']==[0,1,5,7] for c in cfgs[1:])


@pytest.mark.parametrize('key,value',[
    ('nRounds',101),('snapshotAnchorMu',.02),('localEpochs',1),('batchSize',256),
    ('seed',13),('learningRate',.01),('channelNoiseSigmaAbsolute',.02),
    ('ebmMode','legacy_gradient_scale'),('logitCalibrationTau',2.),('liveTrackingEnabled',True),
    ('resultDirectory','newResults/IID/A'),('optimizerStepsPerNode',[50]*10)])
def test_horizon_rejects_unapproved_changes(key,value):
    from basil_core.dirichlet_horizon_study import validate_configuration
    with pytest.raises(ValueError):validate_configuration(dict(configs()[3],**{key:value}))


@pytest.mark.parametrize('module',['basil_core.dirichlet_horizon_study','scripts.prepare_dirichlet_horizon_queues',
    'scripts.run_dirichlet_horizon_condition','gui.work_progress'])
def test_horizon_imports_are_lazy(module):
    subprocess.run([sys.executable,'-c',f'import importlib,sys;importlib.import_module({module!r});'
        'assert not ({"tensorflow","numpy","matplotlib","wandb"} & sys.modules.keys())'],check=True)


def test_dry_run_is_zero_training_and_500_blocked(tmp_path):
    file=tmp_path/'D.json';file.write_text(json.dumps(configs(500)[3]))
    proc=subprocess.run([sys.executable,'scripts/run_dirichlet_horizon_condition.py','--config',str(file)],
        capture_output=True,text=True,check=True)
    data=json.loads(proc.stdout)
    assert not data['trainingStarted'] and not data['executionReady']
    assert data['rounds']==500 and data['wandbEnabled']


def test_500_requires_all_100_rounds_without_an_accuracy_gate(monkeypatch,tmp_path):
    import basil_core.dirichlet_horizon_study as study
    root=tmp_path/'evidence.json';receipt=tmp_path/'receipt.json'
    monkeypatch.setattr(study,'EVIDENCE',root);monkeypatch.setattr(study,'RECEIPT',receipt)
    monkeypatch.setattr(study,'execution_fingerprints',lambda _:{'source':'digest'})
    evidence=dict(status='protocol_and_numerical_checks_passed',partitionHash='a'*64,initialModelHash='b'*64,
        implementationSha256={'source':'digest'},manifests={})
    root.write_text(json.dumps(evidence));digest=study.checksum(root)
    cfg=study.configurations('a'*64,'b'*64,digest,500)[0]
    with pytest.raises(ValueError,match='ALL A/B/C/D'):study.validate_evidence(cfg)
    receipt.write_text(json.dumps(dict(status='protocol_and_numerical_checks_passed',baseEvidenceSha256=digest,
        implementationSha256={'source':'digest'},manifests={})))
    with pytest.raises(ValueError,match='All four'):study.validate_evidence(cfg)


def test_tracking_does_not_reuse_iid_settings_or_ids():
    from basil_core.dirichlet_horizon_study import tracking_settings
    assert tracking_settings(configs()[0])=={'enabled':False}
    settings=tracking_settings(configs(500)[3])
    assert settings['group']=='dirichlet_prior_anchor_500r_v1'
    assert settings['preparedRuns']=={} and settings['project']=='Research Paper'


def test_exact_work_units_weight_unequal_nodes():
    from gui.work_progress import completed_work
    cfg=configs()[0]
    first=completed_work(cfg,1)
    assert first['completedOptimizerSteps']==30 and first['totalOptimizerSteps']==51000
    assert first['progress']==pytest.approx(30/51000)
    assert first['progress']!=1/1000
    fifth=completed_work(cfg,5)
    assert fifth['completedOptimizerSteps']==230
    assert completed_work(cfg,10)['completedOptimizerSteps']==510
    assert completed_work(cfg,1000)['progress']==1.
    for count in (-1,1001):
        with pytest.raises(ValueError):completed_work(cfg,count)


def test_restored_work_uses_checkpoint_cursor_not_old_elapsed(monkeypatch):
    from gui.services.iid_execution_service import IidExecutionHandle
    from gui.services.execution_service import ExecutionService
    from gui.state.application_state import ApplicationState,ExecutionState
    cfg=configs()[0];state=ApplicationState(execution=ExecutionState.RUNNING)
    service=ExecutionService(state)
    handle=IidExecutionHandle.__new__(IidExecutionHandle);handle.config=cfg;handle.sink=service.post
    handle._dispatch({'restored':dict(progress=.001,round=0,priorRuntimeSeconds=100.,startProgress=.001,
        accuracyHistory=[],activationHistory=[(.1,.2)],recoveryTotal=0,recoveryCompleted=0)})
    service.dispatch_pending()
    assert state.completed_activations==1 and state.completed_optimizer_steps==30
    assert state.session_start_progress==pytest.approx(30/51000)
    assert state.experiment_eta_seconds is None
    handle._dispatch(dict(round=0,node=1,accuracy=.3))
    service.dispatch_pending()
    assert state.completed_activations==2 and state.completed_optimizer_steps==75
    assert state.progress==pytest.approx(75/51000)


def test_gui_horizon_can_run_only_after_approval(monkeypatch):
    import basil_core.dirichlet_horizon_study as study
    from gui.state.application_state import ApplicationState
    from gui.state.experiment_state import from_persisted
    monkeypatch.setattr(study,'validate_evidence',lambda cfg:None)
    state=ApplicationState(experiment=from_persisted(configs()[0]))
    assert state.can_run


def test_new_run_resets_work_counts():
    from gui.state.application_state import ApplicationState,ExecutionState
    from gui.state.experiment_state import from_persisted
    state=ApplicationState(experiment=from_persisted(configs()[0]),execution=ExecutionState.COMPLETED,
        completed_activations=1000,completed_optimizer_steps=51000)
    state.transition(ExecutionState.PREPARING)
    assert state.completed_activations==state.completed_optimizer_steps==0
    assert state.total_activations==1000 and state.total_optimizer_steps==51000


def test_total_queue_counts_include_completed_current_and_pending():
    from gui.state.queue_state import QueueEntry,QueueStatus
    from gui.state.experiment_state import from_persisted
    from gui.state.application_state import ApplicationState,ExecutionState
    from gui.work_progress import queue_work,queue_eta_seconds
    entries=[QueueEntry(from_persisted(c)) for c in configs()]
    entries[0].status=QueueStatus.COMPLETED
    entries[1].status=QueueStatus.RUNNING
    state=ApplicationState(execution=ExecutionState.RUNNING,completed_rounds=1,
        completed_activations=11,completed_optimizer_steps=540,progress=540/51000,
        elapsed_seconds=54.)
    for entry in entries:entry.estimated_remaining_seconds=5100.
    work=queue_work(entries,state)
    assert work['completedRuns']==1 and work['totalRuns']==4
    assert work['completedRounds']==101 and work['totalRounds']==400
    assert work['completedActivations']==1011 and work['totalActivations']==4000
    assert work['completedOptimizerSteps']==51540 and work['totalOptimizerSteps']==204000
    assert queue_eta_seconds(entries,state)==pytest.approx(5046+5100+5100)


def test_completed_horizon_entries_remain_in_saved_queue(tmp_path):
    from gui.services.queue_service import QueueService
    from gui.state.queue_state import QueueState,QueueEntry,QueueStatus
    from gui.state.experiment_state import from_persisted
    entry=QueueEntry(from_persisted(configs()[0]),status=QueueStatus.COMPLETED)
    service=QueueService(tmp_path/'queue.json');service.save(QueueState([entry]))
    assert service.load().entries[0].status is QueueStatus.COMPLETED


@pytest.mark.parametrize('rounds',[100,500])
def test_completed_display_needs_no_execution_approval_or_checkpoint(monkeypatch,tmp_path,rounds):
    from tests.test_round_accuracy import make_run
    from gui.services.queue_service import QueueService
    from gui.state.queue_state import QueueState,QueueEntry,QueueStatus
    from gui.state.experiment_state import from_persisted
    from gui.state.application_state import ApplicationState
    import basil_core.dirichlet_horizon_study as study
    cfg=configs(rounds)[0]
    directory=make_run(tmp_path,horizon=rounds)
    (directory/'run.json').write_text(json.dumps(dict(status='completed',config=cfg,
        completedRounds=rounds,completedActivations=rounds*10)))
    approvals=[]
    def stale(_):
        approvals.append(True)
        raise ValueError('Source approval is stale')
    monkeypatch.setattr(study,'validate_evidence',stale)
    entry=QueueEntry(from_persisted(cfg))
    QueueService.describe_recovery(QueueState([entry]),tmp_path)
    assert not approvals and entry.status is QueueStatus.COMPLETED
    assert entry.completed_rounds==rounds and entry.completed_activations==rounds*10
    assert entry.completed_optimizer_steps==entry.total_optimizer_steps==rounds*510
    assert entry.estimated_remaining_seconds==0.
    assert entry.execution_hint.startswith('Recorded completed')
    assert not ApplicationState(experiment=from_persisted(cfg)).can_run


@pytest.mark.parametrize('broken',['counter','config','csv'])
def test_completed_display_refuses_inconsistent_published_results(monkeypatch,tmp_path,broken):
    from tests.test_round_accuracy import make_run
    from gui.services.queue_service import QueueService
    from gui.state.queue_state import QueueState,QueueEntry,QueueStatus
    from gui.state.experiment_state import from_persisted
    cfg=configs(500)[0]
    directory=make_run(tmp_path,horizon=500)
    manifest=dict(status='completed',config=cfg,completedRounds=500,completedActivations=5000)
    if broken=='counter':manifest['completedActivations']=4999
    if broken=='config':manifest['config']=dict(cfg,snapshotAnchorMu=.02)
    if broken=='csv':(directory/'round_metrics.csv').write_text('round,mean_node_accuracy\n')
    (directory/'run.json').write_text(json.dumps(manifest))
    entry=QueueEntry(from_persisted(cfg))
    QueueService.describe_recovery(QueueState([entry]),tmp_path)
    assert entry.status is QueueStatus.PENDING
    assert entry.execution_hint.startswith('Recovery blocked:')


def test_eta_learns_actual_unequal_node_costs_and_ignores_startup():
    from gui.state.application_state import ApplicationState,ExecutionState
    from gui.state.experiment_state import from_persisted
    from gui.work_progress import completed_work
    cfg=configs()[0]
    state=ApplicationState(experiment=from_persisted(cfg),execution=ExecutionState.RUNNING,total_activations=1000)
    elapsed=200. # Deliberately large startup, not a recurring node cost.
    for activation in range(1,12):
        if activation>1:elapsed+=(activation-1)%10+1
        state.elapsed_seconds=elapsed;state.completed_activations=activation
        state.progress=completed_work(cfg,activation)['progress'];state.observe_completed_work()
    # Cursor=11: 98 full cycles plus nodes 1..9, with node costs 1..10 sec.
    assert state.experiment_eta_seconds==pytest.approx(98*55+54)
    state.elapsed_seconds+=2
    assert state.experiment_eta_seconds==pytest.approx(98*55+52)


@pytest.mark.skipif(not os.environ.get('DISPLAY'),reason='GUI display required')
def test_long_horizon_dashboard_keeps_plot_and_both_counters_visible(monkeypatch,tmp_path):
    import tkinter as tk
    from gui.views.application_shell import ApplicationShell
    from gui.services.result_service import ResultService
    from gui.state.application_state import ApplicationState,ExecutionState
    from gui.state.queue_state import QueueState,QueueEntry,QueueStatus
    from gui.state.experiment_state import from_persisted
    from gui.theme import apply_theme
    import basil_core.dirichlet_horizon_study as study
    monkeypatch.setattr(study,'validate_evidence',lambda _:None)
    def forbidden(*args,**kwargs):raise AssertionError('UI test may not start workers')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    cfgs=configs()
    queue=QueueState([QueueEntry(from_persisted(c)) for c in cfgs])
    queue.entries[0].status=QueueStatus.RUNNING
    state=ApplicationState(experiment=from_persisted(cfgs[0]),execution=ExecutionState.RUNNING,
        active_experiment_name=study.display_label(cfgs[0]),active_experiment_rounds=100,
        current_round=2,completed_rounds=1,completed_activations=11,total_activations=1000,
        completed_optimizer_steps=540,total_optimizer_steps=51000,progress=540/51000)
    root=tk.Tk()
    try:
        root.geometry('1440x900');root.rowconfigure(0,weight=1);root.columnconfigure(0,weight=1);apply_theme(root)
        commands={name:lambda *args:None for name in ('save','load','queue','run','stop','queue_changed','library','plot','load_queue')}
        shell=ApplicationShell(root,state,queue,ResultService(tmp_path),commands)
        root.update_idletasks();root.update()
        dashboard=shell.views[state.workspace]
        assert dashboard.chart.winfo_height()>100
        assert '540/51,000' in dashboard.work_counts.cget('text')
        assert '540/204,000' in dashboard.queue_counts.cget('text')
        assert '1/400' in dashboard.queue_counts.cget('text')
    finally:root.destroy()
