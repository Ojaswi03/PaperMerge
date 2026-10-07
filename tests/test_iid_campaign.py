"""Preparation contracts: no training, workers or research queue execution."""
import copy
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from basil_core.iid_campaign import (
    CAMPAIGN_ID, CONFIG_ROOT, QUEUE_PATH, CONDITIONS, campaign_configs,
    verify_campaign, gui_approved,
)
from gui.services import ConfigService, QueueService
from gui.state import ApplicationState, ExecutionState, QueueStatus, Workspace
from gui.state.experiment_state import from_persisted, to_persisted, validate_experiment

PARTITION_HASH='db79edd6b35d709b0a40fcc9ca5b5253e1da52a414cfba30a87de6aff4339b29'
INITIAL_HASH='7c717dbbd9bd1f2f230b1ff3d288bd8fec33dcd3ec730eb43e8a14e496cdc2df'


@pytest.fixture
def configs():
    return campaign_configs(PARTITION_HASH, INITIAL_HASH)


def test_four_conditions_share_fixed_protocol_and_exact_contrasts(configs):
    differences=verify_campaign(configs)
    assert differences['C/D']==['ebmMode','localObjective']
    assert differences['B/C']==['channelNoiseSigmaAbsolute','useChannelNoise']
    assert differences['A/B']==['actualAttackerCount','attackHidden','attackerCount','attackerIds','resolvedAttackerIds']
    for c in configs:
        assert gui_approved(c) and c['researchValid']
        assert c['experimentProtocol']=='sequential_basil_iid_v1'
        assert c['nRounds']==100 and c['localEpochs']==5 and c['batchSize']==512
        assert c['assumedByzantineCount']==4 and c['basilMemorySize']==5
        assert c['snapshotSelection'] and c['aggregationMode']=='strict_sequential_handoff'
        assert c['executionStartMode']=='fresh'
        assert c['expectedPartitionHash']==PARTITION_HASH
        assert c['expectedInitialModelHash']==INITIAL_HASH
        assert c['snapshotAnchorMu']==c['snapshotAnchorMuMax']==c['monteCarloNoiseSamples']==0
    assert configs[0]['resolvedAttackerIds']==[]
    assert all(c['resolvedAttackerIds']==[0,1,5,7] for c in configs[1:])
    assert all(c['attackStartRound']==20 for c in configs)
    assert configs[2]['useChannelNoise'] and configs[2]['ebmMode']=='none'
    assert configs[3]['localObjective']=='source_ebm'
    assert configs[2]['channelNoiseSigmaAbsolute']**2==.0001


@pytest.mark.parametrize('field,value',[
    ('nRounds',50),('localEpochs',3),('basilMemorySize',1),('snapshotSelection',False),
    ('partitionStrategy','one_class_per_node'),('executionStartMode','resume'),
    ('channelNoiseSigmaAbsolute',.0001),('localObjective','snapshot_anchor_only'),
    ('executionMode','multiprocessing'),('researchValid',False),('resultDirectory','experiments/results4/test'),
])
def test_campaign_refuses_protocol_or_routing_mutations(configs,field,value):
    changed=dict(configs[3],**{field:value})
    assert not gui_approved(changed)


def test_unpaired_seed_and_order_are_rejected(configs):
    changed=copy.deepcopy(configs);changed[2]['unexpectedShuffleSeed']=18
    with pytest.raises(ValueError,match='Unexpected scientific differences'):verify_campaign(changed)
    with pytest.raises(ValueError,match='Queue order'):verify_campaign(configs[::-1])


def test_gui_config_load_save_and_idle_queue_roundtrip(configs,tmp_path):
    service=ConfigService()
    for c in configs:
        cfg=from_persisted(c)
        assert not validate_experiment(cfg)
        assert to_persisted(cfg)==c
        path=tmp_path/f"{c['conditionId']}.json"
        service.save(path,cfg)
        assert to_persisted(service.load(path))==c
        state=ApplicationState(experiment=cfg)
        assert state.execution is ExecutionState.IDLE and state.worker_count==0 and state.can_run
    path=tmp_path/'queue.json';path.write_text(json.dumps(configs))
    queue_service=QueueService(path);queue=queue_service.load()
    assert [e.config.extra['conditionId'] for e in queue.entries]==list('ABCD')
    assert all(e.status is QueueStatus.PENDING for e in queue.entries)
    queue.entries[0].status=QueueStatus.RUNNING
    queue.entries[1].status=QueueStatus.FAILED
    queue_service.save(queue);loaded=queue_service.load()
    assert loaded.entries[0].status is QueueStatus.STOPPED
    assert loaded.entries[1].status is QueueStatus.FAILED
    assert loaded.entries[2].status is QueueStatus.PENDING


def test_prepared_files_and_preset_match():
    prepared=json.loads(QUEUE_PATH.read_text())
    verify_campaign(prepared)
    for c in prepared:
        assert json.loads((CONFIG_ROOT/f"{c['experimentName']}.json").read_text())==c
    live=QueueService(QUEUE_PATH).load()
    assert [e.config.extra['conditionId'] for e in live.entries]==list('ABCD')
    assert all(e.status is QueueStatus.PENDING for e in live.entries)


def test_condition_cli_defaults_to_validation_only(configs,tmp_path,capsys,monkeypatch):
    from scripts.run_iid_condition import main
    def forbidden(*args,**kwargs):raise AssertionError('Preparation must never start a process')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    path=tmp_path/'D.json';path.write_text(json.dumps(configs[3]))
    main(['--config',str(path)])
    assert json.loads(capsys.readouterr().out)==dict(status='valid',condition='D',rounds=100,executionStarted=False,startMode='fresh')


def test_no_lambda_and_no_ebm_leakage_without_optimizer_steps():
    import tensorflow as tf
    from basil_core.iid_study import IidWorker
    model=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=13))])
    worker=IidWorker(model);initial=worker.export()
    x=tf.ones([2,32,32,3])*.2;y=tf.constant([0,1]);key=tf.constant([1,2]);out=[]
    for legacy in (1.,999999.):
        grads,values=worker._gradients(x,y,key,'gradient_norm_objective','paper_absolute_gaussian',tf.constant(.01),tf.constant(legacy))
        # Calculate the intended SGD update algebraically; never execute training.
        out.append(([g.numpy() for g in grads],[float(v) for v in values],
                    [p-.05*g.numpy() for p,g in zip(initial,grads)]))
    assert out[0][1]==out[1][1]
    assert out[0][1][3]==pytest.approx(.0001,rel=1e-6)
    for a,b in zip(out[0][0]+out[0][2],out[1][0]+out[1][2]):np.testing.assert_array_equal(a,b)
    clean,ce_values=worker._gradients(x,y,key,'none','paper_absolute_gaussian',tf.constant(.01),tf.constant(999999.))
    assert float(ce_values[3])==float(ce_values[6])==0
    assert float(ce_values[0])==float(ce_values[2])
    for a,b in zip(initial,worker.export()):np.testing.assert_array_equal(a,b)
    assert worker.optimizer is None


def test_duplicate_run_and_empty_queue_do_not_start(monkeypatch):
    from gui.app import PaperMergeApp
    starts=[]
    app=SimpleNamespace(state=SimpleNamespace(can_run=False),_start_entry=starts.append,_queue_running=False)
    PaperMergeApp.run(app)
    app.state=SimpleNamespace(can_run=True,workspace=Workspace.QUEUE)
    app.queue_state=SimpleNamespace(entries=[])
    PaperMergeApp.run(app)
    assert starts==[]


def test_queue_chaining_is_cancelable_and_never_starts_during_poll(configs,tmp_path,monkeypatch):
    from gui.app import PaperMergeApp
    from gui.state import QueueState
    monkeypatch.setattr('gui.app.PROJECT_ROOT',tmp_path)
    queue=QueueState()
    for c in configs:queue.add(from_persisted(c))
    queue.entries[0].status=QueueStatus.RUNNING
    scheduled=[];starts=[]
    state=ApplicationState(experiment=queue.entries[0].config,execution=ExecutionState.COMPLETED)
    app=SimpleNamespace(state=state,queue_state=queue,_queue_running=True,_active_queue_entry=queue.entries[0],
        _worker_handle=None,execution=SimpleNamespace(dispatch_pending=lambda:None),
        queue_changed=lambda:None,root=SimpleNamespace(after=lambda delay,fn:scheduled.append((delay,fn))),
        _start_entry=starts.append,_poll=lambda:None)
    PaperMergeApp._poll(app)
    assert starts==[] and queue.entries[0].status is QueueStatus.COMPLETED
    assert scheduled[0][0]==100 and app._queue_running
    app._queue_running=False;scheduled[0][1]()
    assert starts==[]


def test_execution_reader_dispatches_events_without_starting_a_process(tmp_path):
    import io
    import queue
    from gui.services.iid_execution_service import IidExecutionHandle
    handle=IidExecutionHandle.__new__(IidExecutionHandle)
    sent=[];handle.sink=sent.append;handle.events=queue.Queue()
    handle.events.put(7)  # Non-event JSON must not break main-thread polling.
    handle.events.put({'round':0,'node':0})
    handle.events.put({'roundComplete':dict(round=0,averageAccuracy=.2,worstNodeAccuracy=.1)})
    handle._finished=False;handle._stopping=False;handle.config_path=tmp_path/'D.json'
    handle.pid_path=tmp_path/'D.pid.json';handle.pid_path.write_text('{}')
    handle.process=SimpleNamespace(poll=lambda:0,stdout=io.StringIO(),wait=lambda:0)
    handle.reader=SimpleNamespace(is_alive=lambda:False)
    handle.poll();handle.poll()
    assert [e.kind for e in sent].count('execution_completed')==1
    assert [e.payload['progress'] for e in sent if e.kind=='progress_updated']==[.001,.01]
    assert not handle.active and not handle.pid_path.exists()


@pytest.mark.skipif(not os.environ.get('DISPLAY'),reason='Optional GUI integration requires a display')
def test_real_gui_loads_four_idle_configs_and_presets_without_launch(monkeypatch):
    import tkinter as tk
    from basil_core.research_protocol import validate_config
    from gui.app import PaperMergeApp
    def forbidden(*args,**kwargs):raise AssertionError('GUI startup must never start research execution')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    monkeypatch.setattr(PaperMergeApp,'_launch',forbidden)
    monkeypatch.setattr(QueueService,'save',lambda *args:None)
    monkeypatch.setattr('gui.app.filedialog.askopenfilename',lambda **kwargs:str(QUEUE_PATH))
    monkeypatch.setattr('gui.app.messagebox.askyesno',lambda *args,**kwargs:True)
    root=tk.Tk()
    try:
        app=PaperMergeApp(root);root.update_idletasks();root.update()
        assert app.state.execution is ExecutionState.IDLE and app._worker_handle is None and not app._queue_running
        app.load_queue();root.update_idletasks();root.update()
        assert app.state.workspace is Workspace.QUEUE
        view=app.shell.views[Workspace.QUEUE]
        rows=[view.tree.item(i,'values') for i in view.tree.get_children()]
        assert len(rows)==4 and all(view.tree.set(i,'rounds')=='100' and view.tree.set(i,'status')=='Pending' for i in view.tree.get_children())
        for entry in app.queue_state.entries:
            assert view.tree.set(entry.entry_id,'done')==f'{entry.completed_rounds}/100'
            assert view.tree.set(entry.entry_id,'device')=='CPU'
        assert [row[1][0] for row in rows]==list('ABCD')
        assert app.shell.command_bar.run['text']=='Run Queue'
        assert str(app.shell.command_bar.run['state'])=='normal'
        assert not app._queue_running and app.state.worker_count==0
        builder=app.shell.views[Workspace.BUILDER]
        assert len(builder.iid_presets)==8
        for label in builder.iid_presets:
            builder.preset.set(label);builder._apply_preset()
            assert gui_approved(to_persisted(app.state.experiment))
            assert app.state.execution is ExecutionState.IDLE
        gpu_queue=QUEUE_PATH.with_name('iid_abcd_100r_gpu.json')
        monkeypatch.setattr('gui.app.filedialog.askopenfilename',lambda **kwargs:str(gpu_queue))
        app.load_queue();root.update_idletasks();root.update()
        assert not app._queue_running and app.state.worker_count==0
        assert [e.config.extra['conditionId'] for e in app.queue_state.entries]==list('ABCD')
        for entry in app.queue_state.entries:
            assert view.tree.set(entry.entry_id,'done')==f'{entry.completed_rounds}/100'
            assert view.tree.set(entry.entry_id,'device')=='GPU'
        assert 'GPU requested' in app.shell.command_bar.status.cget('text')
    finally:root.destroy()


def test_prespecified_convergence_windows_and_signed_contrasts():
    from reporting.iid_campaign_plots import convergence,scientific_contrasts
    curve=np.linspace(.1,.7,100)
    metrics={'fullTestAccuracy':curve,'worstNodeAccuracy':curve-.02}
    a=convergence(metrics,attack_applicable=False)
    assert a['mean_rounds_0_19']==pytest.approx(curve[:20].mean())
    assert a['mean_rounds_20_99']==pytest.approx(curve[20:].mean())
    assert a['late_round_mean']==pytest.approx(curve[90:100].mean())
    assert a['late_round_change']==pytest.approx(curve[90:].mean()-curve[80:90].mean())
    stats={letter:dict(a,final_accuracy=a['final_accuracy']+offset) for letter,offset in zip('ABCD',(0,-.1,-.2,-.3))}
    contrasts=scientific_contrasts(stats)
    assert contrasts['ebm_effect']['differences']['final_accuracy']==pytest.approx(-.1)
    assert contrasts['total_robustness_gap']['differences']['final_accuracy']==pytest.approx(-.3)
    assert 'not the cost of BASIL' in contrasts['attack_effect']['meaning']
    with pytest.raises(ValueError,match='100 actual finite rounds'):
        convergence(dict(metrics,fullTestAccuracy=curve[:50]),attack_applicable=True)


def test_campaign_plot_contract_uses_100_rounds_and_marks_only_attacked_conditions(configs,tmp_path,monkeypatch):
    from reporting import iid_study_plots as plots
    from reporting.iid_campaign_plots import comparison_plots
    calls=[]
    def capture(output,name,series,**kwargs):calls.append((name,series,kwargs))
    monkeypatch.setattr(plots,'plot',capture)
    monkeypatch.setattr('reporting.iid_campaign_plots.plot',capture)
    metrics=dict(fullTestAccuracy=np.linspace(.1,.5,100),averageAccuracy=np.linspace(.09,.49,100),
        worstNodeAccuracy=np.linspace(.08,.48,100),fullTestLoss=np.linspace(2.3,1.,100),
        fullTestPerClassAccuracy=np.full((100,10),.5),roundNodeAccuracy=np.full((100,10),.5))
    columns=('honest_selection_rate','byzantine_selection_rate','attacked_source_selection_rate','mean_snapshot_age',
        'mean_local_ce','ebm_penalty','ordinary_gradient_norm','ebm_correction_norm','ebm_correction_ratio',
        'channel_noise_norm','noise_to_model_norm_ratio')
    loaded={}
    for c in configs:
        source=tmp_path/c['conditionId'];source.mkdir()
        with (source/'round_metrics.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
            writer.writerows({k:.1 for k in columns} for _ in range(100))
        manifest=dict(config=c,nRounds=100,experimentName=c['experimentName'],attackStartRound=20,
            iidPartitionHash=PARTITION_HASH,initialModelHash=INITIAL_HASH)
        calls.clear();plots.generate_research_plots(source,tmp_path/'plots',manifest,metrics)
        names={name for name,_,_ in calls}
        assert {'full_test_accuracy_vs_round','mean_node_accuracy_vs_round','worst_node_accuracy_vs_round',
            'full_test_loss_vs_round','per_class_accuracy_vs_round','final_per_class_accuracy',
            'honest_snapshot_selection_rate_vs_round','selected_snapshot_age_vs_round'}<=names
        assert all(kw['attack_start']==(None if c['conditionId']=='A' else 20) for _,_,kw in calls)
        if c['conditionId'] in 'CD':assert {'channel_noise_norm_vs_round','noise_to_model_norm_ratio_vs_round'}<=names
        if c['conditionId']=='D':assert {'ebm_penalty_vs_round','ebm_correction_ratio_vs_round','ebm_correction_norm_vs_round','ordinary_gradient_norm_vs_round'}<=names
        if c['conditionId']=='C':assert 'ebm_penalty_vs_round' not in names
        loaded[c['conditionId']]=(manifest,metrics,[])
    calls.clear();comparison_plots(loaded,tmp_path/'comparison')
    assert {name for name,_,_ in calls}=={'ABCD_full_test_accuracy_comparison','ABCD_mean_node_accuracy_comparison',
        'ABCD_worst_node_accuracy_comparison','ABCD_final_per_class_comparison'}
    assert all(len(series)==4 for _,series,_ in calls)
    assert all(len(values)==100 for _,series,_ in calls[:3] for values in series.values())
