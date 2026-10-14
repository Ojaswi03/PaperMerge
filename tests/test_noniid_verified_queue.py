"""Scientific pairing, atomic recovery and GUI routing; no research SGD."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from basil_core import noniid_verified_study as protocol
from scripts import run_noniid_verified_condition as worker
from gui.services.queue_service import QueueService
from gui.state.experiment_state import from_persisted,to_persisted,validate_experiment


@pytest.fixture
def configs():return protocol.configurations('a'*64,'b'*64)


def test_four_conditions_are_paired_and_gui_valid(configs):
    assert protocol.verify_pairing(configs)['C/D']==['ebmMode','localObjective']
    for cfg in configs:
        protocol.validate_configuration(cfg)
        state=from_persisted(cfg)
        assert not validate_experiment(state) and to_persisted(state)==cfg
        assert (state.rounds,state.local_epochs,state.batch_size,state.basil_memory_size)==(30,5,512,5)
        assert cfg['snapshotAnchorMu']==.01 and not cfg['liveTrackingEnabled']
        assert cfg['gradientClipNorm'] is None and cfg['momentum']==0 and cfg['weightDecayCoefficient']==0
        assert not cfg['attackGaussian'] and cfg['experimentProtocol']==protocol.PROTOCOL
    assert configs[0]['resolvedAttackerIds']==[]
    assert all(c['resolvedAttackerIds']==[0,1,5,7] for c in configs[1:])
    assert all(c['ebmMode']=='none' for c in configs[:3])
    assert configs[3]['ebmMode']=='gradient_norm_objective'
    assert configs[2]['channelNoiseSigmaAbsolute']**2 == .0001


@pytest.mark.parametrize('key,value',[
    ('nRounds',31),('localEpochs',1),('partitionStrategy','iid'),
    ('channelNoiseSemantics','relative_l2_gaussian'),('gradientClipNorm',1.),
    ('liveTrackingEnabled',True),('snapshotAnchorMu',.1),('helperFaults',{}),
    ('resultDirectory','newResults/IID/wrong')])
def test_fixed_config_rejects_protocol_changes(configs,key,value):
    bad=dict(configs[3],**{key:value})
    with pytest.raises(ValueError):protocol.validate_configuration(bad)
    assert validate_experiment(from_persisted(bad))


def test_delayed_helper_faults_and_actually_corrupted_definition(configs):
    cfg=configs[1]
    assert worker.helper_faults(cfg,0)==worker.helper_faults(cfg,19)=={}
    assert set(worker.helper_faults(cfg,20))=={0,1,5,7}
    assert worker.helper_faults(configs[0],29)=={}
    warmup=worker.selected_telemetry(cfg,1,20,dict(sender=0,round=19,hash='h',records=[]))
    active=worker.selected_telemetry(cfg,1,20,dict(sender=0,round=20,hash='h',records=[]))
    assert warmup['selectedAttackerDesignated'] and not warmup['selectedActuallyCorrupted']
    assert active['selectedActuallyCorrupted']


def test_default_worker_dry_run_is_lazy_and_never_trains(tmp_path,configs):
    path=tmp_path/'config.json';path.write_text(json.dumps(configs[0]))
    code=('import sys; from scripts import run_noniid_verified_condition as s; '
        'assert not ({"tensorflow","numpy","matplotlib","wandb"}&sys.modules.keys()); '
        f'from pathlib import Path; s.ROOT=Path({str(tmp_path)!r}); '
        f's.main(["--config",{str(path)!r}])')
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['trainingRunsStarted']==0


def checkpoint_setup(tmp_path,config):
    state=dict(config=config,sourceHashes=protocol.source_hashes(),status='stopped',roundMetrics=[])
    runner=SimpleNamespace(round=0,node=0,step=3,save=lambda p:worker.atomic_json(p,{'test':True,'step':runner.step}))
    worker.commit(runner,state,tmp_path)
    return runner,state


def test_two_slot_commit_survives_pointer_failure(tmp_path,configs,monkeypatch):
    runner,state=checkpoint_setup(tmp_path,configs[0]);original=copy.deepcopy(state)
    original_pointer=(tmp_path/'progress.json').read_bytes()
    write=worker.atomic_json
    def fail_pointer(path,value):
        if Path(path).name=='progress.json':raise OSError('injected pointer failure')
        write(path,value)
    monkeypatch.setattr(worker,'atomic_json',fail_pointer)
    runner.step=4
    with pytest.raises(OSError):worker.commit(runner,state,tmp_path)
    assert state==original and (tmp_path/'progress.json').read_bytes()==original_pointer
    plan=protocol.recovery_plan(tmp_path,configs[0])
    assert plan['mode']=='checkpoint' and plan['step']==3
    assert (tmp_path/'checkpoint_1.json').exists()


def test_corrupt_checkpoint_and_source_changes_fail_closed(tmp_path,configs,monkeypatch):
    _,state=checkpoint_setup(tmp_path,configs[0])
    checkpoint=tmp_path/state['checkpoint'];old=checkpoint.read_bytes();checkpoint.write_text('corrupt')
    with pytest.raises(ValueError,match='corrupt'):protocol.recovery_plan(tmp_path,configs[0])
    checkpoint.write_bytes(old)
    monkeypatch.setattr(protocol,'source_hashes',lambda:{'changed':'code'})
    with pytest.raises(ValueError,match='source mismatch'):protocol.recovery_plan(tmp_path,configs[0])


def test_queue_status_round_trip_and_fresh_hint(tmp_path,configs):
    from gui.state.queue_state import QueueStatus
    path=tmp_path/'queue.json';path.write_text(json.dumps(configs))
    service=QueueService(path);state=service.load()
    QueueService.describe_recovery(state,tmp_path)
    assert len(state.entries)==4 and all(e.execution_hint=='Fresh R0→R29 — GPU' for e in state.entries)
    state.entries[0].status=QueueStatus.STOPPED;service.save(state)
    restored=service.load()
    assert restored.entries[0].status is QueueStatus.STOPPED
    assert [to_persisted(e.config) for e in restored.entries]==configs


def test_noniid_handle_launches_registered_worker_only(tmp_path,configs,monkeypatch):
    from gui.services import iid_execution_service as transport
    from gui.services import noniid_execution_service as service
    commands=[]
    class Reader:
        def __init__(self,*a,**k):pass
        def start(self):pass
    class Process:
        pid=123456
        def __init__(self,command,**kwargs):commands.append(command)
    monkeypatch.setattr(service,'ROOT',tmp_path)
    monkeypatch.setattr(transport.subprocess,'Popen',Process)
    monkeypatch.setattr(transport.threading,'Thread',Reader)
    handle=service.NoniidExecutionHandle(configs[0],lambda event:None)
    assert commands[0][1]==str(tmp_path/'scripts/run_noniid_verified_condition.py')
    assert '--execute' in commands[0] and '--recover' not in commands[0]
    assert handle.config_path.is_relative_to(tmp_path/'gui/worker_state/noniid_verified')


def test_epoch_event_has_fractional_progress_without_fabricated_round_mean(configs):
    from gui.services.noniid_execution_service import NoniidExecutionHandle
    from gui.services.execution_service import ExecutionService
    from gui.state import ApplicationState,ExecutionState
    state=ApplicationState(experiment=from_persisted(configs[0]),execution=ExecutionState.RUNNING)
    service=ExecutionService(state)
    handle=NoniidExecutionHandle.__new__(NoniidExecutionHandle);handle.config=configs[0];handle.sink=service.post
    handle._dispatch(dict(event='epoch_complete',round=2,node=3,epoch=1,accuracy=.2))
    service.dispatch_pending()
    assert state.progress==pytest.approx(2.32/30)
    assert state.average_accuracy is None and state.completed_rounds==0
    assert state.latest_node_accuracy==.2


def test_complete_requires_exact_horizon_and_round_metrics(tmp_path,configs):
    runner,state=checkpoint_setup(tmp_path,configs[0]);state['status']='completed'
    worker.commit(runner,state,tmp_path)
    with pytest.raises(ValueError,match='False completion'):protocol.recovery_plan(tmp_path,configs[0])


def test_plotting_rejects_partial_results(tmp_path,configs):
    from reporting.noniid_verified_plots import report_condition
    (tmp_path/'run.json').write_text(json.dumps(dict(status='completed',roundMetrics=[],config=configs[0])))
    with pytest.raises(ValueError,match='Thirty measured'):report_condition(tmp_path,tmp_path/'plots')

def test_thirty_round_worker_orchestration_and_exact_stop_resume(tmp_path,monkeypatch):
    """Exercise every cursor/attack transition with fake updates, no scientific SGD."""
    import numpy as np
    import tensorflow as tf
    from basil_core import research_protocol,replicated_training_service,isolated_replicas,iid_runtime,verified_cart_runner,parallel_verifier_service
    from scripts import run_verified_cart_learning,check_verified_cart_full_batch
    from reporting import noniid_verified_plots

    monkeypatch.setattr(worker,'ROOT',tmp_path)
    gate=tmp_path/'gui/worker_state/cart_full_batch_threat_v1/summary.json'
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps(dict(status='passed',sourceHashes={},allFiftyUpdatesExactlyMatch=True,incorrectProposalRecovered=True)))
    monkeypatch.setattr(verified_cart_runner,'fingerprints',lambda:{})
    monkeypatch.setattr(protocol,'require_transport_proof',lambda:'transport-proof')
    monkeypatch.setattr(run_verified_cart_learning,'require_proof',lambda:'proof')
    monkeypatch.setattr(iid_runtime,'prepare_environment',lambda device:None)
    monkeypatch.setattr(iid_runtime,'configure_tensorflow',lambda *a:None)
    monkeypatch.setattr(iid_runtime,'runtime_metadata',lambda *a:dict(device='GPU',deviceName='synthetic fixture',numericalExecution={}))
    # Broadcast immutable synthetic pixels: no full image allocation or real training.
    labels=np.repeat(np.arange(10,dtype=np.int32),5000)
    tests=np.repeat(np.arange(10,dtype=np.int32),1000)
    images=np.broadcast_to(np.zeros((1,32,32,3),np.float32),(50000,32,32,3))
    test_images=np.broadcast_to(np.zeros((1,32,32,3),np.float32),(10000,32,32,3))
    def load(path,**kw):
        return {'x_train_float32.npy':images,'y_train_int32.npy':labels,
                'x_test_float32.npy':test_images,'y_test_int32.npy':tests}[Path(path).name]
    monkeypatch.setattr(np,'load',load)
    monkeypatch.setattr(np,'savez',lambda *a,**kw:None)
    monkeypatch.setattr(replicated_training_service,'TrainingReference',
        lambda *a,**kw:SimpleNamespace(root='synthetic-train',images=images,labels=labels))
    initial=np.zeros(117706,np.float32)
    monkeypatch.setattr(research_protocol,'build_model',lambda *a:SimpleNamespace(trainable_variables=[tf.Variable(initial)]))
    audit,_=__import__('basil_core.noniid_study',fromlist=['partition_audit']).partition_audit(labels,tests)
    config=protocol.configurations(audit['partitionHash'],research_protocol.params_hash([initial]))[3]
    fault_phases=[]
    class Pool:
        def __init__(self,*a,**kw):
            fault_phases.append(kw['faults']);self.endpoints={}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def close(self):pass
    class Service:
        public_keys={}
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def close(self):pass
    monkeypatch.setattr(isolated_replicas,'IsolatedReplicaPool',Pool)
    monkeypatch.setattr(parallel_verifier_service,'ParallelVerifierService',Service)
    monkeypatch.setattr(check_verified_cart_full_batch,'health',lambda pool:{})
    stop_at=[None]
    committed=[]
    class Runner:
        def __init__(self,*a,**kw):
            self.round=self.node=self.step=0;self.anchor=None;self.selection=None
            self.params=[initial.copy()];self.records=[]
        def _record_membership(self):pass
        def select(self,probes):
            assert sorted(probes)==list(range(self.node+1) if self.round==0 else range(10))
            self.anchor=[initial.copy()]
            self.selection=dict(sender=(self.node-1)%10,round=self.round if self.node else self.round-1,hash='selected',
                records=[dict(senderId=(self.node-i)%10,balancedTrainingLoss=2.,snapshotRound=self.round-1) for i in range(1,6)])
            return self.selection
        def advance(self):
            committed.append((self.round,self.node,self.step))
            self.records.append(dict(samples=392 if self.step%10==9 else 512,crossEntropy=2.,
                ebmPenalty=.0001,ebmCorrectionNorm=.001,ebmCorrectionRatio=.001))
            self.step+=1
            if stop_at[0] is not None and len(committed)==stop_at[0]:
                __import__('os').kill(__import__('os').getpid(),__import__('signal').SIGTERM)
            return dict(objective=2.)
        def finish_activation(self):
            assert len(self.records)==50
            assert [r['samples'] for r in self.records]==([512]*9+[392])*5
            self.node+=1
            if self.node==10:self.node=0;self.round+=1
            self.anchor=None;self.step=0;self.selection=None
            return [dict(receiver=i,noise=dict(relativeNoiseL2=.01,noiseL2Norm=.1),attackActive=self.round>=20,receivedHash='test') for i in range(5)]
        def save(self,path):
            worker.atomic_json(path,dict(round=self.round,node=self.node,step=self.step,anchor=self.anchor is not None,
                selection=self.selection,records=self.records))
        def restore(self,path):
            data=json.loads(path.read_text())
            self.round,self.node,self.step=data['round'],data['node'],data['step']
            self.anchor=[initial.copy()] if data['anchor'] else None
            self.selection=data['selection'];self.records=data['records']
    monkeypatch.setattr(verified_cart_runner,'VerifiedCartRunner',Runner)
    def evaluate(*a,node,epoch,**kw):
        return dict(node=node,epoch=epoch,accuracy=.1,perClass=[.1]*10,loss=2.,
            confusion=np.eye(10,dtype=int).tolist(),parameterHash='synthetic')
    monkeypatch.setattr(run_verified_cart_learning,'evaluate',evaluate)
    plots=[]
    monkeypatch.setattr(noniid_verified_plots,'report_condition',lambda *a:plots.append('condition'))
    monkeypatch.setattr(noniid_verified_plots,'report_comparison',lambda *a:plots.append('comparison'))
    # Stop inside an epoch, then verify the next cursor starts at the next update.
    stop_at[0]=13
    assert worker.execute(config)==130
    directory=tmp_path/config['resultDirectory']
    plan=protocol.recovery_plan(directory,config)
    assert plan['mode']=='checkpoint' and plan['step']==13
    stop_at[0]=None
    assert worker.execute(config)==0
    assert committed==[(r,n,s) for r in range(30) for n in range(10) for s in range(50)]
    state=json.loads((directory/'progress.json').read_text())
    assert state['cursor']==[30,0,0] and len(state['roundMetrics'])==30
    assert len(state['archiveHashes'])==300
    assert len(list(directory.glob('checkpoint_*.json')))==2
    assert plots==['condition','comparison']
    assert fault_phases==[{}, {}, {0:'poison',1:'forged',5:'withhold',7:'exit'}]
    assert protocol.recovery_plan(directory,config)['mode']=='completed'

@pytest.mark.skipif(not __import__('os').environ.get('DISPLAY'),reason='Optional Tk integration requires a display')
def test_gui_opens_prepared_noniid_queue_idle_and_launches_only_on_click(tmp_path,configs,monkeypatch):
    import tkinter as tk
    from gui.app import PaperMergeApp
    from gui.state import Workspace
    from gui.services import noniid_execution_service
    monkeypatch.setattr('gui.app.PROJECT_ROOT',tmp_path)
    queue=tmp_path/'gui/queue_state.json';queue.parent.mkdir(parents=True);queue.write_text(json.dumps(configs))
    launches=[]
    class FakeHandle:
        active=False
        def __init__(self,cfg,sink):launches.append(cfg['conditionId'])
        def stop(self):pass
    monkeypatch.setattr(noniid_execution_service,'NoniidExecutionHandle',FakeHandle)
    root=tk.Tk()
    try:
        app=PaperMergeApp(root)
        root.update_idletasks();root.update()
        assert launches==[] and not app._queue_running
        assert len(app.queue_state.entries)==4 and app.state.can_run
        view=app.shell.views[Workspace.QUEUE]
        assert len(view.tree.get_children())==4
        assert all(view.tree.item(i,'values')[4]=='0/30' for i in view.tree.get_children())
        assert all('Non-IID' in view.tree.item(i,'values')[2] for i in view.tree.get_children())
        presets=app.shell.views[Workspace.BUILDER].noniid_presets
        assert len([k for k in presets if 'Non-IID CART' in k])==4
        app.shell.navigate(Workspace.QUEUE)
        app.run()  # Fake handle only: confirms manual button routing.
        assert launches==['A'] and app._queue_running
        assert app.state.active_experiment_name==configs[0]['experimentName']
        app._queue_running=False
    finally:root.destroy()


def test_completed_plot_sources_and_network_replay_are_consistent(tmp_path,configs):
    from reporting.noniid_verified_plots import report_condition,report_comparison
    from gui.services.noniid_telemetry import load_replay
    import numpy as np
    for cfg in configs:
        directory=tmp_path/cfg['resultDirectory'];directory.mkdir(parents=True)
        rows=[]
        for r in range(30):
            row=dict(round=r,full_test_accuracy=.1+r*.001,mean_node_accuracy=.2,
                worst_node_accuracy=.1,full_test_loss=2.,per_node_accuracy=[.2]*10,
                per_node_per_class_accuracy=[[.2]*10]*10,corrupted_selection_rate=0.,
                attacker_designated_selection_rate=.4,selected_snapshot_age=1.,
                channel_noise_norm=.1,noise_to_model_norm_ratio=.01,
                ebm_penalty=.0001,ebm_correction_norm=.001,ebm_correction_ratio=.001)
            row.update({f'class_{i}_accuracy':.2 for i in range(10)});rows.append(row)
            for node in range(10):
                selection=dict(selectedSender=(node-1)%10,selectedSnapshotRound=r if node else r-1,
                    selectedHash='test',selectedActuallyCorrupted=False,
                    candidates=[dict(senderId=(node-i)%10,balancedTrainingLoss=2.,snapshotRound=r-1) for i in range(1,6)])
                epochs=[dict(epoch=e,accuracy=.2,perClass=[.2]*10,
                    confusion=np.eye(10,dtype=int).tolist()) for e in range(6)]
                activation=dict(round=r,node=node,selection=selection,epochs=epochs,
                    outgoing=[dict(receiver=i,attackActive=False,noise=dict(relativeNoiseL2=.01,noiseL2Norm=.1)) for i in range(5)])
                (directory/f'activation_{r*10+node:04d}.json').write_text(json.dumps(activation))
        (directory/'run.json').write_text(json.dumps(dict(status='completed',config=cfg,roundMetrics=rows)))
    cfg=configs[3];directory=tmp_path/cfg['resultDirectory']
    result=report_condition(directory,tmp_path/cfg['plotDirectory'])
    assert result['rounds']==30 and result['plots']>=20
    replay=load_replay(directory/'run.json')
    assert replay['metrics']['avg_history'].shape==(30,) and len(replay['researchRecords'])==300
    assert replay['researchRecords'][(29,9)]['globalTestAccuracy']==.2
    comparison=report_comparison(tmp_path,cfg)
    assert comparison['status']=='completed'
    assert all(v==0. for v in comparison['contrastsPercentagePoints'].values())
    assert len(list((tmp_path/cfg['comparisonPlotsDirectory']).glob('*.png')))==4



