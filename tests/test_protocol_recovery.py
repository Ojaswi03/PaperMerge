"""Recovery contracts use scalar simulated workers, never CIFAR training."""
import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from basil_core.protocol_checkpoint import (
    load_checkpoint, save_checkpoint, recovery_plan, read_activation_records,
    verify_replayed_activation,
)
from basil_core import research_protocol as protocol
from basil_core.iid_campaign import campaign_configs
from gui.services import QueueService
from gui.services.execution_service import ExecutionEvent, ExecutionService
from gui.state import ApplicationState, ExecutionState, QueueState
from gui.state.experiment_state import from_persisted
from gui.views.dashboard_view import ACCURACY_SERIES


@pytest.fixture
def simulation(monkeypatch):
    config=campaign_configs('a'*64,'b'*64)[3]
    config.update(nRounds=3,diagnosticTestSamplesPerClass=1,attackStartRound=1)
    monkeypatch.setattr(protocol,'validate_config',lambda config:None)
    monkeypatch.setattr(protocol,'build_model',lambda *args:object())
    monkeypatch.setattr(protocol,'iid_partition_indices',lambda labels,*args:np.array_split(np.arange(len(labels)),10))
    monkeypatch.setattr(protocol,'evaluate_params',lambda worker,params,*args,details=False:
        (float(params[0][0]/100),np.full(10,params[0][0]/100,np.float32),1.) if details else
        (float(params[0][0]/100),np.full(10,params[0][0]/100,np.float32)))

    class Worker:
        def __init__(self,model):self.params=[np.zeros(1,np.float32)]
        def export(self):return protocol.copy_params(self.params)
        def probe(self,params,x,y):return float(abs(params[0][0]-5)),.1
        def train(self,params,x,y,**kwargs):
            assert kwargs['epochs']==5 and kwargs['batch_size']==512
            if kwargs.get('should_stop') and kwargs['should_stop']():raise InterruptedError('simulated stop')
            trained=[params[0]+np.float32(.01*(kwargs['node_id']+1))]
            value=kwargs['evaluation']()
            return trained,dict(loss=1.,accuracy=.1,steps=50,beforeTraining=value,
                epochs=[dict(epoch=i+1,samplesSeen=5000,batchSizes=[512]*9+[392],**value) for i in range(5)],
                optimizerTelemetry=[dict(ebmCoefficient=.0001,effectiveCoordinateSigma=.01,
                    baseGradientNorm=1.,robustObjectiveGradientNorm=1.,ebmPenalty=.0001,
                    ebmObjective=1.0001,ebmCorrectionNorm=.001,ebmCorrectionRatio=.001) for _ in range(50)],
                baseGradientNorm=1.,robustGradientNorm=1.)
    data=(np.zeros((100,1),np.float32),np.tile(np.arange(10),10),
          np.zeros((10,1),np.float32),np.arange(10))
    return config,data,Worker


@pytest.mark.parametrize('stop_after',[1,9,10,14,29,30])
def test_checkpoint_resume_matches_uninterrupted_ring_and_noise(simulation,tmp_path,stop_after):
    config,data,worker=simulation
    full=protocol.run_research_protocol(config,*data,worker_factory=worker)
    completed=[]; path=tmp_path/'checkpoint.npz'
    def save(state):save_checkpoint(path,config,state,partition_hash='partition',initial_hash='initial')
    if stop_after==30:
        protocol.run_research_protocol(config,*data,worker_factory=worker,checkpoint_callback=save)
    else:
        with pytest.raises(InterruptedError):
            protocol.run_research_protocol(config,*data,worker_factory=worker,
                activation_callback=completed.append,checkpoint_callback=save,
                should_stop=lambda:len(completed)>=stop_after)
    state=load_checkpoint(path,config,partition_hash='partition',initial_hash='initial')
    assert state['next_activation']==stop_after
    resumed=protocol.run_research_protocol(config,*data,worker_factory=worker,resume_state=state)
    assert resumed['telemetry']==full['telemetry']
    for key,value in full.items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,resumed[key])
    for a,b in zip(full['finalNodeParams'],resumed['finalNodeParams']):
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


def test_checkpoint_after_last_node_before_round_evaluation(simulation,tmp_path):
    config,data,worker=simulation
    path=tmp_path/'checkpoint.npz'
    def interrupt(state):
        if state['next_activation']==10 and state['completed_rounds']==0:
            save_checkpoint(path,config,state,partition_hash='p',initial_hash='i')
            raise InterruptedError('power interruption after last node')
    with pytest.raises(InterruptedError):protocol.run_research_protocol(config,*data,worker_factory=worker,checkpoint_callback=interrupt)
    state=load_checkpoint(path,config,partition_hash='p',initial_hash='i')
    rounds=[]
    result=protocol.run_research_protocol(config,*data,worker_factory=worker,resume_state=state,round_callback=rounds.append)
    assert [r['round'] for r in rounds]==[0,1,2] and len(result['telemetry'])==30


def test_round_11_memory_contract_survives_checkpoint_resume(simulation,tmp_path):
    config,data,worker=simulation;config=dict(config,nRounds=13)
    records=[];path=tmp_path/'checkpoint.npz'
    def save(state):
        if state['next_activation']==110:
            save_checkpoint(path,config,state,partition_hash='p',initial_hash='i')
    with pytest.raises(InterruptedError):
        protocol.run_research_protocol(config,*data,worker_factory=worker,
            activation_callback=records.append,checkpoint_callback=save,should_stop=lambda:len(records)>=110)
    state=load_checkpoint(path,config,partition_hash='p',initial_hash='i')
    assert [(s.sender_id,s.round_id) for s in state['memories'][0].candidates()]==[(i,10) for i in range(5,10)]
    seen=[]
    def inspect(state):
        if state['next_activation']==111:
            seen.extend((s.sender_id,s.round_id) for s in state['memories'][1].candidates())
    protocol.run_research_protocol(config,*data,worker_factory=worker,resume_state=state,checkpoint_callback=inspect)
    assert seen==[(6,10),(7,10),(8,10),(9,10),(0,11)]


def test_checkpoint_rejects_config_and_partition_changes(simulation,tmp_path):
    config,data,worker=simulation;path=tmp_path/'checkpoint.npz'
    protocol.run_research_protocol(config,*data,worker_factory=worker,
        checkpoint_callback=lambda state:save_checkpoint(path,config,state,partition_hash='p',initial_hash='i'))
    with pytest.raises(ValueError,match='configuration'):load_checkpoint(path,dict(config,seed=9),partition_hash='p',initial_hash='i')
    with pytest.raises(ValueError,match='partition'):load_checkpoint(path,config,partition_hash='changed',initial_hash='i')


def test_legacy_reconstruction_requires_identical_records(simulation):
    config,data,worker=simulation
    original=protocol.run_research_protocol(config,*data,worker_factory=worker)['telemetry']
    replay=[]
    def verify(record):
        verify_replayed_activation(record,original[len(replay)]);replay.append(record)
    protocol.run_research_protocol(config,*data,worker_factory=worker,activation_callback=verify)
    assert replay==original
    changed=copy.deepcopy(original[0]);changed['trainedOutputHash']='different'
    with pytest.raises(ValueError,match='trainedOutputHash'):verify_replayed_activation(changed,original[0])


def test_recovery_plan_is_read_only_and_falls_back_to_previous_checkpoint(simulation,tmp_path):
    config,data,worker=simulation;path=tmp_path/'checkpoint.npz'
    full=protocol.run_research_protocol(config,*data,worker_factory=worker,
        checkpoint_callback=lambda state:save_checkpoint(path,config,state,partition_hash='p',initial_hash='i'))
    (tmp_path/'run.json').write_text(json.dumps(dict(config=config,status='stopped',completedRounds=3,iidPartitionHash='p',initialModelHash='i')))
    log=tmp_path/'partial_activation_telemetry.jsonl'
    log.write_text(''.join(json.dumps(r)+'\n' for r in full['telemetry']))
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    plan=recovery_plan(tmp_path,config)
    assert plan['mode']=='checkpoint' and plan['startActivation']==30
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    path.write_bytes(b'corrupt')
    assert recovery_plan(tmp_path,config)['checkpoint'].endswith('checkpoint.previous.npz')
    path.unlink();(tmp_path/'checkpoint.previous.npz').unlink()
    assert recovery_plan(tmp_path,config)['mode']=='verified_replay'
    assert read_activation_records(log)==full['telemetry']


def test_restored_progress_and_eta_use_only_new_session_work():
    state=ApplicationState(execution=ExecutionState.RUNNING,elapsed_seconds=1100.)
    service=ExecutionService(state)
    service.post(ExecutionEvent('execution_restored',dict(progress=.5,round=50,priorRuntimeSeconds=1000.,
        startProgress=.5,accuracyHistory=[(50,.5,.4)],activationHistory=[(50.,.55)],
        recoveryTotal=500,recoveryCompleted=500)))
    service.dispatch_pending();state.progress=.6
    assert state.experiment_eta_seconds==pytest.approx(400.)
    assert state.average_accuracy==.5 and state.worst_accuracy==.4
    service.post(ExecutionEvent('recovery_updated',dict(completed=4,total=10)))
    service.dispatch_pending();assert '4/10' in state.status


def test_queue_clock_spans_conditions_and_freezes_when_stopped(monkeypatch):
    clock=iter([100.,125.,160.])
    monkeypatch.setattr('gui.state.application_state.time.monotonic',lambda:next(clock))
    state=ApplicationState()
    state.begin_queue();state.tick()
    assert state.queue_elapsed_seconds==25.
    state.end_queue()
    assert state.queue_elapsed_seconds==60. and not state.queue_active
    state.tick();assert state.queue_elapsed_seconds==60.


def test_external_graceful_stop_does_not_leave_execution_running():
    state=ApplicationState(execution=ExecutionState.RUNNING)
    service=ExecutionService(state);service._worker=object()
    service.post(ExecutionEvent('execution_stopped'))
    service.dispatch_pending()
    assert state.execution is ExecutionState.STOPPED and service._worker is None


def test_completed_outputs_are_not_requeued(tmp_path):
    config=campaign_configs('a'*64,'b'*64)[0]
    root=tmp_path;output=root/config['resultDirectory'];output.mkdir(parents=True)
    (output/'run.json').write_text(json.dumps(dict(config=config,status='completed')))
    queue=QueueState();entry=queue.add(from_persisted(config))
    QueueService.describe_recovery(queue,root)
    assert entry.status.value=='completed' and entry.execution_hint=='Already completed'


def test_curve_colors_and_line_styles_are_distinct():
    assert len({color for color,*_ in ACCURACY_SERIES})==3
    assert len({dash for _,dash,*_ in ACCURACY_SERIES})==3


@pytest.mark.parametrize('mode',['none','gradient_norm_objective'])
def test_tiny_tensorflow_gradients_and_next_update_survive_checkpoint(simulation,tmp_path,mode):
    import tensorflow as tf
    from basil_core.iid_study import IidWorker
    config,data,mock_worker=simulation
    states=[]
    protocol.run_research_protocol(config,*data,worker_factory=mock_worker,
        checkpoint_callback=lambda state:states.append(copy.deepcopy(state)))
    def model():
        return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),
            tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=13))])
    original=IidWorker(model());params=original.export()
    state=states[-1]
    state.update(next_activation=0,completed_rounds=0,telemetry=[],
        node_params=[protocol.copy_params(params) for _ in range(10)],
        memories=protocol.initialize_memories(params))
    path=tmp_path/'checkpoint.npz'
    save_checkpoint(path,config,state,partition_hash='p',initial_hash='i')
    loaded=load_checkpoint(path,config,partition_hash='p',initial_hash='i')
    restored=IidWorker(model());restored.load(loaded['node_params'][0])
    x=tf.ones([2,32,32,3])*.2;y=tf.constant([0,1]);key=tf.constant([1,2])
    def evaluate(worker):
        grads,values=worker._gradients(x,y,key,mode,'paper_absolute_gaussian',tf.constant(.01),tf.constant(999999.))
        return [g.numpy() for g in grads],[float(v) for v in values],[p-.05*g.numpy() for p,g in zip(worker.export(),grads)]
    a,b=evaluate(original),evaluate(restored)
    assert a[1]==b[1]
    for group in (0,2):
        for first,second in zip(a[group],b[group]):np.testing.assert_array_equal(first,second)


@pytest.mark.parametrize('stop_after',[0,14])
def test_serial_runner_recovers_without_replacing_recorded_evidence(simulation,tmp_path,monkeypatch,stop_after):
    from scripts import run_iid_basil_pair as runner
    config,data,worker=simulation
    config.update(resultDirectory=str(tmp_path/'result'),plotDirectory=str(tmp_path/'plots'))
    initial=[np.zeros(1,np.float32)]
    config['expectedInitialModelHash']=protocol.params_hash(initial)
    config['expectedPartitionHash']='partition'
    monkeypatch.setattr(runner,'validate_config',lambda cfg:None)
    monkeypatch.setattr(runner,'iid_output',lambda path,**kwargs:__import__('pathlib').Path(path))
    monkeypatch.setattr(runner,'IidWorker',worker)
    monkeypatch.setattr(runner,'build_model',lambda *args:object())
    monkeypatch.setattr(runner,'evaluate_params',lambda *args:(0.,np.zeros(10)))
    monkeypatch.setattr(runner,'ProtocolAudit',lambda *args:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr('subprocess.check_output',lambda *args,**kwargs:'test-commit')
    monkeypatch.setattr('reporting.iid_study_plots.generate_run_plots',lambda *args:None)
    monkeypatch.setattr('reporting.iid_campaign_plots.report_campaign',lambda **kwargs:None)
    real_engine=protocol.run_research_protocol
    captured=[]
    def interrupt(cfg,*data,**kwargs):
        callback=kwargs['activation_callback']
        def activation(record):callback(record);captured.append(record)
        kwargs.update(activation_callback=activation,should_stop=lambda:len(captured)>=stop_after,audit=None)
        return real_engine(cfg,*data,**kwargs)
    monkeypatch.setattr(runner,'run_research_protocol',interrupt)
    audit={'partitionHash':'partition'}
    with pytest.raises(InterruptedError):runner.execute(config,data,audit,family='research')
    output=tmp_path/'result'
    log_before=(output/'partial_activation_telemetry.jsonl').read_bytes() if stop_after else b''
    csv_before=(output/'round_metrics.csv').read_bytes()
    # Emulate legacy runs that predate checkpoints; mathematical replay must
    # verify their existing evidence rather than silently restarting the files.
    (output/'checkpoint.npz').unlink();(output/'checkpoint.previous.npz').unlink(missing_ok=True)
    monkeypatch.setattr(runner,'run_research_protocol',lambda cfg,*data,**kwargs:
        real_engine(cfg,*data,**dict(kwargs,audit=None)))
    metadata=runner.execute(config,data,audit,family='research',recover=True)
    assert metadata['status']=='completed' and metadata['completedRounds']==3
    assert (output/'partial_activation_telemetry.jsonl').read_bytes().startswith(log_before)
    assert (output/'round_metrics.csv').read_bytes().startswith(csv_before)
    assert len(read_activation_records(output/'partial_activation_telemetry.jsonl'))==30
    with (output/'round_metrics.csv').open() as handle:
        assert [int(r['round']) for r in csv.DictReader(handle)]==[0,1,2]
    assert list((output/'recovery_attempts').glob('*/run_before_recovery.json'))


@pytest.mark.parametrize('saved_checkpoint',[False,True])
def test_cpu_state_transfers_to_separate_gpu_output_without_training_on_gpu(simulation,tmp_path,monkeypatch,saved_checkpoint):
    from basil_core.iid_device_transition import transfer_cpu_state
    from basil_core.research_audit import array_hash
    import hashlib
    config,data,worker=simulation
    config.update(resultDirectory='source',plotDirectory='plots')
    config['expectedInitialModelHash']=protocol.params_hash([np.zeros(1,np.float32)])
    source=tmp_path/'source';source.mkdir();records=[];states=[];rounds=[]
    with pytest.raises(InterruptedError):
        protocol.run_research_protocol(config,*data,worker_factory=worker,activation_callback=records.append,
            round_callback=rounds.append,checkpoint_callback=lambda value:states.append(dict(value)),
            should_stop=lambda:len(records)>=14)
    (source/'partial_activation_telemetry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with (source/'round_metrics.csv').open('w',newline='') as handle:
        keys=('round','full_test_accuracy','full_test_loss','mean_node_accuracy','worst_node_accuracy')
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        writer.writerows(dict(round=r['round'],full_test_accuracy=r['fullTestAccuracy'],full_test_loss=r['fullTestLoss'],
            mean_node_accuracy=r['averageAccuracy'],worst_node_accuracy=r['worstNodeAccuracy']) for r in rounds)
    hashes={key:array_hash(value) for key,value in zip(('trainImagesHash','trainLabelsHash','testImagesHash','testLabelsHash'),data)}
    import tensorflow as tf
    manifest=dict(config=config,device='CPU',status='stopped',completedRounds=1,runtimeSeconds=10.,
        dataSha256=hashes,iidPartitionHash='p',initialModelHash=config['expectedInitialModelHash'],tensorflowVersion=tf.__version__)
    (source/'run.json').write_text(json.dumps(manifest))
    if saved_checkpoint:save_checkpoint(source/'checkpoint.npz',config,states[-1],partition_hash='p',initial_hash=manifest['initialModelHash'])
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    monkeypatch.setattr('basil_core.iid_study.IidWorker',worker)
    monkeypatch.setattr('basil_core.research_audit.ProtocolAudit',lambda *args:SimpleNamespace(close=lambda:None))
    engine=protocol.run_research_protocol
    calls=[]
    def cpu_engine(*args,**kwargs):
        calls.append(args[0]);kwargs['audit']=None
        return engine(*args,**kwargs)
    monkeypatch.setattr(protocol,'run_research_protocol',cpu_engine)
    gpu=dict(config,executionDevice='GPU',executionMode='deterministic_single_process_gpu',resultDirectory='destination',
        recoverySourceDirectory='source',deviceTransition='cpu_to_gpu_verified_state')
    runtime_info=dict(device='GPU',numericalExecution={'test':True})
    transfer_cpu_state(tmp_path,gpu,data,{'partitionHash':'p'},runtime_info)
    if saved_checkpoint:assert not calls
    else:assert len(calls)==1
    output=tmp_path/'destination'
    restored=load_checkpoint(output/'checkpoint.npz',gpu,partition_hash='p',initial_hash=manifest['initialModelHash'])
    assert restored['next_activation']==14 and restored['telemetry']==records
    for first,second in zip(states[-1]['node_params'],restored['node_params']):
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    new=json.loads((output/'run.json').read_text())
    assert new['deviceTransition']['exactStateTransfer'] and not new['deviceTransition']['bitIdenticalCpuContinuation']
    assert new['executionDeviceSegments'][0]['endExclusive']==14
    assert (output/'round_metrics.csv').read_bytes()==(source/'round_metrics.csv').read_bytes()
