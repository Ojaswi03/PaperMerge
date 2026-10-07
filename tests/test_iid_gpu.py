"""Device/routing contracts are headless and never launch research training."""
import copy
import json
from types import SimpleNamespace

import pytest

from basil_core.iid_campaign import campaign_configs, gui_approved, verify_campaign
from basil_core import iid_runtime as runtime
from gui.services import QueueService
from gui.state import ApplicationState, ExecutionState
from gui.state.experiment_state import from_persisted, to_persisted
from gui.services.execution_service import ExecutionService, ExecutionEvent


def fake_tf(*, gpu=True, tf32=False):
    return SimpleNamespace(__version__='test-tf',config=SimpleNamespace(
        get_logical_device_configuration=lambda device:[SimpleNamespace(memory_limit=runtime.GPU_MEMORY_LIMIT_MB)],
        list_physical_devices=lambda kind: [SimpleNamespace(name='/physical_device:GPU:0')] if gpu else [],
        experimental=SimpleNamespace(get_device_details=lambda device:{'device_name':'Test GPU'},
            tensor_float_32_execution_enabled=lambda:tf32),
        threading=SimpleNamespace(get_intra_op_parallelism_threads=lambda:1,
            get_inter_op_parallelism_threads=lambda:1)),
        sysconfig=SimpleNamespace(get_build_info=lambda:dict(cuda_version='12.5.1',cudnn_version='9')))


def verification(tmp_path,monkeypatch):
    from basil_core.protocol_checkpoint import implementation_fingerprint
    report=dict(status='passed',numericalExecution=dict(policyRevision=runtime.POLICY_REVISION,
        device='GPU',gpuName='Test GPU',tensorflowVersion='test-tf',tensorFloat32=False,
        intraOpThreads=1,interOpThreads=1,cudaVersion='12.5.1',cudnnVersion='9',gpuWorkerSha256=runtime.gpu_worker_hash(),
        gpuMemoryLimitMb=runtime.GPU_MEMORY_LIMIT_MB),
        implementationSha256=implementation_fingerprint())
    path=tmp_path/'gpu.json';path.write_text(json.dumps(report))
    monkeypatch.setattr(runtime,'GPU_VERIFICATION',path)
    return path,report


@pytest.mark.parametrize('device',['CPU','GPU'])
def test_explicit_environment_and_device_setting(monkeypatch,device):
    # Limit changes to this test, not the pytest process's real device policy.
    monkeypatch.setattr(runtime.os,'environ',{})
    values=runtime.prepare_environment(device)
    assert values['CUDA_VISIBLE_DEVICES']==('0' if device=='GPU' else '-1')
    assert values['PAPERMERGE_IID_DEVICE']==device
    assert values['TF_DETERMINISTIC_OPS']=='1'
    assert values['TF_NUM_INTRAOP_THREADS']==values['TF_NUM_INTEROP_THREADS']=='1'
    assert runtime.execution_device({})=='CPU'
    with pytest.raises(ValueError):runtime.execution_device({'executionDevice':'auto'})


def test_gpu_fails_closed_for_unavailable_or_unverified_cuda(tmp_path,monkeypatch):
    config={'executionDevice':'GPU'}
    with pytest.raises(RuntimeError,match='no CUDA'):runtime.runtime_metadata(fake_tf(gpu=False),config)
    monkeypatch.setattr(runtime,'GPU_VERIFICATION',tmp_path/'missing.json')
    with pytest.raises(RuntimeError,match='verification'):runtime.runtime_metadata(fake_tf(),config)
    with pytest.raises(RuntimeError,match='TF32'):runtime.runtime_metadata(fake_tf(tf32=True),config)


def test_gpu_metadata_verification_matches_exact_hardware_policy(tmp_path,monkeypatch):
    path,report=verification(tmp_path,monkeypatch)
    value=runtime.runtime_metadata(fake_tf(),{'executionDevice':'GPU'})
    assert value['device']=='GPU' and value['gpuVerified'] and value['deviceName']=='Test GPU'
    assert value['tensorflowDevice']=='/GPU:0'
    report['numericalExecution']['gpuName']='Different GPU';path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match='verification'):runtime.runtime_metadata(fake_tf(),{'executionDevice':'GPU'})


def test_cpu_manifest_is_not_falsely_gpu_verified():
    value=runtime.runtime_metadata(fake_tf(),{})
    assert value['device']=='CPU' and not value['gpuVerified']
    assert value['tensorflowDevice']=='/CPU:0'


def test_recovery_cannot_cross_device_or_gpu_policy(tmp_path,monkeypatch):
    verification(tmp_path,monkeypatch)
    gpu=runtime.runtime_metadata(fake_tf(),{'executionDevice':'GPU'})
    cpu=runtime.runtime_metadata(fake_tf(),{})
    runtime.check_recovery_device({'device':'CPU'},cpu)
    runtime.check_recovery_device({'device':'GPU','numericalExecution':gpu['numericalExecution']},gpu)
    for old,new in (({'device':'CPU'},gpu),({'device':'GPU'},cpu)):
        with pytest.raises(ValueError,match='exactly resume'):runtime.check_recovery_device(old,new)
    with pytest.raises(ValueError,match='same validated'):
        runtime.check_recovery_device({'device':'GPU','numericalExecution':{}},gpu)


def test_gpu_four_conditions_keep_equations_pairing_and_separate_paths(tmp_path):
    cpu=campaign_configs('a'*64,'b'*64)
    gpu=campaign_configs('a'*64,'b'*64,device='GPU')
    assert verify_campaign(cpu)==verify_campaign(gpu)
    for a,b in zip(cpu,gpu):
        assert gui_approved(b)
        assert b['executionDevice']=='GPU'
        changed={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        assert changed=={'executionDevice','executionMode','experimentName','runId','displayName',
            'resultDirectory','plotDirectory','comparisonResultsDirectory','comparisonPlotsDirectory'}
        assert b['resultDirectory']==a['resultDirectory']+'_gpu'
        assert b['localEpochs']==5 and b['batchSize']==512 and b['basilMemorySize']==5
        assert b['expectedInitialModelHash']==a['expectedInitialModelHash']
        assert to_persisted(from_persisted(b))==b
    changed=copy.deepcopy(gpu);changed[2]['executionDevice']='CPU'
    assert not gui_approved(changed[2])
    with pytest.raises(ValueError):verify_campaign(changed)
    path=tmp_path/'queue.json';path.write_text(json.dumps(gpu))
    state=QueueService(path).load();QueueService.describe_recovery(state,tmp_path)
    assert [e.config.extra['conditionId'] for e in state.entries]==list('ABCD')
    assert all(e.status.value=='pending' and e.execution_hint.endswith('GPU') for e in state.entries)
    assert all(e.estimated_remaining_seconds is None for e in state.entries)


def test_live_device_event_replaces_requested_label_without_widget_dependency():
    state=ApplicationState();service=ExecutionService(state)
    service.post(ExecutionEvent('device_status_changed',dict(device='GPU',deviceName='RTX 4070 Ti')))
    service.dispatch_pending()
    assert state.execution_device=='GPU' and state.device_name=='RTX 4070 Ti'
    state.transition(ExecutionState.PREPARING)
    assert state.execution_device==state.device_name==''


def test_gpu_gui_launcher_does_not_remask_cuda_or_run_automatically(tmp_path,monkeypatch):
    import gui.services.iid_execution_service as service
    cfg=campaign_configs('a'*64,'b'*64,device='GPU')[0]
    monkeypatch.setattr(service,'ROOT',tmp_path)
    monkeypatch.setattr(service,'WORKER_ROOT',tmp_path/'workers')
    launches=[]
    def launch(command,**kwargs):
        launches.append((command,kwargs))
        return SimpleNamespace(pid=1234)
    monkeypatch.setattr(service.subprocess,'Popen',launch)
    monkeypatch.setattr(service.threading,'Thread',lambda **kwargs:SimpleNamespace(start=lambda:None))
    assert not launches
    events=[];service.IidExecutionHandle(cfg,events.append)
    command,options=launches[0]
    assert '--execute' in command and '--recover' not in command
    assert options['env']['CUDA_VISIBLE_DEVICES']=='0'
    assert options['env']['PAPERMERGE_IID_DEVICE']=='GPU'
    assert cfg['runId'] in (tmp_path/'workers'/f"{cfg['runId']}.json").read_text()


def test_gpu_report_routing_cannot_overwrite_cpu_report(tmp_path):
    from reporting.iid_campaign_plots import report_campaign
    value=report_campaign(tmp_path,device='GPU')
    assert value['device']=='GPU' and all(v=='not_started' for v in value['status'].values())
    assert (tmp_path/'newResults/IID/ABCD_100r_comparison_gpu/analysis.json').exists()
    assert not (tmp_path/'newResults/IID/ABCD_100r_comparison').exists()


def test_insufficient_gpu_memory_refuses_start(monkeypatch):
    monkeypatch.setattr(runtime.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='4000\n'))
    with pytest.raises(RuntimeError,match='at least 5120'):runtime.free_gpu_memory()
    monkeypatch.setattr(runtime.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='9000\n'))
    assert runtime.free_gpu_memory()==9000


def test_gpu_memory_cap_is_required(tmp_path,monkeypatch):
    verification(tmp_path,monkeypatch)
    tf=fake_tf();tf.config.get_logical_device_configuration=lambda gpu:None
    with pytest.raises(RuntimeError,match='memory cap'):runtime.runtime_metadata(tf,{'executionDevice':'GPU'})


def test_round_counter_ignores_partial_activations():
    state=ApplicationState();service=ExecutionService(state)
    service.post(ExecutionEvent('progress_updated',dict(round=1,progress=.001)))
    service.dispatch_pending()
    assert state.current_round==1 and state.completed_rounds==0
    service.post(ExecutionEvent('progress_updated',dict(round=1,progress=.01,completedRounds=1)))
    service.dispatch_pending()
    assert state.completed_rounds==1


def test_cpu_transition_queue_pairing_and_source_guard():
    configs=campaign_configs('a'*64,'b'*64,device='GPU',continue_existing=True)
    verify_campaign(configs)
    assert all(gui_approved(c) for c in configs)
    for c in configs:
        assert c['resultDirectory']==c['recoverySourceDirectory']+'_gpu'
        assert c['deviceTransition']=='cpu_to_gpu_verified_state'
    bad=dict(configs[0],recoverySourceDirectory='experiments/results4')
    assert not gui_approved(bad)


def test_gpu_oom_saves_cpu_evidence_without_allocating_or_replaying(tmp_path):
    import numpy as np
    import tensorflow as tf
    from basil_core.iid_gpu_worker import GpuProtocolAudit
    audit=GpuProtocolAudit(tmp_path/'oom')
    audit.current=dict(round=3,node=8,epoch=2,batch=0)
    audit.pending=dict(x=np.zeros((1,32,32,3),np.float32),y=np.array([8]),key=np.array([1,2]),
        params=[np.array([1.,2.],np.float32)])
    def forbidden(*args,**kwargs):raise AssertionError('OOM handling must not allocate GPU tensors or replay.')
    worker=SimpleNamespace(export=forbidden,load=forbidden,_batch_gradients=forbidden)
    try:
        audit.failure(worker,tf.errors.ResourceExhaustedError(None,None,'Synthetic OOM; no memory exhausted'))
    finally:audit.close()
    saved=json.loads((tmp_path/'oom/failure.json').read_text())
    assert saved['category']=='resource_exhausted' and saved['graphReplaySkipped']
    assert saved['round']==3 and saved['node']==8 and saved['epoch']==2
    with np.load(tmp_path/'oom/failure_batch.npz') as batch:
        np.testing.assert_array_equal(batch['weight_0'],[1.,2.])
