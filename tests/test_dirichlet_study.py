"""Local-only mixed-label protocol, pairing and GUI contracts; no research run."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from basil_core import dirichlet_study as study
from gui.state.experiment_state import from_persisted,to_persisted,validate_experiment


@pytest.fixture
def configs():return study.configurations('a'*64,'b'*64)


def test_pairing_and_local_information_boundary(configs):
    assert study.verify_pairing(configs)=={
        'A/B':['attackHidden','attackerCount','resolvedAttackerIds'],
        'B/C':['channelNoiseSigmaAbsolute','useChannelNoise'],
        'C/D':['ebmMode','localObjective']}
    for cfg in configs:
        study.validate_configuration(cfg)
        state=from_persisted(cfg)
        assert not validate_experiment(state)
        assert to_persisted(state)==cfg
        assert (cfg['localEpochs'],cfg['batchSize'],cfg['basilMemorySize'])==(5,512,5)
        assert cfg['snapshotAnchorMu']==.01 and not cfg['liveTrackingEnabled']
        assert cfg['partitionStrategy']=='dirichlet' and cfg['dirichletAlpha']==.2
        assert cfg['informationModel']=='receiver_local_train_only' and cfg['helperExecution']=='disabled'
        assert cfg['snapshotSelectionRule']=='lowest_receiver_local_loss'
        assert cfg['researchValid'] is False
        assert study.gui_approved(cfg)
        assert 'Certified CART' not in state.dirty_summary
    assert configs[0]['resolvedAttackerIds']==[]
    assert all(c['resolvedAttackerIds']==[0,1,5,7] for c in configs[1:])
    assert configs[3]['channelNoiseSigmaAbsolute']**2==.0001


@pytest.mark.parametrize('key,value',[
    ('nRounds',31),('dirichletAlpha',.5),('partitionStrategy','one_class_per_node'),
    ('snapshotSelectionRule','certified_balanced_available_owner_train_ce'),
    ('helperExecution','concurrent'),('snapshotAnchorMu',.1),('liveTrackingEnabled',True),
    ('localEpochs',1),('gradientClipNorm',1.),('resultDirectory','newResults/IID/wrong'),
    ('diagnosticSamplesPerClass',512),('recoverySourceDirectory','newResults/nonIID/old')])
def test_refuse_silent_protocol_or_output_changes(configs,key,value):
    cfg=dict(configs[3],**{key:value})
    with pytest.raises(ValueError):study.validate_configuration(cfg)
    assert validate_experiment(from_persisted(cfg))


def test_existing_dirichlet_partitioner_exact_stream_and_coverage():
    import numpy as np
    from basil_core.data.cifar import _dirichletPartition
    from basil_core.research_protocol import keyed_rng
    y=np.repeat(np.arange(10),5000);ty=np.repeat(np.arange(10),1000)
    audit,chunks=study.partition_audit(y,ty)
    expected=_dirichletPartition(y,10,.2,keyed_rng(2025,'dirichlet_partition'))
    assert audit['assignedSamples']==audit['uniqueSamples']==50000
    assert audit['testClassCounts']==[1000]*10 and audit['testUsage']=='evaluation_only'
    np.testing.assert_array_equal(np.sort(np.concatenate(chunks)),np.arange(50000))
    for a,b,node in zip(chunks,expected,audit['nodes']):
        np.testing.assert_array_equal(a,b)
        assert sum(node['classCounts'])==node['samples']==len(a)
        assert node['batchesPerEpoch']==(len(a)+511)//512
        assert node['optimizerStepsPerActivation']==5*node['batchesPerEpoch']
        assert 1<=node['finalBatchSize']<=512
    assert len({len(c) for c in chunks})>1
    assert any(np.count_nonzero(n['classCounts'])<10 for n in audit['nodes'])
    assert study.partition_audit(y,ty)[0]['partitionHash']==audit['partitionHash']
    assert study.partition_audit(y,ty,2026)[0]['partitionHash']!=audit['partitionHash']


@pytest.mark.parametrize('samples',[989,9660])
def test_unequal_local_dataset_five_true_epochs(samples):
    import numpy as np
    import tensorflow as tf
    from basil_core.noniid_study import SnapshotAnchorWorker
    from basil_core.research_protocol import params_hash
    model=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),
        tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10)])
    worker=SnapshotAnchorWorker(model,mu=.01)
    start=worker.export();x=np.zeros((samples,32,32,3),np.float32);y=np.arange(samples,dtype=np.int32)%10
    _,metrics=worker.train(start,x,y,seed=2025,round_id=0,node_id=0,epochs=5,batch_size=512,
        learning_rate=.05,ebm_mode='none',semantics='paper_absolute_gaussian',sigma=0.)
    assert metrics['steps']==5*((samples+511)//512)
    for epoch in metrics['epochs']:
        assert epoch['samplesSeen']==samples
        assert epoch['batchSizes']==[512]*(samples//512)+([samples%512] if samples%512 else [])
    assert params_hash([a.numpy() for a in worker.reference])==params_hash(start)
    assert float(worker.optimizer.momentum)==0.


def test_no_lambda_and_fresh_idle_queue(configs,tmp_path):
    from basil_core.protocol_checkpoint import checkpoint_implementation
    from gui.services.queue_service import QueueService
    for value in [1,999999]:study.validate_configuration(dict(configs[3],legacyEbmLambda=value))
    assert 'basil_core/noniid_study.py' in checkpoint_implementation(configs[3])
    assert study.recovery_plan(tmp_path/'new',configs[3])['mode']=='fresh'
    path=tmp_path/'queue.json';path.write_text(json.dumps(configs))
    queue=QueueService(path).load();QueueService.describe_recovery(queue,tmp_path)
    assert len(queue.entries)==4 and all(e.status.value=='pending' for e in queue.entries)
    assert all(e.execution_hint.startswith('Fresh') for e in queue.entries)
    QueueService(path).save(queue)
    assert [to_persisted(e.config) for e in QueueService(path).load().entries]==configs


def test_cli_dry_run_starts_no_training(configs,tmp_path):
    path=tmp_path/'config.json';path.write_text(json.dumps(configs[3]))
    # A developer's completed D directory is real evidence, not this fixture.
    # Relocate only the child's output lookup; keep CLI parsing/validation real.
    code=('import sys;from pathlib import Path;from scripts import run_dirichlet_condition as worker;'
        f'worker.ROOT=Path({str(tmp_path)!r});'
        f'sys.argv=["run_dirichlet_condition.py","--config",{str(path)!r}];worker.main()')
    result=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,check=True)
    payload=json.loads(result.stdout)
    assert payload['trainingStarted'] is False and payload['rounds']==30
    assert payload['recovery']['mode']=='fresh'


def test_gpu_handle_uses_only_local_runner(configs,monkeypatch,tmp_path):
    from types import SimpleNamespace
    from gui.services.dirichlet_execution_service import DirichletExecutionHandle
    import gui.services.iid_execution_service as base
    calls=[]
    monkeypatch.setattr(base.subprocess,'Popen',lambda command,**kwargs:
        calls.append((command,kwargs)) or SimpleNamespace(pid=123,stdout=[]))
    monkeypatch.setattr(base.threading.Thread,'start',lambda self:None)
    # Leave the actual CLI/family routing intact but relocate runtime sidecars.
    monkeypatch.setattr(study,'ROOT',tmp_path)
    import gui.services.dirichlet_execution_service as adapter
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    handle=DirichletExecutionHandle(configs[3],lambda event:None)
    assert calls[0][0][1].endswith('scripts/run_dirichlet_condition.py')
    assert '--execute' in calls[0][0] and '--recover' not in calls[0][0]
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES']=='0'
    handle.pid_path.unlink()


def test_local_selection_handoff_memory_attack_and_stateful_resume(configs,monkeypatch):
    import copy
    import numpy as np
    import basil_core.research_protocol as engine
    probes=[];inputs=[];noise_inputs=[];legacy=[]
    class Worker:
        def __init__(self,model):self.params=[np.array([0.],np.float32)]
        def export(self):return engine.copy_params(self.params)
        def probe(self,params,x,y):
            probes.append((x.copy(),y.copy()))
            return float(params[0][0]**2),.1
        def train(self,params,x,y,**kw):
            inputs.append(engine.params_hash(params));legacy.append(kw['legacy_lambda'])
            self.params=[params[0]+1]
            ev=kw['evaluation'](); count=len(y);batches=[512]*(count//512)+([count%512] if count%512 else [])
            return self.export(),dict(loss=1.,accuracy=.1,steps=5*len(batches),beforeTraining=ev,
                epochs=[dict(epoch=i+1,localTrainingLoss=1.,samplesSeen=count,batchSizes=batches,**ev) for i in range(5)],
                optimizerTelemetry=[dict(ebmCoefficient=.0001,effectiveCoordinateSigma=.01)],
                baseGradientNorm=1.,robustGradientNorm=1.,snapshotAnchorReferenceHash=engine.params_hash(params))
    monkeypatch.setattr(engine,'build_model',lambda *a:None)
    monkeypatch.setattr(engine,'evaluate_params',lambda *a,**kw:(.1,np.full(10,.1),1.) if kw.get('details') else (.1,np.full(10,.1)))
    monkeypatch.setattr(engine,'applyAttack',lambda params,*a,**kw:[params[0]+1000])
    def channel(params,**kw):
        noise_inputs.append(params[0].copy())
        return [params[0]+.25],dict(coordinateSigma=kw['sigma'],noiseL2Norm=.25,relativeNoiseL2=.01)
    monkeypatch.setattr(engine,'apply_channel_noise',channel)
    cfg=dict(configs[3],nRounds=21,legacyEbmLambda=999999)
    y=np.repeat(np.arange(10,dtype=np.int32),5000);x=np.arange(50000,dtype=np.float32).reshape(-1,1)
    ty=np.repeat(np.arange(10,dtype=np.int32),1000);tx=-np.arange(1,10001,dtype=np.float32).reshape(-1,1)
    result=engine.run_research_protocol(cfg,x,y,tx,ty,worker_factory=Worker)
    records=result['telemetry']
    assert len(records)==210 and inputs==[r['inputHash'] for r in records] and set(legacy)=={0.}
    assert records[110]['memorySenderIds']==[5,6,7,8,9]
    assert records[110]['memorySnapshotRounds']==[10]*5
    assert records[111]['memorySenderIds']==[6,7,8,9,0]
    assert records[111]['memorySnapshotRounds']==[10,10,10,10,11]
    _,indices=study.partition_audit(y,ty)
    for i,r in enumerate(records):
        batch=engine.keyed_rng(2025,'snapshot_selection_batch',r['nodeId']).choice(indices[r['nodeId']],512,replace=False)
        for px,py in probes[i*5:(i+1)*5]:
            np.testing.assert_array_equal(px,x[batch]);np.testing.assert_array_equal(py,y[batch])
            assert px.min()>=0  # TEST images have negative synthetic IDs.
        assert not np.array_equal(batch,indices[r['nodeId']][:512])
        assert r['snapshotAnchorReferenceHash']==r['inputHash']
        active=r['nodeId'] in cfg['resolvedAttackerIds'] and r['round']>=20
        assert r['attackActive']==active
        assert engine.params_hash([noise_inputs[i*5]-(1000 if active else 0)])==r['trainedOutputHash']
    # Interrupt without committing part of an activation, restore all rolling
    # states/arrays/telemetry, then compare against uninterrupted logical execution.
    completed=[];checkpoint=[]
    with pytest.raises(InterruptedError):
        engine.run_research_protocol(cfg,x,y,tx,ty,worker_factory=Worker,
            activation_callback=completed.append,should_stop=lambda:len(completed)==15,
            checkpoint_callback=lambda state:checkpoint.append(copy.deepcopy(state)))
    assert checkpoint[-1]['next_activation']==15
    restored=engine.run_research_protocol(cfg,x,y,tx,ty,worker_factory=Worker,resume_state=checkpoint[-1])
    assert restored['telemetry']==records
    np.testing.assert_array_equal(restored['roundNodeAccuracy'],result['roundNodeAccuracy'])


@pytest.mark.parametrize('module',['basil_core.dirichlet_study','scripts.run_dirichlet_condition',
    'scripts.prepare_dirichlet_queue','gui.services.dirichlet_execution_service'])
def test_new_modules_keep_heavy_imports_lazy(module):
    code=f"import importlib,sys;importlib.import_module({module!r});assert not ({{'numpy','tensorflow','wandb','matplotlib'}} & sys.modules.keys())"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
