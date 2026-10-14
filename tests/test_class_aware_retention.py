"""Class-aware method contracts, with synthetic data and no research launches."""
import copy
import json
from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')
import pytest
from basil_core.class_aware_retention import (
    REVISION, ClassAwareRetentionWorker, diagnostic_config, normalized_importance,
    protection_weights, state_hash, validate_state, weighted_anchor_gradients,
)
from basil_core.noniid_study import anchored_gradients
from basil_core.protocol_checkpoint import save_checkpoint,load_checkpoint
from basil_core import research_protocol as p


def toy():
    import tensorflow as tf
    model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(2,use_bias=False)])
    model.trainable_variables[0].assign([[.4,-.2]])
    return model


def state_for(params, label=9, value=3.):
    return dict(revision=REVISION,classes={str(label):[np.full_like(v,value) for v in params]},visits={str(label):25000})


def test_weighted_penalty_and_gradient_match_analytic_reference():
    from basil_core.trainer import lossFn
    import tensorflow as tf
    model=toy();x=tf.constant([[1.],[2.]]);y=tf.constant([0,1])
    reference=[tf.constant([[-.1,.3]])];importance=[tf.constant([[2.,7.]])]
    with tf.GradientTape() as tape:ce=lossFn(y,model(x))
    base=tape.gradient(ce,model.trainable_variables)
    grads,values,actual_base=weighted_anchor_gradients(model,x,y,reference,importance,mu=.01)
    drift=model.trainable_variables[0].numpy()-reference[0].numpy()
    expected=base[0].numpy()+.01*importance[0].numpy()*drift
    np.testing.assert_allclose(grads[0],expected,rtol=1e-6,atol=1e-7)
    np.testing.assert_array_equal(actual_base[0],base[0])
    assert float(values[2])==pytest.approx(float(ce)+.005*np.sum(importance[0].numpy()*drift**2),rel=1e-6)
    # Nonuniform importance really differs from an ordinary scalar anchor.
    uniform,_=anchored_gradients(model,x,y,reference,mu=.01,sigma_e=0.,ebm=False)
    assert not np.allclose(grads[0],uniform[0],rtol=1e-6,atol=1e-7)


def test_empty_history_equals_original_uniform_anchor():
    import tensorflow as tf
    model=toy();params=[v.numpy() for v in model.trainable_variables]
    weights,prior=protection_weights(None,params,0)
    assert prior==[]
    reference=[tf.constant([[.3,-.1]])]
    x=tf.constant([[1.],[2.]]);y=tf.constant([0,1])
    expected,values=anchored_gradients(model,x,y,reference,mu=.01,sigma_e=0.,ebm=False)
    actual,weighted,_=weighted_anchor_gradients(model,x,y,reference,weights,mu=.01)
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_allclose(weighted[:13],values,rtol=1e-6,atol=1e-7)


def test_importance_excludes_current_class_and_does_not_mutate_input():
    params=[np.zeros((2,),np.float32)]
    state=state_for(params,9,3.)
    state['classes']['0']=[np.array([100.,100.],np.float32)];state['visits']['0']=25000
    before=state_hash(state)
    weights,prior=protection_weights(state,params,0)
    np.testing.assert_array_equal(weights[0],[4.,4.]);assert prior==[9]
    weights[0][:]=17
    assert state_hash(state)==before


def test_normalization_is_training_only_and_predeclared_bounded():
    result=normalized_importance([np.array([1.,3.],np.float64)])
    np.testing.assert_array_equal(result[0],[.5,1.5])
    result=normalized_importance([np.r_[1.,np.zeros(999)]])
    assert result[0].max()==100. and np.all(result[0]>=0)
    np.testing.assert_array_equal(normalized_importance([np.zeros(2)])[0],[0.,0.])
    for bad in (np.array([np.nan]),np.array([np.inf]),np.array([-1.])):
        with pytest.raises(ValueError):normalized_importance([bad])


@pytest.mark.parametrize('field,value',[('classImportanceCap',50.),('snapshotAnchorMu',.02),
    ('useChannelNoise',True),('attackerCount',4),('researchValid',True),('nRounds',50),
    ('localEpochs',3),('localObjective','source_ebm_snapshot_anchor'),('liveTrackingEnabled',True)])
def test_config_rejects_unapproved_mixing(field,value):
    config=diagnostic_config();p.validate_config(config)
    config[field]=value
    with pytest.raises(ValueError):p.validate_config(config)


def image_model():
    import tensorflow as tf
    return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10,kernel_initializer='ones',bias_initializer='zeros')])


def test_five_complete_epochs_reference_and_auxiliary_state_frozen():
    worker=ClassAwareRetentionWorker(image_model())
    params=worker.export();state=state_for(params)
    state['classes']['0']=[np.full_like(v,5.) for v in params];state['visits']['0']=25000
    original=state_hash(state)
    worker.receive_retention_state(state)
    _,record=worker.train(params,np.zeros((5000,32,32,3),np.float32),np.zeros(5000,np.int32),
        seed=2025,round_id=0,node_id=0,epochs=5,batch_size=512,learning_rate=.05,
        ebm_mode='none',semantics='paper_absolute_gaussian',sigma=0.)
    assert record['steps']==50
    assert all(e['samplesSeen']==5000 and e['batchSizes']==[512]*9+[392] for e in record['epochs'])
    assert record['snapshotAnchorReferenceHash']==p.params_hash(params)
    assert state_hash(state)==original
    assert record['classRetention']['protectedClasses']==[9]
    assert record['classRetention']['outputClasses']==[0,9]
    assert worker.output_state['visits']=={'9':25000,'0':25000}
    assert record['optimizerTelemetry'][0]['snapshotAnchorPenalty']==0.
    assert record['optimizerTelemetry'][-1]['snapshotAnchorPenalty']>0.
    assert all(s['importanceWeightMax']==4. for s in record['optimizerTelemetry'])
    assert all(a.max()<=100 for a in worker.output_state['classes']['0'])
    assert any(not np.array_equal(a,b) for a,b in zip(worker.output_state['classes']['0'],state['classes']['0']))
    assert worker.optimizer.momentum==0 and worker.optimizer.clipnorm is None
    assert worker.optimizer.weight_decay is None and int(worker.optimizer.iterations)==50


def test_importance_accumulates_base_ce_not_proximal_gradient():
    from basil_core.trainer import lossFn
    import tensorflow as tf
    from basil_core.research_audit import augment_batch
    worker=ClassAwareRetentionWorker(image_model())
    for ref,w in zip(worker.reference,worker.model.trainable_variables):ref.assign(w+.01)
    for mask in worker.diagonal:mask.assign(tf.ones_like(mask)*3.)
    images=tf.zeros((2,32,32,3));labels=tf.zeros(2,tf.int32);key=tf.constant([7,8])
    with tf.GradientTape() as tape:ce=lossFn(labels,worker.model(augment_batch(images,key),training=True))
    base=tape.gradient(ce,worker.model.trainable_variables)
    applied,_=worker._gradients(images,labels,key,'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(99.))
    for accumulator,gradient in zip(worker.moments,base):
        np.testing.assert_allclose(accumulator.numpy(),2*np.asarray(gradient,dtype=np.float64)**2,rtol=1e-6,atol=1e-10)
    assert any(not np.allclose(a.numpy(),2*np.asarray(g,dtype=np.float64)**2,rtol=1e-6,atol=1e-10)
        for a,g in zip(worker.moments,applied))


def test_new_method_ignores_legacy_lambda_and_accumulates_base_not_anchor():
    import tensorflow as tf
    outcomes=[]
    for legacy in (1.,999999.):
        worker=ClassAwareRetentionWorker(image_model())
        for ref,w in zip(worker.reference,worker.model.trainable_variables):ref.assign(w+.01)
        for mask in worker.diagonal:mask.assign(tf.ones_like(mask)*3.)
        gradients,values=worker._gradients(tf.ones((2,32,32,3)),tf.zeros(2,tf.int32),
            tf.constant([7,8]),'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(legacy))
        tf.keras.optimizers.SGD(.05).apply_gradients(zip(gradients,worker.model.trainable_variables))
        outcomes.append((gradients,values,worker.export(),[a.numpy() for a in worker.moments]))
    for a,b in zip(outcomes[0],outcomes[1]):
        for x,y in zip(a,b):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))


@pytest.mark.parametrize('bad',[np.nan,np.inf,-1.,101.])
def test_invalid_importance_is_not_silently_accepted(bad):
    params=[np.zeros(2,np.float32)];state=state_for(params)
    state['classes']['9'][0][0]=bad
    with pytest.raises(ValueError):validate_state(state,params)


def test_memory_refresh_and_checkpoint_restore_all_class_state(tmp_path):
    params=[np.array([.2,.3],np.float32)];memories=p.initialize_memories(params)
    state=state_for(params)
    snapshot=p.Snapshot(0,11,params,retention_state=state)
    p.broadcast(memories,snapshot)
    for receiver in range(1,6):assert state_hash(memories[receiver].snapshots[0].retention_state)==state_hash(state)
    config=diagnostic_config()
    checkpoint=dict(next_activation=0,completed_rounds=0,node_params=[params]*10,
        memories=memories,telemetry=[])
    from basil_core.protocol_checkpoint import STATE_ARRAYS
    for name in STATE_ARRAYS:checkpoint[name]=np.zeros((30,10),np.float32)
    path=tmp_path/'checkpoint.npz'
    save_checkpoint(path,config,checkpoint,partition_hash='p',initial_hash='i')
    restored=load_checkpoint(path,config,partition_hash='p',initial_hash='i')
    for receiver in range(1,6):
        assert state_hash(restored['memories'][receiver].snapshots[0].retention_state)==state_hash(state)
    # Missing auxiliary state is a failure, not weights-only "resume".
    with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
    meta=json.loads(str(arrays['metadata']));meta.pop('retentionStates')
    arrays['metadata']=np.asarray(json.dumps(meta));np.savez(path,**arrays)
    with pytest.raises(ValueError,match='retention'):load_checkpoint(path,config,partition_hash='p',initial_hash='i')


def test_interrupted_class_aware_ring_matches_uninterrupted(tmp_path,monkeypatch):
    config=diagnostic_config();config.update(nRounds=3,diagnosticTestSamplesPerClass=1,diagnosticSamplesPerNode=2)
    monkeypatch.setattr(p,'validate_config',lambda cfg:None)
    monkeypatch.setattr(p,'build_model',lambda *args:object())
    monkeypatch.setattr(p,'evaluate_params',lambda worker,params,*args,details=False:
        (.1,np.full(10,.1,np.float32),1.) if details else (.1,np.full(10,.1,np.float32)))
    class Worker:
        def __init__(self,model):self.params=[np.zeros(2,np.float32)];self.state=None
        def export(self):return p.copy_params(self.params)
        def receive_retention_state(self,state):self.state=validate_state(state,self.params)
        def export_retention_state(self):return self.state
        def probe(self,params,x,y):return float(abs(params[0][0]-.1)),.1
        def train(self,params,x,y,**kw):
            incoming=state_hash(self.state);node=kw['node_id']
            self.state['classes'][str(node)]=[np.array([node/10,1.],np.float32)]
            self.state['visits'][str(node)]=10
            value=kw['evaluation']()
            return [params[0]+np.float32(.001*(node+1))],dict(loss=1.,accuracy=.1,steps=50,
                beforeTraining=value,epochs=[dict(epoch=e+1,**value) for e in range(5)],
                optimizerTelemetry=[dict(ebmCoefficient=0.,effectiveCoordinateSigma=0.)],
                baseGradientNorm=1.,robustGradientNorm=1.,snapshotAnchorReferenceHash=p.params_hash(params),
                classRetention=dict(inputStateHash=incoming,outputStateHash=state_hash(self.state)))
    data=(np.zeros((20,1),np.float32),np.repeat(np.arange(10),2),np.zeros((10,1),np.float32),np.arange(10))
    full=p.run_research_protocol(config,*data,worker_factory=Worker)
    records=[];path=tmp_path/'checkpoint.npz'
    with pytest.raises(InterruptedError):
        p.run_research_protocol(config,*data,worker_factory=Worker,activation_callback=records.append,
            should_stop=lambda:len(records)==14,
            checkpoint_callback=lambda state:save_checkpoint(path,config,state,partition_hash='p',initial_hash='i'))
    state=load_checkpoint(path,config,partition_hash='p',initial_hash='i')
    resumed=p.run_research_protocol(config,*data,worker_factory=Worker,resume_state=state)
    assert full['telemetry']==resumed['telemetry']
    for key,value in full.items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,resumed[key])
    for a,b in zip(full['finalNodeParams'],resumed['finalNodeParams']):
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
