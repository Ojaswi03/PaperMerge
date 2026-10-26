"""Independent CART objective, data isolation, rollback and exact resume checks."""
import json
import subprocess
import sys

import pytest

from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')

from basil_core.cart_local_memory import LocalClassMemory, regression_weights, retained_gradients, state_hash


def test_memory_probes_are_local_stratified_repeatable_and_rng_isolated():
    from basil_core.research_protocol import keyed_rng
    labels=np.repeat(np.arange(3,dtype=np.int32),[2,17,32])
    x=np.arange(len(labels)*2,dtype=np.float32).reshape(-1,2)
    before=keyed_rng(2025,'augmentation',1,2,3,4).integers(0,100,10)
    memory=LocalClassMemory()
    probe=memory.prepare(0,x,labels)
    np.testing.assert_array_equal(np.bincount(labels[probe],minlength=10),[2,16,16,0,0,0,0,0,0,0])
    np.testing.assert_array_equal(probe,LocalClassMemory().prepare(0,x,labels))
    np.testing.assert_array_equal(before,keyed_rng(2025,'augmentation',1,2,3,4).integers(0,100,10))
    assert len(np.unique(probe))==len(probe)
    changed=x.copy();changed[-1,0]+=1
    with pytest.raises(ValueError,match='identity'):
        memory.prepare(0,changed,labels)


def memory_fixture():
    import tensorflow as tf
    labels=np.array([0,0,0,1],np.int32)
    memory=LocalClassMemory()
    probe=memory.prepare(0,np.ones((4,2),np.float32),labels)
    y=labels[probe]
    logs=tf.nn.log_softmax(np.zeros((4,10),np.float32)).numpy()
    memory.admit(0,y,logs,'a'*64,0,'selected')
    return memory,y,logs


def test_admission_uses_measured_class_ce_rejects_ties_and_can_mix_teachers():
    import tensorflow as tf
    memory,y,logs=memory_fixture()
    assert memory.admit(0,y,logs,'b'*64,0,'trained')==[]
    logits=np.zeros((4,10),np.float32);logits[y==1,1]=2
    assert memory.admit(0,y,tf.nn.log_softmax(logits).numpy(),'b'*64,0,'trained')==[1]
    records=memory.nodes['0']['classes']
    assert records['0']['modelHash']=='a'*64 and records['1']['modelHash']=='b'*64
    restored=LocalClassMemory(state=json.loads(json.dumps(memory.export())))
    assert restored.export()==memory.export()
    bad=memory.export();bad['nodes']['0']['classes']['1']['loss']=0.
    with pytest.raises(ValueError,match='corrupt'):
        LocalClassMemory(state=bad)


def test_regression_is_class_balanced_and_absent_classes_are_unavailable():
    import tensorflow as tf
    memory,y,logs=memory_fixture()
    bad=np.zeros((4,10),np.float32)
    bad[np.arange(4),y]=-1
    teacher,weights,records=memory.controls(0,y,tf.nn.log_softmax(bad).numpy())
    assert {r['classId'] for r in records}=={0,1}
    assert weights[y==0].sum()==pytest.approx(weights[y==1].sum())
    assert np.all(weights>0)
    np.testing.assert_array_equal(teacher,logs)
    np.testing.assert_array_equal(regression_weights([1.,.5,0.],[.6,.6,0.]),[np.float32(.4),0.,0.])
    assert all(r['weight']<=1 for r in records)
    for invalid in ([np.nan],[-1.]):
        with pytest.raises(ValueError):regression_weights(invalid,[0.])


@pytest.mark.parametrize('coefficient',[0.,.0001])
def test_exact_gradient_matches_hessian_and_closed_form_kl(coefficient):
    import tensorflow as tf
    from basil_core.dirichlet_calibration import calibration_offsets
    model=tf.keras.Sequential([tf.keras.layers.Input((2,)),tf.keras.layers.Dense(10,use_bias=False)])
    w=model.trainable_variables[0]
    w.assign(np.arange(20,dtype=np.float32).reshape(2,10)/20)
    x=tf.constant([[1.,.5],[.3,1.]])
    y=tf.constant([0,4]);px=tf.constant([[.2,.7],[.8,.4]])
    teacher=tf.nn.log_softmax(tf.constant(np.arange(20,dtype=np.float32).reshape(2,10)/30))
    row=tf.constant([.2,.4]);reference=[tf.ones_like(w)*.1]
    offsets=calibration_offsets(np.arange(1,11),method='prior_adjustment')
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            f=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=model(x)-offsets))
        g=inner.gradient(f,w)
    h=outer.jacobian(g,w).numpy().reshape(20,20)
    # Closed form KL gradient for a linear softmax, independent of the new loss implementation.
    kl=px.numpy().T @ (row.numpy()[:,None]*(tf.nn.softmax(model(px)).numpy()-tf.exp(teacher).numpy()))
    actual,values=retained_gradients(model,x,y,reference,offsets,px,teacher,row,coefficient=coefficient)
    expected=g.numpy().ravel()+2*coefficient*h@g.numpy().ravel()+.01*(w.numpy()-.1).ravel()+kl.ravel()
    np.testing.assert_allclose(actual[0].numpy().ravel(),expected,atol=2e-7,rtol=2e-5)
    assert float(values[16])==pytest.approx(np.linalg.norm(kl),rel=2e-5)


def test_beta_zero_and_zero_weights_exactly_preserve_base_and_teacher_is_frozen():
    import tensorflow as tf
    from basil_core.dirichlet_calibration import calibrated_anchor_gradients
    model=tf.keras.Sequential([tf.keras.layers.Input((2,)),tf.keras.layers.Dense(10,kernel_initializer='ones')])
    x=tf.ones((3,2));y=tf.constant([0,1,2]);ref=[tf.identity(v) for v in model.trainable_variables]
    teacher=tf.Variable(tf.nn.log_softmax(tf.reshape(tf.range(30,dtype=tf.float32),(3,10))))
    weights=tf.Variable([.1,.2,.3]);offset=tf.zeros(10)
    base,old=calibrated_anchor_gradients(model,x,y,ref,offset,coefficient=.0001)
    for beta,row in [(0.,weights),(1.,tf.zeros(3))]:
        grad,values=retained_gradients(model,x,y,ref,offset,x,teacher,row,beta=beta,coefficient=.0001)
        for a,b in zip(base,grad):np.testing.assert_array_equal(a.numpy(),b.numpy())
        np.testing.assert_array_equal(np.array(old),np.array(values[:15]))
    with tf.GradientTape(persistent=True) as tape:
        _,values=retained_gradients(model,x,y,ref,offset,x,teacher,weights)
        objective=values[2]
    assert tape.gradient(objective,teacher) is None
    assert tape.gradient(objective,weights) is None


def tiny_model():
    import tensorflow as tf
    return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4,activation='tanh',kernel_initializer=tf.keras.initializers.Constant(.03)),
        tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.Constant(.02))])


def test_retention_gradient_reaches_encoder_and_classifier():
    import tensorflow as tf
    model=tiny_model();x=tf.ones((3,32,32,3));y=tf.constant([0,1,2])
    ref=[tf.identity(v) for v in model.trainable_variables]
    teacher=tf.nn.log_softmax(tf.reshape(tf.range(30,dtype=tf.float32),(3,10)))
    # Nonidentical output weights make the encoder correction observable.
    model.trainable_variables[-2].assign(np.arange(40,dtype=np.float32).reshape(4,10)/40)
    from basil_core.dirichlet_calibration import calibrated_anchor_gradients
    base,_=calibrated_anchor_gradients(model,x,y,ref,tf.zeros(10))
    retained,_=retained_gradients(model,x,y,ref,tf.zeros(10),x,teacher,tf.ones(3)/3)
    assert np.linalg.norm((retained[0]-base[0]).numpy())>1e-5
    assert np.linalg.norm((retained[-1]-base[-1]).numpy())>1e-5


def test_worker_interruption_rolls_back_all_teacher_state_and_keeps_five_epochs():
    from basil_core.cart_local_memory import LocalMemoryCartWorker
    worker=LocalMemoryCartWorker(tiny_model())
    params=worker.export();x=np.ones((19,32,32,3),np.float32);y=np.arange(19,dtype=np.int32)%10
    kwargs=dict(seed=2025,round_id=0,node_id=0,epochs=5,batch_size=512,learning_rate=.05,
        ebm_mode='none',semantics='paper_absolute_gaussian',sigma=0.)
    initial=worker.export_local_memory()
    with pytest.raises(InterruptedError):worker.train(params,x,y,should_stop=lambda:True,**kwargs)
    assert worker.export_local_memory()==initial
    _,record=worker.train(params,x,y,**kwargs)
    assert record['steps']==5 and len(record['cartMemory']['epochControls'])==6
    assert record['cartMemory']['probeCount']==19
    assert all(e['batchSizes']==[19] for e in record['epochs'])
    from basil_core.research_protocol import params_hash
    assert params_hash([v.numpy() for v in worker.reference])==params_hash(params)


def test_config_is_explicit_bounded_and_cannot_be_promoted():
    from basil_core.cart_memory_study import configurations
    from basil_core.research_protocol import validate_config
    from basil_core.dirichlet_study import gui_approved
    configs=configurations('a'*64,'b'*64,'CPU')
    for config in configs:
        validate_config(config)
        assert not gui_approved(config)
        for field,value in [('cartBeta',2.),('nRounds',100),('researchValid',True),('attackStartRound',20)]:
            with pytest.raises(ValueError):validate_config(dict(config,**{field:value}))
    assert configs[0]['cartBeta']==0 and configs[1]['cartBeta']==1
    assert [c['conditionId'] for c in configs]==list('AABCD')


def test_checkpoint_rejects_missing_and_future_teachers():
    from basil_core.cart_memory_study import validate_memory_checkpoint
    memory,_,_=memory_fixture()
    validate_memory_checkpoint(memory.export(),1,2025)
    with pytest.raises(ValueError,match='lost'):
        validate_memory_checkpoint(dict(revision=memory.export()['revision'],seed=2025,nodes={}),1,2025)
    bad=memory.export();bad['nodes']['0']['classes']['0']['activation']=10
    with pytest.raises(ValueError,match='future'):validate_memory_checkpoint(bad,1,2025)


def test_ring_checkpoint_restores_identical_memory_weights_and_metrics(tmp_path,monkeypatch):
    import basil_core.research_protocol as protocol
    from basil_core.cart_local_memory import LocalMemoryCartWorker
    from basil_core.cart_memory_study import configurations
    from basil_core.protocol_checkpoint import save_checkpoint,load_checkpoint
    monkeypatch.setattr(protocol,'build_model',lambda *_:tiny_model())
    config=configurations('a'*64,'b'*64,'CPU')[1]
    rng=np.random.default_rng(4)
    x=rng.uniform(0,1,(1000,32,32,3)).astype(np.float32);y=np.repeat(np.arange(10,dtype=np.int32),100)
    tx=x[:20];ty=y[:20]
    make=lambda m:LocalMemoryCartWorker(m)
    uninterrupted=protocol.run_research_protocol(config,x,y,tx,ty,worker_factory=make)
    cursor=[]
    def completed(record):cursor.append(record['nodeId'])
    def checkpoint(state):save_checkpoint(tmp_path/'checkpoint.npz',config,state,partition_hash='a'*64,initial_hash='b'*64)
    with pytest.raises(InterruptedError):
        protocol.run_research_protocol(config,x,y,tx,ty,worker_factory=make,activation_callback=completed,
            should_stop=lambda:len(cursor)>=3,checkpoint_callback=checkpoint)
    restored=load_checkpoint(tmp_path/'checkpoint.npz',config,partition_hash='a'*64,initial_hash='b'*64)
    assert restored['next_activation']==3 and set(restored['cart_memory']['nodes'])=={'0','1','2'}
    seen=[]
    resumed=protocol.run_research_protocol(config,x,y,tx,ty,worker_factory=make,resume_state=restored,
        activation_callback=lambda r:seen.append((r['round'],r['nodeId'])))
    assert seen[0]==(0,3) and len(seen)==27
    for field in ('averageAccuracy','selectedSources','selectedSnapshotRounds','epochOverallAccuracy'):
        np.testing.assert_array_equal(uninterrupted[field],resumed[field])
    for first,second in zip(uninterrupted['finalNodeParams'],resumed['finalNodeParams']):
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    assert uninterrupted['telemetry']==resumed['telemetry']


@pytest.mark.parametrize('module',['basil_core.cart_local_memory','basil_core.cart_memory_study','scripts.run_cart_memory_smoke'])
def test_new_modules_are_lazy(module):
    subprocess.run([sys.executable,'-c',f'import {module};import sys;'
        'assert not ({"tensorflow","numpy","wandb","matplotlib"}&sys.modules.keys())'],check=True)


def test_default_cli_starts_no_training():
    result=subprocess.run([sys.executable,'scripts/run_cart_memory_smoke.py'],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['trainingStarted'] is False


def test_full_learning_profile_keeps_original_data_and_separates_smoke_schedule():
    from basil_core.cart_memory_study import configurations
    from basil_core.research_protocol import validate_config
    configs=configurations('a'*64,'b'*64,'CPU','clean_learning')
    assert [c['runId'] for c in configs]==['A_beta0','A']
    for config in configs:
        validate_config(config)
        assert config['nRounds']==3 and config['attackStartRound']==20
        assert not any(k.startswith('diagnostic') for k in config)
        assert config['evaluationSamples']==10000 and config['trainingSamples']==50000
        assert config['localEpochs']==5 and config['snapshotAnchorMu']==.01
        assert config['logitCalibrationMethod']=='prior_adjustment'
        assert config['noniidStudyId']=='cart_local_class_memory_v1_clean_learning_3r'
    result=subprocess.run([sys.executable,'scripts/run_cart_memory_smoke.py','--profile','clean_learning'],
        capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['conditions']==['A_beta0','A']


def test_reporting_cannot_claim_success_from_partial_run(tmp_path):
    from scripts.report_cart_memory_smoke import review
    path=tmp_path/'A';path.mkdir()
    (path/'run.json').write_text(json.dumps(dict(status='running',completedActivations=29,completedRounds=2)))
    with pytest.raises(ValueError,match='completed'):
        review(tmp_path)


def test_test_only_callback_cannot_change_local_memory_controls():
    from basil_core.cart_local_memory import LocalMemoryCartWorker
    x=np.ones((19,32,32,3),np.float32);y=np.arange(19,dtype=np.int32)%10
    kwargs=dict(seed=2025,round_id=0,node_id=0,epochs=5,batch_size=512,learning_rate=.05,
        ebm_mode='none',semantics='paper_absolute_gaussian',sigma=0.)
    outputs=[]
    for score in (0.,1.):
        worker=LocalMemoryCartWorker(tiny_model())
        trained,record=worker.train(worker.export(),x,y,evaluation=lambda:dict(globalTestAccuracy=score),**kwargs)
        outputs.append((trained,record['cartMemory'],worker.export_local_memory()))
    for first,second in zip(outputs[0][0],outputs[1][0]):np.testing.assert_array_equal(first,second)
    assert outputs[0][1:]==outputs[1][1:]


def test_executed_source_archives_are_never_replaced(tmp_path,monkeypatch):
    import hashlib
    import scripts.run_cart_memory_smoke as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    (tmp_path/'source.py').write_text('original source\n')
    digest=hashlib.sha256((tmp_path/'source.py').read_bytes()).hexdigest()
    archive=tmp_path/'archive'
    runner.archive_sources(archive,{'source.py':digest})
    runner.archive_sources(archive,{'source.py':digest})
    (archive/digest).write_text('corrupt archive\n')
    with pytest.raises(ValueError,match='refusing replacement'):
        runner.archive_sources(archive,{'source.py':digest})
    assert (archive/digest).read_text()=='corrupt archive\n'


def test_recovery_identity_mismatch_does_not_replace_existing_evidence(tmp_path):
    from scripts.run_cart_memory_smoke import require_identity, keep_json
    path=tmp_path/'verification.json'
    keep_json(path,dict(implementationSha256={'worker':'a'*64},runtime={'device':'GPU'}))
    original=path.read_bytes()
    require_identity(path,dict(implementationSha256={'worker':'a'*64}))
    with pytest.raises(ValueError,match='identity'):
        require_identity(path,dict(implementationSha256={'worker':'b'*64}))
    with pytest.raises(ValueError,match='replace'):
        keep_json(path,dict(implementationSha256={'worker':'b'*64}))
    assert path.read_bytes()==original
