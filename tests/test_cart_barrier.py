"""TRAIN isolation and independent gradient contracts for the separate candidate."""
import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize('coefficient',[0.,.0001])
def test_barrier_gradient_matches_closed_form_active_class_and_hessian(coefficient):
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_barrier import barrier_gradients,SLACK
    model=tf.keras.Sequential([tf.keras.layers.Input((2,)),tf.keras.layers.Dense(10,use_bias=False)])
    w=model.trainable_variables[0];w.assign(np.arange(20,dtype=np.float32).reshape(2,10)/20)
    x=tf.constant([[1.,.5],[.3,1.]]);y=tf.constant([0,4])
    px=tf.constant([[.2,.7],[.8,.4],[.3,.5]]);py=tf.constant([0,0,1])
    average=np.zeros((10,3),np.float32);average[0,:2]=.5;average[1,2]=1
    mask=tf.constant([1.,1.]+[0.]*8)
    losses=tf.linalg.matvec(average,tf.nn.sparse_softmax_cross_entropy_with_logits(labels=py,logits=model(px))).numpy()
    best=losses.copy();best[0]+=.2;best[1]-=.2
    reference=[tf.ones_like(w)*.1]
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            f=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=model(x)))
        g=inner.gradient(f,w)
    h=outer.jacobian(g,w).numpy().reshape(20,20)
    p=tf.nn.softmax(model(px)).numpy()
    target=np.eye(10,dtype=np.float32)[py.numpy()]
    retention=px.numpy().T@((p-target)*np.array([[0.],[0.],[.5]],np.float32))
    gradients,values=barrier_gradients(model,x,y,reference,tf.zeros(10),px,py,average,best,mask,coefficient=coefficient)
    expected=g.numpy().ravel()+2*coefficient*h@g.numpy().ravel()+.01*(w.numpy()-.1).ravel()+retention.ravel()
    np.testing.assert_allclose(gradients[0].numpy().ravel(),expected,atol=2e-7,rtol=2e-5)
    assert float(values[15])==pytest.approx((.2-SLACK)/2,abs=2e-7)
    assert float(values[17])==.5


def test_no_barrier_when_class_loss_is_within_slack_and_beta_zero_exact():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_barrier import barrier_gradients
    from basil_core.dirichlet_calibration import calibrated_anchor_gradients
    model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(10)])
    x=tf.ones((2,1));y=tf.constant([0,1]);ref=[tf.identity(v) for v in model.trainable_variables]
    average=np.zeros((10,2),np.float32);average[0,0]=1;average[1,1]=1
    best=np.ones(10,np.float32)*100
    base,values=calibrated_anchor_gradients(model,x,y,ref,tf.zeros(10),coefficient=.0001)
    for beta in (0.,1.):
        actual,result=barrier_gradients(model,x,y,ref,tf.zeros(10),x,y,average,best,
            tf.constant([1.,1.]+[0.]*8),beta=beta,coefficient=.0001)
        for first,second in zip(base,actual):np.testing.assert_array_equal(first.numpy(),second.numpy())
        np.testing.assert_array_equal(np.array(values),np.array(result[:15]))


def test_train_validation_split_is_disjoint_balanced_and_deterministic():
    import numpy as np
    from basil_core.cart_validation import split_training
    labels=np.repeat(np.arange(10,dtype=np.int32),5000)
    fit,validation=split_training(labels)
    assert len(fit)==49000 and len(validation)==1000
    assert not np.intersect1d(fit,validation).size
    np.testing.assert_array_equal(np.bincount(labels[validation]),[100]*10)
    np.testing.assert_array_equal(np.bincount(labels[fit]),[4900]*10)
    np.testing.assert_array_equal(validation,split_training(labels)[1])


def test_validation_configuration_fixed_and_cannot_be_test_or_production():
    from basil_core.cart_validation import configurations
    from basil_core.research_protocol import validate_config
    for config in configurations('a'*64,'b'*64,'CPU'):
        validate_config(config)
        for key,value in [('officialTestUsed',True),('nRounds',30),('cartBeta',10.),('researchValid',True),
                          ('cartBarrierSlack',.01),('trainingSamples',50000)]:
            with pytest.raises(ValueError):validate_config(dict(config,**{key:value}))


def test_barrier_worker_rolls_back_and_produces_five_full_epochs():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_barrier import BarrierCartWorker,REVISION
    net=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10,kernel_initializer='ones')])
    worker=BarrierCartWorker(net);params=worker.export()
    x=np.ones((19,32,32,3),np.float32);y=np.arange(19,dtype=np.int32)%10
    kwargs=dict(seed=2025,round_id=0,node_id=0,epochs=5,batch_size=512,learning_rate=.05,
        ebm_mode='none',semantics='paper_absolute_gaussian',sigma=0.)
    old=worker.export_local_memory()
    with pytest.raises(InterruptedError):worker.train(params,x,y,should_stop=lambda:True,**kwargs)
    assert worker.export_local_memory()==old
    _,record=worker.train(params,x,y,**kwargs)
    assert record['steps']==5 and record['cartMemory']['lossRevision']==REVISION
    assert record['cartMemory']['retentionForwardImages']==800
    assert len(record['cartMemory']['epochControls'])==6


@pytest.mark.parametrize('module',['basil_core.cart_barrier','basil_core.cart_validation','scripts.run_cart_train_validation'])
def test_new_modules_lazy(module):
    subprocess.run([sys.executable,'-c',f'import {module};import sys;'
        'assert not ({"tensorflow","numpy","wandb","matplotlib"}&sys.modules.keys())'],check=True)


def test_default_validation_cli_does_not_train_or_load_test():
    result=subprocess.run([sys.executable,'scripts/run_cart_train_validation.py'],capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['trainingStarted'] is False and plan['officialTestLoaded'] is False


def test_validation_report_rejects_a_completed_summary_missing_comparison_arms(tmp_path):
    from scripts.report_cart_train_validation import review
    (tmp_path/'split.json').write_text(json.dumps(dict(fitIndices=list(range(49000)),
        validationIndices=list(range(49000,50000)))))
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='completed',stage='budgeted_clean',
        horizon=10,outcomes={'prior':{}})))
    with pytest.raises(ValueError,match='predeclared set'):
        review(tmp_path)


def test_validation_runtime_identity_binds_gpu_and_settings_without_live_load(monkeypatch):
    from types import SimpleNamespace
    import basil_core.execution_profile as profiles
    from scripts.run_cart_train_validation import runtime_identity
    monkeypatch.delenv(profiles.ENV,raising=False)
    device=dict(index=0,uuid='GPU-abc123',name='test GPU',driver_version='test driver',
                **{'memory.free':10000,'utilization.gpu':0})
    monkeypatch.setattr(profiles,'gpu_inventory',lambda:[device])
    tf=SimpleNamespace(__version__='test',sysconfig=SimpleNamespace(get_build_info=lambda:{}),
        config=SimpleNamespace(threading=SimpleNamespace(get_intra_op_parallelism_threads=lambda:1,
            get_inter_op_parallelism_threads=lambda:1),
            experimental=SimpleNamespace(tensor_float_32_execution_enabled=lambda:False)))
    np=SimpleNamespace(__version__='test')
    original=runtime_identity(tf,np,'GPU')
    assert original['gpu']['uuid']=='GPU-abc123' and original['gpuMemoryLimitMiB']==6144
    device['memory.free']=9000;device['utilization.gpu']=95
    assert original==runtime_identity(tf,np,'GPU')
    device['uuid']='GPU-def456'
    assert original!=runtime_identity(tf,np,'GPU')
    def no_gpu_query():raise AssertionError('CPU identity must not require NVIDIA tools')
    monkeypatch.setattr(profiles,'gpu_inventory',no_gpu_query)
    assert runtime_identity(tf,np,'CPU')['gpu'] is None


def test_budgeted_barrier_correction_is_bounded_without_scaling_ebm_or_base():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_barrier import barrier_gradients
    from basil_core.dirichlet_calibration import calibrated_anchor_gradients
    model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(10,kernel_initializer='ones')])
    x=tf.constant([[1.],[2.]]);y=tf.constant([0,1]);ref=[tf.identity(v) for v in model.trainable_variables]
    average=np.zeros((10,2),np.float32);average[0,0]=1.;average[1,1]=1.
    base,values=calibrated_anchor_gradients(model,x,y,ref,tf.zeros(10),coefficient=.0001)
    actual,result=barrier_gradients(model,x,y,ref,tf.zeros(10),x,y,average,tf.ones(10),
        tf.constant([1.,1.]+[0.]*8),coefficient=.0001,correction_ratio_cap=.1)
    change=tf.linalg.global_norm([a-b for a,b in zip(actual,base)])
    assert float(change)<=.1*float(values[4])+1e-7
    assert float(result[18])<=1 and float(result[16])==pytest.approx(float(change),abs=1e-7)


def test_noisy_budgeted_profiles_are_paired_and_fixed_before_execution():
    from basil_core.cart_validation import configurations
    from basil_core.research_protocol import validate_config
    configs=configurations('a'*64,'b'*64,'CPU','noisy_attack')
    assert [c['runId'] for c in configs]==['prior_B','budgeted_B','prior_C','budgeted_C','prior_D','budgeted_D']
    for c in configs:
        validate_config(c)
        assert c['nRounds']==5 and c['attackStartRound']==1
        assert c['useChannelNoise']==(c['conditionId'] in 'CD')
        assert c['channelNoiseSigmaAbsolute']==(.01 if c['conditionId'] in 'CD' else 0.)
        assert c['officialTestUsed'] is False


@pytest.mark.parametrize('condition',['C','D'])
def test_budgeted_ring_resume_restores_exact_updates_and_class_memory(tmp_path,monkeypatch,condition):
    import numpy as np
    import tensorflow as tf
    import basil_core.research_protocol as protocol
    from basil_core.cart_validation import configurations,make_worker
    from basil_core.protocol_checkpoint import save_checkpoint,load_checkpoint

    def tiny_model(*_):
        return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=17))])

    monkeypatch.setattr(protocol,'build_model',tiny_model)
    config=next(c for c in configurations('a'*64,'b'*64,'CPU','noisy_attack')
                if c['runId']=='budgeted_'+condition)
    x=np.random.default_rng(4).uniform(0,1,(1000,32,32,3)).astype(np.float32)
    y=np.repeat(np.arange(10,dtype=np.int32),100)
    make=lambda model:make_worker(model,config)
    complete=protocol.run_research_protocol(config,x,y,x[:20],y[:20],worker_factory=make)
    completed=[]
    def checkpoint(state):
        save_checkpoint(tmp_path/'checkpoint.npz',config,state,partition_hash='a'*64,initial_hash='b'*64)
    # Interrupt after the first attacked round begins, with populated local
    # teachers and channel-noise history, rather than only during initialization.
    with pytest.raises(InterruptedError):
        protocol.run_research_protocol(config,x,y,x[:20],y[:20],worker_factory=make,
            activation_callback=lambda record:completed.append(record),
            should_stop=lambda:len(completed)>=13,checkpoint_callback=checkpoint)
    restored=load_checkpoint(tmp_path/'checkpoint.npz',config,partition_hash='a'*64,initial_hash='b'*64)
    assert restored['next_activation']==13
    seen=[]
    resumed=protocol.run_research_protocol(config,x,y,x[:20],y[:20],worker_factory=make,
        resume_state=restored,activation_callback=lambda record:seen.append((record['round'],record['nodeId'])))
    assert seen[0]==(1,3) and len(seen)==37
    for field in ('averageAccuracy','worstNodeAccuracy','selectedSources','epochOverallAccuracy'):
        np.testing.assert_array_equal(complete[field],resumed[field])
    for expected,actual in zip(complete['finalNodeParams'],resumed['finalNodeParams']):
        for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b)
    assert complete['telemetry']==resumed['telemetry']
