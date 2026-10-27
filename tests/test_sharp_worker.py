from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')
import pytest


def test_second_order_rla_and_proximal_gradient_match_analytic_binary_model():
    import tensorflow as tf
    from basil_core.sharp_worker import objective_gradients
    model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(2,use_bias=False)])
    model.layers[-1].kernel.assign([[.3,-.2]])
    anchor=(tf.constant([[.1,.4]],tf.float32),)
    gradients,values=objective_gradients(model,tf.constant([[1.]]),tf.constant([0]),anchor,tf.constant(.12),tf.constant(.7))
    p=np.exp([.3,-.2]);p/=p.sum();g=p-np.array([1.,0.]);h=np.diag(p)-np.outer(p,p)
    expected=g+2*.12*(h@g)+.7*(np.array([.3,-.2])-np.array([.1,.4]))
    np.testing.assert_allclose(gradients[0].numpy()[0],expected,rtol=1e-6,atol=1e-7)
    assert float(values[3])>0


@pytest.mark.parametrize('classes',[10,100])
def test_cart_repairs_classifier_without_changing_backbone(classes):
    import tensorflow as tf
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_study import configuration
    from basil_core.sharp_cart import transport,probes
    from basil_core.sharp_math import class_ce
    cfg=configuration('cifar100' if classes==100 else 'mnist','iid',2025,'attack_noise_defense_rla_cart',.01,nodes=100)
    inputs=tf.keras.Input((classes,))
    features=tf.keras.layers.Dense(classes,use_bias=False)(inputs)
    model=tf.keras.Model(inputs,tf.keras.layers.Dense(classes)(features))
    model.layers[-2].kernel.assign(np.eye(classes,dtype=np.float32))
    model.layers[-1].kernel.assign(3*np.eye(classes,dtype=np.float32));model.layers[-1].bias.assign(np.zeros(classes))
    worker=SharpWorker(cfg,model=model)
    y=np.repeat(np.arange(classes,dtype=np.int32),4);x=np.eye(classes,dtype=np.float32)[y]
    fit,check=probes(y,2025,99)
    assert not np.intersect1d(fit,check).size
    reference=worker.export();_,losses=class_ce(worker.outputs(reference,x[fit]),y[fit])
    old=dict(params=reference,losses=losses.tolist(),allowance=np.zeros(classes).tolist())
    params=worker.tensors(reference);params[-2]=params[-2].copy();params[-2][0,0]=-3.;params[-2][0,1]=3.
    proposal=np.concatenate([v.ravel() for v in params])
    final,memory,e=transport(worker,proposal,old,x,y,99,199)
    assert e['scale']>0 and e['checkAfter']<e['checkBefore'] and e['classGates'][0]>0
    end=worker.dimension-sum(worker.sizes[-2:])
    np.testing.assert_array_equal(final[:end],proposal[:end])
    assert e['fitCheckDisjoint'] and len(e['classGates'])==classes
    import json
    json.dumps(e,allow_nan=False)
    json.dumps({k:v for k,v in memory.items() if k!='params'},allow_nan=False)


def test_real_cart_rounds_commit_serializable_checkpoints(tmp_path):
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_study import configuration
    from basil_core.sharp_protocol import run,save_state,load_state
    cfg=configuration('mnist','iid',2025,'attack_noise_defense_rla_cart',.01,nodes=4,rounds=2)
    worker=SharpWorker(cfg)
    y=np.tile(np.arange(2,dtype=np.int32),40)
    x=np.zeros((len(y),28,28,1),np.float32)
    x[y==0,:14]=1.;x[y==1,14:]=1.
    chunks=np.array_split(np.arange(len(y)),4)
    path=tmp_path/'checkpoint.npz';contract=dict(config=cfg,source={'test':'actual_tensorflow'})
    state=run(cfg,worker,x,y,chunks,x[:8],y[:8],checkpoint=lambda s:save_state(path,s,contract))
    restored=load_state(path,contract,worker.dimension)
    assert restored['cursor']==8 and len(restored['references'])==4
    assert len(restored['roundMetrics'])==2
    np.testing.assert_array_equal(state['nodeParams'],restored['nodeParams'])


def test_new_worker_takes_five_steps_not_five_epochs():
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_study import configuration
    cfg=configuration('mnist','iid',2025,'clean',0.,nodes=10,rounds=1)
    worker=SharpWorker(cfg)
    x=np.random.default_rng(2).uniform(size=(30,28,28,1)).astype(np.float32);y=np.arange(30,dtype=np.int32)%10
    initial=worker.export();trained,record=worker.train(initial,x,y,0,0,0.,np.arange(14))
    assert record['optimizerSteps']==5 and len(record['steps'])==5
    assert not np.array_equal(initial,trained) and record['proximalMu']==0
