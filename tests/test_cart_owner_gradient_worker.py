"""Runner/worker contracts, synthetic only; never starts a research queue."""
import json
import subprocess
import sys

import pytest


def test_default_plan_is_lazy_bounded_and_clean():
    process=subprocess.run([sys.executable,'-c',
        "from scripts.run_cart_owner_gradient_diagnostic import main; main(); import sys; assert 'tensorflow' not in sys.modules; assert 'numpy' not in sys.modules; assert 'wandb' not in sys.modules"],
        check=True,capture_output=True,text=True)
    plan=json.loads(process.stdout)
    cfg=plan['config']
    assert plan['trainingRunsStarted']==0 and cfg['nRounds']==1 and cfg['localEpochs']==5
    assert cfg['helperSamples']==16 and cfg['basilMemorySize']==5
    assert not any(cfg[k] for k in ('channelNoise','ebm','liveTrackingEnabled','featureReplay','representationPreservation','queueStarted','automaticPromotion'))
    assert cfg['actualAttackers']==[] and cfg['gradientClipNorm'] is None


def synthetic_worker():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_gradient_worker import OwnerGradientCartWorker
    def model():
        return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3,activation='tanh',kernel_initializer=tf.keras.initializers.Constant(.0001)),
            tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.RandomUniform(-.1,.1,seed=3))])
    worker=OwnerGradientCartWorker(model(),model())
    params=worker.export(); worker.load(params)
    for a,p in zip(worker.reference,params): a.assign(p+.001)
    worker.anchor_mu.assign(.007)
    worker.source_x=np.random.default_rng(9).uniform(0,1,(16,32,32,3)).astype(np.float32)
    worker.source_y=np.zeros(16,np.int32); worker.owner_pools=[np.arange(16)]*10
    worker.older_classes=(0,); worker.receiver=1; worker.seed=2025; worker.round_id=0
    return worker


def test_applied_gradient_is_class_mean_plus_frozen_anchor_not_sample_mean():
    import numpy as np
    import tensorflow as tf
    from basil_core.research_audit import augment_batch
    worker=synthetic_worker(); x=worker.source_x[:3]; y=tf.ones(3,tf.int32); key=tf.constant([7,8])
    actual,values=worker._gradients(tf.constant(x),y,key,'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(1.))
    owner=worker.owner_records[0]['owners'][0]
    images=augment_batch(worker.source_x[owner['trainingIndices']],owner['augmentationKey'])
    local=augment_batch(x,key)
    with tf.GradientTape() as tape:
        a=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=worker.model(local)))
        b=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros(16,tf.int32),logits=worker.model(images)))
        anchor=.007/2*tf.add_n([tf.reduce_sum((w-r)**2) for w,r in zip(worker.model.trainable_variables,worker.reference)])
        total=(a+b)/2+anchor
    expected=tape.gradient(total,worker.model.trainable_variables)
    for g,r in zip(actual,expected): np.testing.assert_allclose(g.numpy(),r.numpy(),rtol=2e-5,atol=1e-7)
    assert float(values[2])==pytest.approx(float(total),rel=1e-6)
    assert worker.owner_records[0]['classWeights']=={'0':.5,'1':.5}
    assert worker.owner_records[0]['ownerOptimizerSteps']==0


def test_lambda_ignored_fresh_keys_and_helpers_do_not_update():
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import params_hash
    worker=synthetic_worker(); x=tf.constant(worker.source_x[:3]); y=tf.ones(3,tf.int32); key=tf.constant([7,8])
    before=params_hash(worker.export())
    a,v=worker._gradients(x,y,key,'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(1.))
    first=worker.owner_records[0]
    worker.owner_records=[]
    b,u=worker._gradients(x,y,key,'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(999999.))
    for g,h in zip(a,b): np.testing.assert_array_equal(g.numpy(),h.numpy())
    np.testing.assert_array_equal([float(z) for z in v],[float(z) for z in u])
    assert before==params_hash(worker.export())==params_hash([w.numpy() for w in worker.helper_model.trainable_variables])
    frozen=params_hash([r.numpy() for r in worker.reference])
    tf.keras.optimizers.SGD(.05).apply_gradients(zip(a,worker.model.trainable_variables))
    worker._gradients(x,y,key,'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(1.))
    assert worker.owner_records[-1]['modelHash']!=first['modelHash']
    assert worker.owner_records[-1]['owners'][0]['augmentationKey']!=first['owners'][0]['augmentationKey']
    assert params_hash([r.numpy() for r in worker.reference])==frozen


def test_own_only_is_ordinary_ce_plus_cart_anchor():
    import numpy as np
    import tensorflow as tf
    worker=synthetic_worker(); worker.older_classes=()
    x=tf.constant(worker.source_x[:3]); y=tf.ones(3,tf.int32); key=tf.constant([7,8])
    local,_,_=worker._local_ce(x,y,key)
    actual,_=worker._gradients(x,y,key,'none','paper_absolute_gaussian',tf.constant(0.),tf.constant(0.))
    for g,a,w,r in zip(local,actual,worker.model.trainable_variables,worker.reference):
        np.testing.assert_array_equal(a.numpy(),(g+worker.anchor_mu*(w-r)).numpy())


@pytest.mark.parametrize('mode,sigma',[('gradient_norm_objective',0.),('none',.01)])
def test_noise_or_ebm_is_rejected_in_bounded_candidate(mode,sigma):
    import tensorflow as tf
    worker=synthetic_worker()
    with pytest.raises(ValueError,match='clean CE'):
        worker._gradients(tf.constant(worker.source_x[:3]),tf.ones(3,tf.int32),tf.constant([1,2]),
            mode,'paper_absolute_gaussian',tf.constant(sigma),tf.constant(1.))


def test_future_owners_rejected_before_training():
    worker=synthetic_worker()
    with pytest.raises(ValueError,match='already visited'):
        worker.train([],None,None,source_x=None,source_y=None,owner_pools=None,older_classes=(2,),
            node_id=1,round_id=0)


def test_summary_refuses_incomplete_evidence():
    from scripts.report_cart_owner_gradient_diagnostic import summarize
    with pytest.raises(ValueError,match='Ten complete'): summarize([])
