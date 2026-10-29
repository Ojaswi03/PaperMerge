import os
import numpy as np
import pytest
from basil_core.cart_local_prior import log_prior


def test_floor_one_prior_uses_local_counts_including_absent_classes():
    actual=np.exp(log_prior(np.array([0,0,0,1]),3))
    np.testing.assert_allclose(actual,[3/5,1/5,1/5],rtol=1e-6)
    for y in [np.array([]),np.array([-1]),np.array([3]),np.array([.5])]:
        with pytest.raises(ValueError):log_prior(y,3)


def test_adjusted_hessian_vector_gradient_matches_explicit_toy_hessian():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    from basil_core.cart_local_prior import objective_gradients
    model=tf.keras.Sequential([tf.keras.layers.Input((2,),dtype=tf.float64),
        tf.keras.layers.Dense(3,dtype=tf.float64,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=71))])
    images=tf.constant([[.3,.8],[-.2,.5]],tf.float64);labels=tf.constant([0,2]);prior=tf.constant([-.2,-1.,-2.],tf.float64)
    variables=model.trainable_variables
    anchor=tuple(tf.identity(v)-tf.cast(.02,tf.float64) for v in variables)
    coefficient=tf.constant(.03,tf.float64);mu=tf.constant(.2,tf.float64)
    with tf.GradientTape(persistent=True) as outer:
        with tf.GradientTape() as inner:
            ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=model(images)+prior))
        g=inner.gradient(ce,variables);flat=tf.concat([tf.reshape(v,[-1]) for v in g],0)
    h=tf.concat([tf.reshape(outer.jacobian(flat,v),[len(flat),-1]) for v in variables],1)
    prox=tf.concat([tf.reshape(v-a,[-1]) for v,a in zip(variables,anchor)],0)
    expected=flat+2*coefficient*tf.linalg.matvec(h,flat)+mu*prox
    actual,_=objective_gradients(model,images,labels,anchor,coefficient,mu,prior)
    np.testing.assert_allclose(tf.concat([tf.reshape(v,[-1]) for v in actual],0),expected,atol=1e-12,rtol=1e-12)
