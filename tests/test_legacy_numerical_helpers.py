"""Reproductions of failures found in the repository-wide numerical audit."""
import pytest


@pytest.mark.parametrize('sigma',[0.,.1])
def test_standalone_ebm_keras3_value_and_second_order_gradient(sigma):
    import numpy as np
    import tensorflow as tf
    from noise_comm.ebm_tf import ebm_regularized_loss
    model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(2,use_bias=False)])
    variable=model.trainable_variables[0]
    variable.assign([[.4,-.2]])
    x=tf.constant([[1.],[2.]]);y=tf.constant([0,1])
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce=loss_fn(y,model(x))
        first=inner.gradient(ce,variable)
    hessian=outer.jacobian(first,variable).numpy().reshape(2,2)
    with tf.GradientTape() as applied:
        objective=ebm_regularized_loss(loss_fn,model,x,y,sigma)
    gradient=applied.gradient(objective,variable)
    expected=first.numpy().ravel()+2*sigma**2*hessian@first.numpy().ravel()
    np.testing.assert_allclose(gradient.numpy().ravel(),expected,atol=1e-7,rtol=1e-6)
    assert float(objective)==pytest.approx(float(ce)+sigma**2*float(tf.reduce_sum(first**2)),rel=1e-6)


@pytest.mark.parametrize('stateful',[False,True])
def test_failed_wcm_restores_parameters_and_applies_the_announced_fallback(monkeypatch,stateful):
    import numpy as np
    import tensorflow as tf
    import basil_core.trainer as trainer
    def model():
        layers=[tf.keras.layers.Input((1,))]
        if stateful:layers += [tf.keras.layers.BatchNormalization(),tf.keras.layers.Dropout(.25,seed=17)]
        return tf.keras.Sequential(layers+[tf.keras.layers.Dense(2,
            kernel_initializer=tf.keras.initializers.Constant(.2))])
    ordinary,wcm=model(),model()
    batch=[(tf.constant([[1.],[2.]]),tf.constant([0,1]))]
    trainer.localUpdate(ordinary,batch,1,.05,stepsPerEpoch=1)
    def fail(**kwargs):
        for variable in kwargs['model'].variables:variable.assign_add(tf.ones_like(variable)*3)
        raise RuntimeError('Deliberate failure during an inner solve')
    monkeypatch.setattr(trainer,'WCM_AVAILABLE',True)
    monkeypatch.setattr(trainer,'wcmStep',fail)
    trainer.localUpdate(wcm,batch,1,.05,stepsPerEpoch=1,noiseModel='wcm',sigma=.1)
    for first,second in zip(ordinary.variables,wcm.variables):
        np.testing.assert_array_equal(first.numpy(),second.numpy())


def test_historical_microbatch_ebm_is_not_misidentified_as_full_batch_ebm():
    import numpy as np
    import tensorflow as tf
    from basil_core.experiment_engine import _gradient_norm_regularized_gradients
    from noise_comm.ebm_tf import ebm_regularized_loss
    model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(2,use_bias=False)])
    model.trainable_variables[0].assign([[.4,-.2]])
    x=tf.constant([[.1],[2.],[.2],[3.]]);y=tf.constant([0,0,1,1])
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    chunked=_gradient_norm_regularized_gradients(model,model.trainable_variables,x,y,.1,micro_batch_size=2)
    with tf.GradientTape() as tape:
        full=ebm_regularized_loss(loss_fn,model,x,y,.1**.5)
    exact=tape.gradient(full,model.trainable_variables)
    assert np.linalg.norm(chunked[0].numpy()-exact[0].numpy())>1e-4
