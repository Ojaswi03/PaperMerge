"""Independent CE/EBM/CART paths and checkpointed class memory."""
import pytest


def small_model():
    import tensorflow as tf
    return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=23))])


@pytest.mark.parametrize('condition',['clean','noise_ebm','attack_noise_basil_ebm_cart'])
def test_actual_worker_five_epochs_finite_updates_and_no_hidden_prior_or_anchor(condition):
    import numpy as np
    from basil_core.factorial_study import configuration,CONDITIONS
    from basil_core.factorial_worker import make_worker
    chosen=next(c for c in CONDITIONS if c.name==condition)
    cfg=configuration(chosen,'dirichlet',2025,.01 if chosen.noise else 0.,stage='smoke',device='CPU')
    worker=make_worker(small_model(),cfg)
    x=np.random.default_rng(23).uniform(0,1,(11,32,32,3)).astype(np.float32)
    y=np.arange(11,dtype=np.int32)%3
    initial=worker.export()
    final,row=worker.train(initial,x,y,seed=2025,round_id=0,node_id=0,epochs=5,
        batch_size=128,learning_rate=.01,ebm_mode=cfg['ebmMode'],semantics='paper_absolute_gaussian',
        sigma=cfg['channelNoiseSigmaAbsolute'])
    assert row['steps']==5
    assert all(e['samplesSeen']==11 for e in row['epochs'])
    assert worker.optimizer.momentum==.9
    assert any(not np.array_equal(a,b) for a,b in zip(initial,final))
    for step in row['optimizerTelemetry']:
        assert step['ebmCoefficient']==pytest.approx(.0001 if chosen.noise_mitigation else 0.)
        assert all(np.isfinite(v).all() for v in step.values())
    if chosen.cart:
        assert worker.mu==0 and not worker.calibration_enabled
        assert row['logitCalibrationOffsets']==[0.]*10
        assert row['cartMemory']['probeCount']==11
        assert row['cartMemory']['epochControls']


def test_coupled_l2_is_kernel_only_and_common_to_every_arm():
    import numpy as np
    import tensorflow as tf
    from basil_core.factorial_worker import regularize
    model=small_model()
    zeros=[tf.zeros_like(v) for v in model.trainable_variables]
    values=[tf.constant(0.)]*6
    actual,_=regularize(model,zeros,values,.0005)
    np.testing.assert_allclose(actual[0].numpy(),.0005*model.trainable_variables[0].numpy())
    np.testing.assert_array_equal(actual[1].numpy(),np.zeros(10))


@pytest.mark.parametrize('module',['basil_core.factorial_study','basil_core.factorial_model','basil_core.factorial_worker','scripts.run_factorial_study'])
def test_modules_have_no_import_time_numerical_side_effects(module):
    import subprocess
    import sys
    subprocess.run([sys.executable,'-c',f'import {module};import sys;'
        'assert not ({"tensorflow","numpy","wandb","matplotlib"}&sys.modules.keys())'],check=True)
