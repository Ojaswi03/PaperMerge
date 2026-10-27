"""Graph execution retains SGD updates, momentum and fresh activation state."""
import pytest


def test_compiled_sgd_matches_eager_parameters_momentum_and_reset():
    import numpy as np
    import tensorflow as tf
    from basil_core.compiled_optimizer import CompiledSGD
    from tests.test_factorial_worker import small_model
    first=small_model();second=small_model()
    eager=tf.keras.optimizers.SGD(.03,momentum=.9)
    compiled=CompiledSGD(second,.03,.9)
    rng=np.random.default_rng(164)
    for _ in range(50):
        gradients=[tf.constant(rng.normal(size=v.shape).astype(np.float32)) for v in first.trainable_variables]
        eager.apply_gradients(zip(gradients,first.trainable_variables))
        compiled.apply_gradients(zip(gradients,second.trainable_variables))
    for a,b in zip(first.get_weights(),second.get_weights()):np.testing.assert_array_equal(a,b)
    for a,b in zip(eager.momentums,compiled.momentums):np.testing.assert_array_equal(a.numpy(),b.numpy())
    assert int(compiled.iterations)==50
    fresh=CompiledSGD(second,.03,.9)
    assert int(fresh.iterations)==0
    assert all(np.all(v.numpy()==0) for v in fresh.momentums)
    with pytest.raises(ValueError):compiled.apply_gradients(zip(gradients,first.trainable_variables))
