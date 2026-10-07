import json

import numpy as np
import pytest
import tensorflow as tf

from basil_core.research_audit import ProtocolAudit, augment_batch, tensor_stats
from basil_core.research_protocol import (
    SequentialWorker, apply_channel_noise, evaluate_params,
    gradient_norm_objective_gradients, keyed_rng, parameter_stats, tensor_noise_variance,
)


def test_finite_float32_weights_do_not_overflow_norm_or_channel():
    weights = [np.full(1000, 1e20, np.float32)]
    dimension, norm = parameter_stats(weights)
    assert dimension == 1000 and norm == pytest.approx(np.sqrt(1000) * 1e20)
    perturbed, stats = apply_channel_noise(weights, semantics="relative_l2_gaussian",
        sigma=.4, rng=keyed_rng(2025, "channel_noise", 0, 0, 1))
    assert np.isfinite(perturbed[0]).all()
    assert np.isfinite(stats["noiseL2Norm"])
    assert stats["relativeNoiseL2"] == pytest.approx(.4, rel=.08)


def test_variance_accumulation_preserves_finite_coefficient_and_stop_gradient():
    model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(100, use_bias=False)])
    model.trainable_variables[0].assign(tf.fill((1, 100), 1e19))
    with tf.GradientTape() as tape:
        variance = tensor_noise_variance(model, "relative_l2_gaussian", .2)
    assert np.isfinite(float(variance)) and float(variance) == pytest.approx(4e36, rel=1e-6)
    assert tape.gradient(variance, model.trainable_variables) == [None]


def test_full_ebm_equals_gradient_plus_hessian_vector_correction():
    model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(2, use_bias=False)])
    model.trainable_variables[0].assign([[.4, -.2]])
    x, y, c = tf.constant([[1.], [2.]]), tf.constant([0, 1]), .3
    weight = model.trainable_variables[0]
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=model(x)))
        base = inner.gradient(loss, weight)
    hessian = outer.jacobian(base, weight).numpy().reshape(2, 2)
    robust, *_ = gradient_norm_objective_gradients(model, x, y, c)
    expected = base.numpy().reshape(-1) + 2 * c * hessian @ base.numpy().reshape(-1)
    assert np.allclose(robust[0].numpy().reshape(-1), expected, rtol=1e-5, atol=1e-6)
    assert not np.allclose(expected, (1 + c) * base.numpy().reshape(-1))


def test_evaluation_rejects_nonfinite_logits_instead_of_fabricating_accuracy():
    class Worker:
        def load(self, params): pass
        def predict(self, x): return tf.fill((len(x), 10), float("nan"))
    with pytest.raises(tf.errors.InvalidArgumentError, match="Non-finite evaluation logits"):
        evaluate_params(Worker(), [], np.zeros((2, 32, 32, 3), np.float32), np.array([0, 1]))


def test_confusions_are_evaluation_only_and_cover_all_five_epochs(tmp_path):
    model = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)),
        tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(10)])
    worker = SequentialWorker(model)
    audit = ProtocolAudit(tmp_path / "audit")
    worker.audit = audit
    audit.activation(0, 0)
    x, y = np.zeros((10, 32, 32, 3), np.float32), np.arange(10, dtype=np.int32)
    def evaluation():
        accuracy, per_class = evaluate_params(worker, worker.export(), x, y)
        audit.evaluation()
        return {"globalTestAccuracy": accuracy, "perClassGlobalTestAccuracy": per_class.tolist()}
    worker.train(worker.export(), x[:1], y[:1], seed=2025, round_id=0, node_id=0,
        epochs=5, batch_size=512, learning_rate=.05, ebm_mode="none", evaluation=evaluation)
    audit.close()
    matrices = json.loads((tmp_path / "audit/confusions.json").read_text())
    assert [m["stage"] for m in matrices] == list(range(6))
    assert all(np.array(m["confusion"]).sum() == 10 for m in matrices)
    assert int(worker.optimizer.iterations) == 5


def test_augmentation_is_keyed_and_audit_does_not_advance_random_streams():
    x = tf.reshape(tf.linspace(0., 1., 32 * 32 * 3), (1, 32, 32, 3))
    key = tf.constant([123, 456])
    first = augment_batch(x, key)
    _ = keyed_rng(2025, "unrelated").normal(size=100)
    assert np.array_equal(first.numpy(), augment_batch(x, key).numpy())


def test_tensor_statistics_distinguish_nan_infinity_and_finite_values():
    stats = tensor_stats(np.array([1., -2., np.inf, -np.inf, np.nan], np.float32))
    assert (stats["nan"], stats["positiveInf"], stats["negativeInf"]) == (1, 1, 1)
    assert stats["finiteMin"] == -2 and stats["finiteMax"] == 1


@pytest.mark.parametrize("mode", ["none", "legacy_gradient_scale", "gradient_norm_objective"])
def test_failure_replay_does_not_substitute_full_ebm_for_other_modes(tmp_path, mode):
    class Model:
        def __init__(self):
            self.model = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)),
                tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(10)])
        @property
        def trainable_variables(self): return self.model.trainable_variables
        def __call__(self, x, training=False): return self.model(x, training=training)
    worker = SequentialWorker(Model())
    audit = ProtocolAudit(tmp_path / mode)
    pending = {"params": worker.export(), "x": np.zeros((1, 32, 32, 3), np.float32),
        "y": np.zeros(1, np.int32), "key": np.array([1, 2], np.int32),
        "context": {"mode": mode, "semantics": "paper_absolute_gaussian", "sigma": .2, "legacy_lambda": 25.}}
    audit.replay_values(worker, pending)
    audit.close()
    values = json.loads((tmp_path / mode / "failure_tensor_values.json").read_text())
    if mode == "gradient_norm_objective":
        assert values["objective"] > values["crossEntropy"]
        assert len(values["secondOrderCorrections"]) == 2
        assert values["coefficient"] == pytest.approx(.04)
    else:
        assert values["objective"] == values["crossEntropy"]
        assert values["secondOrderCorrections"] == []
        factor = 2 if mode == "legacy_gradient_scale" else 1
        assert values["robustGradientNorm"] == pytest.approx(factor * values["baseGradientNorm"])
