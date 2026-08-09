import gc
import json
import unittest

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from gui.campaign4 import make_config

if tf is not None:
    from basil_core.campaign4_engine import (
        _base_and_regularizer_gradients,
        _bounded_adaptive_coefficient,
        _relative_l2_channel_noise,
        run_campaign_four,
    )
    from basil_core.trainer import lossFn


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class Campaign4EngineTests(unittest.TestCase):
    class TinyModel:
        def __init__(self):
            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(4,)),
                    tf.keras.layers.Dense(12, activation="relu"),
                    tf.keras.layers.Dense(10),
                ]
            )

        @property
        def trainable_weights(self):
            return self.model.trainable_weights

        def __call__(self, values, training=False):
            return self.model(values, training=training)

    def tearDown(self):
        tf.keras.backend.clear_session()
        gc.collect()

    def _data(self, seed):
        rng = np.random.default_rng(seed)
        train_loaders = []
        probe_batches = []
        class_counts = []
        client_indices = []
        for node in range(3):
            labels = np.tile(np.arange(10, dtype=np.int32), 2)
            features = rng.normal(size=(20, 4)).astype(np.float32)
            features[:, 0] += labels.astype(np.float32) / 10.0
            train_loaders.append(
                tf.data.Dataset.from_tensor_slices((features, labels))
                .repeat()
                .batch(4, drop_remainder=True)
            )
            probe_batches.append((features, labels))
            class_counts.append(np.bincount(labels, minlength=10))
            client_indices.append(np.arange(node * 20, (node + 1) * 20))
        labels = np.tile(np.arange(10, dtype=np.int32), 3)
        features = rng.normal(size=(30, 4)).astype(np.float32)
        test_loader = tf.data.Dataset.from_tensor_slices((features, labels)).batch(10)
        return train_loaders, test_loader, {
            "probeBatches": probe_batches,
            "classCounts": np.asarray(class_counts, dtype=np.int32),
            "clientIndices": client_indices,
        }

    def _config(self, **overrides):
        config = make_config(
            split="nonIID",
            approach=overrides.pop("approach", "merged"),
            environment=overrides.pop("environment", "hidden_noise"),
            mitigation=overrides.pop("mitigation", "ss_ebm"),
            sigma=overrides.pop("sigma", 0.4),
            seed=overrides.pop("seed", 77),
            rounds=2,
        )
        config.update(
            {
                "nNodes": 3,
                "nRounds": 2,
                "localEpochs": 1,
                "stepsPerEpoch": 1,
                "batchSize": 4,
                "internalMicroBatchSize": 4,
                "basilMemorySize": 2,
                "attackerIds": "1" if config["attackHidden"] else "",
                "attackHiddenStart": 0,
                "useLrDecay": False,
            }
        )
        config.update(overrides)
        return config

    def test_second_order_gradient_matches_declared_objective(self):
        tf.keras.utils.set_random_seed(13)
        model = self.TinyModel().model
        weights = model.trainable_weights
        x = tf.constant(np.arange(32, dtype=np.float32).reshape(8, 4) / 31.0)
        y = tf.constant(np.arange(8, dtype=np.int32) % 4)
        base, regularizer = _base_and_regularizer_gradients(
            model, weights, x, y, micro_batch_size=8
        )
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                loss = lossFn(y, model(x, training=True))
            expected_base = inner.gradient(
                loss, weights, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
            norm_squared = tf.add_n(
                [tf.reduce_sum(tf.square(value)) for value in expected_base]
            )
        expected_regularizer = outer.gradient(
            norm_squared,
            weights,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        for actual, expected in zip(base, expected_base):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)
        for actual, expected in zip(regularizer, expected_regularizer):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)

    def test_adaptive_coefficient_is_bounded_and_rate_limited(self):
        value = _bounded_adaptive_coefficient(
            previous=tf.constant(0.001),
            stress=tf.constant(0.6),
            base_norm=tf.constant(2.0),
            regularizer_norm=tf.constant(0.01),
            target_ratio_base=tf.constant(0.05),
            stress_gain=tf.constant(0.25),
            ratio_min=tf.constant(0.02),
            ratio_max=tf.constant(0.35),
            coefficient_min=tf.constant(1e-6),
            coefficient_max=tf.constant(0.01),
            beta=tf.constant(0.9),
            max_change_factor=tf.constant(2.0),
        )
        self.assertGreaterEqual(float(value.numpy()), 0.0005)
        self.assertLessEqual(float(value.numpy()), 0.002)

    def test_relative_noise_is_deterministic_and_has_requested_norm(self):
        params = [tf.reshape(tf.range(1, 101, dtype=tf.float32), (10, 10))]
        first, _, first_relative = _relative_l2_channel_noise(
            params, sigma=0.6, base_seed=5, seed_parts=("link", 1)
        )
        second, _, second_relative = _relative_l2_channel_noise(
            params, sigma=0.6, base_seed=5, seed_parts=("link", 1)
        )
        np.testing.assert_array_equal(first[0].numpy(), second[0].numpy())
        self.assertAlmostEqual(float(first_relative.numpy()), float(second_relative.numpy()))
        self.assertAlmostEqual(float(first_relative.numpy()), 0.6, delta=0.12)

    def test_run_emits_node_events_and_complete_telemetry(self):
        config = self._config()
        train, test, metadata = self._data(config["seed"])
        events = []
        rounds = []
        result = run_campaign_four(
            config=config,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
            node_callback=events.append,
            round_callback=lambda *values: rounds.append(values),
        )
        self.assertFalse(result["stopped"])
        self.assertEqual(result["metrics"]["avg_history"].shape, (2,))
        self.assertEqual(result["metrics"]["per_node_history"].shape, (2, 3))
        self.assertEqual(result["telemetry"]["stress_ema"].shape, (2, 3))
        self.assertEqual(
            result["telemetry"]["ebm_requested_coefficient"].shape, (2, 3)
        )
        self.assertEqual(result["telemetry"]["outgoing_noise_norm"].shape, (2, 3, 2))
        self.assertEqual(result["telemetry"]["candidate_senders"].shape, (2, 3, 2))
        self.assertEqual(len(events), 6)
        self.assertEqual(len(rounds), 2)
        self.assertEqual(len(rounds[-1][-1]), 3)
        self.assertIn("selection", events[-1])
        self.assertIn("ebm", events[-1])
        self.assertIn("outgoing", events[-1])
        encoded = json.dumps(events[-1])
        self.assertLess(len(encoded), 20_000)
        self.assertNotIn("params", encoded.lower())

    def test_weight_decay_reduces_final_model_norm(self):
        train, test, metadata = self._data(seed=99)
        config_off = self._config(
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=99,
            nRounds=6,
            weightDecayCoefficient=0.0,
        )
        config_on = self._config(
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=99,
            nRounds=6,
            weightDecayCoefficient=0.5,
        )
        result_off = run_campaign_four(
            config=config_off,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
        )
        result_on = run_campaign_four(
            config=config_on,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
        )
        norm_off = float(result_off["telemetry"]["model_norm"][-1].mean())
        norm_on = float(result_on["telemetry"]["model_norm"][-1].mean())
        self.assertLess(norm_on, norm_off)


if __name__ == "__main__":
    unittest.main()
