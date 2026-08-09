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
        _weight_decay_control_step,
        run_campaign_four,
        SharedDeviceWorker,
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

    def test_weight_decay_control_step_is_bounded_and_rate_limited(self):
        applied, smoothed = _weight_decay_control_step(
            previous_coefficient=1e-6,
            previous_smoothed_ratio=1.0,
            model_norm=346.6,
            model_norm_round0=56.4,
            target_ratio=2.0,
            gain=0.025,
            coefficient_min=1e-6,
            coefficient_max=0.05,
            beta=0.9,
            max_change_factor=2.0,
        )
        # growth_ratio = 346.6/56.4 approx 6.15; smoothed = 0.9*1.0 + 0.1*6.15 = 1.515
        # desired = clip(0.025*(1.515-2.0), 1e-6, 0.05) -> clipped to the coefficient_min floor
        # since (1.515-2.0) is negative -- rate-limited against a previous of 1e-6
        self.assertGreaterEqual(applied, 1e-6)
        self.assertLessEqual(applied, 1e-6 * 2.0)
        self.assertAlmostEqual(smoothed, 1.515, places=3)

        # A node already at a high coefficient facing continued high growth
        # should ramp toward the ceiling, not snap there in one round.
        applied_high, _ = _weight_decay_control_step(
            previous_coefficient=0.01,
            previous_smoothed_ratio=4.0,
            model_norm=346.6,
            model_norm_round0=56.4,
            target_ratio=2.0,
            gain=0.025,
            coefficient_min=1e-6,
            coefficient_max=0.05,
            beta=0.9,
            max_change_factor=2.0,
        )
        self.assertGreater(applied_high, 0.01)
        self.assertLessEqual(applied_high, 0.01 * 2.0)

        # Zero previous must not permanently lock the coefficient at zero
        # (the multiplicative-rate-limiter degeneracy this floor exists to avoid).
        applied_from_floor, _ = _weight_decay_control_step(
            previous_coefficient=1e-6,
            previous_smoothed_ratio=5.0,
            model_norm=346.6,
            model_norm_round0=56.4,
            target_ratio=2.0,
            gain=0.025,
            coefficient_min=1e-6,
            coefficient_max=0.05,
            beta=0.9,
            max_change_factor=2.0,
        )
        self.assertGreater(applied_from_floor, 1e-6)

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

    def test_weight_decay_variable_updates_live_without_retrace(self):
        tf.keras.utils.set_random_seed(21)
        model = self.TinyModel()
        worker = SharedDeviceWorker(
            model,
            lr0=0.1,
            momentum=0.0,
            micro_batch_size=4,
            jit_compile=False,
            adaptive_config={
                "adaptiveEbmTargetRatioBase": 0.05,
                "adaptiveEbmStressGain": 0.25,
                "adaptiveEbmRatioMin": 0.02,
                "adaptiveEbmRatioMax": 0.35,
                "adaptiveEbmCoefficientMin": 1e-6,
                "adaptiveEbmCoefficientMax": 0.01,
                "adaptiveEbmBeta": 0.9,
                "adaptiveEbmMaxChangeFactor": 2.0,
            },
        )
        params = worker.export()
        optimizer_state = worker.zero_optimizer_state()
        x = tf.ones((4, 4), dtype=tf.float32)
        y = tf.zeros(4, dtype=tf.int32)

        params_zero_decay, optimizer_state, _ = worker.train(
            params,
            optimizer_state=optimizer_state,
            data_iterator=iter([(x, y)] * 10),
            first_batch=(x, y),
            total_steps=5,
            lr=0.1,
            ebm_mode="none",
            initial_coefficient=0.0,
            stress=0.0,
            prox_mu=0.0,
            weight_decay_coefficient=0.0,
        )
        norm_zero_decay = float(tf.linalg.global_norm(params_zero_decay).numpy())

        params_with_decay, _, _ = worker.train(
            params,
            optimizer_state=optimizer_state,
            data_iterator=iter([(x, y)] * 10),
            first_batch=(x, y),
            total_steps=5,
            lr=0.1,
            ebm_mode="none",
            initial_coefficient=0.0,
            stress=0.0,
            prox_mu=0.0,
            weight_decay_coefficient=0.5,
        )
        norm_with_decay = float(tf.linalg.global_norm(params_with_decay).numpy())

        self.assertLess(norm_with_decay, norm_zero_decay)

    def test_adaptive_weight_decay_ramps_up_when_norm_grows(self):
        train, test, metadata = self._data(seed=55)
        config = self._config(
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=55,
            nRounds=6,
            adaptiveWeightDecayMode="adaptive",
            adaptiveWeightDecayTargetRatio=1.1,
            adaptiveWeightDecayGain=0.05,
            adaptiveWeightDecayCoefficientMin=1e-6,
            adaptiveWeightDecayCoefficientMax=0.05,
            adaptiveWeightDecayBeta=0.9,
            adaptiveWeightDecayMaxChangeFactor=2.0,
        )
        result = run_campaign_four(
            config=config,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
        )
        coefficient = result["telemetry"]["adaptive_weight_decay_coefficient"]
        growth_ratio = result["telemetry"]["adaptive_weight_decay_growth_ratio"]
        self.assertEqual(coefficient.shape, (6, 3))
        self.assertEqual(growth_ratio.shape, (6, 3))
        # Round 0 has no prior-round norm yet -- must start at the exact 0.0
        # default (not the coefficient_min floor, which would falsely claim
        # "evidence of growth" before any has been observed).
        self.assertTrue(np.all(coefficient[0] == 0.0))
        # From round 1 on, the controller is active -- every value must stay
        # within the configured bounds.
        self.assertTrue(np.all(coefficient[1:] >= 1e-6 - 1e-9))
        self.assertTrue(np.all(coefficient <= 0.05 + 1e-9))

    def test_adaptive_weight_decay_off_matches_mode_none_exactly(self):
        train, test, metadata = self._data(seed=56)
        base_kwargs = dict(
            environment="noise",
            mitigation="none",
            sigma=0.4,
            seed=56,
            nRounds=4,
        )
        config_none = self._config(**base_kwargs, adaptiveWeightDecayMode="none")
        result_none = run_campaign_four(
            config=config_none,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
        )
        self.assertTrue(
            np.all(result_none["telemetry"]["adaptive_weight_decay_coefficient"] == 0.0)
        )


if __name__ == "__main__":
    unittest.main()
