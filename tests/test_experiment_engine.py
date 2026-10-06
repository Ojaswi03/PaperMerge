import gc
from collections import OrderedDict
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from gui.baseline_study import make_config

if tf is not None:
    from basil_core.experiment_engine import (
        _accuracy_from_confusion,
        _gradient_norm_regularized_gradients,
        _mean_reference_supported_gap,
        _selected_snapshot,
        RingSnapshot,
        run_campaign_three,
    )
    from basil_core.class_registry import ClassRegistry
    from basil_core.trainer import lossFn


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class CampaignEngineTests(unittest.TestCase):
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
        tf.keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        train_loaders = []
        probe_batches = []
        class_counts = []
        client_indices = []
        for node in range(3):
            labels = np.tile(np.arange(10, dtype=np.int32), 2)
            features = rng.normal(size=(20, 4)).astype(np.float32)
            features[:, 0] += labels.astype(np.float32) / 10.0
            loader = (
                tf.data.Dataset.from_tensor_slices((features, labels))
                .shuffle(20, seed=seed + node, reshuffle_each_iteration=True)
                .repeat()
                .batch(4, drop_remainder=True)
            )
            train_loaders.append(loader)
            probe_batches.append((features, labels))
            class_counts.append(np.bincount(labels, minlength=10))
            client_indices.append(np.arange(node * 20, (node + 1) * 20))

        test_labels = np.tile(np.arange(10, dtype=np.int32), 3)
        test_features = rng.normal(size=(30, 4)).astype(np.float32)
        test_features[:, 0] += test_labels.astype(np.float32) / 10.0
        test_loader = tf.data.Dataset.from_tensor_slices(
            (test_features, test_labels)
        ).batch(10)
        metadata = {
            "probeBatches": probe_batches,
            "classCounts": np.asarray(class_counts, dtype=np.int32),
            "clientIndices": client_indices,
        }
        return train_loaders, test_loader, metadata

    def _small_config(self, **values):
        config = make_config(
            split="nonIID",
            approach=values.pop("approach"),
            environment=values.pop("environment"),
            mitigation=values.pop("mitigation"),
            sigma=values.pop("sigma", 0.0),
            seed=values.pop("seed", 77),
            rounds=2,
            gamma=values.pop("gamma", 0.2),
        )
        config.update(
            {
                "nNodes": 3,
                "nRounds": 2,
                "localEpochs": 1,
                "stepsPerEpoch": 1,
                "batchSize": 4,
                "basilMemorySize": 2,
                "attackerIds": "1" if config["attackHidden"] else "",
                "attackHiddenStart": 0,
                "useLrDecay": False,
            }
        )
        config.update(values)
        return config

    def _run(self, config):
        train, test, metadata = self._data(config["seed"])
        return run_campaign_three(
            config=config,
            model_class=self.TinyModel,
            train_loaders=train,
            test_loader=test,
            data_metadata=metadata,
        )

    def test_clean_shared_worker_produces_complete_metrics(self):
        result = self._run(
            self._small_config(
                approach="merged",
                environment="clean",
                mitigation="none",
            )
        )
        self.assertFalse(result["stopped"])
        self.assertEqual(result["avg_history"].shape, (2,))
        self.assertEqual(result["worst_history"].shape, (2,))
        self.assertEqual(result["per_node_history"].shape, (2, 3))
        self.assertEqual(result["final_class_accuracy"].shape, (3, 10))
        self.assertEqual(result["confusion"].shape, (3, 10, 10))
        self.assertEqual(len(result["initialization_hash"]), 64)
        expected_avg, expected_worst, expected_nodes = _accuracy_from_confusion(
            result["confusion"]
        )
        self.assertAlmostEqual(float(result["final_avg"]), expected_avg)
        self.assertAlmostEqual(float(result["final_worst"]), expected_worst)
        np.testing.assert_allclose(
            result["final_node_accuracy"],
            expected_nodes,
            rtol=0.0,
            atol=0.0,
        )

    def test_cart_ss_ebm_is_reproducible_for_same_seed(self):
        config = self._small_config(
            approach="cart",
            environment="hidden_noise",
            mitigation="ss_ebm",
            sigma=0.2,
        )
        first = self._run(config)
        tf.keras.backend.clear_session()
        second = self._run(config)
        np.testing.assert_allclose(
            first["avg_history"],
            second["avg_history"],
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            first["mu_history"],
            second["mu_history"],
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(first["initialization_hash"], second["initialization_hash"])
        self.assertEqual(first["selected_sources"].shape, (2, 3))
        self.assertGreater(float(np.max(first["mu_history"])), 0.0)
        self.assertAlmostEqual(float(first["ebm_coefficient"]), 0.001)

        # Node 0 may bootstrap from self before any message exists. By round 2,
        # every SS decision must select a received neighbor.
        receivers = np.arange(3, dtype=np.int32)
        self.assertTrue(np.all(first["selected_sources"][1] != receivers))

    def test_ebm_gradient_matches_stated_gradient_norm_objective(self):
        tf.keras.utils.set_random_seed(91)
        model = self.TinyModel().model
        weights = model.trainable_weights
        x = tf.constant(
            np.arange(32, dtype=np.float32).reshape(8, 4) / 31.0
        )
        y = tf.constant(np.arange(8, dtype=np.int32) % 4)
        coefficient = tf.constant(0.16, dtype=tf.float32)

        actual = _gradient_norm_regularized_gradients(
            model,
            weights,
            x,
            y,
            coefficient,
        )
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                loss = lossFn(y, model(x, training=True))
            base = inner_tape.gradient(
                loss,
                weights,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            objective = loss + coefficient * tf.add_n(
                [tf.reduce_sum(tf.square(value)) for value in base]
            )
        expected = outer_tape.gradient(
            objective,
            weights,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        for observed, reference in zip(actual, expected):
            np.testing.assert_allclose(
                observed.numpy(),
                reference.numpy(),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_cart_gap_requires_reference_to_reproduce_registry_claim(self):
        registry = ClassRegistry(3)
        registry.update(
            nodeId=2,
            myClassAcc=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            support=np.asarray([10, 10, 10], dtype=np.int32),
            roundId=4,
        )
        gap = _mean_reference_supported_gap(
            registry,
            current_acc=np.asarray([0.4, 0.4, 0.4], dtype=np.float32),
            current_support=np.asarray([10, 10, 10], dtype=np.int32),
            reference_acc=np.asarray([0.6, 0.3, 0.4], dtype=np.float32),
            reference_support=np.asarray([10, 10, 10], dtype=np.int32),
        )
        self.assertAlmostEqual(gap, (0.2 + 0.0 + 0.0) / 3.0, places=6)

    def test_ss_guard_filters_implausible_low_loss_neighbor(self):
        honest = RingSnapshot(
            sender_id=1,
            round_id=2,
            params=[np.asarray([1.1, 0.9], dtype=np.float32)],
        )
        malicious = RingSnapshot(
            sender_id=2,
            round_id=2,
            params=[np.asarray([-1.0, -1.0], dtype=np.float32)],
        )
        node = SimpleNamespace(
            params=[np.asarray([1.0, 1.0], dtype=np.float32)],
            memory=OrderedDict([(1, honest), (2, malicious)]),
        )

        class FakeWorker:
            @staticmethod
            def batch_loss(params, _batch):
                return float(np.mean(params[0]))

        unguarded = _selected_snapshot(
            node,
            FakeWorker(),
            batch=None,
            use_snapshots=True,
            node_count=3,
        )
        guarded = _selected_snapshot(
            node,
            FakeWorker(),
            batch=None,
            use_snapshots=True,
            node_count=3,
            max_relative_distance=0.5,
        )
        self.assertEqual(unguarded.sender_id, 2)
        self.assertEqual(guarded.sender_id, 1)


if __name__ == "__main__":
    unittest.main()
