import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from argparse import Namespace
import subprocess
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from basil_core.trainer import lossFn
from basil_core.wcm_pilot import run_wcm_pilot
from noise_comm.wcm import (
    WcmState,
    fullModelL2Norm,
    paperSequence,
    sampleBoundaryPayload,
    scaSurrogateGradients,
    updateGradientEstimate,
    validatePaperExponents,
    wcmLocalUpdate,
    wcmStep,
)
from scripts import run_wcm_pilot as pilot_runner


class TinyModel:
    def __init__(self):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(
                    10,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(
                        seed=17
                    ),
                ),
            ]
        )

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def __call__(self, x, training=False):
        return self.model(x, training=training)


def _tiny_data():
    x = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    return x, y


class WcmMathTests(unittest.TestCase):
    def test_boundary_sampler_uses_one_full_model_sphere(self):
        params = [
            np.ones((2, 3), dtype=np.float32),
            np.full((4,), 2.0, dtype=np.float32),
        ]
        absolute = sampleBoundaryPayload(
            params,
            0.75,
            np.random.default_rng(11),
            relative_radius=False,
        )
        self.assertAlmostEqual(fullModelL2Norm(absolute), 0.75, places=6)

        relative = sampleBoundaryPayload(
            params,
            0.4,
            np.random.default_rng(11),
            relative_radius=True,
        )
        self.assertAlmostEqual(
            fullModelL2Norm(relative),
            0.4 * fullModelL2Norm(params),
            places=5,
        )
        repeated = sampleBoundaryPayload(
            params,
            0.4,
            np.random.default_rng(11),
            relative_radius=True,
        )
        for left, right in zip(relative, repeated):
            np.testing.assert_array_equal(left, right)

    def test_recursive_gradient_estimate_matches_equation_32(self):
        updated = updateGradientEstimate(
            [np.asarray([1.0, -1.0], dtype=np.float32)],
            [np.asarray([3.0, 1.0], dtype=np.float32)],
            0.25,
        )
        np.testing.assert_allclose(
            updated[0],
            np.asarray([1.5, -0.5], dtype=np.float32),
        )

    def test_surrogate_gradient_matches_equation_31(self):
        gradients = scaSurrogateGradients(
            noisy_gradients=[np.asarray([2.0], dtype=np.float32)],
            candidate=[np.asarray([3.0], dtype=np.float32)],
            reference=[np.asarray([1.0], dtype=np.float32)],
            previous_gradient_estimate=[
                np.asarray([4.0], dtype=np.float32)
            ],
            rho=0.5,
            penalty=0.1,
            extra_prox_mu=0.2,
        )
        np.testing.assert_allclose(
            gradients[0],
            np.asarray([3.8], dtype=np.float32),
            rtol=1e-6,
        )

    def test_paper_sequences_enforce_lemma_7(self):
        validatePaperExponents(0.6, 0.8)
        self.assertEqual(paperSequence(0, 0.6), 1.0)
        self.assertLess(paperSequence(5, 0.8), paperSequence(1, 0.8))
        with self.assertRaises(ValueError):
            validatePaperExponents(0.8, 0.6)
        with self.assertRaises(ValueError):
            validatePaperExponents(0.4, 0.8)


class WcmRunnerSafetyTests(unittest.TestCase):
    @staticmethod
    def _runner_args(**overrides):
        values = {
            "approach": "cart",
            "environment": "hidden_noise",
            "snapshot_selection": True,
            "sigma": 0.6,
            "seed": 2025,
            "rounds": 30,
            "cart_gamma": 0.0005,
            "wcm_penalty": 0.1,
            "wcm_rho_exponent": 0.6,
            "wcm_gamma_exponent": 0.8,
            "wcm_boundary_samples": 1,
            "wcm_radius_multiplier": 1.0,
            "device": "gpu",
            "gpu_memory_mb": 3800,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_gpu_process_probe_fails_closed(self):
        with mock.patch.object(
            pilot_runner.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 5),
        ):
            with self.assertRaises(RuntimeError):
                pilot_runner._active_gpu_processes()

    def test_config_hash_ignores_derived_run_id(self):
        first = {"seed": 2025, "wcmPenalty": 0.1}
        second = {
            "seed": 2025,
            "wcmPenalty": 0.1,
            "runId": "derived-value",
        }
        self.assertEqual(
            pilot_runner._config_hash(first),
            pilot_runner._config_hash(second),
        )

    def test_result_path_isolated_by_parameterized_run_id(self):
        base = {
            "nonIID": True,
            "environment": "hidden_noise",
            "approach": "cart",
            "channelNoiseSigma": 0.6,
            "snapshotSelection": True,
            "seed": 2025,
        }
        first = dict(base, runId="wcm-r1-first")
        second = dict(base, runId="wcm-r1-second")
        first_metrics, _ = pilot_runner._result_paths(first)
        second_metrics, _ = pilot_runner._result_paths(second)
        self.assertNotEqual(first_metrics, second_metrics)
        self.assertEqual(first_metrics.parent.name, "wcm-r1-first")

    def test_gpu_memory_limit_is_conservative(self):
        with self.assertRaises(ValueError):
            pilot_runner._validate_args(
                self._runner_args(gpu_memory_mb=5000)
            )

    def test_built_config_uses_wcm_without_modifying_ss_or_ebm(self):
        config = pilot_runner._build_config(self._runner_args())
        self.assertEqual(config["noiseMitigation"], "wcm")
        self.assertTrue(config["snapshotSelection"])
        self.assertTrue(config["attackHidden"])
        self.assertEqual(config["ebmLambda"], 0.0)
        self.assertEqual(
            config["wcmInnerOptimizer"],
            "plain_gradient_descent_no_momentum",
        )
        self.assertTrue(config["runId"].startswith("wcm-r1-"))


class WcmTensorFlowTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(123)

    def tearDown(self):
        tf.keras.backend.clear_session()

    def test_local_update_initializes_state_and_conditional_step(self):
        model = TinyModel()
        x, y = _tiny_data()
        reference_norm = fullModelL2Norm(
            [value.numpy() for value in model.trainable_variables]
        )
        first_state, first = wcmLocalUpdate(
            model=model,
            lossFn=lossFn,
            batches=[(x, y), (x, y)],
            state=WcmState(),
            uncertainty_radius=0.2,
            penalty=0.1,
            inner_lr=0.01,
            rho_exponent=0.6,
            gamma_exponent=0.8,
            boundary_samples=2,
            rng=np.random.default_rng(9),
        )
        self.assertEqual(first_state.iteration, 1)
        self.assertEqual(
            len(first_state.gradient_estimate),
            len(model.trainable_variables),
        )
        self.assertAlmostEqual(
            first["boundary_norm"],
            0.2 * reference_norm,
            places=5,
        )
        self.assertAlmostEqual(first["gamma"], 1.0)

        second_state, second = wcmLocalUpdate(
            model=model,
            lossFn=lossFn,
            batches=[(x, y)],
            state=first_state,
            uncertainty_radius=0.2,
            penalty=0.1,
            inner_lr=0.01,
            rho_exponent=0.6,
            gamma_exponent=0.8,
            boundary_samples=1,
            rng=np.random.default_rng(10),
        )
        self.assertEqual(second_state.iteration, 2)
        self.assertLess(second["gamma"], 1.0)
        self.assertLessEqual(
            second["update_distance"],
            second["candidate_distance"] + 1e-7,
        )

    def test_legacy_step_now_initializes_recursive_gradient(self):
        model = TinyModel()
        x, y = _tiny_data()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        result = wcmStep(
            model=model,
            optimizer=optimizer,
            lossFn=lossFn,
            x=x,
            y=y,
            sigma=0.2,
            S=1,
            rho=0.5,
            lam=0.1,
            wPrev=None,
            gPrev=None,
        )
        self.assertIsNotNone(result["gPrev"])
        self.assertEqual(
            len(result["gPrev"]),
            len(model.trainable_variables),
        )

    def test_tiny_ring_pilot_runs_without_campaign_three_changes(self):
        x, y = _tiny_data()
        loaders = [
            tf.data.Dataset.from_tensor_slices((x, y))
            .repeat()
            .batch(4)
            for _ in range(2)
        ]
        test_loader = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
        class_counts = np.zeros((2, 10), dtype=np.int32)
        class_counts[:, 0] = 2
        class_counts[:, 1] = 2
        metadata = {
            "probeBatches": [(x, y), (x, y)],
            "classCounts": class_counts,
        }
        config = {
            "seed": 77,
            "approach": "merged",
            "environment": "noise",
            "noiseMitigation": "wcm",
            "snapshotSelection": False,
            "useBasil": False,
            "basilMemorySize": 1,
            "useChannelNoise": True,
            "channelNoiseSigma": 0.1,
            "channelNoiseSemantics": "relative_l2_per_link",
            "attackHidden": False,
            "attackHiddenStart": 20,
            "attackerIds": "",
            "nNodes": 2,
            "nRounds": 2,
            "localEpochs": 1,
            "stepsPerEpoch": 1,
            "learningRate": 0.01,
            "momentum": 0.0,
            "useLrDecay": False,
            "distillStrength": 0.0,
            "cartEmaBeta": 0.85,
            "verifyThreshold": 0.05,
            "wcmPenalty": 0.1,
            "wcmRhoExponent": 0.6,
            "wcmGammaExponent": 0.8,
            "wcmBoundarySamples": 1,
            "wcmRadiusMultiplier": 1.0,
        }
        result = run_wcm_pilot(
            config=config,
            model_class=TinyModel,
            train_loaders=loaders,
            test_loader=test_loader,
            data_metadata=metadata,
        )
        self.assertFalse(result["stopped"])
        self.assertEqual(result["avg_history"].shape, (2,))
        self.assertEqual(result["wcm_rho_history"].shape, (2,))
        self.assertTrue(np.isfinite(float(result["final_avg"])))
        self.assertEqual(int(result["peak_gpu_bytes"]), 0)


if __name__ == "__main__":
    unittest.main()
