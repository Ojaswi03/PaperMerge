import json
from pathlib import Path
import tempfile
import unittest

from gui.campaign3 import make_config
from gui.runtime_estimator import RuntimeEstimator


class RuntimeEstimatorTests(unittest.TestCase):
    def _write_runtime(self, root, config, seconds, *, wall=True):
        run_dir = Path(root) / config["runId"]
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "status": "completed",
            "config": config,
            "runtimeSeconds": float(seconds),
        }
        if wall:
            metadata["wallRuntimeSeconds"] = float(seconds)
        (run_dir / "run.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

    def test_exact_same_round_history_is_preferred(self):
        target = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="ebm",
            sigma=0.2,
            seed=2026,
            rounds=100,
        )
        calibration = dict(target, nRounds=30, seed=2025)
        confirmation = dict(target, seed=2027)
        with tempfile.TemporaryDirectory() as directory:
            self._write_runtime(directory, calibration, 900.0)
            self._write_runtime(directory, confirmation, 3000.0)
            estimator = RuntimeEstimator(directory)
            estimate = estimator.estimate(target)
            self.assertEqual(estimate.seconds, 3000.0)
            self.assertEqual(estimate.sample_count, 1)
            self.assertIn("same rounds", estimate.basis)

    def test_queue_makespan_uses_lane_scheduling_and_active_elapsed(self):
        configs = [
            make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="none",
                seed=seed,
                rounds=100,
            )
            for seed in (2026, 2027, 2028)
        ]
        with tempfile.TemporaryDirectory() as directory:
            for config in configs:
                self._write_runtime(directory, config, 100.0)
            estimator = RuntimeEstimator(directory)
            one_lane = estimator.estimate_queue(configs, lanes=1)
            two_lanes = estimator.estimate_queue(configs, lanes=2)
            active = estimator.estimate_queue(
                configs,
                lanes=2,
                active_elapsed={configs[0]["runId"]: 40.0},
            )
            self.assertEqual(one_lane.seconds, 300.0)
            self.assertEqual(one_lane.work_seconds, 300.0)
            self.assertEqual(two_lanes.seconds, 200.0)
            self.assertEqual(two_lanes.work_seconds, 300.0)
            self.assertEqual(active.seconds, 160.0)
            self.assertEqual(active.work_seconds, 260.0)

    def test_ebm_and_standard_runtime_histories_do_not_mix(self):
        standard = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="none",
            sigma=0.2,
            seed=2026,
            rounds=100,
        )
        ebm = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="ebm",
            sigma=0.2,
            seed=2026,
            rounds=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            self._write_runtime(directory, standard, 1000.0)
            self._write_runtime(directory, ebm, 4000.0)
            estimator = RuntimeEstimator(directory)
            self.assertEqual(estimator.estimate(standard).seconds, 1000.0)
            self.assertEqual(estimator.estimate(ebm).seconds, 4000.0)

    def test_queue_eta_respects_one_ebm_at_a_time(self):
        ebm = [
            make_config(
                split="nonIID",
                approach="merged",
                environment="noise",
                mitigation="ebm",
                sigma=0.2,
                seed=seed,
                rounds=100,
            )
            for seed in (2026, 2027)
        ]
        standard = [
            make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="none",
                seed=seed,
                rounds=100,
            )
            for seed in (2026, 2027)
        ]
        with tempfile.TemporaryDirectory() as directory:
            for config in ebm:
                self._write_runtime(directory, config, 100.0)
            for config in standard:
                self._write_runtime(directory, config, 40.0)
            estimator = RuntimeEstimator(directory)
            unrestricted = estimator.estimate_queue(ebm + standard, lanes=2)
            constrained = estimator.estimate_queue(
                standard + ebm,
                lanes=2,
                max_concurrent_ebm=1,
                allow_mixed_ebm_standard=True,
                allow_dual_standard=True,
                prioritize_ebm=True,
            )
            self.assertEqual(unrestricted.seconds, 140.0)
            self.assertEqual(constrained.seconds, 200.0)

    def test_queue_eta_applies_measured_pair_slowdowns_only_while_overlapped(self):
        configs = [
            make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="none",
                seed=seed,
                rounds=100,
            )
            for seed in (2026, 2027, 2028)
        ]
        with tempfile.TemporaryDirectory() as directory:
            for config in configs:
                self._write_runtime(directory, config, 100.0)
            estimator = RuntimeEstimator(directory)
            estimate = estimator.estimate_queue(
                configs,
                lanes=2,
                runtime_slowdowns={"dualStandard": 1.5},
            )
            # Two jobs share the GPU for 150 s; the final job then runs alone.
            self.assertEqual(estimate.seconds, 250.0)

    def test_validated_full_round_canary_replaces_generic_default(self):
        config = make_config(
            split="nonIID",
            approach="cart",
            environment="hidden_noise",
            mitigation="ss_ebm",
            sigma=0.6,
            seed=2026,
            rounds=100,
        )
        config.update(
            {
                "campaignVersion": 4,
                "performanceProfileId": "bf16-test",
                "precisionProfile": "mixed_bfloat16",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            estimator = RuntimeEstimator(directory)
            estimator.set_execution_profile(
                {
                    "status": "validated",
                    "selectedProfile": {"profileId": "bf16-test"},
                    "precisionValidation": {
                        "status": "validated",
                        "cases": [
                            {
                                "caseId": "joint_cart_sigma_0_6",
                                "passed": True,
                                "candidateWallSeconds": 1640.0,
                            }
                        ],
                    },
                }
            )
            estimate = estimator.estimate(config)
            self.assertEqual(estimate.seconds, 1640.0)
            self.assertIn("precision canary", estimate.basis)

            estimator.set_execution_profile({"status": "not_benchmarked"})
            fallback = estimator.estimate(config)
            self.assertNotEqual(fallback.seconds, 1640.0)
            self.assertEqual(fallback.basis, "conservative default")


if __name__ == "__main__":
    unittest.main()
