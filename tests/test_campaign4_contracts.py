import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gui.campaign4 import (
    CAMPAIGN_ID,
    CONFIG_ROOT,
    PLOT_ROOT,
    RESULT_ROOT,
    WORKER_ROOT,
    adaptive_method_contract,
    apply_performance_profile,
    build_diagnostic,
    build_main_confirmation,
    build_performance_benchmark,
    build_static_controls,
    config_hash,
    current_source_hashes,
    diagnostic_completion,
    freeze_campaign_state,
    is_completed,
    is_confirmation_frozen,
    load_campaign_state,
    make_config,
    make_run_id,
    result_paths,
    write_json_atomic,
    write_npz_atomic,
)
from plots.plotCampaign4 import (
    clear_record_cache,
    generate_campaign4_plots,
    load_records,
)
from gui.network_view import CampaignNetworkView


def _write_synthetic_completed(config, root, final):
    rounds = int(config["nRounds"])
    history = np.linspace(0.1, final, rounds, dtype=np.float32)
    metrics_path, run_path, telemetry_path = result_paths(config, root=root)
    write_npz_atomic(
        metrics_path,
        avg_history=history,
        worst_history=np.maximum(0.0, history - 0.05),
        per_node_history=np.tile(history[:, None], (1, 10)),
        final_avg=np.float32(final),
        final_worst=np.float32(max(0.0, final - 0.05)),
        final_node_accuracy=np.full(10, final, dtype=np.float32),
        final_class_accuracy=np.full((10, 10), final, dtype=np.float32),
        confusion=np.zeros((10, 10, 10), dtype=np.int64),
        class_counts=np.ones((10, 10), dtype=np.int32),
        learning_curve_auc=np.float32(np.mean(history)),
        runtime_seconds=np.float64(10.0),
        peak_gpu_bytes=np.int64(1024),
    )
    write_npz_atomic(
        telemetry_path,
        outgoing_relative_noise=np.zeros((rounds, 10, 5), dtype=np.float32),
        stress_ema=np.zeros((rounds, 10), dtype=np.float32),
        ebm_coefficient=np.zeros((rounds, 10), dtype=np.float32),
        clip_fraction=np.zeros((rounds, 10), dtype=np.float32),
        selected_sources=np.zeros((rounds, 10), dtype=np.int32),
        attack_active=np.zeros((rounds, 10), dtype=np.bool_),
        selection_fallback=np.zeros((rounds, 10), dtype=np.bool_),
    )
    write_json_atomic(
        run_path,
        {
            "status": "completed",
            "campaignId": CAMPAIGN_ID,
            "runId": config["runId"],
            "configHash": config_hash(config),
            "config": config,
            "sourceHashes": current_source_hashes(),
        },
    )


class Campaign4ContractTests(unittest.TestCase):
    def test_paths_are_isolated_from_prior_campaigns(self):
        self.assertEqual(CONFIG_ROOT, Path("gui/configs/campaign4"))
        self.assertEqual(RESULT_ROOT, Path("experiments/results4/campaign4/gui"))
        self.assertEqual(WORKER_ROOT, Path("experiments/results4/campaign4/workers"))
        self.assertEqual(PLOT_ROOT, Path("plots4/campaign4"))
        for path in (CONFIG_ROOT, RESULT_ROOT, WORKER_ROOT, PLOT_ROOT):
            self.assertNotIn("results3", str(path))
            self.assertNotIn("plots3", str(path))

    def test_preset_counts_and_run_ids_are_stable(self):
        main = build_main_confirmation()
        static = build_static_controls()
        merged_diagnostic = build_diagnostic("merged")
        cart_diagnostic = build_diagnostic("cart")
        benchmark = build_performance_benchmark()
        self.assertEqual(len(main), 396)
        self.assertEqual(len(static), 60)
        self.assertEqual(len(merged_diagnostic), 39)
        self.assertEqual(len(cart_diagnostic), 39)
        self.assertEqual(len(benchmark), 4)
        combined = main + static + merged_diagnostic + cart_diagnostic + benchmark
        self.assertEqual(len({config["runId"] for config in combined}), len(combined))
        self.assertEqual(
            [config["runId"] for config in build_main_confirmation()],
            [config["runId"] for config in main],
        )

    def test_environment_and_mitigation_semantics_are_explicit(self):
        clean = make_config(
            split="nonIID",
            approach="merged",
            environment="clean",
            mitigation="none",
            seed=1,
        )
        self.assertEqual(clean["aggregationMode"], "pairwise_consensus")
        self.assertTrue(clean["protocolMatchedClean"])
        self.assertFalse(clean["attackHidden"])
        self.assertFalse(clean["useChannelNoise"])
        self.assertEqual(clean["ebmMode"], "none")

        joint = make_config(
            split="IID",
            approach="cart",
            environment="hidden_noise",
            mitigation="ss_ebm",
            sigma=0.6,
            seed=2,
        )
        self.assertTrue(joint["attackHidden"])
        self.assertEqual(joint["attackHiddenStart"], 20)
        self.assertTrue(joint["useChannelNoise"])
        self.assertEqual(joint["channelNoiseStart"], 0)
        self.assertTrue(joint["snapshotSelection"])
        self.assertEqual(joint["ebmMode"], "adaptive")
        self.assertEqual(joint["batchSize"], 512)
        self.assertEqual(joint["internalMicroBatchSize"], 512)
        self.assertEqual(joint["ebmGradientBatchSemantics"], "exact_configured_batch")
        self.assertEqual(joint["nRounds"], 100)

    def test_weight_decay_defaults_to_zero_and_is_named_when_enabled(self):
        default_config = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=2025,
        )
        self.assertEqual(default_config["weightDecayCoefficient"], 0.0)
        self.assertNotIn("_wd", default_config["conditionId"])
        self.assertNotIn("weight_decay", default_config["experimentName"])

        decayed_config = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=2025,
            weight_decay_coefficient=1e-4,
        )
        self.assertEqual(decayed_config["weightDecayCoefficient"], 1e-4)
        self.assertTrue(decayed_config["conditionId"].endswith("_wd_0_0001"))
        self.assertIn("weight_decay=0.0001", decayed_config["experimentName"])
        self.assertNotEqual(decayed_config["runId"], default_config["runId"])

        combined = (
            build_main_confirmation()
            + build_static_controls()
            + build_diagnostic("merged")
            + build_diagnostic("cart")
            + build_performance_benchmark()
        )
        self.assertTrue(
            all(config["weightDecayCoefficient"] == 0.0 for config in combined)
        )
        self.assertEqual(len(build_main_confirmation()), 396)
        self.assertEqual(len(build_static_controls()), 60)
        self.assertEqual(len(build_diagnostic("merged")), 39)
        self.assertEqual(len(build_diagnostic("cart")), 39)
        self.assertEqual(len(build_performance_benchmark()), 4)

    def test_adaptive_weight_decay_defaults_off_and_requires_noise(self):
        default_config = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=2025,
        )
        self.assertEqual(default_config["adaptiveWeightDecayMode"], "none")
        self.assertEqual(default_config["adaptiveWeightDecayTargetRatio"], 1.1)
        self.assertEqual(default_config["adaptiveWeightDecayGain"], 0.05)
        self.assertEqual(default_config["adaptiveWeightDecayCoefficientMin"], 1e-6)
        self.assertEqual(default_config["adaptiveWeightDecayCoefficientMax"], 0.05)
        self.assertEqual(default_config["adaptiveWeightDecayBeta"], 0.9)
        self.assertEqual(default_config["adaptiveWeightDecayMaxChangeFactor"], 2.0)

        adaptive_config = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="none",
            sigma=0.6,
            seed=2025,
            adaptive_weight_decay_mode="adaptive",
        )
        self.assertEqual(adaptive_config["adaptiveWeightDecayMode"], "adaptive")
        self.assertNotEqual(adaptive_config["runId"], default_config["runId"])

        with self.assertRaises(ValueError):
            make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="none",
                seed=2025,
                adaptive_weight_decay_mode="adaptive",
            )

    def test_invalid_cross_hazard_mitigations_are_rejected(self):
        with self.assertRaises(ValueError):
            make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="ss",
                seed=1,
            )
        with self.assertRaises(ValueError):
            make_config(
                split="nonIID",
                approach="merged",
                environment="noise",
                mitigation="ebm",
                sigma=0.6,
                seed=1,
                internal_micro_batch=256,
            )
        with self.assertRaises(ValueError):
            make_config(
                split="nonIID",
                approach="merged",
                environment="noise",
                mitigation="ebm",
                sigma=0.6,
                seed=1,
                phase="oom_recovery",
                internal_micro_batch=256,
            )
        with self.assertRaises(ValueError):
            make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="none",
                seed=1,
                precision="mixed_float16",
            )
        with self.assertRaises(ValueError):
            make_config(
                split="nonIID",
                approach="merged",
                environment="hidden",
                mitigation="ebm",
                seed=1,
            )

    def test_validated_bfloat16_profile_keeps_official_ebm_full_batch(self):
        config = make_config(
            split="nonIID",
            approach="cart",
            environment="hidden_noise",
            mitigation="ss_ebm",
            ebm_mode="adaptive",
            sigma=0.6,
            seed=2025,
            phase="diagnostic",
        )
        profiled = apply_performance_profile(
            config,
            {
                "status": "validated",
                "selectedProfile": {
                    "profileId": "bf16-async-test",
                    "internalMicroBatchSize": 256,
                    "precisionProfile": "mixed_bfloat16",
                    "jitCompile": False,
                    "gpuAllocator": "cuda_malloc_async",
                },
            },
        )
        self.assertEqual(profiled["batchSize"], 512)
        self.assertEqual(profiled["internalMicroBatchSize"], 512)
        self.assertEqual(profiled["precisionProfile"], "mixed_bfloat16")
        self.assertEqual(profiled["gpuAllocator"], "cuda_malloc_async")
        self.assertNotEqual(profiled["runId"], config["runId"])

    def test_completion_requires_metrics_metadata_and_telemetry(self):
        config = make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="ebm",
            sigma=0.4,
            seed=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, run, telemetry = result_paths(config, root=root)
            write_npz_atomic(metrics, avg_history=np.zeros(2, dtype=np.float32))
            write_json_atomic(
                run,
                {
                    "status": "completed",
                    "campaignId": CAMPAIGN_ID,
                    "runId": config["runId"],
                    "configHash": config_hash(config),
                    "config": config,
                },
            )
            self.assertFalse(is_completed(config, root=root))
            write_npz_atomic(telemetry, stress_ema=np.zeros((2, 10), dtype=np.float32))
            self.assertTrue(is_completed(config, root=root))

    def test_confirmation_requires_complete_diagnostics_and_exact_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            result_root = temporary / "results4"
            state_path = temporary / "campaign_state.json"
            profile = {
                "status": "validated",
                "selectedProfile": {
                    "profileId": "test-full-batch",
                    "internalMicroBatchSize": 512,
                    "precisionProfile": "float32",
                    "jitCompile": False,
                },
            }
            completion = diagnostic_completion(
                result_root=result_root,
                execution_profile=profile,
            )
            self.assertEqual(completion["expected"], 78)
            self.assertEqual(completion["completed"], 0)
            with self.assertRaises(RuntimeError):
                freeze_campaign_state(
                    rationale="diagnostic decision",
                    noninferiority_margin=0.05,
                    result_root=result_root,
                    state_path=state_path,
                    execution_profile=profile,
                )

            diagnostics = [
                apply_performance_profile(config, profile)
                for config in build_diagnostic("merged")
                + build_diagnostic("cart")
            ]
            for config in diagnostics:
                _write_synthetic_completed(config, result_root, 0.5)
            completion = diagnostic_completion(
                result_root=result_root,
                execution_profile=profile,
            )
            self.assertTrue(completion["complete"])
            self.assertFalse(diagnostic_completion(result_root=result_root)["complete"])

            _, first_run_path, _ = result_paths(diagnostics[0], root=result_root)
            first_metadata = json.loads(first_run_path.read_text(encoding="utf-8"))
            first_metadata["sourceHashes"] = {"stale": "source"}
            write_json_atomic(first_run_path, first_metadata)
            mismatch = diagnostic_completion(
                result_root=result_root,
                execution_profile=profile,
            )
            self.assertFalse(mismatch["complete"])
            self.assertEqual(mismatch["sourceMismatchRunIds"], [diagnostics[0]["runId"]])
            _write_synthetic_completed(diagnostics[0], result_root, 0.5)

            state = freeze_campaign_state(
                rationale="Accepted from seed-2025 diagnostics only.",
                noninferiority_margin=0.05,
                result_root=result_root,
                state_path=state_path,
                execution_profile=profile,
            )
            self.assertEqual(state["diagnosticRunIds"], completion["completedRunIds"])
            self.assertEqual(
                state["adaptiveMethodContract"],
                adaptive_method_contract(profile),
            )
            self.assertEqual(load_campaign_state(state_path), state)
            self.assertTrue(
                is_confirmation_frozen(
                    result_root=result_root,
                    state_path=state_path,
                    execution_profile=profile,
                )
            )
            self.assertFalse(
                is_confirmation_frozen(
                    result_root=result_root,
                    state_path=state_path,
                )
            )

            state["adaptiveMethodContractHash"] = "stale"
            write_json_atomic(state_path, state)
            self.assertFalse(
                is_confirmation_frozen(
                    result_root=result_root,
                    state_path=state_path,
                    execution_profile=profile,
                )
            )

    def test_generated_manifest_matches_library(self):
        root = Path(__file__).resolve().parents[1] / CONFIG_ROOT
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        files = [path for path in root.glob("**/*.json") if path.name != "manifest.json"]
        self.assertEqual(manifest["campaignId"], CAMPAIGN_ID)
        self.assertEqual(manifest["configCount"], 538)
        self.assertEqual(len(files), 538)

    def test_plotter_is_incremental_and_uses_only_explicit_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            result_root = temporary / "results4"
            plot_root = temporary / "plots4"
            configs = []
            for approach in ("merged", "cart"):
                for environment, mitigation, sigma, final in (
                    ("clean", "none", 0.0, 0.75),
                    ("hidden", "ss", 0.0, 0.60),
                    ("noise", "ebm", 0.4, 0.58),
                    ("hidden_noise", "ss_ebm", 0.4, 0.55),
                ):
                    config = make_config(
                        split="nonIID",
                        approach=approach,
                        environment=environment,
                        mitigation=mitigation,
                        sigma=sigma,
                        seed=2026,
                        rounds=100,
                    )
                    configs.append(config)
                    _write_synthetic_completed(
                        config,
                        result_root,
                        final + (0.02 if approach == "cart" else 0.0),
                    )
            clear_record_cache()
            first = generate_campaign4_plots(
                mode="both",
                split="nonIID",
                only_changed=True,
                formats=("png",),
                result_root=result_root,
                plot_root=plot_root,
            )
            second = generate_campaign4_plots(
                mode="both",
                split="nonIID",
                only_changed=True,
                formats=("png",),
                result_root=result_root,
                plot_root=plot_root,
            )
            self.assertEqual(first["records"], len(configs))
            self.assertGreater(len(first["generated"]), 5)
            self.assertEqual(first["errors"], [])
            self.assertEqual(second["generated"], [])
            self.assertGreater(len(second["skipped"]), 5)
            self.assertTrue(all(str(plot_root) in path for path in first["generated"]))

    def test_plotter_separates_incompatible_execution_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            result_root = temporary / "results4"
            plot_root = temporary / "plots4"
            first = make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="none",
                seed=2026,
                rounds=100,
            )
            second = dict(first)
            second["jitCompile"] = True
            second["seed"] = 2027
            second["runId"] = make_run_id(second)
            _write_synthetic_completed(first, result_root, 0.70)
            _write_synthetic_completed(second, result_root, 0.71)
            clear_record_cache()
            result = generate_campaign4_plots(
                mode="paper",
                split="nonIID",
                only_changed=True,
                formats=("png",),
                result_root=result_root,
                plot_root=plot_root,
            )
            self.assertEqual(result["errors"], [])
            self.assertTrue(result["warnings"])
            self.assertTrue(
                all("/profiles/" in path for path in result["generated"])
            )

    def test_plotter_separates_weight_decay_from_no_mitigation(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            result_root = temporary / "results4"
            plot_root = temporary / "plots4"
            baseline = make_config(
                split="nonIID",
                approach="merged",
                environment="noise",
                mitigation="none",
                sigma=0.6,
                seed=2025,
                rounds=100,
                optimizer_state_mode="persistent",
            )
            decayed = make_config(
                split="nonIID",
                approach="merged",
                environment="noise",
                mitigation="none",
                sigma=0.6,
                seed=2025,
                rounds=100,
                optimizer_state_mode="persistent",
                weight_decay_coefficient=1e-4,
            )
            self.assertNotEqual(baseline["runId"], decayed["runId"])
            _write_synthetic_completed(baseline, result_root, 0.10)
            _write_synthetic_completed(decayed, result_root, 0.30)
            clear_record_cache()
            result = generate_campaign4_plots(
                mode="paper",
                split="nonIID",
                only_changed=True,
                formats=("png",),
                result_root=result_root,
                plot_root=plot_root,
            )
            self.assertEqual(result["errors"], [])
            self.assertTrue(result["warnings"])
            self.assertTrue(
                all("/profiles/" in path for path in result["generated"])
            )

    def test_plotter_rejects_short_scientific_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            result_root = Path(directory) / "results4"
            config = make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="none",
                seed=2026,
                rounds=3,
            )
            _write_synthetic_completed(config, result_root, 0.7)
            clear_record_cache()
            self.assertEqual(load_records(result_root=result_root), [])

    def test_network_reducer_ignores_stale_node_events(self):
        class Value:
            def get(self):
                return 0

        config = {"runId": "run", "nNodes": 1}
        view = object.__new__(CampaignNetworkView)
        view.runs = {
            0: {
                "config": config,
                "nodes": {0: {"nodeId": 0, "round": 5, "value": "new"}},
                "round": 5,
                "activeNode": 0,
            }
        }
        view.active_lane = Value()
        view.request_redraw = lambda: None
        view.apply_node_update(
            0,
            config,
            {"nodeId": 0, "round": 4, "value": "stale"},
        )
        self.assertEqual(view.runs[0]["nodes"][0]["value"], "new")


if __name__ == "__main__":
    unittest.main()
