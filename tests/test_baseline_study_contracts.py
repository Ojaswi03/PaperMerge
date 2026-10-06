import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from basil_core.class_registry import ClassRegistry
from gui.baseline_study import (
    CART_GAMMA_CANDIDATES,
    CART_LOW_NOISE_EBM_COEFFICIENT,
    CART_LOW_NOISE_CONFIRMATION_PATCH,
    CART_LOW_NOISE_REFINEMENT_GAMMAS,
    CART_LOW_NOISE_REFINEMENT_PATCH,
    PLOT_ROOT,
    PROTOCOL_REVISION,
    RESULT_ROOT,
    build_calibration,
    build_cart_addon,
    build_cart_controls,
    build_cart_low_noise_refinement,
    build_core_confirmation,
    build_iid_controls,
    build_merged_core,
    build_repair,
    config_hash,
    ebm_objective_coefficient,
    freeze_calibration_if_ready,
    is_completed,
    make_config,
    make_run_id,
    result_run_dir,
    result_paths,
    write_json_atomic,
    write_npz_atomic,
)
from reporting.baseline_study_plots import (
    clear_record_cache,
    generate_campaign3_live_plots,
    generate_campaign3_plots,
    load_records,
)


def _synthetic_metrics(config):
    environment_base = {
        "clean": 0.72,
        "hidden": 0.18,
        "noise": 0.58,
        "hidden_noise": 0.14,
    }[config["environment"]]
    mitigation = (
        "ss_ebm"
        if config["snapshotSelection"] and config["noiseMitigation"] == "ebm"
        else "ss"
        if config["snapshotSelection"]
        else "ebm"
        if config["noiseMitigation"] == "ebm"
        else "none"
    )
    gain = {"none": 0.0, "ss": 0.20, "ebm": 0.08, "ss_ebm": 0.30}[mitigation]
    sigma = float(config["channelNoiseSigma"])
    approach_gain = 0.025 if config["approach"] == "cart" else 0.0
    seed_delta = 0.004 * (int(config["seed"]) % 3)
    final = float(np.clip(environment_base + gain + approach_gain - 0.08 * sigma + seed_delta, 0, 1))
    rounds = int(config["nRounds"])
    history = np.linspace(0.1, final, rounds, dtype=np.float32)
    per_node = np.stack(
        [history - 0.002 * node for node in range(10)],
        axis=1,
    ).astype(np.float32)
    class_accuracy = np.stack(
        [
            np.clip(final - 0.01 * node + np.linspace(-0.04, 0.04, 10), 0, 1)
            for node in range(10)
        ]
    ).astype(np.float32)
    confusion = np.zeros((10, 10, 10), dtype=np.int64)
    for node in range(10):
        np.fill_diagonal(confusion[node], 80)
        confusion[node] += 2
    return {
        "avg_history": history,
        "worst_history": np.clip(history - 0.03, 0, 1),
        "per_node_history": per_node,
        "final_avg": np.float32(final),
        "final_worst": np.float32(max(0.0, final - 0.03)),
        "final_node_accuracy": per_node[-1],
        "final_class_accuracy": class_accuracy,
        "confusion": confusion,
        "class_counts": np.arange(100, 200, dtype=np.int32).reshape(10, 10),
        "mu_history": np.linspace(0.0, 0.15, rounds, dtype=np.float32),
        "accepted_claims": np.arange(rounds, dtype=np.int32) % 5,
        "rejected_claims": np.arange(rounds, dtype=np.int32) % 2,
        "registry_coverage": np.linspace(0.0, 1.0, rounds, dtype=np.float32),
        "selected_sources": np.zeros((rounds, 10), dtype=np.int32),
        "runtime_seconds": np.float64(12.0 + sigma),
        "peak_gpu_bytes": np.int64(512 * 1024 * 1024),
        "learning_curve_auc": np.float32(np.mean(history)),
    }


def _write_completed(config, root):
    metrics_path, run_path = result_paths(config, root=root)
    write_npz_atomic(metrics_path, **_synthetic_metrics(config))
    write_json_atomic(
        run_path,
        {
            "schemaVersion": 3,
            "campaignId": config["campaignId"],
            "runId": config["runId"],
            "status": "completed",
            "configHash": config_hash(config),
            "config": config,
        },
    )


class CampaignConfigTests(unittest.TestCase):
    def test_gui_campaign_path_has_no_unbound_tensorflow_seed_call(self):
        gui_source = (
            Path(__file__).resolve().parents[1] / "gui" / "experiment_app.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("tf.keras.utils.set_random_seed", gui_source)

    def test_preset_counts_and_run_ids_are_stable(self):
        calibration = build_calibration()
        merged = build_merged_core()
        cart = build_cart_addon(0.2)
        cart_controls = build_cart_controls(0.2)
        iid = build_iid_controls(0.2)
        repair = build_repair()
        refinement = build_cart_low_noise_refinement()
        core_confirmation = build_core_confirmation(0.2)

        self.assertEqual(len(calibration), 49)
        self.assertEqual(len(repair), 6)
        self.assertEqual(len(refinement), 6)
        self.assertEqual(len(merged), 99)
        self.assertEqual(len(cart), 99)
        self.assertEqual(len(cart_controls), 9)
        self.assertEqual(len(iid), 48)
        self.assertEqual(len(core_confirmation), 105)
        self.assertEqual(
            len({config["runId"] for config in core_confirmation}),
            105,
        )
        combined = calibration + merged + cart + iid
        self.assertEqual(len({config["runId"] for config in combined}), len(combined))
        self.assertEqual(
            [config["runId"] for config in build_calibration()],
            [config["runId"] for config in calibration],
        )
        self.assertIn("r2", PROTOCOL_REVISION)
        self.assertEqual(RESULT_ROOT, Path("experiments/results3/r2/gui"))
        self.assertEqual(PLOT_ROOT, Path("plots3/r2"))
        self.assertTrue(
            {config["runId"] for config in repair}.issubset(
                {config["runId"] for config in calibration}
            )
        )
        self.assertTrue(
            all(
                config["channelNoiseSigma"] == 0.2
                and config["noiseMitigation"] == "ebm"
                and config.get("protocolPatch")
                for config in repair
            )
        )
        self.assertTrue(
            all("patch_" in str(result_run_dir(config)) for config in repair)
        )
        self.assertTrue(
            {config["runId"] for config in refinement}.issubset(
                {config["runId"] for config in calibration}
            )
        )
        self.assertTrue(
            {config["runId"] for config in core_confirmation}.issubset(
                {
                    config["runId"]
                    for config in merged + cart + iid
                }
            )
        )
        self.assertTrue(
            {config["runId"] for config in cart_controls}.issubset(
                {config["runId"] for config in cart}
            )
        )
        self.assertEqual(
            {
                (
                    config["environment"],
                    (
                        "ss"
                        if config["snapshotSelection"]
                        else "none"
                    ),
                )
                for config in cart_controls
            },
            {("clean", "none"), ("hidden", "none"), ("hidden", "ss")},
        )
        self.assertEqual(
            {
                float(config["distillStrength"])
                for config in refinement
            },
            set(CART_LOW_NOISE_REFINEMENT_GAMMAS),
        )
        self.assertTrue(
            all(
                config.get("protocolPatch")
                == CART_LOW_NOISE_REFINEMENT_PATCH
                for config in refinement
            )
        )
        for config in refinement:
            self.assertEqual(config["channelNoiseSigma"], 0.2)
            if config["noiseMitigation"] == "ebm":
                self.assertAlmostEqual(
                    config["ebmTargetCoefficient"],
                    CART_LOW_NOISE_EBM_COEFFICIENT,
                )
                self.assertEqual(
                    config["ebmCoefficientSchedule"],
                    "cart_low_noise_refinement",
                )
            else:
                self.assertFalse(config["noiseMitigation"] == "ebm")

    def test_core_confirmation_preserves_the_claim_matrix(self):
        schedule = {
            "0.2": 0.0003,
            "0.4": 0.001,
            "0.6": 0.0005,
            "default": 0.001,
        }
        configs = build_core_confirmation(schedule)
        for seed in (2026, 2027, 2028):
            seeded = [config for config in configs if config["seed"] == seed]
            self.assertEqual(len(seeded), 35)
            self.assertEqual(
                len(
                    [
                        config
                        for config in seeded
                        if config["environment"] == "clean"
                    ]
                ),
                1,
            )
            self.assertEqual(
                {
                    config["channelNoiseSigma"]
                    for config in seeded
                    if config["approach"] == "cart"
                },
                {0.2, 0.3, 0.4, 0.5, 0.6},
            )
            self.assertTrue(all(config["nonIID"] for config in seeded))

        cart_low_noise_both = [
            config
            for config in configs
            if config["approach"] == "cart"
            and config["conditionId"] == "hidden_noise_sigma_0_2_ss_ebm"
        ]
        self.assertEqual(len(cart_low_noise_both), 3)
        self.assertTrue(
            all(
                config["protocolPatch"] == CART_LOW_NOISE_CONFIRMATION_PATCH
                and abs(config["ebmTargetCoefficient"] - 0.00025) < 1e-12
                and abs(config["distillStrength"] - 0.0003) < 1e-12
                for config in cart_low_noise_both
            )
        )

    def test_environment_and_mitigation_contracts(self):
        configs = (
            build_calibration()
            + build_merged_core()
            + build_cart_addon(0.2)
            + build_iid_controls(0.2)
        )
        attack_flags = (
            "attackGaussian",
            "attackSignFlip",
            "attackModelPoison",
            "attackScaling",
            "attackAlie",
            "attackIpm",
            "attackNoiseAmp",
        )
        for config in configs:
            self.assertTrue(all(not config[name] for name in attack_flags))
            self.assertEqual(config["attackHidden"], "hidden" in config["environment"])
            if config["attackHidden"]:
                self.assertEqual(config["attackHiddenStart"], 20)
            self.assertEqual(
                config["useChannelNoise"],
                config["environment"] in ("noise", "hidden_noise"),
            )
            self.assertEqual(
                config["snapshotSelection"],
                config["useBasil"],
            )
            if config["snapshotSelection"]:
                self.assertTrue(config["attackHidden"])
            if config["noiseMitigation"] == "ebm":
                self.assertTrue(config["useChannelNoise"])
                expected_coefficient = (
                    CART_LOW_NOISE_EBM_COEFFICIENT
                    if config.get("protocolPatch")
                    in {
                        CART_LOW_NOISE_REFINEMENT_PATCH,
                        CART_LOW_NOISE_CONFIRMATION_PATCH,
                    }
                    else ebm_objective_coefficient(
                        config["channelNoiseSigma"]
                    )
                )
                coefficient = (
                    config["ebmLambda"]
                    * config["channelNoiseSigma"] ** 2
                )
                self.assertAlmostEqual(
                    coefficient,
                    expected_coefficient,
                )
                self.assertAlmostEqual(
                    config["ebmTargetCoefficient"],
                    expected_coefficient,
                )
                self.assertEqual(
                    config["ebmObjective"],
                    "F_plus_lambda_sigma2_gradient_norm_squared",
                )
            if config["snapshotSelection"]:
                self.assertEqual(
                    config["snapshotSelectionRule"],
                    "plausibility_guard_then_lowest_loss_received_neighbor",
                )
                self.assertEqual(
                    config["snapshotPlausibilityGuard"],
                    "relative_l2_channel_budget",
                )
            if config["environment"] == "clean":
                self.assertFalse(config["attackHidden"])
                self.assertFalse(config["useChannelNoise"])
                self.assertFalse(config["snapshotSelection"])
                self.assertEqual(config["noiseMitigation"], "none")
                self.assertEqual(config["aggregationMode"], "full_consensus")
            else:
                self.assertEqual(
                    config["aggregationMode"],
                    "pairwise_consensus",
                )

    def test_cart_gamma_schedule_maps_intermediate_noise_levels(self):
        schedule = {
            "0.2": 0.00025,
            "0.4": 0.001,
            "0.6": 0.0005,
            "default": 0.001,
        }
        configs = build_cart_addon(schedule)
        observed = {}
        for config in configs:
            if config["seed"] != configs[0]["seed"]:
                continue
            key = (
                config["environment"],
                float(config["channelNoiseSigma"]),
            )
            observed.setdefault(key, float(config["distillStrength"]))

        self.assertEqual(observed[("clean", 0.0)], 0.001)
        self.assertEqual(observed[("hidden", 0.0)], 0.001)
        self.assertEqual(observed[("noise", 0.2)], 0.00025)
        self.assertEqual(observed[("noise", 0.3)], 0.00025)
        self.assertEqual(observed[("noise", 0.4)], 0.001)
        self.assertEqual(observed[("noise", 0.5)], 0.0005)
        self.assertEqual(observed[("noise", 0.6)], 0.0005)

    def test_cart_confirmation_propagates_low_noise_ebm_calibration(self):
        cart_configs = build_cart_addon(
            {
                "0.2": 0.0003,
                "0.4": 0.001,
                "0.6": 0.0005,
                "default": 0.001,
            }
        )
        low_noise_ebm = [
            config
            for config in cart_configs
            if abs(float(config["channelNoiseSigma"]) - 0.2) < 1e-12
            and config["noiseMitigation"] == "ebm"
        ]
        self.assertEqual(len(low_noise_ebm), 9)
        for config in low_noise_ebm:
            self.assertAlmostEqual(
                config["ebmTargetCoefficient"],
                CART_LOW_NOISE_EBM_COEFFICIENT,
            )
            self.assertAlmostEqual(config["ebmLambda"], 0.00625)
            self.assertEqual(
                config.get("protocolPatch"),
                CART_LOW_NOISE_CONFIRMATION_PATCH,
            )
            self.assertEqual(
                config["ebmCoefficientSchedule"],
                "cart_low_noise_method_specific_confirmation",
            )

        merged_low_noise_ebm = [
            config
            for config in build_merged_core()
            if abs(float(config["channelNoiseSigma"]) - 0.2) < 1e-12
            and config["noiseMitigation"] == "ebm"
        ]
        self.assertEqual(len(merged_low_noise_ebm), 9)
        self.assertTrue(
            all(
                abs(float(config["ebmTargetCoefficient"]) - 0.001)
                < 1e-12
                for config in merged_low_noise_ebm
            )
        )

    def test_result_completion_requires_matching_metadata(self):
        config = make_config(
            split="nonIID",
            approach="merged",
            environment="clean",
            mitigation="none",
            seed=7,
            rounds=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(is_completed(config, root=root))
            _write_completed(config, root)
            self.assertTrue(is_completed(config, root=root))
            _, run_path = result_paths(config, root=root)
            metadata = json.loads(run_path.read_text(encoding="utf-8"))
            metadata["status"] = "stopped"
            write_json_atomic(run_path, metadata)
            self.assertFalse(is_completed(config, root=root))

    def test_calibration_freezes_predeclared_nonzero_gamma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            state_path = Path(directory) / "campaign_state.json"
            for config in build_calibration():
                metrics = _synthetic_metrics(config)
                gamma = float(config.get("distillStrength", 0.0))
                if config["approach"] == "cart":
                    if config["environment"] == "clean":
                        metrics["final_avg"] = np.float32(0.70 - 0.01 * gamma)
                    elif config["noiseMitigation"] == "ebm":
                        metrics["final_avg"] = np.float32(
                            0.50
                            if (
                                abs(gamma - 0.0003) < 1e-12
                                and abs(
                                    float(config["channelNoiseSigma"]) - 0.2
                                )
                                < 1e-12
                            )
                            else 0.30
                            + (
                                0.18
                                if abs(gamma - 0.0005) < 1e-12
                                else 0.0
                            )
                        )
                metrics_path, run_path = result_paths(config, root=root)
                write_npz_atomic(metrics_path, **metrics)
                write_json_atomic(
                    run_path,
                    {
                        "status": "completed",
                        "runId": config["runId"],
                        "configHash": config_hash(config),
                        "config": config,
                    },
                )
            state = freeze_calibration_if_ready(
                result_root=root,
                state_path=state_path,
            )
            self.assertEqual(state["status"], "frozen", state)
            self.assertAlmostEqual(state["cartGamma"], 0.0005)
            self.assertEqual(
                state["cartGammaBySigma"],
                {"0.2": 0.0003, "0.4": 0.0005, "0.6": 0.0005},
            )
            self.assertIn(0.0005, CART_GAMMA_CANDIDATES)

    def test_calibration_blocks_when_ebm_component_regresses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            state_path = Path(directory) / "campaign_state.json"
            for config in build_calibration():
                metrics = _synthetic_metrics(config)
                if (
                    config["approach"] == "merged"
                    and config["environment"] == "noise"
                    and config["noiseMitigation"] == "ebm"
                    and abs(float(config["channelNoiseSigma"]) - 0.4) < 1e-12
                ):
                    metrics["final_avg"] = np.float32(0.1)
                metrics_path, run_path = result_paths(config, root=root)
                write_npz_atomic(metrics_path, **metrics)
                write_json_atomic(
                    run_path,
                    {
                        "status": "completed",
                        "runId": config["runId"],
                        "configHash": config_hash(config),
                        "config": config,
                    },
                )

            state = freeze_calibration_if_ready(
                result_root=root,
                state_path=state_path,
            )
            self.assertEqual(state["status"], "blocked")
            self.assertEqual(state["cartGamma"], 0.0)
            self.assertTrue(state["blockedReasons"])
            self.assertLess(state["mergedNoiseOnlyGains"]["0.4"], 0.0)

    def test_plot_loader_excludes_superseded_low_noise_calibration(self):
        current = next(
            config
            for config in build_repair()
            if config["approach"] == "merged"
            and config["environment"] == "noise"
        )
        obsolete = dict(current)
        obsolete.pop("protocolPatch")
        obsolete["ebmTargetCoefficient"] = 0.00025
        obsolete["ebmLambda"] = 0.00025 / (0.2 ** 2)
        obsolete["runId"] = make_run_id(obsolete)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_completed(obsolete, root)
            _write_completed(current, root)
            records = load_records(root, phase="calibration")
            self.assertEqual(
                [record.config["runId"] for record in records],
                [current["runId"]],
            )


class ClassRegistryTests(unittest.TestCase):
    def test_unknown_classes_do_not_create_training_signal(self):
        registry = ClassRegistry(3)
        registry.update(
            nodeId=0,
            myClassAcc=np.asarray([0.9, 0.8, np.nan]),
            support=np.asarray([12, 0, 0]),
            roundId=4,
        )
        np.testing.assert_array_equal(registry.class_best_support, [12, 0, 0])
        weights = registry.trustWeights(
            np.asarray([0.4, 0.1, 0.1]),
            np.asarray([5, 5, 5]),
        )
        np.testing.assert_allclose(weights, [0.5, 0.0, 0.0])

    def test_verified_merge_copies_atomic_registry_metadata(self):
        local = ClassRegistry(2)
        incoming = ClassRegistry(2)
        incoming.update(
            nodeId=3,
            myClassAcc=np.asarray([0.75, 0.85]),
            support=np.asarray([20, 20]),
            roundId=9,
        )
        accepted, rejected = local.verifyAndMerge(
            incoming,
            receivedModelClassAcc=np.asarray([0.74, 0.20]),
            verifyThreshold=0.05,
            receivedSupport=np.asarray([8, 8]),
        )
        self.assertEqual((accepted, rejected), (1, 1))
        self.assertEqual(local.class_best_source[0], 3)
        self.assertEqual(local.class_best_round[0], 9)
        self.assertEqual(local.class_best_support[0], 20)
        self.assertEqual(local.class_best_source[1], -1)


class CampaignPlotTests(unittest.TestCase):
    def test_record_loader_reuses_unchanged_metric_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(
                split="nonIID",
                approach="cart",
                environment="clean",
                mitigation="none",
                seed=2026,
                rounds=5,
                gamma=0.001,
            )
            _write_completed(config, root)
            clear_record_cache()
            import reporting.baseline_study_plots as campaign_plots

            with mock.patch.object(
                campaign_plots,
                "_load_npz",
                wraps=campaign_plots._load_npz,
            ) as loader:
                self.assertEqual(len(load_records(root)), 1)
                first_count = loader.call_count
                self.assertEqual(len(load_records(root)), 1)
                self.assertEqual(loader.call_count, first_count)
            clear_record_cache()

    def test_live_plotter_requests_changed_png_previews_only(self):
        import reporting.baseline_study_plots as campaign_plots

        expected = {
            "records": 0,
            "generated": [],
            "skipped": [],
            "errors": [],
            "plotRoot": "unused",
        }
        with mock.patch.object(
            campaign_plots,
            "_generate_campaign3_plots_unlocked",
            return_value=expected,
        ) as generator:
            result = generate_campaign3_live_plots(split="nonIID")
        self.assertIs(result, expected)
        self.assertEqual(generator.call_args.kwargs["mode"], "both")
        self.assertEqual(generator.call_args.kwargs["formats"], ("png",))
        self.assertTrue(generator.call_args.kwargs["only_changed"])

    def test_plotter_isolated_formats_and_incremental_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result_root = base / "results3" / "gui"
            plot_root = base / "plots3"
            for seed in (101, 102):
                for approach in ("merged", "cart"):
                    _write_completed(
                        make_config(
                            split="nonIID",
                            approach=approach,
                            environment="clean",
                            mitigation="none",
                            seed=seed,
                            rounds=5,
                            gamma=0.2,
                        ),
                        result_root,
                    )
                    for mitigation in ("none", "ss"):
                        _write_completed(
                            make_config(
                                split="nonIID",
                                approach=approach,
                                environment="hidden",
                                mitigation=mitigation,
                                seed=seed,
                                rounds=5,
                                gamma=0.2,
                            ),
                            result_root,
                        )
                    for sigma in (0.2, 0.3, 0.4, 0.5, 0.6):
                        for environment, mitigations in (
                            ("noise", ("none", "ebm")),
                            ("hidden_noise", ("none", "ss", "ebm", "ss_ebm")),
                        ):
                            for mitigation in mitigations:
                                _write_completed(
                                    make_config(
                                        split="nonIID",
                                        approach=approach,
                                        environment=environment,
                                        mitigation=mitigation,
                                        sigma=sigma,
                                        seed=seed,
                                        rounds=5,
                                        gamma=0.2,
                                    ),
                                    result_root,
                                )

            first = generate_campaign3_plots(
                mode="both",
                result_root=result_root,
                plot_root=plot_root,
                only_changed=True,
            )
            self.assertEqual(first["errors"], [])
            self.assertGreaterEqual(len(first["generated"]), 15)
            familiar = {
                str(
                    Path("images")
                    / "gui"
                    / "nonIID"
                    / "cifar10"
                    / "hidden"
                    / approach
                    / "paper"
                    / filename
                )
                for approach in ("merged", "cart")
                for filename in (
                    "experiments_avg",
                    "experiments_avg_zoom",
                    "grid_avg",
                    "final_accuracy_avg",
                    "improvement_over_no_mitigation_avg",
                    "seed_profiles_final_accuracy",
                    "ablation_groups_avg",
                )
            }
            self.assertTrue(familiar.issubset(set(first["generated"])))
            self.assertTrue((plot_root / "manifest.json").exists())
            self.assertTrue(
                (plot_root / "tables" / "nonIID" / "cifar10" / "summary.csv").exists()
            )
            for stem in first["generated"]:
                for extension in ("png", "pdf", "eps"):
                    self.assertTrue((plot_root / stem).with_suffix(f".{extension}").exists())

            second = generate_campaign3_plots(
                mode="both",
                result_root=result_root,
                plot_root=plot_root,
                only_changed=True,
            )
            self.assertEqual(second["errors"], [])
            self.assertEqual(second["generated"], [])
            self.assertEqual(set(second["skipped"]), set(first["generated"]))


if __name__ == "__main__":
    unittest.main()
