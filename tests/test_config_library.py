import json
from pathlib import Path
import unittest

from gui.baseline_study import build_approach_confirmation, frozen_cart_schedule
from gui.config_library import (
    APPROACHES,
    CURRENT_CONFIG_ROOT,
    CURRENT_LIBRARY_VERSION,
    SPLITS,
    build_current_configs,
    config_filename,
)


class CurrentConfigLibraryTests(unittest.TestCase):
    def test_matrix_counts_and_filenames_are_unique(self):
        expected_counts = {
            "basil": 54,
            "noisy": 66,
            "merged": 99,
            "cart": 99,
        }
        for split in SPLITS:
            for approach in APPROACHES:
                with self.subTest(split=split, approach=approach):
                    configs = build_current_configs(split, approach)
                    filenames = [config_filename(config) for config in configs]
                    self.assertEqual(len(configs), expected_counts[approach])
                    self.assertEqual(len(set(filenames)), len(filenames))
                    self.assertEqual(
                        {int(config["seed"]) for config in configs},
                        {2026, 2027, 2028},
                    )

    def test_every_config_obeys_environment_and_fixed_training_contracts(self):
        attack_flags = (
            "attackGaussian",
            "attackSignFlip",
            "attackModelPoison",
            "attackScaling",
            "attackAlie",
            "attackIpm",
            "attackNoiseAmp",
        )
        for split in SPLITS:
            for approach in APPROACHES:
                for config in build_current_configs(split, approach):
                    context = (
                        split,
                        approach,
                        config["seed"],
                        config["conditionId"],
                    )
                    with self.subTest(context=context):
                        self.assertEqual(config["dataset"], "cifar10")
                        self.assertEqual(config["approach"], approach)
                        self.assertEqual(config["nonIID"], split == "nonIID")
                        self.assertEqual(config["nNodes"], 10)
                        self.assertEqual(config["nRounds"], 100)
                        self.assertEqual(config["localEpochs"], 5)
                        self.assertEqual(config["stepsPerEpoch"], 5)
                        self.assertEqual(config["batchSize"], 512)
                        self.assertAlmostEqual(config["learningRate"], 0.05)
                        self.assertAlmostEqual(config["momentum"], 0.9)
                        self.assertTrue(all(not config[name] for name in attack_flags))

                        has_hidden = config["environment"] in (
                            "hidden",
                            "hidden_noise",
                        )
                        has_noise = config["environment"] in (
                            "noise",
                            "hidden_noise",
                        )
                        self.assertEqual(config["attackHidden"], has_hidden)
                        self.assertEqual(config["useChannelNoise"], has_noise)
                        self.assertEqual(
                            bool(config["snapshotSelection"]),
                            bool(config["useBasil"]),
                        )
                        if config["snapshotSelection"]:
                            self.assertTrue(has_hidden)
                        if config["noiseMitigation"] == "ebm":
                            self.assertTrue(has_noise)
                        if has_hidden:
                            self.assertEqual(config["attackHiddenStart"], 20)
                            self.assertEqual(config["attackerIds"], "1,4,6,8")
                        else:
                            self.assertEqual(config["attackerIds"], "")

                        if config["environment"] == "clean":
                            self.assertFalse(config["attackHidden"])
                            self.assertFalse(config["useChannelNoise"])
                            self.assertFalse(config["snapshotSelection"])
                            self.assertEqual(config["noiseMitigation"], "none")

    def test_merged_and_cart_files_are_exact_campaign_r2_confirmations(self):
        gamma = frozen_cart_schedule()
        self.assertIsNotNone(gamma)
        for split in SPLITS:
            self.assertEqual(
                build_current_configs(split, "merged"),
                build_approach_confirmation(
                    "merged",
                    split=split,
                    gamma=0.0,
                ),
            )
            self.assertEqual(
                build_current_configs(split, "cart"),
                build_approach_confirmation(
                    "cart",
                    split=split,
                    gamma=gamma,
                ),
            )

    def test_standalone_paper_baselines_keep_defenses_in_scope(self):
        for split in SPLITS:
            basil = build_current_configs(split, "basil")
            noisy = build_current_configs(split, "noisy")
            self.assertTrue(
                all(config["noiseMitigation"] == "none" for config in basil)
            )
            self.assertTrue(
                all(not config["snapshotSelection"] for config in noisy)
            )
            for config in basil + noisy:
                self.assertEqual(
                    config["configLibraryVersion"],
                    CURRENT_LIBRARY_VERSION,
                )
                self.assertEqual(
                    config["executionProtocol"],
                    "standalone_paper_baseline",
                )
                if config["noiseMitigation"] == "ebm":
                    sigma = float(config["channelNoiseSigma"])
                    self.assertAlmostEqual(config["ebmLambda"] * sigma**2, 1.0)
                    self.assertAlmostEqual(config["ebmTargetScale"], 2.0)

    def test_synced_json_library_matches_the_generators(self):
        manifest_path = CURRENT_CONFIG_ROOT / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["libraryVersion"], CURRENT_LIBRARY_VERSION)

        expected_manifest_counts = {}
        for split in SPLITS:
            for approach in APPROACHES:
                folder = CURRENT_CONFIG_ROOT / split / approach
                configs = build_current_configs(split, approach)
                expected = {
                    config_filename(config): config for config in configs
                }
                actual_paths = sorted(folder.glob("*.json"))
                self.assertEqual(
                    {path.name for path in actual_paths},
                    set(expected),
                )
                for path in actual_paths:
                    with path.open("r", encoding="utf-8") as handle:
                        self.assertEqual(json.load(handle), expected[path.name])
                expected_manifest_counts[f"{split}/{approach}"] = len(configs)

        self.assertEqual(manifest["counts"], expected_manifest_counts)

    def test_gui_defaults_to_campaign4_multi_select_library(self):
        source = (
            Path(__file__).resolve().parents[1] / "gui" / "experiment_app.py"
        ).read_text(encoding="utf-8")
        self.assertIn("sourceVar2   = tk.StringVar(value='campaign4')", source)
        self.assertIn('(\"Campaign 4\", \"campaign4\")', source)
        self.assertIn("selectmode=tk.EXTENDED", source)
        self.assertIn("self._configsByEstimatedDuration", source)
        self.assertIn("tree.bind('<B1-Motion>', dragRow", source)

    def test_campaign_worker_events_feed_the_log_and_live_chart(self):
        source = (
            Path(__file__).resolve().parents[1] / "gui" / "experiment_app.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _logCampaignConfiguration", source)
        self.assertIn("def _recordCampaignRound", source)
        self.assertIn('float(payload["worstAccuracy"])', source)
        self.assertIn("self._recordCampaignRound,", source)
        self.assertIn("self._startCampaignLiveRun,", source)
        self.assertIn("STARTING CAMPAIGN {campaignVersion} EXPERIMENT", source)
        self.assertIn("self.networkView.apply_node_update", source)


if __name__ == "__main__":
    unittest.main()
