import unittest

from gui.adaptive_study import make_config
from gui.execution_policy import (
    is_ebm_config,
    memory_limit_for_config,
    select_next_config,
    settings_from_profile,
)


def _config(environment, mitigation, *, sigma=0.0, seed=2025):
    mode = "adaptive" if mitigation in ("ebm", "ss_ebm") else "none"
    return make_config(
        split="nonIID",
        approach="cart",
        environment=environment,
        mitigation=mitigation,
        ebm_mode=mode,
        sigma=sigma,
        seed=seed,
        phase="diagnostic",
    )


class Campaign4ExecutionTests(unittest.TestCase):
    def _profile(self):
        return {
            "status": "validated",
            "gpuMemoryLimitMb": 7600,
            "concurrencyProfile": {
                "status": "validated",
                "profileId": "test-two-lane",
                "recommendedLanes": 2,
                "maxConcurrentEbm": 1,
                "allowMixedEbmStandard": True,
                "allowDualStandard": True,
                "prioritizeEbm": True,
                "concurrencySlowdown": 1.2,
                "measuredSpeedup": 1.4,
                "memoryLimitsMb": {"standard": 3100, "ebm": 5100},
            },
        }

    def test_profile_must_validate_concurrency_before_two_lanes_are_enabled(self):
        profile = self._profile()
        settings = settings_from_profile(profile)
        self.assertEqual(settings["lanes"], 2)
        self.assertEqual(settings["maxConcurrentEbm"], 1)

        profile["concurrencyProfile"]["status"] = "rejected"
        self.assertEqual(settings_from_profile(profile)["lanes"], 1)

    def test_resource_class_selects_the_measured_memory_cap(self):
        settings = settings_from_profile(self._profile())
        standard = _config("hidden", "ss")
        ebm = _config("noise", "ebm", sigma=0.6)
        self.assertFalse(is_ebm_config(standard))
        self.assertTrue(is_ebm_config(ebm))
        self.assertEqual(memory_limit_for_config(standard, settings), 3100)
        self.assertEqual(memory_limit_for_config(ebm, settings), 5100)

    def test_scheduler_starts_ebm_then_overlaps_only_standard_work(self):
        settings = settings_from_profile(self._profile())
        standard = _config("clean", "none", seed=2025)
        first_ebm = _config("noise", "ebm", sigma=0.4, seed=2025)
        second_ebm = _config("noise", "ebm", sigma=0.6, seed=2025)
        queue = [standard, first_ebm, second_ebm]

        selected = select_next_config(
            queue,
            active_configs=[],
            active_object_ids=set(),
            active_run_ids=set(),
            campaign_version=4,
            settings=settings,
        )
        self.assertIs(selected, first_ebm)

        selected = select_next_config(
            queue,
            active_configs=[first_ebm],
            active_object_ids={id(first_ebm)},
            active_run_ids={first_ebm["runId"]},
            campaign_version=4,
            settings=settings,
        )
        self.assertIs(selected, standard)

        selected = select_next_config(
            [first_ebm, second_ebm],
            active_configs=[first_ebm],
            active_object_ids={id(first_ebm)},
            active_run_ids={first_ebm["runId"]},
            campaign_version=4,
            settings=settings,
        )
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
