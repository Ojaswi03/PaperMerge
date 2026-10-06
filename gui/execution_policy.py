"""Validated resource policy helpers for Campaign 4 queue execution."""

from __future__ import annotations

from typing import Iterable


DEFAULT_GPU_MEMORY_MB = 7600


def is_ebm_config(config: dict) -> bool:
    """Return whether a config executes the second-order EBM training path."""
    return bool(
        config.get("useChannelNoise")
        and str(config.get("noiseMitigation", "none")) == "ebm"
        and float(config.get("ebmLambda", 0.0)) > 0.0
    )


def settings_from_profile(profile: dict | None) -> dict:
    """Build conservative queue settings from a measured machine profile."""
    profile = profile if isinstance(profile, dict) else {}
    concurrency = profile.get("concurrencyProfile")
    concurrency = concurrency if isinstance(concurrency, dict) else {}
    limits = concurrency.get("memoryLimitsMb")
    limits = limits if isinstance(limits, dict) else {}
    slowdowns = concurrency.get("runtimeSlowdowns")
    slowdowns = slowdowns if isinstance(slowdowns, dict) else {}

    backend_valid = profile.get("status") == "validated"
    concurrency_valid = concurrency.get("status") == "validated"
    requested_lanes = int(concurrency.get("recommendedLanes", 1))
    lanes = 2 if backend_valid and concurrency_valid and requested_lanes >= 2 else 1
    default_limit = max(
        0,
        int(profile.get("gpuMemoryLimitMb", DEFAULT_GPU_MEMORY_MB)),
    )
    standard_limit = max(0, int(limits.get("standard", default_limit)))
    ebm_limit = max(0, int(limits.get("ebm", default_limit)))

    return {
        "lanes": lanes,
        "gpuMemoryLimitMb": max(standard_limit, ebm_limit),
        "memoryLimitsMb": {
            "standard": standard_limit,
            "ebm": ebm_limit,
        },
        "maxConcurrentEbm": (
            max(1, min(lanes, int(concurrency.get("maxConcurrentEbm", 1))))
            if lanes > 1
            else 1
        ),
        "allowMixedEbmStandard": bool(
            lanes > 1 and concurrency.get("allowMixedEbmStandard", False)
        ),
        "allowDualStandard": bool(
            lanes > 1 and concurrency.get("allowDualStandard", False)
        ),
        "prioritizeEbm": bool(
            lanes > 1 and concurrency.get("prioritizeEbm", True)
        ),
        "concurrencySlowdown": (
            max(1.0, float(concurrency.get("concurrencySlowdown", 1.0)))
            if lanes > 1
            else 1.0
        ),
        "runtimeSlowdowns": {
            "mixedEbm": max(1.0, float(slowdowns.get("mixedEbm", 1.0))),
            "mixedStandard": max(
                1.0,
                float(slowdowns.get("mixedStandard", 1.0)),
            ),
            "dualStandard": max(
                1.0,
                float(slowdowns.get("dualStandard", 1.0)),
            ),
            "dualEbm": max(1.0, float(slowdowns.get("dualEbm", 1.0))),
        },
        "measuredSpeedup": float(concurrency.get("measuredSpeedup", 0.0)),
        "profileId": str(concurrency.get("profileId", "single_lane")),
        "status": (
            str(concurrency.get("status", "not_benchmarked"))
            if backend_valid
            else "backend_not_validated"
        ),
    }


def memory_limit_for_config(config: dict, settings: dict) -> int:
    limits = settings.get("memoryLimitsMb", {})
    key = "ebm" if is_ebm_config(config) else "standard"
    return max(
        0,
        int(limits.get(key, settings.get("gpuMemoryLimitMb", DEFAULT_GPU_MEMORY_MB))),
    )


def _leading_campaign_block(configs: Iterable[dict], campaign_version: int) -> list[dict]:
    block = []
    for config in configs:
        if int(config.get("campaignVersion", 0)) != int(campaign_version):
            break
        block.append(config)
    return block


def select_next_config(
    configs: Iterable[dict],
    *,
    active_configs: Iterable[dict],
    active_object_ids: set[int],
    active_run_ids: set[str],
    campaign_version: int,
    settings: dict,
) -> dict | None:
    """Select the next config while obeying a validated two-lane policy.

    When mixed execution is validated, an EBM job is started first so lighter
    standard work can overlap its longer runtime. Queue order remains stable
    within each resource class.
    """
    candidates = [
        config
        for config in _leading_campaign_block(configs, campaign_version)
        if id(config) not in active_object_ids
        and not (
            config.get("runId")
            and str(config.get("runId")) in active_run_ids
        )
    ]
    if not candidates:
        return None
    if int(settings.get("lanes", 1)) <= 1:
        return candidates[0]

    active = list(active_configs)
    active_ebm = sum(1 for config in active if is_ebm_config(config))
    active_standard = len(active) - active_ebm
    ebm_candidates = [config for config in candidates if is_ebm_config(config)]
    standard_candidates = [config for config in candidates if not is_ebm_config(config)]
    max_ebm = max(1, int(settings.get("maxConcurrentEbm", 1)))
    allow_mixed = bool(settings.get("allowMixedEbmStandard", False))
    allow_dual_standard = bool(settings.get("allowDualStandard", False))

    if not active:
        if settings.get("prioritizeEbm", True) and ebm_candidates:
            return ebm_candidates[0]
        return candidates[0]

    if active_ebm:
        if active_ebm < max_ebm and ebm_candidates:
            return ebm_candidates[0]
        if allow_mixed and standard_candidates:
            return standard_candidates[0]
        return None

    if active_standard:
        if allow_mixed and active_ebm < max_ebm and ebm_candidates:
            return ebm_candidates[0]
        if allow_dual_standard and standard_candidates:
            return standard_candidates[0]
        return None

    return candidates[0]


__all__ = [
    "DEFAULT_GPU_MEMORY_MB",
    "is_ebm_config",
    "memory_limit_for_config",
    "select_next_config",
    "settings_from_profile",
]
