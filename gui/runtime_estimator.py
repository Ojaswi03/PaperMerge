"""Empirical runtime estimates for experiment queues.

Estimates are based on completed ``run.json`` records and are matched by the
parts of a configuration that materially change runtime. The estimator reports
a range because GPU load, thermal state, and first-run compilation vary.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


DEFAULT_RESULT_ROOT = Path("experiments") / "results3" / "r2"
WORKER_PROFILE_PATH = DEFAULT_RESULT_ROOT / "worker_profile.json"
ISOLATED_WORKER_OVERHEAD_SECONDS = 15.0


@dataclass(frozen=True)
class RuntimeEstimate:
    seconds: float
    low_seconds: float
    high_seconds: float
    sample_count: int
    basis: str


@dataclass(frozen=True)
class QueueEstimate:
    seconds: float
    low_seconds: float
    high_seconds: float
    work_seconds: float
    low_work_seconds: float
    high_work_seconds: float
    item_count: int
    lanes: int
    historical_samples: int


@dataclass(frozen=True)
class _RuntimeRecord:
    config: dict
    seconds: float
    includes_worker_overhead: bool


def _active_ebm(config: dict) -> bool:
    return bool(
        config.get("useChannelNoise")
        and config.get("noiseMitigation") == "ebm"
        and float(config.get("ebmLambda", 0.0)) > 0.0
    )


def _signature(config: dict) -> tuple:
    return (
        int(config.get("campaignVersion", 0)),
        str(config.get("dataset", "cifar10")),
        str(config.get("approach", "merged")),
        _active_ebm(config),
        str(config.get("ebmMode", "static" if _active_ebm(config) else "none")),
        bool(config.get("snapshotSelection", config.get("useBasil", False))),
        bool(config.get("useChannelNoise", False)),
        str(config.get("environment", "")) == "clean",
        bool(config.get("nonIID", True)),
        int(config.get("batchSize", 512)),
        int(config.get("internalMicroBatchSize", 128)),
        str(config.get("precisionProfile", "float32")),
        bool(config.get("jitCompile", False)),
        str(config.get("gpuAllocator", "bfc")),
        str(config.get("optimizerStateMode", "legacy")),
        int(config.get("nNodes", 10)),
        int(config.get("localEpochs", 5)),
        int(config.get("stepsPerEpoch", 5)),
    )


def _work_units(config: dict) -> float:
    return float(
        max(1, int(config.get("nRounds", 1)))
        * max(1, int(config.get("nNodes", 1)))
        * max(1, int(config.get("localEpochs", 1)))
        * max(1, int(config.get("stepsPerEpoch", 1)))
    )


def _load_records(result_root: Path) -> list[_RuntimeRecord]:
    records: list[_RuntimeRecord] = []
    for run_path in result_root.glob("**/run.json"):
        try:
            with run_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError):
            continue
        if metadata.get("status") != "completed":
            continue
        config = metadata.get("config")
        if not isinstance(config, dict):
            continue
        wall_runtime = metadata.get("wallRuntimeSeconds")
        engine_runtime = metadata.get("runtimeSeconds")
        try:
            if wall_runtime is not None:
                seconds = float(wall_runtime)
                includes_worker_overhead = True
            else:
                seconds = float(engine_runtime)
                includes_worker_overhead = False
        except (TypeError, ValueError):
            continue
        if math.isfinite(seconds) and seconds > 0.0:
            records.append(
                _RuntimeRecord(
                    config=dict(config),
                    seconds=seconds,
                    includes_worker_overhead=includes_worker_overhead,
                )
            )
    return records


def _matching_tier(target: dict, record: dict) -> int | None:
    target_signature = _signature(target)
    record_signature = _signature(record)
    if target_signature == record_signature:
        return 0

    # Keep approach, EBM, SS, split, batch, node count, and local work.
    important = (
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    )
    if all(target_signature[index] == record_signature[index] for index in important):
        return 1

    # Fall back to the same approach and expensive algorithmic path.
    if (
        target_signature[0:6] == record_signature[0:6]
        and target_signature[8] == record_signature[8]
    ):
        return 2

    if target_signature[0:5] == record_signature[0:5]:
        return 3
    if (
        target_signature[0] == record_signature[0]
        and target_signature[3:5] == record_signature[3:5]
    ):
        return 4
    return None


def _scaled_seconds(target: dict, record: _RuntimeRecord) -> float:
    scaled = record.seconds * _work_units(target) / _work_units(record.config)
    if not record.includes_worker_overhead:
        scaled += ISOLATED_WORKER_OVERHEAD_SECONDS
    return scaled


class RuntimeEstimator:
    def __init__(
        self,
        result_root: Path | str | Iterable[Path | str] = DEFAULT_RESULT_ROOT,
    ):
        if isinstance(result_root, (str, Path)):
            self.result_roots = (Path(result_root),)
        else:
            self.result_roots = tuple(Path(value) for value in result_root)
        self.result_root = self.result_roots[0]
        self.records: list[_RuntimeRecord] = []
        self.execution_calibrations: dict[tuple[int, str, bool], float] = {}
        self.reload()

    def reload(self) -> None:
        self.records = [
            record
            for root in self.result_roots
            for record in _load_records(root)
        ]

    def set_execution_profile(self, profile: dict | None) -> None:
        """Load full-round no-save runtime canaries from a machine profile."""
        self.execution_calibrations = {}
        profile = profile if isinstance(profile, dict) else {}
        selected = profile.get("selectedProfile")
        validation = profile.get("precisionValidation")
        if (
            profile.get("status") != "validated"
            or not isinstance(selected, dict)
            or not isinstance(validation, dict)
            or validation.get("status") != "validated"
        ):
            return
        profile_id = str(selected.get("profileId", ""))
        if not profile_id:
            return
        for case in validation.get("cases", []):
            if not isinstance(case, dict) or not case.get("passed"):
                continue
            case_id = str(case.get("caseId", ""))
            is_ebm = "joint" in case_id or "ebm" in case_id
            try:
                seconds = float(case["candidateWallSeconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(seconds) and seconds > 0.0:
                self.execution_calibrations[(4, profile_id, is_ebm)] = seconds

    def estimate(self, config: dict) -> RuntimeEstimate:
        candidates: list[tuple[_RuntimeRecord, int]] = []
        for record in self.records:
            tier = _matching_tier(config, record.config)
            if tier is not None:
                candidates.append((record, tier))

        if not candidates:
            calibration_key = (
                int(config.get("campaignVersion", 0)),
                str(config.get("performanceProfileId", "")),
                _active_ebm(config),
            )
            calibration = self.execution_calibrations.get(calibration_key)
            if calibration is not None:
                scale = _work_units(config) / (100 * 10 * 5 * 5)
                expected = calibration * scale
                return RuntimeEstimate(
                    seconds=expected,
                    low_seconds=expected * 0.90,
                    high_seconds=expected * 1.15,
                    sample_count=1,
                    basis="validated 100-round precision canary",
                )
            # Conservative defaults derived from this repository's 100-round
            # campaign runs. They are used only before any history exists.
            baseline = 3900.0 if _active_ebm(config) else 1500.0
            scale = _work_units(config) / (100 * 10 * 5 * 5)
            expected = baseline * scale + ISOLATED_WORKER_OVERHEAD_SECONDS
            return RuntimeEstimate(
                seconds=expected,
                low_seconds=expected * 0.70,
                high_seconds=expected * 1.45,
                sample_count=0,
                basis="conservative default",
            )

        best_tier = min(tier for _, tier in candidates)
        tier_records = [record for record, tier in candidates if tier == best_tier]
        same_round_records = [
            record
            for record in tier_records
            if int(record.config.get("nRounds", 0))
            == int(config.get("nRounds", 0))
        ]
        if same_round_records:
            tier_records = same_round_records
            round_basis = "same rounds"
        else:
            round_basis = "round-scaled"

        values = [_scaled_seconds(config, record) for record in tier_records]
        expected = float(statistics.median(values))
        if len(values) > 1:
            deviations = [abs(value - expected) for value in values]
            robust_sigma = 1.4826 * float(statistics.median(deviations))
            relative_margin = max(0.10, min(0.45, 1.28 * robust_sigma / expected))
        else:
            relative_margin = 0.12 if same_round_records else 0.22

        return RuntimeEstimate(
            seconds=expected,
            low_seconds=max(0.0, expected * (1.0 - relative_margin)),
            high_seconds=expected * (1.0 + relative_margin),
            sample_count=len(tier_records),
            basis=f"history tier {best_tier}, {round_basis}",
        )

    def estimate_queue(
        self,
        configs: Iterable[dict],
        *,
        lanes: int = 1,
        active_elapsed: dict[str, float] | None = None,
        concurrency_slowdown: float = 1.0,
        max_concurrent_ebm: int | None = None,
        allow_mixed_ebm_standard: bool = True,
        allow_dual_standard: bool = True,
        prioritize_ebm: bool = False,
        runtime_slowdowns: dict[str, float] | None = None,
    ) -> QueueEstimate:
        queue = list(configs)
        lanes = max(1, int(lanes))
        slowdown = max(1.0, float(concurrency_slowdown)) if lanes > 1 else 1.0
        max_concurrent_ebm = (
            lanes
            if max_concurrent_ebm is None
            else max(1, min(lanes, int(max_concurrent_ebm)))
        )
        runtime_slowdowns = (
            runtime_slowdowns if isinstance(runtime_slowdowns, dict) else {}
        )
        mixed_ebm_slowdown = max(
            1.0,
            float(runtime_slowdowns.get("mixedEbm", slowdown)),
        )
        mixed_standard_slowdown = max(
            1.0,
            float(runtime_slowdowns.get("mixedStandard", slowdown)),
        )
        dual_standard_slowdown = max(
            1.0,
            float(runtime_slowdowns.get("dualStandard", slowdown)),
        )
        dual_ebm_slowdown = max(
            1.0,
            float(runtime_slowdowns.get("dualEbm", slowdown)),
        )
        active_elapsed = active_elapsed or {}
        estimates = [(config, self.estimate(config)) for config in queue]

        def makespan(selector) -> float:
            active_jobs = []
            pending = []
            for config, estimate in estimates:
                run_id = str(config.get("runId", ""))
                duration = selector(estimate)
                if run_id and run_id in active_elapsed:
                    elapsed = max(0.0, float(active_elapsed[run_id]))
                    active_jobs.append(
                        {
                            "remaining": max(0.0, duration - elapsed),
                            "ebm": _active_ebm(config),
                        }
                    )
                else:
                    pending.append(
                        {
                            "duration": duration,
                            "ebm": _active_ebm(config),
                        }
                    )

            now = 0.0

            def candidate_index():
                if not pending:
                    return None
                active_ebm = sum(1 for job in active_jobs if job["ebm"])
                active_standard = len(active_jobs) - active_ebm

                if not active_jobs:
                    if prioritize_ebm:
                        for index, job in enumerate(pending):
                            if job["ebm"]:
                                return index
                    return 0

                if active_ebm:
                    if active_ebm < max_concurrent_ebm:
                        for index, job in enumerate(pending):
                            if job["ebm"]:
                                return index
                    if allow_mixed_ebm_standard:
                        for index, job in enumerate(pending):
                            if not job["ebm"]:
                                return index
                    return None

                if active_standard:
                    if allow_mixed_ebm_standard:
                        for index, job in enumerate(pending):
                            if job["ebm"]:
                                return index
                    if allow_dual_standard:
                        for index, job in enumerate(pending):
                            if not job["ebm"]:
                                return index
                    return None
                return 0

            while pending or active_jobs:
                while pending and len(active_jobs) < lanes:
                    index = candidate_index()
                    if index is None:
                        break
                    job = pending.pop(index)
                    active_jobs.append(
                        {
                            "remaining": job["duration"],
                            "ebm": job["ebm"],
                        }
                    )
                if not active_jobs:
                    # Invalid policies must not turn an ETA call into a hang.
                    job = pending.pop(0)
                    active_jobs.append(
                        {
                            "remaining": job["duration"],
                            "ebm": job["ebm"],
                        }
                    )

                if len(active_jobs) == 1:
                    slowdowns_for_active = [1.0]
                elif all(job["ebm"] for job in active_jobs):
                    slowdowns_for_active = [dual_ebm_slowdown] * len(active_jobs)
                elif all(not job["ebm"] for job in active_jobs):
                    slowdowns_for_active = [dual_standard_slowdown] * len(active_jobs)
                else:
                    slowdowns_for_active = [
                        mixed_ebm_slowdown
                        if job["ebm"]
                        else mixed_standard_slowdown
                        for job in active_jobs
                    ]
                elapsed_to_finish = min(
                    job["remaining"] * job_slowdown
                    for job, job_slowdown in zip(
                        active_jobs,
                        slowdowns_for_active,
                    )
                )
                for job, job_slowdown in zip(
                    active_jobs,
                    slowdowns_for_active,
                ):
                    job["remaining"] = max(
                        0.0,
                        job["remaining"] - elapsed_to_finish / job_slowdown,
                    )
                now += elapsed_to_finish
                active_jobs = [
                    job
                    for job in active_jobs
                    if job["remaining"] > 1e-9
                ]
            return now

        def remaining_work(selector) -> float:
            total = 0.0
            for config, estimate in estimates:
                duration = selector(estimate)
                run_id = str(config.get("runId", ""))
                if run_id and run_id in active_elapsed:
                    duration = max(
                        0.0,
                        duration - max(0.0, float(active_elapsed[run_id])),
                    )
                total += duration
            return total

        return QueueEstimate(
            seconds=makespan(lambda estimate: estimate.seconds),
            low_seconds=makespan(lambda estimate: estimate.low_seconds),
            high_seconds=makespan(lambda estimate: estimate.high_seconds),
            work_seconds=remaining_work(lambda estimate: estimate.seconds),
            low_work_seconds=remaining_work(
                lambda estimate: estimate.low_seconds
            ),
            high_work_seconds=remaining_work(
                lambda estimate: estimate.high_seconds
            ),
            item_count=len(queue),
            lanes=lanes,
            historical_samples=len(self.records),
        )


def load_worker_profile(path: Path | str = WORKER_PROFILE_PATH) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
        if isinstance(profile, dict):
            return profile
    except (OSError, ValueError):
        pass
    return {
        "status": "not_benchmarked",
        "recommendedLanes": 1,
        "concurrencySlowdown": 1.0,
        "gpuMemoryLimitMb": 4200,
    }


def format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
