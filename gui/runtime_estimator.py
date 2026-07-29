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
        str(config.get("dataset", "cifar10")),
        str(config.get("approach", "merged")),
        _active_ebm(config),
        bool(config.get("snapshotSelection", config.get("useBasil", False))),
        bool(config.get("useChannelNoise", False)),
        str(config.get("environment", "")) == "clean",
        bool(config.get("nonIID", True)),
        int(config.get("batchSize", 512)),
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
    important = (0, 1, 2, 3, 6, 7, 8, 9, 10)
    if all(target_signature[index] == record_signature[index] for index in important):
        return 1

    # Fall back to the same approach and expensive algorithmic path.
    if (
        target_signature[0:4] == record_signature[0:4]
        and target_signature[6] == record_signature[6]
    ):
        return 2

    if target_signature[1:3] == record_signature[1:3]:
        return 3
    if target_signature[2] == record_signature[2]:
        return 4
    return None


def _scaled_seconds(target: dict, record: _RuntimeRecord) -> float:
    scaled = record.seconds * _work_units(target) / _work_units(record.config)
    if not record.includes_worker_overhead:
        scaled += ISOLATED_WORKER_OVERHEAD_SECONDS
    return scaled


class RuntimeEstimator:
    def __init__(self, result_root: Path | str = DEFAULT_RESULT_ROOT):
        self.result_root = Path(result_root)
        self.records: list[_RuntimeRecord] = []
        self.reload()

    def reload(self) -> None:
        self.records = _load_records(self.result_root)

    def estimate(self, config: dict) -> RuntimeEstimate:
        candidates: list[tuple[_RuntimeRecord, int]] = []
        for record in self.records:
            tier = _matching_tier(config, record.config)
            if tier is not None:
                candidates.append((record, tier))

        if not candidates:
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
    ) -> QueueEstimate:
        queue = list(configs)
        lanes = max(1, int(lanes))
        slowdown = max(1.0, float(concurrency_slowdown)) if lanes > 1 else 1.0
        active_elapsed = active_elapsed or {}
        estimates = [(config, self.estimate(config)) for config in queue]

        def makespan(selector) -> float:
            lane_totals = [0.0] * lanes
            pending = []
            for config, estimate in estimates:
                run_id = str(config.get("runId", ""))
                duration = selector(estimate) * slowdown
                if run_id and run_id in active_elapsed:
                    elapsed = max(0.0, float(active_elapsed[run_id]))
                    lane_index = min(range(lanes), key=lane_totals.__getitem__)
                    lane_totals[lane_index] = max(0.0, duration - elapsed)
                else:
                    pending.append(duration)
            for duration in pending:
                lane_index = min(range(lanes), key=lane_totals.__getitem__)
                lane_totals[lane_index] += duration
            return max(lane_totals, default=0.0)

        def remaining_work(selector) -> float:
            total = 0.0
            for config, estimate in estimates:
                duration = selector(estimate) * slowdown
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
