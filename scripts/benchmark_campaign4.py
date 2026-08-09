#!/usr/bin/env python3
"""Benchmark OOM-safe Campaign 4 GPU execution profiles."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gui.campaign4 import (  # noqa: E402
    PERFORMANCE_PROFILE,
    build_performance_benchmark,
    make_run_id,
    write_json_atomic,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-mb", type=int, default=7600)
    parser.add_argument("--include-xla", action="store_true")
    parser.add_argument("--include-cuda-malloc-async", action="store_true")
    parser.add_argument("--only-cuda-malloc-async", action="store_true")
    parser.add_argument("--include-mixed-bfloat16", action="store_true")
    parser.add_argument("--only-mixed-bfloat16", action="store_true")
    parser.add_argument("--accuracy-tolerance", type=float, default=0.03)
    parser.add_argument("--profile", default=str(PERFORMANCE_PROFILE))
    return parser.parse_args()


def _profile_candidates(
    include_xla,
    include_cuda_malloc_async,
    only_cuda_malloc_async=False,
    include_mixed_bfloat16=False,
    only_mixed_bfloat16=False,
):
    if only_mixed_bfloat16:
        return [
            {
                "internalMicroBatchSize": 512,
                "precisionProfile": "mixed_bfloat16",
                "jitCompile": False,
                "gpuAllocator": "cuda_malloc_async",
            }
        ]
    if only_cuda_malloc_async:
        return [
            {
                "internalMicroBatchSize": 512,
                "precisionProfile": "float32",
                "jitCompile": False,
                "gpuAllocator": "cuda_malloc_async",
            }
        ]
    profiles = [
        {
            "internalMicroBatchSize": 512,
            "precisionProfile": "float32",
            "jitCompile": False,
            "gpuAllocator": "bfc",
        }
    ]
    if include_cuda_malloc_async:
        profiles.append(
            {
                "internalMicroBatchSize": 512,
                "precisionProfile": "float32",
                "jitCompile": False,
                "gpuAllocator": "cuda_malloc_async",
            }
        )
    if include_xla:
        profiles.append(
            {
                "internalMicroBatchSize": 512,
                "precisionProfile": "float32",
                "jitCompile": True,
                "gpuAllocator": "bfc",
            }
        )
    if include_mixed_bfloat16:
        profiles.append(
            {
                "internalMicroBatchSize": 512,
                "precisionProfile": "mixed_bfloat16",
                "jitCompile": False,
                "gpuAllocator": "cuda_malloc_async",
            }
        )
    return profiles


def _profile_id(profile):
    return (
        f"mb{profile['internalMicroBatchSize']}_"
        f"{profile['precisionProfile']}_"
        f"{'xla' if profile['jitCompile'] else 'no_xla'}_"
        f"{profile.get('gpuAllocator', 'bfc')}"
    )


def _command(config_path, output_path, memory_mb):
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_campaign4_worker.py"),
        "--config",
        str(config_path),
        "--gpu-memory-mb",
        str(memory_mb),
        "--no-save",
        "--output-json",
        str(output_path),
        "--suppress-node-events",
    ]


def _run(config, directory, name, memory_mb):
    config_path = directory / f"{name}.config.json"
    output_path = directory / f"{name}.summary.json"
    write_json_atomic(config_path, config)
    started = time.perf_counter()
    process = subprocess.run(
        _command(config_path, output_path, memory_mb),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    summary = None
    if process.returncode == 0 and output_path.exists():
        summary = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        "returnCode": int(process.returncode),
        "elapsedSeconds": float(elapsed),
        "summary": summary,
        "logTail": "\n".join(process.stdout.splitlines()[-30:]),
    }


def _configured(config, profile, rounds):
    result = dict(config)
    result.update(profile)
    result["nRounds"] = int(rounds)
    result["phase"] = "performance_benchmark"
    result["autoPlotCampaign4"] = False
    result["runId"] = make_run_id(result)
    return result


def main():
    args = _parse_args()
    if args.rounds <= 0 or args.repeats <= 0:
        raise SystemExit("--rounds and --repeats must be positive.")
    if args.gpu_memory_mb < 4096:
        raise SystemExit("Campaign 4 benchmarking requires at least a 4096 MB cap.")

    comparison_profile_id = None
    external_baseline = {}
    previous_profile = None
    if args.only_cuda_malloc_async or args.only_mixed_bfloat16:
        try:
            previous = json.loads(Path(args.profile).read_text(encoding="utf-8"))
            previous_profile = previous
            previous_selected = previous.get("selectedProfile") or {}
            previous_id = previous_selected.get("profileId")
            previous_results = {
                item.get("profileId"): item
                for item in previous.get("profiles", [])
                if isinstance(item, dict)
            }
            previous_result = previous_results.get(previous_id, {})
            for case_index, values in previous_result.get("caseMedians", {}).items():
                external_baseline[int(case_index)] = float(values["accuracy"])
            if external_baseline:
                comparison_profile_id = str(previous_id)
        except (OSError, ValueError, TypeError, KeyError):
            external_baseline = {}

    candidates = _profile_candidates(
        args.include_xla,
        args.include_cuda_malloc_async,
        args.only_cuda_malloc_async,
        args.include_mixed_bfloat16,
        args.only_mixed_bfloat16,
    )
    base_configs = build_performance_benchmark(rounds=args.rounds)
    results = []
    baseline_accuracy = dict(external_baseline)

    with tempfile.TemporaryDirectory(prefix="papermerge-campaign4-benchmark-") as raw:
        directory = Path(raw)
        for profile_index, candidate in enumerate(candidates):
            profile_id = _profile_id(candidate)
            print(f"\nProfile {profile_id}", flush=True)
            warm_config = _configured(base_configs[0], candidate, 1)
            warm = _run(
                warm_config,
                directory,
                f"{profile_id}.warm",
                args.gpu_memory_mb,
            )
            profile_runs = []
            if warm["returnCode"] == 0:
                for repeat in range(args.repeats):
                    for case_index, base in enumerate(base_configs):
                        config = _configured(base, candidate, args.rounds)
                        run = _run(
                            config,
                            directory,
                            f"{profile_id}.r{repeat}.c{case_index}",
                            args.gpu_memory_mb,
                        )
                        run.update(
                            {
                                "repeat": repeat,
                                "case": case_index,
                                "experimentName": config["experimentName"],
                            }
                        )
                        profile_runs.append(run)
                        print(
                            f"  repeat {repeat + 1}/{args.repeats}, case {case_index + 1}/4: "
                            f"exit={run['returnCode']} time={run['elapsedSeconds']:.1f}s",
                            flush=True,
                        )

            succeeded = bool(profile_runs) and all(run["returnCode"] == 0 for run in profile_runs)
            finite = succeeded and all(
                math.isfinite(float(run["summary"]["finalAverageAccuracy"]))
                for run in profile_runs
            )
            medians = {}
            deviations = []
            if finite:
                for case_index in range(len(base_configs)):
                    case_runs = [run for run in profile_runs if run["case"] == case_index]
                    accuracies = [float(run["summary"]["finalAverageAccuracy"]) for run in case_runs]
                    runtimes = [float(run["elapsedSeconds"]) for run in case_runs]
                    medians[str(case_index)] = {
                        "accuracy": float(statistics.median(accuracies)),
                        "wallSeconds": float(statistics.median(runtimes)),
                        "peakGpuBytes": max(
                            int(run["summary"]["peakGpuBytes"])
                            for run in case_runs
                        ),
                    }
                    if profile_index == 0 and not external_baseline:
                        baseline_accuracy[case_index] = medians[str(case_index)]["accuracy"]
                    elif case_index in baseline_accuracy:
                        deviations.append(
                            abs(medians[str(case_index)]["accuracy"] - baseline_accuracy[case_index])
                        )
            numerical_ok = finite and (
                (profile_index == 0 and not external_baseline)
                or (
                    len(deviations) == len(base_configs)
                    and max(deviations) <= float(args.accuracy_tolerance)
                )
            )
            peak = max(
                (
                    int(run["summary"]["peakGpuBytes"])
                    for run in profile_runs
                    if run.get("summary")
                ),
                default=0,
            )
            total_median = sum(value["wallSeconds"] for value in medians.values())
            results.append(
                {
                    "profileId": profile_id,
                    **candidate,
                    "warmupSucceeded": warm["returnCode"] == 0,
                    "allRunsSucceeded": succeeded,
                    "finite": finite,
                    "numericalGatePassed": numerical_ok,
                    "maximumAccuracyDeviation": max(deviations, default=0.0),
                    "caseMedians": medians,
                    "totalMedianSeconds": float(total_median),
                    "peakGpuBytes": int(peak),
                    "failedLogs": (
                        ([warm["logTail"]] if warm["returnCode"] != 0 else [])
                        + [
                            run["logTail"]
                            for run in profile_runs
                            if run["returnCode"] != 0
                        ]
                    ),
                }
            )

    eligible = [result for result in results if result["allRunsSucceeded"] and result["finite"] and result["numericalGatePassed"]]
    selected = min(eligible, key=lambda result: result["totalMedianSeconds"], default=None)
    fully_validated = selected is not None and args.repeats >= 3
    precision_requires_canary = bool(
        fully_validated
        and selected.get("precisionProfile") != "float32"
    ) if selected else False
    fallback_profile = None
    if previous_profile and previous_profile.get("status") == "validated":
        candidate = previous_profile.get("selectedProfile")
        if isinstance(candidate, dict) and candidate.get("precisionProfile") == "float32":
            fallback_profile = dict(candidate)
    if fallback_profile is None:
        float32_eligible = [
            result
            for result in eligible
            if result.get("precisionProfile") == "float32"
        ]
        fallback = min(
            float32_eligible,
            key=lambda result: result["totalMedianSeconds"],
            default=None,
        )
        if fallback:
            fallback_profile = {
                "internalMicroBatchSize": fallback["internalMicroBatchSize"],
                "precisionProfile": fallback["precisionProfile"],
                "jitCompile": fallback["jitCompile"],
                "gpuAllocator": fallback.get("gpuAllocator", "bfc"),
                "profileId": fallback["profileId"],
            }
    reasons = []
    if selected is None:
        reasons.append("No execution profile passed OOM, finite-value, and numerical gates.")
    if selected is not None and args.repeats < 3:
        reasons.append("At least three repeats are required before the profile is validated.")
    if precision_requires_canary:
        reasons.append(
            "mixed_bfloat16 passed the short execution gate but remains disabled until the 100-round float32-versus-BF16 precision canary passes."
        )
    reasons.append(
        "mixed_float16 rejected: the custom second-order EBM loop does not yet provide the mandatory dynamic loss scaling and finite-gradient gate."
    )
    reasons.append(
        "mixed_bfloat16 is eligible only when float32 variables/gradient accumulation, finite outputs, and the declared accuracy gate all pass."
    )
    reasons.append(
        "Official EBM profiles use one full configured batch of 512 so the second-order term is exactly grad(||grad F_batch||^2); smaller batches define a different objective and are permitted only in isolated performance diagnostics."
    )
    profile = {
        "schemaVersion": 2,
        "campaignVersion": 4,
        "status": (
            "requires_full_round_validation"
            if precision_requires_canary
            else "validated"
            if fully_validated
            else "provisional"
            if selected
            else "benchmark_failed"
        ),
        "benchmarkedAt": datetime.now().astimezone().isoformat(),
        "benchmarkRounds": int(args.rounds),
        "repeats": int(args.repeats),
        "gpuMemoryLimitMb": int(args.gpu_memory_mb),
        "recommendedLanes": 1,
        "concurrencySlowdown": 1.0,
        "selectedProfile": (
            {
                "internalMicroBatchSize": selected["internalMicroBatchSize"],
                "precisionProfile": selected["precisionProfile"],
                "jitCompile": selected["jitCompile"],
                "gpuAllocator": selected.get("gpuAllocator", "bfc"),
                "profileId": selected["profileId"],
            }
            if selected
            else None
        ),
        "profiles": results,
        "accuracyTolerance": float(args.accuracy_tolerance),
        "comparisonBaselineProfileId": comparison_profile_id,
        "fallbackProfile": fallback_profile,
        "precisionValidation": (
            {
                "status": "required",
                "rounds": 100,
                "candidateProfileId": selected["profileId"],
                "baselineProfileId": (
                    fallback_profile.get("profileId")
                    if isinstance(fallback_profile, dict)
                    else comparison_profile_id
                ),
            }
            if precision_requires_canary
            else None
        ),
        "reasons": reasons,
    }
    write_json_atomic(args.profile, profile)
    print(json.dumps(profile, indent=2, sort_keys=True), flush=True)
    return 0 if selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
