#!/usr/bin/env python3
"""Run 100-round float32-versus-BF16 Campaign 4 precision canaries."""

from __future__ import annotations

import argparse
from collections import deque
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

from gui.adaptive_study import (  # noqa: E402
    PERFORMANCE_PROFILE,
    make_config,
    make_run_id,
    write_json_atomic,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=str(PERFORMANCE_PROFILE))
    parser.add_argument("--gpu-memory-mb", type=int, default=7600)
    parser.add_argument("--final-accuracy-tolerance", type=float, default=0.05)
    parser.add_argument("--auc-tolerance", type=float, default=0.05)
    parser.add_argument("--worst-accuracy-tolerance", type=float, default=0.08)
    parser.add_argument("--minimum-speedup", type=float, default=1.15)
    return parser.parse_args()


def _read_profile(path):
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit("Campaign 4 performance profile is missing or invalid.") from error
    candidate = profile.get("selectedProfile")
    fallback = profile.get("fallbackProfile")
    if (
        profile.get("status") != "requires_full_round_validation"
        or not isinstance(candidate, dict)
        or candidate.get("precisionProfile") != "mixed_bfloat16"
        or not isinstance(fallback, dict)
        or fallback.get("precisionProfile") != "float32"
    ):
        raise SystemExit(
            "The profile does not contain a pending BF16 candidate and a validated "
            "float32 fallback. Run benchmark_adaptive_study.py --only-mixed-bfloat16 "
            "--repeats 3 first."
        )
    return profile, fallback, candidate


def _cases():
    return [
        make_config(
            split="nonIID",
            approach="cart",
            environment="clean",
            mitigation="none",
            seed=2025,
            rounds=100,
            phase="precision_validation",
        ),
        make_config(
            split="nonIID",
            approach="cart",
            environment="hidden_noise",
            mitigation="ss_ebm",
            ebm_mode="adaptive",
            sigma=0.6,
            seed=2025,
            rounds=100,
            phase="precision_validation",
        ),
    ]


def _configured(base, execution_profile):
    config = dict(base)
    for key in (
        "internalMicroBatchSize",
        "precisionProfile",
        "jitCompile",
        "gpuAllocator",
    ):
        if key in execution_profile:
            config[key] = execution_profile[key]
    config["performanceProfileId"] = execution_profile.get("profileId")
    config["runId"] = make_run_id(config)
    config["autoPlotCampaign4"] = False
    return config


def _command(config_path, output_path, memory_mb):
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_adaptive_worker.py"),
        "--config",
        str(config_path),
        "--gpu-memory-mb",
        str(int(memory_mb)),
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
    process = subprocess.Popen(
        _command(config_path, output_path, memory_mb),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    tail = deque(maxlen=60)
    if process.stdout is not None:
        for line in process.stdout:
            text = line.rstrip()
            tail.append(text)
            if text.startswith("[campaign4 round "):
                try:
                    round_id = int(text.split("[campaign4 round ", 1)[1].split("]", 1)[0])
                except (ValueError, IndexError):
                    round_id = -1
                if round_id == 0 or (round_id + 1) % 10 == 0:
                    print(f"    {text}", flush=True)
    return_code = process.wait()
    summary = None
    if return_code == 0 and output_path.exists():
        summary = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        "returnCode": int(return_code),
        "elapsedSeconds": float(time.perf_counter() - started),
        "summary": summary,
        "logTail": "\n".join(tail),
    }


def _finite(summary):
    if not isinstance(summary, dict):
        return False
    try:
        return all(
            math.isfinite(float(summary[key]))
            for key in (
                "finalAverageAccuracy",
                "finalWorstAccuracy",
                "finalLearningCurveAuc",
                "runtimeSeconds",
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def main():
    args = _parse_args()
    profile, fallback, candidate = _read_profile(args.profile)
    results = []
    passed = True

    with tempfile.TemporaryDirectory(prefix="papermerge-campaign4-precision-") as raw:
        directory = Path(raw)
        for case_index, base in enumerate(_cases()):
            case_name = "clean_cart" if case_index == 0 else "joint_cart_sigma_0_6"
            print(f"\n100-round canary: {case_name}", flush=True)
            baseline_config = _configured(base, fallback)
            candidate_config = _configured(base, candidate)
            print("  float32 baseline", flush=True)
            baseline_run = _run(
                baseline_config,
                directory,
                f"{case_name}.float32",
                args.gpu_memory_mb,
            )
            print("  mixed_bfloat16 candidate", flush=True)
            candidate_run = _run(
                candidate_config,
                directory,
                f"{case_name}.bfloat16",
                args.gpu_memory_mb,
            )

            baseline_summary = baseline_run.get("summary")
            candidate_summary = candidate_run.get("summary")
            valid = bool(
                baseline_run["returnCode"] == 0
                and candidate_run["returnCode"] == 0
                and _finite(baseline_summary)
                and _finite(candidate_summary)
            )
            accuracy_delta = auc_delta = worst_delta = math.inf
            speedup = 0.0
            initialization_match = False
            if valid:
                accuracy_delta = abs(
                    float(candidate_summary["finalAverageAccuracy"])
                    - float(baseline_summary["finalAverageAccuracy"])
                )
                auc_delta = abs(
                    float(candidate_summary["finalLearningCurveAuc"])
                    - float(baseline_summary["finalLearningCurveAuc"])
                )
                worst_delta = abs(
                    float(candidate_summary["finalWorstAccuracy"])
                    - float(baseline_summary["finalWorstAccuracy"])
                )
                speedup = baseline_run["elapsedSeconds"] / max(
                    candidate_run["elapsedSeconds"],
                    1e-9,
                )
                initialization_match = (
                    baseline_summary.get("initializationHash")
                    == candidate_summary.get("initializationHash")
                )
            case_passed = bool(
                valid
                and initialization_match
                and accuracy_delta <= float(args.final_accuracy_tolerance)
                and auc_delta <= float(args.auc_tolerance)
                and worst_delta <= float(args.worst_accuracy_tolerance)
                and speedup >= float(args.minimum_speedup)
            )
            passed = passed and case_passed
            result = {
                "caseId": case_name,
                "passed": case_passed,
                "initializationMatch": initialization_match,
                "accuracyDelta": float(accuracy_delta),
                "aucDelta": float(auc_delta),
                "worstAccuracyDelta": float(worst_delta),
                "speedup": float(speedup),
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "baselineWallSeconds": baseline_run["elapsedSeconds"],
                "candidateWallSeconds": candidate_run["elapsedSeconds"],
                "failedLogs": [
                    run["logTail"]
                    for run in (baseline_run, candidate_run)
                    if run["returnCode"] != 0 or not _finite(run.get("summary"))
                ],
            }
            results.append(result)
            print(
                f"  gate={'pass' if case_passed else 'fail'} "
                f"speedup={speedup:.2f}x final_delta={accuracy_delta:.4f} "
                f"auc_delta={auc_delta:.4f} worst_delta={worst_delta:.4f}",
                flush=True,
            )

    validation = {
        "status": "validated" if passed else "rejected",
        "validatedAt": datetime.now().astimezone().isoformat(),
        "rounds": 100,
        "candidateProfileId": candidate.get("profileId"),
        "baselineProfileId": fallback.get("profileId"),
        "finalAccuracyTolerance": float(args.final_accuracy_tolerance),
        "aucTolerance": float(args.auc_tolerance),
        "worstAccuracyTolerance": float(args.worst_accuracy_tolerance),
        "minimumSpeedup": float(args.minimum_speedup),
        "medianSpeedup": float(
            statistics.median(result["speedup"] for result in results)
        ),
        "cases": results,
    }
    updated = dict(profile)
    updated["precisionValidation"] = validation
    updated["recommendedLanes"] = 1
    updated.pop("concurrencyProfile", None)
    reasons = [
        str(reason)
        for reason in updated.get("reasons", [])
        if "remains disabled until the 100-round" not in str(reason)
    ]
    if passed:
        updated["status"] = "validated"
        reasons.append(
            "mixed_bfloat16 passed both 100-round no-save precision canaries; "
            "float32 variables, gradient accumulation, EBM norms, noise, CART "
            "state, and reported metrics remain unchanged."
        )
    else:
        updated["status"] = "validated"
        updated["selectedProfile"] = fallback
        reasons.append(
            "BF16 100-round precision canary failed; the validated float32 fallback was restored."
        )
    updated["reasons"] = reasons
    write_json_atomic(args.profile, updated)
    print(json.dumps(validation, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
