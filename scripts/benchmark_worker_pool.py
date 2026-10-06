#!/usr/bin/env python3
"""Validate whether two isolated Campaign 3 workers improve GPU throughput."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gui.baseline_study import make_config, write_json_atomic
from gui.runtime_estimator import WORKER_PROFILE_PATH


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--gpu-memory-mb", type=int, default=4200)
    parser.add_argument(
        "--minimum-speedup",
        type=float,
        default=1.4,
    )
    parser.add_argument(
        "--profile",
        default=str(WORKER_PROFILE_PATH),
    )
    return parser.parse_args()


def _command(config_path, output_path, memory_mb):
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_baseline_worker.py"),
        "--config",
        str(config_path),
        "--gpu-memory-mb",
        str(memory_mb),
        "--no-save",
        "--output-json",
        str(output_path),
    ]


def _run_one(config_path, output_path, memory_mb):
    return subprocess.run(
        _command(config_path, output_path, memory_mb),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _load_summary(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = _parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive.")
    if args.gpu_memory_mb < 1024:
        raise SystemExit("--gpu-memory-mb must be at least 1024.")

    configs = [
        make_config(
            split="nonIID",
            approach="merged",
            environment="noise",
            mitigation="ebm",
            sigma=0.2,
            seed=seed,
            rounds=args.rounds,
            phase="worker_benchmark",
            gamma=0.0,
        )
        for seed in (9101, 9102)
    ]

    with tempfile.TemporaryDirectory(prefix="papermerge-worker-benchmark-") as directory:
        temp = Path(directory)
        config_paths = []
        for index, config in enumerate(configs):
            path = temp / f"config_{index}.json"
            write_json_atomic(path, config)
            config_paths.append(path)

        # Warm only the immutable CPU cache. The warm-up result is discarded.
        warm_output = temp / "warm.json"
        print("Preparing immutable CIFAR cache and warming one worker...", flush=True)
        warm = _run_one(config_paths[0], warm_output, args.gpu_memory_mb)
        if warm.returncode != 0:
            print(warm.stdout, flush=True)
            raise SystemExit("Worker warm-up failed.")

        sequential_outputs = [temp / f"sequential_{i}.json" for i in range(2)]
        sequential_started = time.perf_counter()
        sequential_processes = [
            _run_one(config_paths[i], sequential_outputs[i], args.gpu_memory_mb)
            for i in range(2)
        ]
        sequential_seconds = time.perf_counter() - sequential_started
        if any(process.returncode != 0 for process in sequential_processes):
            for process in sequential_processes:
                print(process.stdout, flush=True)
            raise SystemExit("Sequential worker benchmark failed.")

        parallel_outputs = [temp / f"parallel_{i}.json" for i in range(2)]
        parallel_started = time.perf_counter()
        processes = [
            subprocess.Popen(
                _command(config_paths[i], parallel_outputs[i], args.gpu_memory_mb),
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            for i in range(2)
        ]
        parallel_logs = []
        for process in processes:
            output, _ = process.communicate()
            parallel_logs.append(output)
        parallel_seconds = time.perf_counter() - parallel_started

        parallel_ok = all(process.returncode == 0 for process in processes)
        if not parallel_ok:
            for output in parallel_logs:
                print(output, flush=True)

        equivalent = False
        if parallel_ok:
            sequential_summaries = [
                _load_summary(path) for path in sequential_outputs
            ]
            parallel_summaries = [
                _load_summary(path) for path in parallel_outputs
            ]
            equivalent = all(
                left["metricsFingerprint"] == right["metricsFingerprint"]
                and left["initializationHash"] == right["initializationHash"]
                for left, right in zip(
                    sequential_summaries,
                    parallel_summaries,
                )
            )

        speedup = (
            sequential_seconds / parallel_seconds
            if parallel_seconds > 0
            else 0.0
        )
        validated = (
            parallel_ok
            and equivalent
            and speedup >= float(args.minimum_speedup)
        )
        reasons = []
        if not parallel_ok:
            reasons.append("At least one concurrent worker failed or exhausted GPU memory.")
        if parallel_ok and not equivalent:
            reasons.append("Concurrent and sequential result fingerprints differed.")
        if speedup < float(args.minimum_speedup):
            reasons.append(
                f"Measured {speedup:.2f}x speedup was below "
                f"the required {float(args.minimum_speedup):.2f}x."
            )

        profile = {
            "schemaVersion": 1,
            "status": "validated" if validated else "single_lane_required",
            "benchmarkedAt": datetime.now().astimezone().isoformat(),
            "benchmarkRounds": int(args.rounds),
            "gpuMemoryLimitMb": int(args.gpu_memory_mb),
            "minimumRequiredSpeedup": float(args.minimum_speedup),
            "sequentialSeconds": float(sequential_seconds),
            "parallelSeconds": float(parallel_seconds),
            "measuredSpeedup": float(speedup),
            "equivalentResults": bool(equivalent),
            "parallelWorkersSucceeded": bool(parallel_ok),
            "recommendedLanes": 2 if validated else 1,
            "concurrencySlowdown": (
                max(1.0, 2.0 / speedup)
                if validated and speedup > 0
                else 1.0
            ),
            "reasons": reasons,
        }
        write_json_atomic(args.profile, profile)

    print(json.dumps(profile, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
