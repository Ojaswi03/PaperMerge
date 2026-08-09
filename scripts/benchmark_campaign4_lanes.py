#!/usr/bin/env python3
"""Validate memory-safe Campaign 4 GPU concurrency without saving results."""

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


MIB = 1024 * 1024


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", default=str(PERFORMANCE_PROFILE))
    parser.add_argument("--calibration-memory-mb", type=int, default=7600)
    parser.add_argument("--per-worker-margin-mb", type=int, default=192)
    parser.add_argument("--system-reserve-mb", type=int, default=256)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    return parser.parse_args()


def _read_profile(path):
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(
            "Run scripts/benchmark_campaign4.py successfully before the lane "
            "benchmark."
        ) from error
    selected = profile.get("selectedProfile")
    if profile.get("status") != "validated" or not isinstance(selected, dict):
        raise SystemExit(
            "Campaign 4 algorithm profile is not validated; concurrency remains disabled."
        )
    return profile, selected


def _configured(config, selected, rounds):
    result = dict(config)
    for key in (
        "internalMicroBatchSize",
        "precisionProfile",
        "jitCompile",
        "gpuAllocator",
    ):
        if key in selected:
            result[key] = selected[key]
    result["nRounds"] = int(rounds)
    result["phase"] = "performance_benchmark"
    result["autoPlotCampaign4"] = False
    result["performanceProfileId"] = selected.get("profileId", "validated")
    result["runId"] = make_run_id(result)
    return result


def _command(config_path, output_path, memory_mb):
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_campaign4_worker.py"),
        "--config",
        str(config_path),
        "--gpu-memory-mb",
        str(int(memory_mb)),
        "--no-save",
        "--output-json",
        str(output_path),
        "--suppress-node-events",
    ]


def _load_summary(path, return_code):
    if return_code != 0 or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _single_run(config, directory, name, memory_mb):
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
    return {
        "returnCode": int(process.returncode),
        "elapsedSeconds": float(time.perf_counter() - started),
        "summary": _load_summary(output_path, process.returncode),
        "logTail": "\n".join(process.stdout.splitlines()[-40:]),
    }


def _pair_run(configs, directory, name, memory_limits):
    entries = []
    started = time.perf_counter()
    for index, (config, memory_mb) in enumerate(zip(configs, memory_limits)):
        config_path = directory / f"{name}.{index}.config.json"
        output_path = directory / f"{name}.{index}.summary.json"
        write_json_atomic(config_path, config)
        process = subprocess.Popen(
            _command(config_path, output_path, memory_mb),
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        entries.append(
            {
                "process": process,
                "outputPath": output_path,
                "finished": None,
            }
        )

    try:
        while any(entry["process"].poll() is None for entry in entries):
            now = time.perf_counter()
            for entry in entries:
                if entry["finished"] is None and entry["process"].poll() is not None:
                    entry["finished"] = now
            time.sleep(0.05)
    except BaseException:
        for entry in entries:
            if entry["process"].poll() is None:
                entry["process"].terminate()
        for entry in entries:
            try:
                entry["process"].wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                entry["process"].kill()
        raise
    finished_at = time.perf_counter()
    runs = []
    for entry in entries:
        process = entry["process"]
        output, _ = process.communicate()
        completed = entry["finished"] or finished_at
        runs.append(
            {
                "returnCode": int(process.returncode),
                "elapsedSeconds": float(completed - started),
                "summary": _load_summary(entry["outputPath"], process.returncode),
                "logTail": "\n".join(output.splitlines()[-40:]),
            }
        )
    return {
        "wallSeconds": float(finished_at - started),
        "runs": runs,
    }


def _gpu_memory():
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        total, used, free = [
            int(value.strip()) for value in process.stdout.splitlines()[0].split(",")
        ]
        return {"totalMiB": total, "usedMiB": used, "freeMiB": free}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _finite_summary(summary):
    if not isinstance(summary, dict):
        return False
    try:
        return all(
            math.isfinite(float(summary[key]))
            for key in (
                "finalAverageAccuracy",
                "finalWorstAccuracy",
                "runtimeSeconds",
                "wallRuntimeSeconds",
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _validate_pair(
    *,
    pair_id,
    indices,
    configs,
    baselines,
    limits,
    directory,
    repeats,
    minimum_speedup,
    gpu_memory,
    system_reserve_mb,
):
    observed_peak_mib = [
        int(math.ceil(int(baselines[index]["summary"]["peakGpuBytes"]) / MIB))
        for index in indices
    ]
    required_free = int(sum(limits) + system_reserve_mb)
    free_mib = int(gpu_memory["freeMiB"]) if gpu_memory else 0
    result = {
        "pairId": pair_id,
        "caseIndices": list(indices),
        "memoryLimitsMb": list(limits),
        "observedSequentialPeakMiB": observed_peak_mib,
        "requiredFreeMiB": required_free,
        "freeMiBAtGate": free_mib,
        "attempted": False,
        "status": "insufficient_vram",
        "repeats": [],
    }
    if gpu_memory is None:
        result["status"] = "gpu_memory_unavailable"
        return result
    if free_mib < required_free:
        return result

    result["attempted"] = True
    for repeat in range(repeats):
        pair = _pair_run(
            [configs[index] for index in indices],
            directory,
            f"{pair_id}.repeat-{repeat}",
            limits,
        )
        checks = []
        for position, case_index in enumerate(indices):
            run = pair["runs"][position]
            baseline = baselines[case_index]
            summary = run.get("summary")
            checks.append(
                bool(
                    run["returnCode"] == 0
                    and _finite_summary(summary)
                    and summary.get("initializationHash")
                    == baseline["summary"].get("initializationHash")
                    and summary.get("scientificFingerprint")
                    == baseline["summary"].get("scientificFingerprint")
                )
            )
        sequential_seconds = sum(
            float(baselines[index]["elapsedSeconds"]) for index in indices
        )
        speedup = sequential_seconds / max(float(pair["wallSeconds"]), 1e-9)
        inflations = [
            pair["runs"][position]["elapsedSeconds"]
            / max(float(baselines[case_index]["elapsedSeconds"]), 1e-9)
            for position, case_index in enumerate(indices)
        ]
        result["repeats"].append(
            {
                "wallSeconds": float(pair["wallSeconds"]),
                "sequentialSeconds": float(sequential_seconds),
                "speedup": float(speedup),
                "runtimeInflations": [float(value) for value in inflations],
                "checksPassed": checks,
                "returnCodes": [run["returnCode"] for run in pair["runs"]],
                "failedLogs": [
                    run["logTail"]
                    for position, run in enumerate(pair["runs"])
                    if run["returnCode"] != 0 or not checks[position]
                ],
            }
        )
        print(
            f"  {pair_id} repeat {repeat + 1}/{repeats}: "
            f"wall={pair['wallSeconds']:.1f}s speedup={speedup:.2f}x "
            f"checks={'pass' if all(checks) else 'fail'} "
            f"exits={[run['returnCode'] for run in pair['runs']]}",
            flush=True,
        )

    all_checks = all(
        all(repeat["checksPassed"]) for repeat in result["repeats"]
    )
    speedups = [repeat["speedup"] for repeat in result["repeats"]]
    median_speedup = float(statistics.median(speedups))
    maximum_inflation = max(
        value
        for repeat in result["repeats"]
        for value in repeat["runtimeInflations"]
    )
    median_inflations = [
        float(
            statistics.median(
                repeat["runtimeInflations"][position]
                for repeat in result["repeats"]
            )
        )
        for position in range(2)
    ]
    result.update(
        {
            "medianSpeedup": median_speedup,
            "maximumRuntimeInflation": float(maximum_inflation),
            "medianRuntimeInflations": median_inflations,
            "fingerprintGatePassed": bool(all_checks),
            "throughputGatePassed": median_speedup >= float(minimum_speedup),
            "status": (
                "validated"
                if all_checks and median_speedup >= float(minimum_speedup)
                else "rejected"
            ),
        }
    )
    return result


def main():
    args = _parse_args()
    if args.rounds <= 0 or args.repeats <= 0:
        raise SystemExit("--rounds and --repeats must be positive.")
    if args.calibration_memory_mb < 4096:
        raise SystemExit("Calibration requires at least a 4096 MB GPU cap.")
    if args.per_worker_margin_mb < 64 or args.system_reserve_mb < 128:
        raise SystemExit("Memory margins are below the allowed safety floor.")

    profile, selected = _read_profile(args.profile)
    base_configs = build_performance_benchmark(rounds=args.rounds)
    configs = [_configured(config, selected, args.rounds) for config in base_configs]
    baselines = {}

    with tempfile.TemporaryDirectory(prefix="papermerge-campaign4-lanes-") as raw:
        directory = Path(raw)
        print("Sequential full-batch calibration", flush=True)
        for index, config in enumerate(configs):
            run = _single_run(
                config,
                directory,
                f"baseline-{index}",
                args.calibration_memory_mb,
            )
            baselines[index] = run
            peak_mib = (
                int(math.ceil(run["summary"]["peakGpuBytes"] / MIB))
                if run.get("summary")
                else 0
            )
            print(
                f"  case {index + 1}/4: exit={run['returnCode']} "
                f"time={run['elapsedSeconds']:.1f}s peak={peak_mib} MiB",
                flush=True,
            )
        if not all(
            run["returnCode"] == 0 and _finite_summary(run.get("summary"))
            for run in baselines.values()
        ):
            failed = [
                run["logTail"]
                for run in baselines.values()
                if run["returnCode"] != 0 or not _finite_summary(run.get("summary"))
            ]
            raise SystemExit(
                "Sequential calibration failed; concurrency was not attempted.\n"
                + "\n\n".join(failed)
            )

        standard_peak = max(
            int(math.ceil(baselines[index]["summary"]["peakGpuBytes"] / MIB))
            for index in (0, 1)
        )
        ebm_peak = max(
            int(math.ceil(baselines[index]["summary"]["peakGpuBytes"] / MIB))
            for index in (2, 3)
        )
        gpu_memory = _gpu_memory()
        selected_margin = int(args.per_worker_margin_mb)
        minimum_margin = 128
        if gpu_memory:
            desired = (
                standard_peak
                + ebm_peak
                + 2 * selected_margin
                + int(args.system_reserve_mb)
            )
            minimum = (
                standard_peak
                + ebm_peak
                + 2 * minimum_margin
                + int(args.system_reserve_mb)
            )
            if desired > int(gpu_memory["freeMiB"]) and minimum <= int(
                gpu_memory["freeMiB"]
            ):
                selected_margin = minimum_margin
        standard_limit = standard_peak + selected_margin
        ebm_limit = ebm_peak + selected_margin
        if gpu_memory:
            print(
                "GPU memory gate: "
                f"{gpu_memory['freeMiB']} MiB free of {gpu_memory['totalMiB']} MiB; "
                f"caps standard={standard_limit} MiB, EBM={ebm_limit} MiB "
                f"({selected_margin} MiB measured-peak margin each)",
                flush=True,
            )

        pair_specs = (
            ("standard_standard", (0, 1), (standard_limit, standard_limit)),
            ("ebm_standard", (2, 0), (ebm_limit, standard_limit)),
            ("ebm_ebm", (2, 3), (ebm_limit, ebm_limit)),
        )
        pair_results = {}
        for pair_id, indices, limits in pair_specs:
            result = _validate_pair(
                pair_id=pair_id,
                indices=indices,
                configs=configs,
                baselines=baselines,
                limits=limits,
                directory=directory,
                repeats=args.repeats,
                minimum_speedup=args.minimum_speedup,
                gpu_memory=gpu_memory,
                system_reserve_mb=args.system_reserve_mb,
            )
            pair_results[pair_id] = result
            if not result["attempted"]:
                print(
                    f"  {pair_id}: skipped ({result['status']}), needs "
                    f"{result['requiredFreeMiB']} MiB free; measured "
                    f"{result['freeMiBAtGate']} MiB",
                    flush=True,
                )

    standard_ok = pair_results["standard_standard"]["status"] == "validated"
    mixed_ok = pair_results["ebm_standard"]["status"] == "validated"
    dual_ebm_ok = pair_results["ebm_ebm"]["status"] == "validated"
    validated_repeats = args.repeats >= 3
    if not validated_repeats:
        standard_ok = mixed_ok = dual_ebm_ok = False

    if dual_ebm_ok:
        selected_pair = pair_results["ebm_ebm"]
        max_ebm = 2
    elif mixed_ok:
        selected_pair = pair_results["ebm_standard"]
        max_ebm = 1
    elif standard_ok:
        selected_pair = pair_results["standard_standard"]
        max_ebm = 1
    else:
        selected_pair = None
        max_ebm = 1

    lanes = 2 if selected_pair is not None else 1

    def pair_inflation(pair_id, position):
        values = pair_results[pair_id].get("medianRuntimeInflations")
        if isinstance(values, list) and len(values) > position:
            return max(1.0, float(values[position]))
        return 1.0

    runtime_slowdowns = {
        "mixedEbm": pair_inflation("ebm_standard", 0),
        "mixedStandard": pair_inflation("ebm_standard", 1),
        "dualStandard": max(
            pair_inflation("standard_standard", 0),
            pair_inflation("standard_standard", 1),
        ),
        "dualEbm": max(
            pair_inflation("ebm_ebm", 0),
            pair_inflation("ebm_ebm", 1),
        ),
    }
    concurrency = {
        "schemaVersion": 1,
        "status": "validated" if lanes == 2 else "single_lane_required",
        "profileId": (
            f"{selected.get('profileId', 'backend')}-resource-aware-2lane"
            if lanes == 2
            else "single_lane"
        ),
        "benchmarkedAt": datetime.now().astimezone().isoformat(),
        "benchmarkRounds": int(args.rounds),
        "repeats": int(args.repeats),
        "recommendedLanes": lanes,
        "maxConcurrentEbm": max_ebm,
        "allowMixedEbmStandard": bool(mixed_ok),
        "allowDualStandard": bool(standard_ok),
        "prioritizeEbm": bool(mixed_ok),
        "memoryLimitsMb": {
            "standard": int(standard_limit),
            "ebm": int(ebm_limit),
        },
        "concurrencySlowdown": (
            max(1.0, float(selected_pair["maximumRuntimeInflation"]))
            if selected_pair
            else 1.0
        ),
        "runtimeSlowdowns": runtime_slowdowns,
        "measuredSpeedup": (
            float(selected_pair["medianSpeedup"]) if selected_pair else 0.0
        ),
        "minimumSpeedup": float(args.minimum_speedup),
        "gpuMemoryAtBenchmark": gpu_memory,
        "requestedPerWorkerMarginMiB": int(args.per_worker_margin_mb),
        "perWorkerMarginMiB": int(selected_margin),
        "systemReserveMiB": int(args.system_reserve_mb),
        "pairResults": pair_results,
        "reasons": [
            "Concurrency changes only process scheduling and allocator caps; "
            f"batch 512, validated {selected.get('precisionProfile', 'float32')} "
            "execution, method equations, data order, and 100-round scientific "
            "contracts remain unchanged.",
            "A pair is enabled only after exact initialization/result fingerprints, finite outputs, VRAM headroom, and throughput gates pass in isolated no-save workers.",
        ],
    }
    updated = dict(profile)
    updated["schemaVersion"] = max(2, int(profile.get("schemaVersion", 1)))
    updated["concurrencyProfile"] = concurrency
    updated["recommendedLanes"] = lanes
    updated["concurrencySlowdown"] = concurrency["concurrencySlowdown"]
    updated["gpuMemoryLimitMb"] = max(standard_limit, ebm_limit)
    write_json_atomic(args.profile, updated)
    print(json.dumps(concurrency, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
