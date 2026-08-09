#!/usr/bin/env python3
"""Run one isolated Campaign 4 configuration and save it atomically."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
EVENT_PREFIX = "@@CAMPAIGN_EVENT@@"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-memory-mb", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "experiments" / "cache" / "cifar10"),
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--suppress-node-events",
        action="store_true",
        help="Disable live node JSON only for non-GUI benchmarks.",
    )
    return parser.parse_args()


def _emit(event, **payload):
    print(
        EVENT_PREFIX + json.dumps({"event": event, **payload}, sort_keys=True),
        flush=True,
    )


def _configure_tensorflow(config, memory_limit_mb):
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    allocator = str(config.get("gpuAllocator", "bfc"))
    if allocator == "cuda_malloc_async":
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    elif allocator == "bfc":
        os.environ.pop("TF_GPU_ALLOCATOR", None)
    else:
        raise ValueError(f"Unsupported Campaign 4 GPU allocator: {allocator}")
    import tensorflow as tf

    precision = str(config.get("precisionProfile", "float32"))
    tf.keras.mixed_precision.set_global_policy(precision)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            if memory_limit_mb > 0:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [
                        tf.config.LogicalDeviceConfiguration(
                            memory_limit=float(memory_limit_mb)
                        )
                    ],
                )
            else:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf, gpus


def _partition_hash(data_metadata):
    import numpy as np

    digest = hashlib.sha256()
    for indices in data_metadata.get("clientIndices", []):
        digest.update(np.asarray(indices, dtype=np.int64).tobytes(order="C"))
    return digest.hexdigest()


def _git_state():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def _fingerprint(groups):
    import numpy as np

    digest = hashlib.sha256()
    for group_name, arrays in groups:
        digest.update(group_name.encode("ascii"))
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            digest.update(name.encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _scientific_fingerprint(metrics, telemetry):
    metric_values = {
        key: value
        for key, value in metrics.items()
        if key not in {"runtime_seconds", "peak_gpu_bytes", "stage_runtime_seconds"}
    }
    telemetry_values = {
        key: value
        for key, value in telemetry.items()
        if not key.endswith("_seconds")
    }
    return _fingerprint(
        (("metrics", metric_values), ("telemetry", telemetry_values))
    )


def _tensorflow_metadata(tf, gpus):
    try:
        build_info = dict(tf.sysconfig.get_build_info())
    except Exception:
        build_info = {}
    logical = tf.config.list_logical_devices("GPU")
    return {
        "tensorflowVersion": tf.__version__,
        "physicalGpus": [device.name for device in gpus],
        "logicalGpus": [device.name for device in logical],
        "cudaVersion": str(build_info.get("cuda_version", "unknown")),
        "cudnnVersion": str(build_info.get("cudnn_version", "unknown")),
    }


def main():
    args = _parse_args()
    wall_started = time.perf_counter()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config.get("campaignVersion", 0)) != 4:
        raise SystemExit("run_campaign4_worker.py accepts Campaign 4 configs only.")
    phase = str(config.get("phase", "confirmation"))
    allowed_phases = {
        "diagnostic",
        "static_control",
        "confirmation",
        "performance_benchmark",
        "precision_validation",
    }
    if phase not in allowed_phases:
        raise SystemExit(f"Unsupported Campaign 4 phase: {phase}")
    if phase in (
        "diagnostic",
        "static_control",
        "confirmation",
        "precision_validation",
    ):
        if int(config.get("nRounds", 0)) != 100:
            raise SystemExit(
                f"Campaign 4 {phase} runs must use exactly 100 rounds. "
                "Use phase=performance_benchmark for short execution checks."
            )
    precision = str(config.get("precisionProfile", "float32"))
    if precision not in ("float32", "mixed_bfloat16"):
        raise SystemExit(
            "Campaign 4 supports float32 or the separately benchmarked "
            "mixed_bfloat16 profile. mixed_float16 remains disabled because "
            "the second-order EBM path has no validated dynamic loss scaling."
        )
    uses_ebm = (
        bool(config.get("useChannelNoise"))
        and str(config.get("noiseMitigation", "none")) == "ebm"
    )
    if (
        uses_ebm
        and int(config.get("internalMicroBatchSize", 0))
        != int(config.get("batchSize", 512))
        and phase != "performance_benchmark"
    ):
        raise SystemExit(
            "Official Campaign 4 EBM runs require internalMicroBatchSize == "
            "batchSize so the saved run uses the declared full-batch "
            "gradient-norm objective."
        )

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        _emit("stop_requested", runId=config.get("runId"))

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    tf, gpus = _configure_tensorflow(config, max(0, int(args.gpu_memory_mb)))

    import numpy as np

    from basil_core.campaign4_engine import run_campaign_four
    from basil_core.data.cifar import loadCifar10, makeLoaders
    from basil_core.models import CIFARModel
    from gui.campaign4 import (
        CAMPAIGN_ID,
        config_hash,
        current_source_hashes,
        is_completed,
        result_paths,
        write_json_atomic,
        write_npz_atomic,
    )

    metrics_path, run_path, telemetry_path = result_paths(config)
    if not args.no_save and is_completed(config):
        _emit("skipped", runId=config["runId"], reason="already_completed")
        return 0

    revision, dirty = _git_state()
    run_metadata = {
        "schemaVersion": 4,
        "campaignId": CAMPAIGN_ID,
        "runId": config.get("runId"),
        "status": "preparing",
        "configHash": config_hash(config),
        "codeRevision": revision,
        "workingTreeDirty": dirty,
        "sourceHashes": current_source_hashes(),
        "startedAt": datetime.now().astimezone().isoformat(),
        "executionMode": "isolated_process",
        "gpuMemoryLimitMb": max(0, int(args.gpu_memory_mb)),
        "gpuAllocator": str(config.get("gpuAllocator", "bfc")),
        "config": dict(config),
    }
    if not args.no_save:
        write_json_atomic(run_path, run_metadata)

    train = test = train_loaders = test_loader = data_metadata = result = None
    serialization_started = None
    try:
        _emit(
            "preparing",
            runId=config["runId"],
            gpuCount=len(gpus),
            gpuMemoryLimitMb=max(0, int(args.gpu_memory_mb)),
            precision=config.get("precisionProfile", "float32"),
            internalMicroBatchSize=int(config.get("internalMicroBatchSize", 128)),
            gpuAllocator=str(config.get("gpuAllocator", "bfc")),
        )
        train, test = loadCifar10(cacheDir=args.cache_dir)
        train_loaders, test_loader, data_metadata = makeLoaders(
            train,
            test,
            batchSize=int(config["batchSize"]),
            iid=not bool(config.get("nonIID", True)),
            nClients=int(config["nNodes"]),
            dirichletAlpha=float(config.get("dirichletAlpha", 0.2)),
            seed=int(config["seed"]),
            returnMetadata=True,
            cacheDir=args.cache_dir,
        )
        run_metadata.update(
            {
                "partitionHash": _partition_hash(data_metadata),
                "status": "running",
                "runtime": _tensorflow_metadata(tf, gpus),
            }
        )
        if not args.no_save:
            write_json_atomic(run_path, run_metadata)

        _emit("started", runId=config["runId"])
        result = run_campaign_four(
            config=config,
            model_class=CIFARModel,
            train_loaders=train_loaders,
            test_loader=test_loader,
            data_metadata=data_metadata,
            stop_callback=lambda: stop_requested,
            round_callback=lambda round_num, avg, worst, total, per_node: _emit(
                "round",
                runId=config["runId"],
                round=int(round_num),
                totalRounds=int(total),
                averageAccuracy=float(avg),
                worstAccuracy=float(worst),
                perNodeAccuracy=np.asarray(per_node, dtype=float).tolist(),
            ),
            node_callback=(
                None
                if args.suppress_node_events
                else lambda payload: _emit(
                    "node_update", runId=config["runId"], **payload
                )
            ),
        )

        if result.get("stopped") or stop_requested:
            completed_rounds = int(len(result.get("metrics", {}).get("avg_history", [])))
            run_metadata.update(
                {
                    "status": "stopped",
                    "completedRounds": completed_rounds,
                    "stoppedAt": datetime.now().astimezone().isoformat(),
                    "wallRuntimeSeconds": float(time.perf_counter() - wall_started),
                }
            )
            if not args.no_save:
                write_json_atomic(run_path, run_metadata)
            _emit("stopped", runId=config["runId"], completedRounds=completed_rounds)
            return 2

        initialization_hash = result["initialization_hash"]
        metrics = result["metrics"]
        telemetry = result["telemetry"]
        fingerprint = _fingerprint((("metrics", metrics), ("telemetry", telemetry)))
        scientific_fingerprint = _scientific_fingerprint(metrics, telemetry)
        serialization_started = time.perf_counter()
        if not args.no_save:
            write_npz_atomic(metrics_path, **metrics)
            write_npz_atomic(telemetry_path, **telemetry)
        serialization_seconds = time.perf_counter() - serialization_started

        run_metadata.update(
            {
                "status": "completed",
                "completedAt": datetime.now().astimezone().isoformat(),
                "initializationHash": initialization_hash,
                "resultFingerprint": fingerprint,
                "scientificFingerprint": scientific_fingerprint,
                "finalAverageAccuracy": float(metrics["final_avg"]),
                "finalWorstAccuracy": float(metrics["final_worst"]),
                "finalLearningCurveAuc": float(metrics["learning_curve_auc"]),
                "runtimeSeconds": float(metrics["runtime_seconds"]),
                "wallRuntimeSeconds": float(time.perf_counter() - wall_started),
                "serializationSeconds": float(serialization_seconds),
                "peakGpuBytes": int(metrics["peak_gpu_bytes"]),
                "artifactPaths": {
                    "metrics": str(metrics_path),
                    "telemetry": str(telemetry_path),
                    "run": str(run_path),
                },
            }
        )
        if not args.no_save:
            write_json_atomic(run_path, run_metadata)

        summary = {
            "status": "completed",
            "runId": config["runId"],
            "resultFingerprint": fingerprint,
            "scientificFingerprint": scientific_fingerprint,
            "initializationHash": initialization_hash,
            "finalAverageAccuracy": float(metrics["final_avg"]),
            "finalWorstAccuracy": float(metrics["final_worst"]),
            "finalLearningCurveAuc": float(metrics["learning_curve_auc"]),
            "runtimeSeconds": float(metrics["runtime_seconds"]),
            "wallRuntimeSeconds": run_metadata["wallRuntimeSeconds"],
            "serializationSeconds": float(serialization_seconds),
            "peakGpuBytes": int(metrics["peak_gpu_bytes"]),
        }
        if args.output_json:
            write_json_atomic(args.output_json, summary)
        _emit("completed", **summary)
        return 0
    except Exception as error:
        run_metadata.update(
            {
                "status": "failed",
                "failedAt": datetime.now().astimezone().isoformat(),
                "wallRuntimeSeconds": float(time.perf_counter() - wall_started),
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        if not args.no_save:
            write_json_atomic(run_path, run_metadata)
        _emit(
            "failed",
            runId=config.get("runId"),
            error=str(error),
            errorType=type(error).__name__,
            resourceExhausted=isinstance(error, tf.errors.ResourceExhaustedError),
        )
        traceback.print_exc()
        return 1
    finally:
        result = None
        data_metadata = None
        test_loader = None
        train_loaders = None
        train = None
        test = None
        try:
            tf.keras.backend.clear_session(free_memory=True)
        except TypeError:
            tf.keras.backend.clear_session()
        for _ in range(3):
            gc.collect()
        _emit("cleanup_complete", runId=config.get("runId"))


if __name__ == "__main__":
    raise SystemExit(main())
