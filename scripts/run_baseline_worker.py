#!/usr/bin/env python3
"""Run and atomically save one Campaign 3 configuration.

The process owns exactly one experiment. Exiting after the result is written
guarantees that CUDA and TensorFlow allocator state cannot accumulate between
queue items.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
EVENT_PREFIX = "@@CAMPAIGN_EVENT@@"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON configuration path")
    parser.add_argument(
        "--gpu-memory-mb",
        type=int,
        default=0,
        help="Per-process GPU memory cap; 0 uses TensorFlow memory growth.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "experiments" / "cache" / "cifar10"),
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _emit(event, **payload):
    print(
        EVENT_PREFIX
        + json.dumps({"event": event, **payload}, sort_keys=True),
        flush=True,
    )


def _configure_gpu(memory_limit_mb):
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf

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
    digest = hashlib.sha256()
    for indices in data_metadata.get("clientIndices", []):
        import numpy as np

        digest.update(np.asarray(indices, dtype=np.int64).tobytes(order="C"))
    return digest.hexdigest()


def _code_revision():
    git_dir = PROJECT_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            return (git_dir / head[5:]).read_text(encoding="utf-8").strip()
        return head
    except OSError:
        return "unknown"


def _metrics_fingerprint(result):
    import numpy as np

    digest = hashlib.sha256()
    for key in (
        "avg_history",
        "worst_history",
        "final_node_accuracy",
        "confusion",
        "mu_history",
        "selected_sources",
    ):
        value = np.asarray(result[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main():
    args = _parse_args()
    wall_started = time.perf_counter()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config.get("campaignVersion", 0)) != 3:
        raise SystemExit("run_baseline_worker.py accepts Campaign 3 configs only.")

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        _emit("stop_requested", runId=config.get("runId"))

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    tf, gpus = _configure_gpu(max(0, int(args.gpu_memory_mb)))

    import numpy as np

    from basil_core.experiment_engine import run_campaign_three
    from basil_core.data.cifar import loadCifar10, makeLoaders
    from basil_core.models import CIFARModel
    from gui.baseline_study import (
        config_hash,
        is_completed,
        result_paths,
        write_json_atomic,
        write_npz_atomic,
    )

    metrics_path, run_path = result_paths(config)
    if not args.no_save and is_completed(config):
        _emit("skipped", runId=config["runId"], reason="already_completed")
        return 0

    started_at = datetime.now().astimezone().isoformat()
    run_metadata = {
        "schemaVersion": 3,
        "campaignId": config.get("campaignId"),
        "runId": config.get("runId"),
        "status": "preparing",
        "configHash": config_hash(config),
        "codeRevision": _code_revision(),
        "startedAt": started_at,
        "executionMode": "isolated_process",
        "gpuMemoryLimitMb": max(0, int(args.gpu_memory_mb)),
        "config": dict(config),
    }
    if not args.no_save:
        write_json_atomic(run_path, run_metadata)

    train = test = train_loaders = test_loader = data_metadata = result = None
    try:
        _emit(
            "preparing",
            runId=config["runId"],
            gpuCount=len(gpus),
            gpuMemoryLimitMb=max(0, int(args.gpu_memory_mb)),
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
        run_metadata["partitionHash"] = _partition_hash(data_metadata)
        run_metadata["status"] = "running"
        if not args.no_save:
            write_json_atomic(run_path, run_metadata)

        _emit("started", runId=config["runId"])
        result = run_campaign_three(
            config=config,
            model_class=CIFARModel,
            train_loaders=train_loaders,
            test_loader=test_loader,
            data_metadata=data_metadata,
            stop_callback=lambda: stop_requested,
            round_callback=lambda round_num, avg, worst, total: _emit(
                "round",
                runId=config["runId"],
                round=int(round_num),
                totalRounds=int(total),
                averageAccuracy=float(avg),
                worstAccuracy=float(worst),
            ),
        )

        if result.get("stopped") or stop_requested:
            run_metadata.update(
                {
                    "status": "stopped",
                    "completedRounds": int(len(result.get("avg_history", []))),
                    "stoppedAt": datetime.now().astimezone().isoformat(),
                    "wallRuntimeSeconds": float(time.perf_counter() - wall_started),
                }
            )
            if not args.no_save:
                write_json_atomic(run_path, run_metadata)
            _emit(
                "stopped",
                runId=config["runId"],
                completedRounds=run_metadata["completedRounds"],
            )
            return 2

        initialization_hash = result.pop("initialization_hash")
        result.pop("stopped", None)
        fingerprint = _metrics_fingerprint(result)
        if not args.no_save:
            write_npz_atomic(metrics_path, **result)

        run_metadata.update(
            {
                "status": "completed",
                "completedAt": datetime.now().astimezone().isoformat(),
                "initializationHash": initialization_hash,
                "metricsFingerprint": fingerprint,
                "finalAverageAccuracy": float(result["final_avg"]),
                "finalWorstAccuracy": float(result["final_worst"]),
                "runtimeSeconds": float(result["runtime_seconds"]),
                "wallRuntimeSeconds": float(time.perf_counter() - wall_started),
                "peakGpuBytes": int(result["peak_gpu_bytes"]),
            }
        )
        if not args.no_save:
            write_json_atomic(run_path, run_metadata)

        summary = {
            "status": "completed",
            "runId": config["runId"],
            "metricsFingerprint": fingerprint,
            "initializationHash": initialization_hash,
            "finalAverageAccuracy": float(result["final_avg"]),
            "finalWorstAccuracy": float(result["final_worst"]),
            "runtimeSeconds": float(result["runtime_seconds"]),
            "wallRuntimeSeconds": run_metadata["wallRuntimeSeconds"],
            "peakGpuBytes": int(result["peak_gpu_bytes"]),
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
