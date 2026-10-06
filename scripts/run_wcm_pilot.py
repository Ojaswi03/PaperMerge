#!/usr/bin/env python3
"""Run one isolated WCM pilot without touching Campaign 3 results.

GPU execution is refused while another compute process is visible. The script
uses one shared model, a bounded activation microbatch, and one process per run.
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
import subprocess
import sys
import tempfile
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULT_ROOT = PROJECT_ROOT / "experiments" / "wcm_pilot" / "r1"
PAPER_PATH = (
    PROJECT_ROOT
    / "Papers"
    / "002-Robust Federated Learning with Noisy Communication.pdf"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run one paper-WCM pilot in an isolated process."
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--approach", choices=("merged", "cart"), default="cart")
    parser.add_argument(
        "--environment",
        choices=("noise", "hidden_noise"),
        default="hidden_noise",
    )
    parser.add_argument("--snapshot-selection", action="store_true")
    parser.add_argument("--sigma", type=float, choices=(0.2, 0.3, 0.4, 0.5, 0.6), required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--cart-gamma", type=float, default=0.0005)
    parser.add_argument("--wcm-penalty", type=float, default=0.1)
    parser.add_argument("--wcm-rho-exponent", type=float, default=0.6)
    parser.add_argument("--wcm-gamma-exponent", type=float, default=0.8)
    parser.add_argument("--wcm-boundary-samples", type=int, default=1)
    parser.add_argument("--wcm-radius-multiplier", type=float, default=1.0)
    parser.add_argument("--gpu-memory-mb", type=int, default=3800)
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "experiments" / "cache" / "cifar10"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def _active_gpu_processes():
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Unable to inspect GPU compute processes with nvidia-smi."
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "nvidia-smi returned an error."
        raise RuntimeError(
            f"Unable to verify GPU availability: {detail}"
        )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _validate_args(args):
    if int(args.rounds) < 1:
        raise ValueError("--rounds must be at least 1.")
    if float(args.cart_gamma) < 0.0:
        raise ValueError("--cart-gamma cannot be negative.")
    if float(args.wcm_penalty) <= 0.0:
        raise ValueError("--wcm-penalty must be positive.")
    if not (
        0.5
        < float(args.wcm_rho_exponent)
        < float(args.wcm_gamma_exponent)
        < 1.0
    ):
        raise ValueError(
            "WCM requires 0.5 < rho exponent < gamma exponent < 1 "
            "(paper Lemma 7)."
        )
    if int(args.wcm_boundary_samples) < 1:
        raise ValueError("--wcm-boundary-samples must be at least 1.")
    if float(args.wcm_radius_multiplier) <= 0.0:
        raise ValueError("--wcm-radius-multiplier must be positive.")
    if args.device == "gpu" and not 512 <= int(args.gpu_memory_mb) <= 4500:
        raise ValueError(
            "--gpu-memory-mb must be between 512 and 4500 for this "
            "isolated pilot."
        )


def _configure_device(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return

    active = _active_gpu_processes()
    if active:
        detail = "\n".join(f"  {line}" for line in active)
        raise SystemExit(
            "Refusing to start WCM while another GPU compute process is active.\n"
            f"{detail}\n"
            "Wait for the GUI queue to finish, then run this command again."
        )
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def _initialize_tensorflow(args):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "gpu":
        if not gpus:
            raise RuntimeError("GPU mode requested, but TensorFlow found no GPU.")
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=float(args.gpu_memory_mb)
                    )
                ],
            )
    return tf, gpus


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path, values):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **values)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _config_hash(config):
    canonical = {
        key: value
        for key, value in config.items()
        if key != "runId"
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return (
        completed.stdout.strip()
        if completed.returncode == 0
        else "unavailable"
    )


def _build_config(args):
    from gui.baseline_study import make_config

    if args.snapshot_selection and args.environment != "hidden_noise":
        raise ValueError("Snapshot Selection requires hidden_noise.")
    base_mitigation = "ss" if args.snapshot_selection else "none"
    config = make_config(
        split="nonIID",
        approach=args.approach,
        environment=args.environment,
        mitigation=base_mitigation,
        sigma=float(args.sigma),
        seed=int(args.seed),
        rounds=int(args.rounds),
        phase="wcm_pilot",
        gamma=(
            float(args.cart_gamma)
            if args.approach == "cart"
            else 0.0
        ),
    )
    mode = "ss_wcm" if args.snapshot_selection else "wcm"
    config.update(
        {
            "campaignId": "wcm_pilot_r1",
            "campaignVersion": 0,
            "wcmPilotVersion": 1,
            "conditionId": (
                f"{args.environment}_sigma_{args.sigma:.1f}_{mode}"
            ),
            "experimentName": (
                f"WCM Pilot R1 | nonIID | {args.approach.upper()} | "
                f"{args.environment} sigma={args.sigma:.1f} | "
                f"{mode.upper()} | seed={args.seed}"
            ),
            "noiseMitigation": "wcm",
            "ebmLambda": 0.0,
            "ebmTargetCoefficient": 0.0,
            "ebmCoefficientSchedule": "disabled_for_wcm_pilot",
            "wcmPaper": (
                "Ang_et_al_2019_sections_III-B_and_V_equations_"
                "10_27_31_32_36b"
            ),
            "wcmUncertaintySet": "relative_full_model_l2_ball",
            "wcmBoundarySampling": "isotropic_full_model_boundary",
            "wcmSurrogateSolve": "finite_local_sgd_approximation",
            "wcmInnerOptimizer": "plain_gradient_descent_no_momentum",
            "wcmMicroBatchSize": 128,
            "wcmPenalty": float(args.wcm_penalty),
            "wcmRhoExponent": float(args.wcm_rho_exponent),
            "wcmGammaExponent": float(args.wcm_gamma_exponent),
            "wcmBoundarySamples": int(args.wcm_boundary_samples),
            "wcmRadiusMultiplier": float(args.wcm_radius_multiplier),
            "autoPlotCampaign3": False,
        }
    )
    config.pop("protocolPatch", None)
    config.pop("runId", None)
    config["runId"] = f"wcm-r1-{_config_hash(config)[:20]}"
    return config


def _result_paths(config):
    split = "nonIID" if config.get("nonIID", True) else "IID"
    mode = (
        "ss_wcm"
        if config.get("snapshotSelection", False)
        else "wcm"
    )
    sigma = f"sigma_{float(config['channelNoiseSigma']):.1f}".replace(".", "_")
    directory = (
        RESULT_ROOT
        / split
        / "cifar10"
        / str(config["environment"])
        / str(config["approach"])
        / sigma
        / mode
        / f"seed_{int(config['seed'])}"
        / str(config["runId"])
    )
    return directory / "metrics.npz", directory / "run.json"


def _partition_hash(data_metadata):
    import numpy as np

    digest = hashlib.sha256()
    for indices in data_metadata.get("clientIndices", []):
        digest.update(
            np.asarray(indices, dtype=np.int64).tobytes(order="C")
        )
    return digest.hexdigest()


def main():
    args = _parse_args()
    _validate_args(args)
    _configure_device(args)
    tf, gpus = _initialize_tensorflow(args)

    from basil_core.data.cifar import loadCifar10, makeLoaders
    from basil_core.models import CIFARModel
    from basil_core.wcm_pilot import run_wcm_pilot

    config = _build_config(args)
    metrics_path, run_path = _result_paths(config)
    if run_path.exists() and not args.force:
        try:
            existing = json.loads(run_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                print(f"Already completed: {run_path}")
                return 0
        except (OSError, ValueError):
            pass

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.perf_counter()
    metadata = {
        "schemaVersion": 1,
        "status": "preparing",
        "startedAt": datetime.now().astimezone().isoformat(),
        "executionMode": "isolated_wcm_pilot",
        "codeRevision": _git_revision(),
        "sourcePaper": str(PAPER_PATH.relative_to(PROJECT_ROOT)),
        "sourcePaperSha256": _file_hash(PAPER_PATH),
        "device": args.device,
        "gpuCount": len(gpus),
        "gpuMemoryLimitMb": (
            int(args.gpu_memory_mb) if args.device == "gpu" else 0
        ),
        "configHash": _config_hash(config),
        "config": config,
    }
    if not args.no_save:
        _atomic_json(run_path, metadata)

    train = test = train_loaders = test_loader = data_metadata = result = None
    try:
        train, test = loadCifar10(cacheDir=args.cache_dir)
        train_loaders, test_loader, data_metadata = makeLoaders(
            train,
            test,
            batchSize=512,
            iid=False,
            nClients=int(config["nNodes"]),
            dirichletAlpha=float(config["dirichletAlpha"]),
            seed=int(config["seed"]),
            returnMetadata=True,
            cacheDir=args.cache_dir,
        )
        metadata["partitionHash"] = _partition_hash(data_metadata)
        metadata["status"] = "running"
        if not args.no_save:
            _atomic_json(run_path, metadata)

        print(json.dumps(config, indent=2, sort_keys=True), flush=True)
        result = run_wcm_pilot(
            config=config,
            model_class=CIFARModel,
            train_loaders=train_loaders,
            test_loader=test_loader,
            data_metadata=data_metadata,
            stop_callback=lambda: stop_requested,
        )
        if result.get("stopped") or stop_requested:
            metadata.update(
                {
                    "status": "stopped",
                    "completedRounds": len(result.get("avg_history", [])),
                    "stoppedAt": datetime.now().astimezone().isoformat(),
                    "wallRuntimeSeconds": time.perf_counter() - started,
                }
            )
            if not args.no_save:
                _atomic_json(run_path, metadata)
            return 2

        initialization_hash = result.pop("initialization_hash")
        result.pop("stopped", None)
        if not args.no_save:
            _atomic_npz(metrics_path, result)
        metadata.update(
            {
                "status": "completed",
                "completedAt": datetime.now().astimezone().isoformat(),
                "initializationHash": initialization_hash,
                "finalAverageAccuracy": float(result["final_avg"]),
                "finalWorstAccuracy": float(result["final_worst"]),
                "runtimeSeconds": float(result["runtime_seconds"]),
                "wallRuntimeSeconds": time.perf_counter() - started,
                "peakGpuBytes": int(result["peak_gpu_bytes"]),
            }
        )
        if not args.no_save:
            _atomic_json(run_path, metadata)
        print(
            f"WCM pilot complete: final average={float(result['final_avg']):.4f}, "
            f"worst={float(result['final_worst']):.4f}"
        )
        print(f"Results: {run_path.parent}")
        return 0
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "failedAt": datetime.now().astimezone().isoformat(),
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wallRuntimeSeconds": time.perf_counter() - started,
            }
        )
        if not args.no_save:
            _atomic_json(run_path, metadata)
        raise
    finally:
        result = data_metadata = test_loader = train_loaders = None
        test = train = None
        try:
            tf.keras.backend.clear_session(free_memory=True)
        except TypeError:
            tf.keras.backend.clear_session()
        for _ in range(3):
            gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
