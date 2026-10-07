#!/usr/bin/env python3
"""Bounded CPU audit runner; never accepts research-valid/production configs."""
from __future__ import annotations

import argparse
import json
import hashlib
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core.protocol_compatibility import normalize_config
from basil_core.artifact_paths import writable_output


def run(config_path, directory, deterministic=False, candidate_evaluation=False, threads=1):
    import numpy as np
    import tensorflow as tf
    from basil_core.data.cifar import _datasetArrays, loadCifar10
    from basil_core.research_audit import ProtocolAudit
    from basil_core.research_protocol import run_research_protocol, validate_config
    config = normalize_config(json.loads(Path(config_path).read_text()))
    validate_config(config)
    if config.get("researchValid", True) or config["nRounds"] > 10:
        raise ValueError("Audit allows only bounded, non-research smoke/preflight configs")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1 if threads == 1 else 2)
    if deterministic:
        tf.keras.utils.set_random_seed(config["seed"])
        tf.config.experimental.enable_op_determinism()
    directory = writable_output(Path(directory))
    directory.mkdir(parents=True, exist_ok=False)
    audit = ProtocolAudit(directory / "trace", candidate_evaluation=candidate_evaluation)
    started = time.perf_counter()
    summary = {"config": config, "tensorflow": tf.__version__, "numpy": np.__version__,
        "deterministic": deterministic, "devices": [d.device_type for d in tf.config.list_physical_devices()],
        "environment": {k: os.environ.get(k) for k in ("TF_ENABLE_ONEDNN_OPTS", "OMP_NUM_THREADS", "TF_DETERMINISTIC_OPS")},
        "intraOpThreads": threads, "interOpThreads": 1 if threads == 1 else 2}
    summary["sourceSha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("basil_core/research_protocol.py", "basil_core/research_audit.py", "basil_core/models.py")}
    try:
        train, test = loadCifar10(cacheDir=ROOT / "experiments/cache/cifar10")
        x, y = _datasetArrays(train)
        tx, ty = _datasetArrays(test)
        if config.get("diagnosticTestSamplesPerClass"):
            count = config["diagnosticTestSamplesPerClass"]
            chosen = np.concatenate([np.flatnonzero(ty == c)[:count] for c in range(10)])
            tx, ty = tx[chosen], ty[chosen]
        result = run_research_protocol(config, x, y, tx, ty, audit=audit)
        np.savez_compressed(directory / "metrics.npz", **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        summary.update(status="completed", finalAverage=float(result["averageAccuracy"][-1]),
            finalWorst=float(result["worstNodeAccuracy"][-1]), finalBest=float(result["bestNodeAccuracy"][-1]))
    except Exception as error:
        summary.update(status="failed", error=str(error))
        (directory / "execution.log").write_text(traceback.format_exc())
    finally:
        audit.close()
        summary["runtimeSeconds"] = time.perf_counter() - started
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(directory), "status": summary["status"], "seconds": summary["runtimeSeconds"]}), flush=True)
    return summary["status"] == "completed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("direct", "spawn", "worker", "pool"), default="direct")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--candidate-evaluation", action="store_true")
    parser.add_argument("--gpu-memory-mb", type=int, default=0)
    parser.add_argument("--threads", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if args.deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    if args.mode == "spawn":
        process = multiprocessing.get_context("spawn").Process(target=spawn_run,
            args=(args.config, args.output, args.deterministic, args.candidate_evaluation, args.threads))
        process.start()
        process.join()
        raise SystemExit(process.exitcode)
    if args.mode == "pool":
        from gui.worker_pool import CampaignWorkerPool
        directory = writable_output(Path(args.output))
        directory.mkdir(parents=True, exist_ok=False)
        arguments = ("--result-root", str(directory / "worker_results"),
                     "--audit-directory", str(directory / "trace"))
        if args.candidate_evaluation:
            arguments += ("--audit-candidates",)
        pool = CampaignWorkerPool(lanes=1, gpu_memory_limit_mb=0,
            work_dir=directory / "worker_state", worker_script=ROOT / "scripts/run_research_protocol.py",
            python_executable=sys.executable, worker_arguments=arguments)
        pool.launch(json.loads(Path(args.config).read_text()))
        while pool.has_active:
            finished = pool.poll_finished()
            if finished:
                raise SystemExit(finished[0].return_code)
            time.sleep(.1)
        return
    if args.mode == "worker":
        command = [sys.executable, str(ROOT / "scripts/run_research_protocol.py"),
            "--config", args.config, "--result-root", str(Path(args.output) / "worker_results"),
            "--audit-directory", str(Path(args.output) / "trace")]
        if args.deterministic:
            command.append("--deterministic")
        if args.candidate_evaluation:
            command.append("--audit-candidates")
        writable_output(Path(args.output)).mkdir(parents=True, exist_ok=False)
        with (Path(args.output) / "worker_stdout.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        raise SystemExit(completed.returncode)
    raise SystemExit(0 if run(args.config, args.output, args.deterministic, args.candidate_evaluation, args.threads) else 1)


def spawn_run(config_path, directory, deterministic, candidate_evaluation, threads):
    raise SystemExit(0 if run(config_path, directory, deterministic, candidate_evaluation, threads) else 1)


if __name__ == "__main__":
    main()
