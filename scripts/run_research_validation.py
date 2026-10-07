#!/usr/bin/env python3
"""Run only targeted scientific-validation configs, never production."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOKE_CONDITIONS = (
    "noise_rel_0.4_ebm", "noise_rel_0.6_ebm", "noise_rel_0.6_legacy", "noise_rel_0.6_none",
    "noise_rel_0.2_ebm", "noise_rel_0.4_none", "noise_rel_0.4_legacy", "noise_abs_0.02_ebm",
    "clean_one_class_per_node", "hidden_ss", "hidden_noise_rel_0.4_ss", "hidden_noise_rel_0.4_ss_ebm",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "preflight", "repro"), required=True)
    parser.add_argument("--output", default="experiments/research_validation_results/corrected")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    from basil_core.artifact_paths import writable_output
    output = writable_output(Path(args.output) / args.suite)
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_ENABLE_ONEDNN_OPTS": "0", "TF_DETERMINISTIC_OPS": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0", "MPLCONFIGDIR": "/tmp/papermerge_mpl"}
    if args.suite == "preflight":
        items = [(f"preflight_clean_{p}", "direct") for p in ("iid", "dirichlet", "one_class_per_node")]
        family = "preflight"
    elif args.suite == "repro":
        items = [("smoke_noise_rel_0.4_ebm", mode) for mode in ("direct", "worker", "spawn", "pool")]
        family = "smoke"
    else:
        items = [("smoke_" + name, "direct") for name in SMOKE_CONDITIONS]
        family = "smoke"
    def execute(item):
        name, mode = item
        label = f"{name}_{mode}"
        command = [sys.executable, str(ROOT / "scripts/audit_research_protocol.py"),
            "--config", str(ROOT / "gui/configs/sequential_basil" / family / f"{name}.json"),
            "--output", str(output / label), "--mode", mode, "--deterministic"]
        if "_ss" in name:
            command.append("--candidate-evaluation")
        with (output / f"{label}.log").open("w") as log:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"{label}: exit {result.returncode}", flush=True)
        return result.returncode
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(execute, items))
    print(f"{args.suite}: {sum(code == 0 for code in codes)} zero exits / {len(codes)} targeted invocations", flush=True)
    # A smoke failing numerically is evidence, not a harness success.
    raise SystemExit(any(codes))


if __name__ == "__main__":
    main()
