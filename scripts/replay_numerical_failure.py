#!/usr/bin/env python3
"""Replay a saved failing batch without performing a training experiment."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--preceding", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    from basil_core.research_audit import ProtocolAudit
    from basil_core.research_protocol import SequentialWorker, build_model
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    trace = Path(args.trace)
    failure = json.loads((trace / "failure.json").read_text())
    if args.preceding:
        failure = json.loads((trace / "batch_trace.jsonl").read_text().splitlines()[-1])
    archive = "preceding_optimizer_batch.npz" if args.preceding else "failure_batch.npz"
    with np.load(trace / archive, allow_pickle=False) as source:
        params = [source[f"weight_{i}"].copy() for i in range(10)]
        pending = {"params": params, "x": source["x"].copy(), "y": source["y"].copy(), "key": source["key"].copy(),
            "context": {key: failure[key] for key in ("round_id", "node_id", "epoch", "batch", "mode", "semantics", "sigma", "legacy_lambda", "learning_rate")}}
    audit = ProtocolAudit(args.output)
    worker = SequentialWorker(build_model("basil_paper_cnn", args.seed))
    worker.load(params)
    audit.pending, audit.current = pending, failure
    try:
        if args.preceding:
            audit.replay_values(worker, pending, name="preceding_optimizer_tensor_values.json")
        else:
            audit.failure(worker, RuntimeError("Replay of saved failure; current source arithmetic"))
    finally:
        audit.close()


if __name__ == "__main__":
    main()
