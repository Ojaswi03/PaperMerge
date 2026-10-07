#!/usr/bin/env python3
"""Build read-only validation summaries and epoch-level evidence plots."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core.artifact_paths import writable_output


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def compare_traces(left, right):
    a, b = read_rows(left / "trace/batch_trace.jsonl"), read_rows(right / "trace/batch_trace.jsonl")
    result = {"left": str(left), "right": str(right), "batchCounts": [len(a), len(b)],
              "identicalCompleteTrace": len(a) == len(b) and bool(a), "earliestDivergence": None}
    keys = ("parameterHashBefore", "imagesHash", "labelsHash", "indicesHash",
            "augmentationKey", "augmentedImagesHash", "gradientHash", "parameterHashAfter")
    for index, (x, y) in enumerate(zip(a, b)):
        different = [key for key in keys if x.get(key) != y.get(key)]
        if different:
            result["identicalCompleteTrace"] = False
            result["earliestDivergence"] = {"index": index,
                **{key: x[key] for key in ("round_id", "node_id", "epoch", "batch")},
                "different": different, "left": x, "right": y}
            break
    initial = [json.loads((p / "trace/initial.json").read_text()) for p in (left, right)]
    result["initialIdentical"] = initial[0] == initial[1]
    links = [read_rows(p / "trace/activation_trace.jsonl") for p in (left, right)]
    result["transmissionHashesIdentical"] = len(links[0]) == len(links[1]) and all(
        [v["transmittedHash"] for v in x["outgoingLinks"]] == [v["transmittedHash"] for v in y["outgoingLinks"]]
        for x, y in zip(*links))
    return result


def candidate_analysis(path):
    rows = json.loads(path.read_text())
    selected = [r for r in rows if r["selected"]]
    if not rows:
        return None
    groups = {}
    for record in rows:
        groups.setdefault((record["round"], record["node"]), []).append(record)
    correct = 0
    informative = []
    for (_round, node), group in groups.items():
        minimum = min(r["receiverLocalLoss"] for r in group)
        tied = [r for r in group if r["receiverLocalLoss"] <= minimum + 1e-8]
        expected = min(tied, key=lambda r: (node - r["senderId"]) % 10)
        actual = next(r for r in group if r["selected"])
        correct += actual["senderId"] == expected["senderId"]
        loss = np.array([r["receiverLocalLoss"] for r in group])
        accuracy = np.array([r["globalAccuracyEvaluationOnly"] for r in group])
        if np.std(loss) > 0 and np.std(accuracy) > 1e-10:
            informative.append(float(np.corrcoef(loss, accuracy)[0, 1]))
    return {"path": str(path), "activations": len(groups), "candidates": len(rows),
        "localMinimumAndTieRuleCorrect": correct,
        "selectedConfiguredByzantine": sum(r["configuredByzantine"] for r in selected),
        "selectedHonestIdentity": sum(not r["configuredByzantine"] for r in selected),
        "selectedActivelyAttacked": sum(r["attackActiveAtSource"] for r in selected),
        "selectedNotActivelyAttacked": sum(not r["attackActiveAtSource"] for r in selected),
        "candidateMeanGlobalAccuracy": float(np.mean([r["globalAccuracyEvaluationOnly"] for r in rows])),
        "selectedMeanGlobalAccuracy": float(np.mean([r["globalAccuracyEvaluationOnly"] for r in selected])),
        "informativeWithinActivationCorrelations": len(informative),
        "meanWithinActivationLossAccuracyCorrelation": float(np.mean(informative)) if informative else None}


def learning_analysis(directory, plot_root=None):
    rows = json.loads((directory / "trace/confusions.json").read_text())
    rounds = max(r["round"] for r in rows) + 1
    matrices = np.zeros((rounds, 10, 6, 10, 10), np.int64)
    for row in rows:
        matrices[row["round"], row["node"], row["stage"]] = row["confusion"]
    totals = matrices.sum(axis=-1)
    per_class = np.divide(np.diagonal(matrices, axis1=-2, axis2=-1), totals,
                          out=np.zeros_like(totals, dtype=float), where=totals > 0)
    prediction_counts = matrices.sum(axis=-2)
    constant = prediction_counts.max(axis=-1) == prediction_counts.sum(axis=-1)
    losses = per_class[:, :, 0, :] - per_class[:, :, 5, :]
    forgetting = np.count_nonzero(losses > .5)
    result = {"path": str(directory), "matrixShape": list(matrices.shape), "rounds": rounds,
        "testImagesPerStage": np.unique(matrices.sum(axis=(-2, -1))).tolist(),
        "afterEpoch5ConstantClassStates": int(constant[:, :, 5].sum()),
        "totalNodeActivations": rounds * 10,
        "beforeToAfterClassDropsAbove50PercentagePoints": int(forgetting),
        "finalAfterTrainingPerClass": per_class[-1, :, 5].tolist()}
    if plot_root is not None and not (plot_root / directory.name).exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        output = writable_output(plot_root / directory.name)
        output.mkdir(parents=True, exist_ok=False)
        for node in range(10):
            fig, axes = plt.subplots(rounds, 6, figsize=(18, max(4, rounds * 3)), squeeze=False)
            for r in range(rounds):
                for stage in range(6):
                    rates = np.divide(matrices[r, node, stage], totals[r, node, stage, :, None],
                        out=np.zeros((10, 10), float), where=totals[r, node, stage, :, None] > 0)
                    axes[r, stage].imshow(rates, vmin=0, vmax=1, cmap="viridis")
                    axes[r, stage].set_title(f"Round {r}: {'before' if stage == 0 else f'epoch {stage}'}")
                    axes[r, stage].set_xticks(range(10)); axes[r, stage].set_yticks(range(10))
                    axes[r, stage].set_xlabel("Predicted class")
                    if stage == 0: axes[r, stage].set_ylabel("True class")
            fig.suptitle(f"{directory.name}: Node {node}; row-normalized confusion matrices (evaluation only)")
            fig.tight_layout(); fig.savefig(output / f"node_{node}_confusions.png", dpi=120); plt.close(fig)
            fig, axes = plt.subplots(rounds, 1, figsize=(11, rounds * 2.5), squeeze=False)
            for r in range(rounds):
                for c in range(10): axes[r, 0].plot(range(6), per_class[r, node, :, c], marker=".", label=str(c))
                axes[r, 0].set_title(f"Round {r}"); axes[r, 0].set_ylim(-.02, 1.02)
                axes[r, 0].set_ylabel("Class accuracy"); axes[r, 0].set_xticks(range(6))
                axes[r, 0].set_xticklabels(["before", "epoch 1", "epoch 2", "epoch 3", "epoch 4", "epoch 5"])
            axes[0, 0].legend(title="CIFAR class ID", ncol=10, loc="upper right")
            fig.suptitle(f"{directory.name}: Node {node}; five-epoch per-class retention")
            fig.tight_layout(); fig.savefig(output / f"node_{node}_per_class.png", dpi=140); plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/research_validation_results")
    parser.add_argument("--output", default="docs/validation_evidence.json")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--plot-root", default="plots/research_validation")
    parser.add_argument("--artifact-manifest", help="Verified pre-edit sha256sum manifest to retain with the evidence")
    args = parser.parse_args()
    root = Path(args.root)
    results = {"runs": [], "reproducibility": [], "selection": [], "learning": []}
    if args.artifact_manifest:
        manifest = Path(args.artifact_manifest)
        subprocess.run(["sha256sum", "-c", str(manifest.resolve())], cwd=ROOT,
            stdout=subprocess.DEVNULL, check=True)
        contents = manifest.read_bytes()
        files = []
        for line in contents.decode().splitlines():
            digest, filename = line.split("  ", 1)
            files.append({"path": filename, "sha256": digest})
        artifact_report = {"verification": "all pre-edit hashes checked and unchanged",
            "fileCount": len(files), "originalManifestSha256": hashlib.sha256(contents).hexdigest(),
            "files": files}
        (ROOT / "docs/validation_artifacts.json").write_text(json.dumps(artifact_report, indent=2) + "\n")
        results["protectedArtifacts"] = {key: value for key, value in artifact_report.items() if key != "files"}
    for path in sorted(root.rglob("summary.json")):
        results["runs"].append({"path": str(path.parent), **json.loads(path.read_text())})
    for path in sorted(root.rglob("worker_results/*/*/run.json")):
        metadata = json.loads(path.read_text())
        results["runs"].append({"path": str(path), **metadata})
    for family in ("raw", "deterministic", "reference_backend"):
        left, right = root / family / "direct_rel_04", root / family / "worker_rel_04"
        if (right / "trace/initial.json").exists():
            results["reproducibility"].append(compare_traces(left, right))
    for family in ("corrected", "serial_cpu", "final_cpu", "keyed_runtime"):
        folder = root / family / "repro"
        left = folder / "smoke_noise_rel_0.4_ebm_direct"
        for mode in ("worker", "spawn", "pool"):
            right = folder / f"smoke_noise_rel_0.4_ebm_{mode}"
            if (right / "trace/initial.json").exists():
                results["reproducibility"].append(compare_traces(left, right))
    for path in sorted(root.rglob("candidate_evaluation.json")):
        result = candidate_analysis(path)
        if result: results["selection"].append(result)
    for path in sorted((root / "corrected/preflight").glob("*/summary.json")):
        if json.loads(path.read_text())["status"] == "completed":
            results["learning"].append(learning_analysis(path.parent,
                Path(args.plot_root) if args.plots else None))
    output=writable_output(Path(args.output));output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Evidence: {len(results['runs'])} runs, {len(results['reproducibility'])} comparisons, "
          f"{len(results['selection'])} selection audits, {len(results['learning'])} preflights")


if __name__ == "__main__": main()
