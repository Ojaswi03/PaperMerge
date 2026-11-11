"""Combine and verify the saved official test evaluations."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVALUATIONS = Path(__file__).resolve().parents[1] / "docs/results/official-test-20261110"
SOURCES = [EVALUATIONS / name for name in (
    "full-official-test-20261110", "basil-full-test-20261110"
)]
OUTPUT = Path(__file__).resolve().parents[1] / ".papermerge/evaluations/regenerated-four-method-test"
METHODS = ["BASIL", "BASIL + ACDS", "FedSharp", "FedSharp+"]
DATASETS = {"mnist": "MNIST", "cifar10": "CIFAR-10"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_results():
    rows = {}
    test_data = None
    predictions_checked = 0
    for source in SOURCES:
        manifest = json.loads((source / "SHA256.json").read_text())
        for name, expected in manifest.items():
            assert sha(source / name) == expected, name
        receipt = json.loads((source / "results.json").read_text())
        assert receipt["status"] == "COMPLETED_OFFICIAL_TEST_EVALUATION"
        assert not receipt["trainingPerformed"] and not receipt["checkpointsModified"]
        assert not receipt["testUsedForTrainingOrSelection"]
        if test_data is None:
            test_data = receipt["testData"]
        assert receipt["testData"] == test_data
        for result in receipt["results"]:
            key = result["dataset"], result["method"]
            assert key not in rows
            assert (result["seed"], result["rounds"], result["partition"],
                    result["attackType"], result["sigma"]) == (2025, 500, "dirichlet", "hidden", .01)
            assert result["evaluationRuntime"]["device"] == "CPU"
            with np.load(source / result["runId"] / "predictions.npz", allow_pickle=False) as archive:
                labels = archive["labels"]
                predicted = archive["predictedClass"]
                loss = archive["crossEntropyLoss"]
                assert labels.shape == (10000,)
                assert hashlib.sha256(labels.tobytes()).hexdigest() == test_data[key[0]]["labelsSha256"]
                assert predicted.shape == loss.shape == (100, 10000)
                assert np.all(predicted < 10) and np.isfinite(loss).all() and np.all(loss >= 0)
                correct = predicted == labels
                clients = result["perClient"]
                assert [row["clientId"] for row in clients] == list(range(100))
                np.testing.assert_array_equal(correct.sum(axis=1), [r["testCorrect"] for r in clients])
                np.testing.assert_allclose(loss.mean(axis=1), [r["meanCrossEntropyLoss"] for r in clients],
                                           rtol=0, atol=1e-12)
                honest = [r["clientId"] for r in clients if not r["attackerIdentityDuringTraining"]]
                assert len(honest) == 67
                for summary, ids in [("allClients", list(range(100))), ("honestClients", honest)]:
                    np.testing.assert_allclose(
                        [correct[ids].mean(), loss[ids].mean()],
                        [result[summary]["meanAccuracy"], result[summary]["meanCrossEntropyLoss"]],
                        rtol=0, atol=1e-12
                    )
                predictions_checked += predicted.size
            rows[key] = result
    assert set(rows) == {(dataset, method) for dataset in DATASETS for method in METHODS}
    return rows, predictions_checked


def render(rows):
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    colors = ["#697b86", "#39856e", "#356CB0", "#8E477D"]
    for index, method in enumerate(METHODS):
        results = [rows[dataset, method]["allClients"] for dataset in DATASETS]
        position = np.arange(2) + (index - 1.5) * .20
        for axis, metric, scale in zip(axes, ["meanAccuracy", "meanCrossEntropyLoss"], [100, 1]):
            values = [result[metric] * scale for result in results]
            bars = axis.bar(position, values, width=.18, color=colors[index], label=method)
            labels = [f"{value:.2f}%" if scale == 100 else f"{value:.3f}" for value in values]
            axis.bar_label(bars, labels=labels, fontsize=10, padding=4)
    for axis in axes:
        axis.set_xticks([0, 1], list(DATASETS.values()))
        axis.grid(axis="y", color="#e0e5e6", linewidth=.6)
        axis.set_axisbelow(True)
    axes[0].set(title="Official test accuracy", ylabel="Mean client accuracy (%)", ylim=(0, 100))
    max_loss = max(row["allClients"]["meanCrossEntropyLoss"] for row in rows.values())
    axes[1].set(title="Official test loss", ylabel="Mean cross-entropy", ylim=(0, max_loss * 1.18))
    fig.suptitle("Final round-500 test results, non-IID, Hidden attack + noise, seed 2025", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=.07, right=.98, top=.82, bottom=.20, wspace=.27)
    for extension in ["png", "svg"]:
        fig.savefig(OUTPUT / f"test_accuracy_and_loss.{extension}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    rows, count = load_results()
    OUTPUT.mkdir(parents=True)
    lines = [
        "# Official test results for four methods", "",
        "Final round-500 models under non-IID partitioning, Hidden attack + noise, and seed 2025.",
        "Each of the 100 client models per method evaluated all 10,000 official test images per dataset.",
        "BASIL + ACDS used its configured fitting-data sharing during training.", "",
        "| Dataset | Method | Test accuracy | Mean test loss | Honest-client accuracy | Round-500 validation |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, label in DATASETS.items():
        for method in METHODS:
            row = rows[dataset, method]
            a, h = row["allClients"], row["honestClients"]
            lines.append(f"| {label} | {method} | {a['meanAccuracy']:.3%} | "
                         f"{a['meanCrossEntropyLoss']:.6f} | {h['meanAccuracy']:.3%} | "
                         f"{a['finalRoundValidationMeanAccuracy']:.3%} |")
    lines += ["", "Test accuracy averages each client model's accuracy with equal weight.",
              "The honest-client column averages the 67 honest clients.",
              "The comparison covers one training seed and one attack setting.",
              "Image 0 appeared in the earlier demonstration. Both evaluations use the frozen models.",
              f"The saved prediction arrays support independent verification of {count:,} client-image predictions.",
              "", "## Source evaluations", ""]
    lines += [f"- [{source.name}](../{source.name}/report.md)" for source in SOURCES]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")
    with (OUTPUT / "per_client.csv").open("w", newline="") as stream:
        writer = None
        for source in SOURCES:
            with (source / "per_client.csv").open(newline="") as original:
                reader = csv.DictReader(original)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    render(rows)
    manifest = {p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()}
    (OUTPUT / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
