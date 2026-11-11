"""Add dataset-level test figures to the saved four-method report."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from matplotlib import pyplot as plt

from reporting.official_test_results import DATASETS, METHODS, OUTPUT, SOURCES, load_results, sha

COLORS = ["#697b86", "#39856e", "#356CB0", "#8E477D"]
CLASSES = {
    "mnist": [str(digit) for digit in range(10)],
    "cifar10": ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                "Dog", "Frog", "Horse", "Ship", "Truck"],
}
SETTING = "Round 500, non-IID, Hidden attack + noise, seed 2025"


def verify_recall(rows):
    for source in SOURCES:
        receipt = json.loads((source / "results.json").read_text())
        for result in receipt["results"]:
            dataset, method = result["dataset"], result["method"]
            with np.load(source / result["runId"] / "predictions.npz", allow_pickle=False) as data:
                labels = data["labels"]
                predicted = data["predictedClass"]
                recall = [float((predicted[:, labels == index] == index).mean())
                          for index in range(10)]
            np.testing.assert_allclose(
                recall, rows[dataset, method]["allClients"]["meanClassRecall"],
                rtol=0, atol=1e-12)


def save(fig, stem):
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"{stem}.{extension}", dpi=600,
                    facecolor="white", bbox_inches="tight", pad_inches=.13)
    plt.close(fig)


def render_accuracy_loss(dataset, label, summaries):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    panels = (
        ("meanAccuracy", 100, "Test accuracy", "Mean client accuracy (%)", 100),
        ("meanCrossEntropyLoss", 1, "Test loss", "Mean cross-entropy",
         max(row["meanCrossEntropyLoss"] for row in summaries) * 1.20),
    )
    for axis, (key, scale, title, ylabel, ymax) in zip(axes, panels):
        values = [row[key] * scale for row in summaries]
        bars = axis.bar(np.arange(4), values, width=.64, color=COLORS)
        labels = [f"{value:.2f}%" if scale == 100 else f"{value:.4f}" for value in values]
        axis.bar_label(bars, labels=labels, padding=5, fontsize=11)
        axis.set(title=title, ylabel=ylabel, ylim=(0, ymax))
        axis.set_xticks(np.arange(4), METHODS, fontsize=10)
        axis.grid(axis="y", color="#e0e5e6", linewidth=.6)
        axis.set_axisbelow(True)
    fig.suptitle(f"{label} official test results", fontsize=17, fontweight="bold", y=.98)
    fig.text(.5, .89, SETTING, ha="center", fontsize=11)
    fig.text(.5, .034, "10,000 test images per client model, averaged over 100 clients per method",
             ha="center", fontsize=10, color="#42535d")
    fig.subplots_adjust(left=.08, right=.98, top=.76, bottom=.15, wspace=.30)
    save(fig, f"{dataset}_test_accuracy_and_loss")


def render_recall(dataset, label, summaries):
    recall = 100 * np.array([row["meanClassRecall"] for row in summaries])
    fig, axis = plt.subplots(figsize=(12.2, 4.7))
    cells = axis.pcolormesh(
        np.arange(11) - .5, np.arange(5) - .5, recall, cmap="Blues",
        vmin=0, vmax=100, shading="flat", edgecolors="white", linewidth=.8)
    axis.set_xlim(-.5, 9.5)
    axis.set_ylim(3.5, -.5)
    for method, row in enumerate(recall):
        for class_id, value in enumerate(row):
            axis.text(class_id, method, f"{value:.1f}", ha="center", va="center",
                      color="white" if value > 55 else "#172b32", fontsize=11)
    axis.set_yticks(np.arange(4), METHODS)
    axis.set_xticks(np.arange(10), CLASSES[dataset], fontsize=10)
    if dataset == "cifar10":
        plt.setp(axis.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    axis.tick_params(length=0)
    axis.spines[["bottom", "left"]].set_visible(False)
    colorbar = fig.colorbar(cells, ax=axis, pad=.02, shrink=.88, aspect=22)
    colorbar.set_label("Recall (%)")
    colorbar.outline.set_visible(False)
    colorbar.solids.set_rasterized(False)
    fig.suptitle(f"{label} official test recall by class", fontsize=17, fontweight="bold", y=.98)
    fig.text(.5, .88, SETTING, ha="center", fontsize=11)
    fig.text(.5, .026, "Recall is the percentage of images in each class classified correctly, averaged over 100 clients.",
             ha="center", fontsize=10, color="#42535d")
    fig.subplots_adjust(left=.14, right=.98, top=.79, bottom=.24 if dataset == "cifar10" else .16)
    save(fig, f"{dataset}_test_class_recall")


def update_report():
    report = OUTPUT / "report.md"
    text = report.read_text()
    heading = "## Test figures"
    section = [heading, ""]
    for dataset, label in DATASETS.items():
        for suffix, title in (("accuracy_and_loss", "Accuracy and loss"), ("class_recall", "Recall by class")):
            stem = f"{dataset}_test_{suffix}"
            section.extend([
                f"### {label} {title.lower()}", "",
                f"![{label} {title.lower()}]({stem}.png)", "",
                f"[PNG]({stem}.png) · [SVG]({stem}.svg)", "",
            ])
    report.write_text(text.split(heading, 1)[0].rstrip() + "\n\n" + "\n".join(section) + "\n")
    files = {path.name: sha(path) for path in sorted(OUTPUT.iterdir())
             if path.is_file() and path.name != "SHA256.json"}
    (OUTPUT / "SHA256.json").write_text(json.dumps(files, indent=2) + "\n")


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.destination.resolve()
    OUTPUT.mkdir(parents=True)
    rows, count = load_results()
    assert count == 8_000_000
    verify_recall(rows)
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    for dataset, label in DATASETS.items():
        summaries = [rows[dataset, method]["allClients"] for method in METHODS]
        render_accuracy_loss(dataset, label, summaries)
        render_recall(dataset, label, summaries)
    (OUTPUT / "report.md").write_text((SOURCES[0].parent / "four-method-official-test-20261110/report.md").read_text())
    update_report()


if __name__ == "__main__":
    main()
