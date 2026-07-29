"""Publication and diagnostic plots for campaign-three experiments.

Only completed runs under ``experiments/results3/r2`` are read. Every generated
figure is written under ``plots3/r2`` as PNG, PDF, and EPS. Confirmation and
calibration runs are never pooled in the same aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable, Iterable

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

try:
    from gui.campaign3 import (
        CAMPAIGN_ID,
        CART_LOW_NOISE_CONFIRMATION_PATCH,
        CART_LOW_NOISE_REFINEMENT_PATCH,
        LOW_NOISE_EBM_PATCH,
        PLOT_ROOT,
        RESULT_ROOT,
        build_calibration,
        config_hash,
        write_json_atomic,
    )
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    from gui.campaign3 import (
        CAMPAIGN_ID,
        CART_LOW_NOISE_CONFIRMATION_PATCH,
        CART_LOW_NOISE_REFINEMENT_PATCH,
        LOW_NOISE_EBM_PATCH,
        PLOT_ROOT,
        RESULT_ROOT,
        build_calibration,
        config_hash,
        write_json_atomic,
    )


PLOT_SCHEMA_VERSION = 7
FORMATS = ("png", "pdf", "eps")
NOISE_LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6)
MITIGATION_ORDER = ("none", "ss", "ebm", "ss_ebm")
MITIGATION_LABELS = {
    "none": "No mitigation",
    "ss": "SS",
    "ebm": "EBM",
    "ss_ebm": "SS+EBM",
}
MITIGATION_COLORS = {
    "none": "#666666",
    "ss": "#0072B2",
    "ebm": "#E69F00",
    "ss_ebm": "#009E73",
}
NOISE_COLORS = {
    0.2: "#0072B2",
    0.3: "#009E73",
    0.4: "#D55E00",
    0.5: "#CC79A7",
    0.6: "#E69F00",
}
MITIGATION_LINESTYLES = {
    "none": ":",
    "ss": "--",
    "ebm": "-.",
    "ss_ebm": "-",
}
APPROACH_LABELS = {"merged": "Merged", "cart": "CART"}
APPROACH_MARKERS = {"merged": "o", "cart": "s"}
APPROACH_LINESTYLES = {"merged": "-", "cart": "--"}
APPROACH_HATCHES = {"merged": "", "cart": "///"}
SEED_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
SEED_MARKERS = ("o", "s", "^", "D", "P", "X", "v")
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class RunRecord:
    config: dict
    metrics: dict[str, np.ndarray]
    run_path: Path
    metrics_path: Path

    @property
    def split(self) -> str:
        return "nonIID" if self.config.get("nonIID", True) else "IID"

    @property
    def approach(self) -> str:
        return str(self.config.get("approach", "merged"))

    @property
    def environment(self) -> str:
        return str(self.config.get("environment", "unknown"))

    @property
    def mitigation(self) -> str:
        has_ss = bool(
            self.config.get(
                "snapshotSelection",
                self.config.get("useBasil", False),
            )
        )
        has_ebm = (
            self.config.get("noiseMitigation") == "ebm"
            and self.config.get("useChannelNoise", False)
        )
        if has_ss and has_ebm:
            return "ss_ebm"
        if has_ss:
            return "ss"
        if has_ebm:
            return "ebm"
        return "none"

    @property
    def sigma(self) -> float:
        if not self.config.get("useChannelNoise", False):
            return 0.0
        return float(self.config.get("channelNoiseSigma", 0.0))

    @property
    def seed(self) -> int:
        return int(self.config.get("seed", 0))

    @property
    def phase(self) -> str:
        return str(self.config.get("phase", "confirmation"))

    def scalar(self, name: str, default: float = math.nan) -> float:
        value = self.metrics.get(name)
        if value is None:
            return float(default)
        return float(np.asarray(value).reshape(-1)[0])


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


_RECORD_CACHE_LOCK = threading.RLock()
_RECORD_CACHE: dict[
    Path,
    tuple[tuple[int, int, int, int], RunRecord | None],
] = {}
_PLOT_GENERATION_LOCK = threading.RLock()


def clear_record_cache() -> None:
    """Clear the in-process result cache used by incremental GUI plotting."""
    with _RECORD_CACHE_LOCK:
        _RECORD_CACHE.clear()


def _record_signature(run_path: Path, metrics_path: Path) -> tuple[int, int, int, int]:
    run_stat = run_path.stat()
    metrics_stat = metrics_path.stat()
    return (
        int(run_stat.st_size),
        int(run_stat.st_mtime_ns),
        int(metrics_stat.st_size),
        int(metrics_stat.st_mtime_ns),
    )


def _validated_record(
    run_path: Path,
    current_calibration_run_ids: set[str],
) -> RunRecord | None:
    metrics_path = run_path.with_name("metrics.npz")
    if not metrics_path.exists():
        return None
    signature = _record_signature(run_path, metrics_path)
    cache_key = run_path.resolve()
    with _RECORD_CACHE_LOCK:
        cached = _RECORD_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

    record: RunRecord | None = None
    try:
        with run_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        config = metadata["config"]
        if metadata.get("status") != "completed":
            return None
        if config.get("campaignId") != CAMPAIGN_ID:
            return None
        if metadata.get("runId") != config.get("runId"):
            return None
        if metadata.get("configHash") != config_hash(config):
            return None
        is_low_noise_ebm = (
            config.get("noiseMitigation") == "ebm"
            and abs(float(config.get("channelNoiseSigma", 0.0)) - 0.2) < 1e-12
        )
        if (
            is_low_noise_ebm
            and config.get("protocolPatch")
            not in {
                LOW_NOISE_EBM_PATCH,
                CART_LOW_NOISE_REFINEMENT_PATCH,
                CART_LOW_NOISE_CONFIRMATION_PATCH,
            }
        ):
            return None
        if (
            config.get("phase") == "calibration"
            and config.get("runId") not in current_calibration_run_ids
        ):
            return None
        record = RunRecord(
            config=config,
            metrics=_load_npz(metrics_path),
            run_path=run_path,
            metrics_path=metrics_path,
        )
        return record
    finally:
        with _RECORD_CACHE_LOCK:
            _RECORD_CACHE[cache_key] = (signature, record)


def load_records(
    result_root: Path | str = RESULT_ROOT,
    *,
    split: str | None = None,
    phase: str | None = None,
) -> list[RunRecord]:
    """Load validated completed records, reusing unchanged metric archives."""
    records: list[RunRecord] = []
    current_calibration_run_ids = {
        config["runId"] for config in build_calibration()
    }
    run_paths = sorted(Path(result_root).glob("**/run.json"))
    live_cache_keys = {path.resolve() for path in run_paths}
    with _RECORD_CACHE_LOCK:
        stale = [
            key
            for key in _RECORD_CACHE
            if key not in live_cache_keys
            and (key == Path(result_root).resolve() or Path(result_root).resolve() in key.parents)
        ]
        for key in stale:
            _RECORD_CACHE.pop(key, None)

    for run_path in run_paths:
        try:
            record = _validated_record(run_path, current_calibration_run_ids)
            if record is None:
                continue
            if split is not None and record.split != split:
                continue
            if phase is not None and record.phase != phase:
                continue
            records.append(record)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records


def _records_fingerprint(records: Iterable[RunRecord], label: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"campaign3-plots-v{PLOT_SCHEMA_VERSION}|{label}".encode("ascii"))
    for record in sorted(records, key=lambda item: str(item.run_path)):
        for path in (record.run_path, record.metrics_path):
            stat = path.stat()
            digest.update(str(path).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


class PlotWriter:
    def __init__(
        self,
        root: Path | str,
        *,
        only_changed: bool,
        formats: Iterable[str] = FORMATS,
    ):
        self.root = Path(root)
        self.only_changed = bool(only_changed)
        self.formats = tuple(dict.fromkeys(str(value).lower() for value in formats))
        unsupported = set(self.formats) - set(FORMATS)
        if not self.formats or unsupported:
            raise ValueError(f"Unsupported plot formats: {sorted(unsupported)}")
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self._load_manifest()
        self.generated: list[str] = []
        self.skipped: list[str] = []

    def _load_manifest(self) -> dict:
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if value.get("schemaVersion") == PLOT_SCHEMA_VERSION:
                return value
        except (OSError, ValueError):
            pass
        return {"schemaVersion": PLOT_SCHEMA_VERSION, "plots": {}}

    def save(
        self,
        figure: Figure,
        relative_stem: Path | str,
        records: Iterable[RunRecord],
    ) -> None:
        stem = self.root / Path(relative_stem)
        key = str(Path(relative_stem))
        record_list = list(records)
        fingerprint = _records_fingerprint(record_list, key)
        outputs = [stem.with_suffix(f".{extension}") for extension in self.formats]
        old = self.manifest["plots"].get(key, {})
        if (
            self.only_changed
            and old.get("fingerprint") == fingerprint
            and set(self.formats).issubset(set(old.get("formats", ())))
            and all(path.exists() for path in outputs)
        ):
            self.skipped.append(key)
            figure.clear()
            return

        stem.parent.mkdir(parents=True, exist_ok=True)
        FigureCanvasAgg(figure)
        for extension, output_path in zip(self.formats, outputs):
            fd, temporary_name = tempfile.mkstemp(
                dir=output_path.parent,
                prefix=f".{output_path.stem}.",
                suffix=f".{extension}",
            )
            os.close(fd)
            temporary_path = Path(temporary_name)
            try:
                figure.savefig(
                    temporary_path,
                    format=extension,
                    dpi=600,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="white",
                    transparent=False,
                )
                os.replace(temporary_path, output_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        self.manifest["plots"][key] = {
            "fingerprint": fingerprint,
            "sources": len(record_list),
            "formats": list(self.formats),
        }
        self.generated.append(key)
        figure.clear()

    def finish(self) -> None:
        write_json_atomic(self.manifest_path, self.manifest)


def _new_figure(
    *,
    width: float = 7.16,
    height: float = 3.0,
    columns: int = 1,
) -> tuple[Figure, list]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "hatch.linewidth": 0.7,
            "savefig.dpi": 600,
        }
    )
    figure = Figure(figsize=(width, height), constrained_layout=True)
    if columns == 1:
        axes = [figure.add_subplot(1, 1, 1)]
    else:
        axes = [figure.add_subplot(1, columns, index + 1) for index in range(columns)]
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, linestyle=":")
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    return figure, axes


def _filter(records: Iterable[RunRecord], **criteria) -> list[RunRecord]:
    result = []
    for record in records:
        if all(getattr(record, key) == value for key, value in criteria.items()):
            result.append(record)
    return result


def _mean_std(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, 0
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return float(np.mean(array)), std, int(array.size)


def _metric_summary(
    records: Iterable[RunRecord],
    metric: str = "final_avg",
) -> tuple[float, float, int]:
    return _mean_std(record.scalar(metric) for record in records)


def _clean_accuracy(
    records: Iterable[RunRecord],
    approach: str,
) -> tuple[float, float, int]:
    return _metric_summary(
        _filter(
            records,
            approach=approach,
            environment="clean",
            mitigation="none",
        )
    )


def _add_clean_line(axis, records, approach: str, *, label: bool = True) -> None:
    mean, _, count = _clean_accuracy(records, approach)
    if count:
        axis.axhline(
            mean,
            color="#111111",
            linestyle=":",
            linewidth=1.2,
            label=f"{APPROACH_LABELS[approach]} clean" if label else None,
            zorder=1,
        )


def _accuracy_axis(axis) -> None:
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Test accuracy")
    axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))


def _legend_if_handles(axis, **kwargs) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, **kwargs)


def _seed_styles(records: Iterable[RunRecord]) -> dict[int, tuple[str, str]]:
    seeds = sorted({record.seed for record in records})
    return {
        seed: (
            SEED_COLORS[index % len(SEED_COLORS)],
            SEED_MARKERS[index % len(SEED_MARKERS)],
        )
        for index, seed in enumerate(seeds)
    }


def _errorbar_series(
    axis,
    records: Iterable[RunRecord],
    *,
    approach: str,
    environment: str,
    mitigation: str,
    label: str | None = None,
    metric: str = "final_avg",
) -> bool:
    x_values = []
    means = []
    stds = []
    for sigma in NOISE_LEVELS:
        subset = _filter(
            records,
            approach=approach,
            environment=environment,
            mitigation=mitigation,
            sigma=sigma,
        )
        mean, std, count = _metric_summary(subset, metric)
        if count:
            x_values.append(sigma)
            means.append(mean)
            stds.append(std)
    if not x_values:
        return False
    axis.errorbar(
        x_values,
        means,
        yerr=stds,
        color=MITIGATION_COLORS[mitigation],
        marker=APPROACH_MARKERS[approach],
        linestyle=APPROACH_LINESTYLES[approach],
        capsize=2.5,
        label=label or MITIGATION_LABELS[mitigation],
    )
    return True


def _paper_dir(split: str) -> Path:
    return Path("images") / "gui" / split / "cifar10" / "hidden" / "comparison" / "paper"


def _diagnostic_dir(split: str) -> Path:
    return (
        Path("images")
        / "gui"
        / split
        / "cifar10"
        / "hidden"
        / "comparison"
        / "diagnostics"
    )


def _familiar_paper_dir(split: str, approach: str) -> Path:
    return (
        Path("images")
        / "gui"
        / split
        / "cifar10"
        / "hidden"
        / approach
        / "paper"
    )


def _plot_component_validation(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    merged = _filter(records, approach="merged")
    if not merged:
        return
    figure, axes = _new_figure(columns=2, height=2.85)

    axis = axes[0]
    plotted = False
    for mitigation in ("none", "ebm"):
        plotted |= _errorbar_series(
            axis,
            merged,
            approach="merged",
            environment="noise",
            mitigation=mitigation,
        )
    _add_clean_line(axis, merged, "merged")
    axis.set_title("(a) Channel noise component")
    axis.set_xlabel("Channel noise sigma")
    _accuracy_axis(axis)
    if plotted:
        axis.legend(frameon=False, loc="best")

    axis = axes[1]
    labels = []
    means = []
    errors = []
    colors = []
    for mitigation in ("none", "ss"):
        subset = _filter(
            merged,
            approach="merged",
            environment="hidden",
            mitigation=mitigation,
        )
        mean, std, count = _metric_summary(subset)
        if count:
            labels.append(MITIGATION_LABELS[mitigation])
            means.append(mean)
            errors.append(std)
            colors.append(MITIGATION_COLORS[mitigation])
    if labels:
        positions = np.arange(len(labels))
        axis.bar(
            positions,
            means,
            yerr=errors,
            color=colors,
            edgecolor="#222222",
            linewidth=0.65,
            capsize=3,
            width=0.62,
        )
        axis.set_xticks(positions, labels)
    _add_clean_line(axis, merged, "merged")
    axis.set_title("(b) Hidden Byzantine component")
    _accuracy_axis(axis)
    axis.legend(frameon=False, loc="best")

    writer.save(figure, _paper_dir(split) / "component_validation_avg", merged)


def _plot_joint_profile(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = [
        record
        for record in records
        if record.environment in ("clean", "hidden_noise")
        and (
            record.environment == "clean"
            or record.mitigation == "ss_ebm"
        )
    ]
    if not _filter(selected, environment="hidden_noise", mitigation="ss_ebm"):
        return
    figure, (axis,) = _new_figure(width=3.5, height=2.8)
    for approach in ("merged", "cart"):
        subset = _filter(selected, approach=approach)
        x_values = []
        means = []
        stds = []
        for sigma in NOISE_LEVELS:
            mean, std, count = _metric_summary(
                _filter(
                    subset,
                    environment="hidden_noise",
                    mitigation="ss_ebm",
                    sigma=sigma,
                )
            )
            if count:
                x_values.append(sigma)
                means.append(mean)
                stds.append(std)
        if x_values:
            axis.errorbar(
                x_values,
                means,
                yerr=stds,
                color=MITIGATION_COLORS["ss_ebm"],
                marker=APPROACH_MARKERS[approach],
                linestyle=APPROACH_LINESTYLES[approach],
                capsize=2.5,
                label=f"{APPROACH_LABELS[approach]} SS+EBM",
            )
            _add_clean_line(axis, subset, approach)
    axis.set_title("Joint hidden-attack and channel-noise robustness")
    axis.set_xlabel("Channel noise sigma")
    _accuracy_axis(axis)
    axis.legend(frameon=False, loc="best")
    writer.save(figure, _paper_dir(split) / "joint_robustness_profile_avg", selected)


def _plot_defense_composition(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = [
        record
        for record in records
        if record.environment in ("clean", "hidden_noise")
    ]
    if not _filter(selected, environment="hidden_noise"):
        return
    figure, axes = _new_figure(columns=2, height=3.0)
    for panel_index, (axis, approach) in enumerate(zip(axes, ("merged", "cart"))):
        approach_records = _filter(selected, approach=approach)
        any_series = False
        for mitigation in MITIGATION_ORDER:
            any_series |= _errorbar_series(
                axis,
                approach_records,
                approach=approach,
                environment="hidden_noise",
                mitigation=mitigation,
            )
        _add_clean_line(axis, approach_records, approach)
        axis.set_title(f"({chr(97 + panel_index)}) {APPROACH_LABELS[approach]}")
        axis.set_xlabel("Channel noise sigma")
        _accuracy_axis(axis)
        if any_series:
            axis.legend(frameon=False, loc="best", ncol=2)
    writer.save(figure, _paper_dir(split) / "defense_composition_avg", selected)


def _paired_by_seed(
    left: Iterable[RunRecord],
    right: Iterable[RunRecord],
    metric: str = "final_avg",
) -> list[float]:
    left_by_seed = {record.seed: record.scalar(metric) for record in left}
    right_by_seed = {record.seed: record.scalar(metric) for record in right}
    return [
        left_by_seed[seed] - right_by_seed[seed]
        for seed in sorted(left_by_seed.keys() & right_by_seed.keys())
    ]


def _plot_cart_lift(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = _filter(
        records,
        environment="hidden_noise",
        mitigation="ss_ebm",
    )
    differences = []
    stds = []
    x_values = []
    used: list[RunRecord] = []
    for sigma in NOISE_LEVELS:
        merged = _filter(selected, approach="merged", sigma=sigma)
        cart = _filter(selected, approach="cart", sigma=sigma)
        paired = _paired_by_seed(cart, merged)
        mean, std, count = _mean_std(paired)
        if count:
            x_values.append(sigma)
            differences.append(mean)
            stds.append(std)
            used.extend(merged)
            used.extend(cart)
    if not x_values:
        return
    figure, (axis,) = _new_figure(width=3.5, height=2.75)
    positions = np.arange(len(x_values))
    colors = [
        MITIGATION_COLORS["ss_ebm"] if value >= 0 else "#B2182B"
        for value in differences
    ]
    bars = axis.bar(
        positions,
        differences,
        yerr=stds,
        color=colors,
        edgecolor="#222222",
        hatch=APPROACH_HATCHES["cart"],
        linewidth=0.65,
        capsize=2.5,
        width=0.65,
    )
    axis.axhline(0.0, color="#111111", linewidth=0.9)
    axis.set_xticks(positions, [f"{sigma:.1f}" for sigma in x_values])
    axis.set_xlabel("Channel noise sigma")
    axis.set_ylabel("CART minus Merged accuracy")
    axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axis.set_title("CART minus Merged with SS+EBM (paired seeds)")
    axis.margins(y=0.18)
    for bar, value in zip(bars, differences):
        axis.annotate(
            f"{value:+.1%}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4 if value >= 0 else -4),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.5,
        )
    writer.save(figure, _paper_dir(split) / "cart_lift_over_merged_avg", used)


def _paired_damage_recovery(
    records: list[RunRecord],
    *,
    approach: str,
    sigma: float,
    mitigation: str,
) -> list[float]:
    clean = {
        record.seed: record.scalar("final_avg")
        for record in _filter(
            records,
            approach=approach,
            environment="clean",
            mitigation="none",
        )
    }
    degraded = {
        record.seed: record.scalar("final_avg")
        for record in _filter(
            records,
            approach=approach,
            environment="hidden_noise",
            mitigation="none",
            sigma=sigma,
        )
    }
    defended = {
        record.seed: record.scalar("final_avg")
        for record in _filter(
            records,
            approach=approach,
            environment="hidden_noise",
            mitigation=mitigation,
            sigma=sigma,
        )
    }
    values = []
    for seed in sorted(clean.keys() & degraded.keys() & defended.keys()):
        damage = clean[seed] - degraded[seed]
        if damage <= 1e-12:
            continue
        raw_recovery = 100.0 * (defended[seed] - degraded[seed]) / damage
        values.append(float(np.clip(raw_recovery, 0.0, 100.0)))
    return values


def _plot_damage_recovery(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    if not _filter(records, environment="hidden_noise"):
        return
    figure, axes = _new_figure(columns=2, height=3.0)
    used = [
        record
        for record in records
        if record.environment in ("clean", "hidden_noise")
    ]
    for panel_index, (axis, approach) in enumerate(zip(axes, ("merged", "cart"))):
        positions = np.arange(len(NOISE_LEVELS), dtype=np.float64)
        width = 0.24
        plotted = False
        for offset_index, mitigation in enumerate(("ss", "ebm", "ss_ebm")):
            means = []
            stds = []
            for sigma in NOISE_LEVELS:
                mean, std, count = _mean_std(
                    _paired_damage_recovery(
                        records,
                        approach=approach,
                        sigma=sigma,
                        mitigation=mitigation,
                    )
                )
                means.append(0.0 if not count else mean)
                stds.append(0.0 if not count else std)
                plotted |= bool(count)
            offset = (offset_index - 1) * width
            axis.bar(
                positions + offset,
                means,
                yerr=stds,
                width=width,
                color=MITIGATION_COLORS[mitigation],
                edgecolor="#222222",
                hatch=APPROACH_HATCHES[approach],
                linewidth=0.55,
                capsize=1.8,
                label=MITIGATION_LABELS[mitigation],
            )
        axis.set_xticks(positions, [f"{sigma:.1f}" for sigma in NOISE_LEVELS])
        axis.set_ylim(0.0, 105.0)
        axis.set_xlabel("Channel noise sigma")
        axis.set_ylabel("Damage recovered")
        axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(100.0))
        axis.set_title(f"({chr(97 + panel_index)}) {APPROACH_LABELS[approach]}")
        if plotted:
            axis.legend(frameon=False, loc="best", ncol=3)
        axis.text(
            0.01,
            0.02,
            "0% = no recovery; values are bounded to [0, 100]",
            transform=axis.transAxes,
            fontsize=6.2,
            color="#444444",
        )
    writer.save(figure, _paper_dir(split) / "damage_recovery_avg", used)


def _history_summary(records: Iterable[RunRecord], name: str) -> tuple[np.ndarray, np.ndarray]:
    histories = [
        np.asarray(record.metrics[name], dtype=np.float64)
        for record in records
        if name in record.metrics
    ]
    if not histories:
        return np.asarray([]), np.asarray([])
    length = min(len(history) for history in histories)
    stacked = np.stack([history[:length] for history in histories])
    return np.mean(stacked, axis=0), np.std(stacked, axis=0, ddof=1 if len(stacked) > 1 else 0)


def _plot_convergence(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    representative_sigma = 0.4
    selected = [
        record
        for record in records
        if record.environment == "clean"
        or (
            record.environment == "hidden_noise"
            and abs(record.sigma - representative_sigma) < 1e-12
        )
    ]
    if not _filter(selected, environment="hidden_noise"):
        return
    figure, axes = _new_figure(columns=2, height=3.0)
    for panel_index, (axis, approach) in enumerate(zip(axes, ("merged", "cart"))):
        approach_records = _filter(selected, approach=approach)
        clean = _filter(
            approach_records,
            environment="clean",
            mitigation="none",
        )
        mean, _ = _history_summary(clean, "avg_history")
        if mean.size:
            axis.plot(
                np.arange(1, len(mean) + 1),
                mean,
                color="#111111",
                linestyle=":",
                label="Clean",
            )
        for mitigation in MITIGATION_ORDER:
            subset = _filter(
                approach_records,
                environment="hidden_noise",
                mitigation=mitigation,
                sigma=representative_sigma,
            )
            mean, _ = _history_summary(subset, "avg_history")
            if mean.size:
                axis.plot(
                    np.arange(1, len(mean) + 1),
                    mean,
                    color=MITIGATION_COLORS[mitigation],
                    linestyle=APPROACH_LINESTYLES[approach],
                    label=MITIGATION_LABELS[mitigation],
                )
        axis.set_title(
            f"({chr(97 + panel_index)}) {APPROACH_LABELS[approach]}, "
            f"sigma={representative_sigma:.1f}"
        )
        attack_records = _filter(
            approach_records,
            environment="hidden_noise",
            sigma=representative_sigma,
        )
        if attack_records:
            attack_start = int(attack_records[0].config.get("attackHiddenStart", 0))
            axis.axvline(
                attack_start + 1,
                color="#B2182B",
                linestyle="-.",
                linewidth=0.9,
                label="Attack starts",
            )
        axis.set_xlabel("Communication round")
        _accuracy_axis(axis)
        _legend_if_handles(axis, frameon=False, loc="best", ncol=2)
    writer.save(figure, _paper_dir(split) / "convergence_representative_avg", selected)


def _plot_average_vs_worst(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = [
        record
        for record in records
        if record.environment in ("clean", "hidden_noise")
    ]
    if not selected:
        return
    figure, (axis,) = _new_figure(width=3.5, height=3.0)
    for approach in ("merged", "cart"):
        for mitigation in MITIGATION_ORDER:
            subset = _filter(selected, approach=approach, mitigation=mitigation)
            if not subset:
                continue
            axis.scatter(
                [record.scalar("final_avg") for record in subset],
                [record.scalar("final_worst") for record in subset],
                marker=APPROACH_MARKERS[approach],
                facecolor=MITIGATION_COLORS[mitigation],
                edgecolor="#222222",
                linewidth=0.45,
                s=24,
            )
    axis.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=0.8)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Average-node accuracy")
    axis.set_ylabel("Worst-node accuracy")
    axis.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axis.set_title("Average versus worst-node performance")
    method_legend = [
        Patch(facecolor=MITIGATION_COLORS[key], label=MITIGATION_LABELS[key])
        for key in MITIGATION_ORDER
    ]
    approach_legend = [
        Line2D(
            [0],
            [0],
            marker=APPROACH_MARKERS[key],
            color="none",
            markerfacecolor="white",
            markeredgecolor="#222222",
            label=APPROACH_LABELS[key],
        )
        for key in ("merged", "cart")
    ]
    axis.legend(
        handles=method_legend + approach_legend,
        frameon=False,
        loc="lower right",
        ncol=2,
    )
    writer.save(figure, _paper_dir(split) / "average_vs_worst", selected)


def _plot_class_retention(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    representative_sigma = 0.2
    selected = _filter(
        records,
        environment="hidden_noise",
        mitigation="ss_ebm",
        sigma=representative_sigma,
    )
    values = {}
    for approach in ("merged", "cart"):
        arrays = [
            np.nanmean(
                np.asarray(record.metrics["final_class_accuracy"], dtype=np.float64),
                axis=0,
            )
            for record in _filter(selected, approach=approach)
            if "final_class_accuracy" in record.metrics
        ]
        if arrays:
            values[approach] = np.nanmean(np.stack(arrays), axis=0)
    if not values:
        return
    figure, (axis,) = _new_figure(width=7.16, height=2.9)
    positions = np.arange(len(CLASS_NAMES))
    width = 0.36
    for index, approach in enumerate(("merged", "cart")):
        if approach not in values:
            continue
        axis.bar(
            positions + (index - 0.5) * width,
            values[approach],
            width=width,
            color="#56B4E9" if approach == "merged" else "#009E73",
            edgecolor="#222222",
            hatch=APPROACH_HATCHES[approach],
            linewidth=0.55,
            label=APPROACH_LABELS[approach],
        )
    axis.set_xticks(positions, CLASS_NAMES, rotation=25, ha="right")
    axis.set_title(
        f"Per-class retention under hidden attack, sigma={representative_sigma:.1f}, SS+EBM"
    )
    _accuracy_axis(axis)
    axis.legend(frameon=False, loc="best")
    writer.save(figure, _paper_dir(split) / "class_retention_avg", selected)


def _plot_history_curve(
    axis,
    records: list[RunRecord],
    *,
    color: str,
    linestyle: str,
    label: str | None = None,
    uncertainty: bool = False,
) -> np.ndarray:
    mean, std = _history_summary(records, "avg_history")
    if not mean.size:
        return mean
    rounds = np.arange(1, len(mean) + 1)
    axis.plot(
        rounds,
        mean,
        color=color,
        linestyle=linestyle,
        label=label,
    )
    if uncertainty and len(records) > 1:
        rgb = np.asarray(mpl.colors.to_rgb(color), dtype=np.float64)
        band_color = tuple(0.86 + 0.14 * rgb)
        axis.fill_between(
            rounds,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=band_color,
            linewidth=0,
        )
    return mean


def _set_zoom_limits(axis, histories: Iterable[np.ndarray]) -> None:
    values = [
        np.asarray(history, dtype=np.float64)
        for history in histories
        if np.asarray(history).size
    ]
    if not values:
        axis.set_ylim(0.0, 1.0)
        return
    combined = np.concatenate(values)
    combined = combined[np.isfinite(combined)]
    if not combined.size:
        axis.set_ylim(0.0, 1.0)
        return
    lower = max(0.0, float(np.min(combined)) - 0.03)
    upper = min(1.0, float(np.max(combined)) + 0.04)
    if upper - lower < 0.15:
        center = 0.5 * (lower + upper)
        lower = max(0.0, center - 0.075)
        upper = min(1.0, center + 0.075)
    axis.set_ylim(lower, upper)


def _familiar_curve_legend(
    axis,
    *,
    mitigations: Iterable[str],
    include_clean: bool,
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=NOISE_COLORS[sigma],
            linestyle="-",
            label=f"sigma={sigma:.1f}",
        )
        for sigma in NOISE_LEVELS
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=MITIGATION_LINESTYLES[mitigation],
            label=MITIGATION_LABELS[mitigation],
        )
        for mitigation in mitigations
    )
    if include_clean:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#111111",
                linestyle=":",
                label="Clean reference",
            )
        )
    axis.legend(
        handles=handles,
        frameon=False,
        loc="best",
        fontsize=5.8,
        ncol=2,
    )


def _plot_familiar_experiments(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
    approach: str,
    *,
    zoom: bool,
) -> None:
    approach_records = _filter(records, approach=approach)
    if not approach_records:
        return
    figure, axes = _new_figure(columns=2, height=3.2)
    used: list[RunRecord] = []
    plotted_any = False
    specifications = (
        ("noise", ("ebm",) if zoom else ("none", "ebm"), "Channel noise only"),
        (
            "hidden_noise",
            ("ss_ebm",) if zoom else ("ss", "ss_ebm"),
            "Hidden attack + channel noise",
        ),
    )
    clean = _filter(
        approach_records,
        environment="clean",
        mitigation="none",
    )

    for panel, (axis, (environment, mitigations, title)) in enumerate(
        zip(axes, specifications)
    ):
        histories: list[np.ndarray] = []
        for sigma in NOISE_LEVELS:
            for mitigation in mitigations:
                subset = _filter(
                    approach_records,
                    environment=environment,
                    mitigation=mitigation,
                    sigma=sigma,
                )
                if not subset:
                    continue
                history = _plot_history_curve(
                    axis,
                    subset,
                    color=NOISE_COLORS[sigma],
                    linestyle=MITIGATION_LINESTYLES[mitigation],
                    uncertainty=zoom,
                )
                histories.append(history)
                used.extend(subset)
                plotted_any = True

        if not zoom and clean:
            clean_history = _plot_history_curve(
                axis,
                clean,
                color="#111111",
                linestyle=":",
                uncertainty=True,
            )
            histories.append(clean_history)
            used.extend(clean)
        if environment == "hidden_noise":
            attack_records = _filter(
                approach_records,
                environment=environment,
            )
            if attack_records:
                attack_start = int(
                    attack_records[0].config.get("attackHiddenStart", 0)
                )
                axis.axvline(
                    attack_start + 1,
                    color="#B2182B",
                    linestyle="-.",
                    linewidth=0.9,
                )
        axis.set_title(f"({chr(97 + panel)}) {title}")
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Average test accuracy")
        axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        if zoom:
            _set_zoom_limits(axis, histories)
        else:
            axis.set_ylim(0.0, 1.0)
        _familiar_curve_legend(
            axis,
            mitigations=mitigations,
            include_clean=not zoom,
        )

    if not plotted_any:
        figure.clear()
        return
    suffix = "experiments_avg_zoom" if zoom else "experiments_avg"
    writer.save(
        figure,
        _familiar_paper_dir(split, approach) / suffix,
        used,
    )


def _plot_familiar_grid(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
    approach: str,
) -> None:
    approach_records = _filter(records, approach=approach)
    joint = _filter(approach_records, environment="hidden_noise")
    if not joint:
        return
    figure, axes = _new_figure(width=7.16, height=2.65, columns=5)
    used: list[RunRecord] = []
    clean = _filter(
        approach_records,
        environment="clean",
        mitigation="none",
    )
    for panel, (axis, sigma) in enumerate(zip(axes, NOISE_LEVELS)):
        plotted = False
        for mitigation in MITIGATION_ORDER:
            subset = _filter(
                joint,
                mitigation=mitigation,
                sigma=sigma,
            )
            if not subset:
                continue
            _plot_history_curve(
                axis,
                subset,
                color=MITIGATION_COLORS[mitigation],
                linestyle=MITIGATION_LINESTYLES[mitigation],
                label=MITIGATION_LABELS[mitigation],
            )
            used.extend(subset)
            plotted = True
        if clean:
            _plot_history_curve(
                axis,
                clean,
                color="#111111",
                linestyle=":",
                label="Clean",
            )
            used.extend(clean)
        axis.axvline(
            int(joint[0].config.get("attackHiddenStart", 0)) + 1,
            color="#B2182B",
            linestyle="-.",
            linewidth=0.7,
        )
        axis.set_title(f"sigma={sigma:.1f}")
        axis.set_xlabel("Round")
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        if panel == 0:
            axis.set_ylabel("Average accuracy")
        else:
            axis.tick_params(labelleft=False)
        if panel == len(NOISE_LEVELS) - 1 and plotted:
            axis.legend(
                frameon=False,
                loc="best",
                fontsize=5.4,
            )
        if not plotted:
            axis.text(
                0.5,
                0.5,
                "Pending",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#777777",
            )
    writer.save(
        figure,
        _familiar_paper_dir(split, approach) / "grid_avg",
        used,
    )


def _plot_familiar_final_accuracy(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
    approach: str,
) -> None:
    approach_records = _filter(records, approach=approach)
    if not approach_records:
        return
    conditions = [
        ("Clean", "clean", "none", 0.0, "#111111"),
        ("Hidden\nSS", "hidden", "ss", 0.0, MITIGATION_COLORS["ss"]),
    ]
    conditions.extend(
        (
            f"Noise {sigma:.1f}\nEBM",
            "noise",
            "ebm",
            sigma,
            MITIGATION_COLORS["ebm"],
        )
        for sigma in NOISE_LEVELS
    )
    conditions.extend(
        (
            f"Joint {sigma:.1f}\nSS+EBM",
            "hidden_noise",
            "ss_ebm",
            sigma,
            MITIGATION_COLORS["ss_ebm"],
        )
        for sigma in NOISE_LEVELS
    )

    labels = []
    means = []
    stds = []
    colors = []
    used: list[RunRecord] = []
    for label, environment, mitigation, sigma, color in conditions:
        subset = _filter(
            approach_records,
            environment=environment,
            mitigation=mitigation,
            sigma=sigma,
        )
        mean, std, count = _metric_summary(subset)
        if not count:
            continue
        labels.append(label)
        means.append(mean)
        stds.append(std)
        colors.append(color)
        used.extend(subset)
    if not labels:
        return

    figure, (axis,) = _new_figure(width=7.16, height=3.25)
    positions = np.arange(len(labels))
    bars = axis.bar(
        positions,
        means,
        yerr=stds,
        color=colors,
        edgecolor="#222222",
        hatch=APPROACH_HATCHES[approach],
        linewidth=0.6,
        capsize=2.2,
        width=0.68,
    )
    for bar, value in zip(bars, means):
        axis.annotate(
            f"{value:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.8,
            rotation=90,
        )
    _add_clean_line(axis, approach_records, approach)
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_title(
        f"{APPROACH_LABELS[approach]} headline final accuracy (mean +/- SD)"
    )
    _accuracy_axis(axis)
    axis.legend(frameon=False, loc="upper right")
    writer.save(
        figure,
        _familiar_paper_dir(split, approach) / "final_accuracy_avg",
        used,
    )


def _plot_paired_gain_panel(
    axis,
    records: list[RunRecord],
    *,
    approach: str,
    environment: str,
    mitigations: tuple[str, ...],
    title: str,
) -> list[RunRecord]:
    positions = np.arange(len(NOISE_LEVELS), dtype=np.float64)
    width = min(0.24, 0.76 / max(1, len(mitigations)))
    used: list[RunRecord] = []
    plotted_values = []
    for method_index, mitigation in enumerate(mitigations):
        label_added = False
        offset = (method_index - (len(mitigations) - 1) / 2.0) * width
        for sigma_index, sigma in enumerate(NOISE_LEVELS):
            baseline = _filter(
                records,
                approach=approach,
                environment=environment,
                mitigation="none",
                sigma=sigma,
            )
            defended = _filter(
                records,
                approach=approach,
                environment=environment,
                mitigation=mitigation,
                sigma=sigma,
            )
            paired = _paired_by_seed(defended, baseline)
            mean, std, count = _mean_std(paired)
            if not count:
                continue
            position = positions[sigma_index] + offset
            bar = axis.bar(
                [position],
                [mean],
                yerr=[std],
                width=width,
                color=MITIGATION_COLORS[mitigation],
                edgecolor="#222222",
                hatch=APPROACH_HATCHES[approach],
                linewidth=0.55,
                capsize=1.8,
                label=(
                    MITIGATION_LABELS[mitigation]
                    if not label_added
                    else None
                ),
            )[0]
            label_added = True
            axis.annotate(
                f"{mean:+.1%}",
                xy=(bar.get_x() + bar.get_width() / 2.0, mean),
                xytext=(0, 3 if mean >= 0 else -3),
                textcoords="offset points",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=5.5,
                rotation=90,
            )
            plotted_values.extend((mean - std, mean + std))
            used.extend(baseline)
            used.extend(defended)
    axis.axhline(0.0, color="#111111", linewidth=0.9)
    axis.set_xticks(positions, [f"{sigma:.1f}" for sigma in NOISE_LEVELS])
    axis.set_xlabel("Channel noise sigma")
    axis.set_ylabel("Paired final-accuracy change")
    axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axis.set_title(title)
    if plotted_values:
        lower = min(0.0, min(plotted_values))
        upper = max(0.0, max(plotted_values))
        span = max(0.03, upper - lower)
        padding = max(0.01, span * 0.12)
        axis.set_ylim(lower - padding, upper + padding)
        _legend_if_handles(axis, frameon=False, loc="best")
    axis.text(
        0.01,
        0.02,
        "0 = matching no-mitigation seed; negative effects retained",
        transform=axis.transAxes,
        fontsize=5.7,
        color="#444444",
    )
    return used


def _plot_familiar_improvement(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
    approach: str,
) -> None:
    approach_records = _filter(records, approach=approach)
    if not approach_records:
        return
    figure, axes = _new_figure(columns=2, height=3.15)
    used = _plot_paired_gain_panel(
        axes[0],
        approach_records,
        approach=approach,
        environment="noise",
        mitigations=("ebm",),
        title="(a) Noise-only mitigation gain",
    )
    used.extend(
        _plot_paired_gain_panel(
            axes[1],
            approach_records,
            approach=approach,
            environment="hidden_noise",
            mitigations=("ss", "ebm", "ss_ebm"),
            title="(b) Joint-environment mitigation gain",
        )
    )
    if not used:
        figure.clear()
        return
    writer.save(
        figure,
        _familiar_paper_dir(split, approach)
        / "improvement_over_no_mitigation_avg",
        used,
    )


def _plot_familiar_seed_profiles(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
    approach: str,
) -> None:
    approach_records = _filter(records, approach=approach)
    specifications = (
        ("noise", "ebm", "(a) Channel noise + EBM"),
        (
            "hidden_noise",
            "ss_ebm",
            "(b) Hidden attack + channel noise + SS+EBM",
        ),
    )
    selected = [
        record
        for record in approach_records
        if any(
            record.environment == environment
            and record.mitigation == mitigation
            for environment, mitigation, _ in specifications
        )
    ]
    if not selected:
        return

    seed_styles = _seed_styles(selected)
    figure, axes = _new_figure(columns=2, height=3.15)
    used: list[RunRecord] = []
    clean = _filter(
        approach_records,
        environment="clean",
        mitigation="none",
    )
    for axis, (environment, mitigation, title) in zip(axes, specifications):
        panel_records = _filter(
            selected,
            environment=environment,
            mitigation=mitigation,
        )
        for seed in sorted({record.seed for record in panel_records}):
            seed_records = sorted(
                _filter(panel_records, seed=seed),
                key=lambda record: record.sigma,
            )
            axis.plot(
                [record.sigma for record in seed_records],
                [record.scalar("final_avg") for record in seed_records],
                color=seed_styles[seed][0],
                marker=seed_styles[seed][1],
                linewidth=1.0,
                label=f"seed={seed}",
                zorder=3,
            )
            used.extend(seed_records)

        x_values = []
        means = []
        stds = []
        counts = []
        for sigma in NOISE_LEVELS:
            mean, std, count = _metric_summary(
                _filter(panel_records, sigma=sigma)
            )
            if count:
                x_values.append(sigma)
                means.append(mean)
                stds.append(std)
                counts.append(count)
        if x_values:
            axis.errorbar(
                x_values,
                means,
                yerr=stds,
                color="#111111",
                marker="_",
                linestyle="--",
                linewidth=1.2,
                capsize=2.5,
                label="Mean +/- SD (all seeds)",
                zorder=2,
            )
            for sigma, mean, std, count in zip(
                x_values,
                means,
                stds,
                counts,
            ):
                axis.annotate(
                    f"n={count}",
                    xy=(sigma, mean + std),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color="#333333",
                )
        _add_clean_line(axis, approach_records, approach)
        axis.set_xticks(NOISE_LEVELS, [f"{sigma:.1f}" for sigma in NOISE_LEVELS])
        axis.set_xlabel("Channel noise sigma")
        axis.set_title(title)
        _accuracy_axis(axis)
        if panel_records:
            _legend_if_handles(axis, frameon=False, loc="best", ncol=2)
        else:
            axis.text(
                0.5,
                0.5,
                "Results pending",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#777777",
            )

    used.extend(clean)
    writer.save(
        figure,
        _familiar_paper_dir(split, approach)
        / "seed_profiles_final_accuracy",
        used,
    )


def _plot_ablation_panel(
    axis,
    records: list[RunRecord],
    *,
    approach: str,
    environment: str,
    mitigations: tuple[str, ...],
    title: str,
) -> list[RunRecord]:
    positions = np.arange(len(NOISE_LEVELS), dtype=np.float64)
    width = min(0.22, 0.78 / max(1, len(mitigations)))
    used: list[RunRecord] = []
    for method_index, mitigation in enumerate(mitigations):
        label_added = False
        offset = (method_index - (len(mitigations) - 1) / 2.0) * width
        for sigma_index, sigma in enumerate(NOISE_LEVELS):
            subset = _filter(
                records,
                approach=approach,
                environment=environment,
                mitigation=mitigation,
                sigma=sigma,
            )
            mean, std, count = _metric_summary(subset)
            if not count:
                continue
            axis.bar(
                [positions[sigma_index] + offset],
                [mean],
                yerr=[std],
                width=width,
                color=MITIGATION_COLORS[mitigation],
                edgecolor="#222222",
                hatch=APPROACH_HATCHES[approach],
                linewidth=0.55,
                capsize=1.8,
                label=(
                    MITIGATION_LABELS[mitigation]
                    if not label_added
                    else None
                ),
            )
            label_added = True
            used.extend(subset)
    _add_clean_line(axis, records, approach)
    axis.set_xticks(positions, [f"{sigma:.1f}" for sigma in NOISE_LEVELS])
    axis.set_xlabel("Channel noise sigma")
    axis.set_title(title)
    _accuracy_axis(axis)
    if used:
        _legend_if_handles(axis, frameon=False, loc="best", ncol=2)
    return used


def _plot_familiar_ablation(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
    approach: str,
) -> None:
    approach_records = _filter(records, approach=approach)
    if not approach_records:
        return
    figure, axes = _new_figure(columns=2, height=3.15)
    used = _plot_ablation_panel(
        axes[0],
        approach_records,
        approach=approach,
        environment="noise",
        mitigations=("none", "ebm"),
        title="(a) Channel noise only",
    )
    used.extend(
        _plot_ablation_panel(
            axes[1],
            approach_records,
            approach=approach,
            environment="hidden_noise",
            mitigations=MITIGATION_ORDER,
            title="(b) Hidden attack + channel noise",
        )
    )
    if not used:
        figure.clear()
        return
    writer.save(
        figure,
        _familiar_paper_dir(split, approach) / "ablation_groups_avg",
        used,
    )


def _plot_familiar_suite(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    for approach in ("merged", "cart"):
        if not _filter(records, approach=approach):
            continue
        _plot_familiar_experiments(
            records,
            writer,
            split,
            approach,
            zoom=False,
        )
        _plot_familiar_experiments(
            records,
            writer,
            split,
            approach,
            zoom=True,
        )
        _plot_familiar_grid(records, writer, split, approach)
        _plot_familiar_final_accuracy(records, writer, split, approach)
        _plot_familiar_improvement(records, writer, split, approach)
        _plot_familiar_seed_profiles(records, writer, split, approach)
        _plot_familiar_ablation(records, writer, split, approach)


def _plot_mitigation_heatmaps(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = _filter(records, environment="hidden_noise")
    if not selected:
        return
    figure, axes = _new_figure(columns=2, height=3.0)
    image = None
    for panel_index, (axis, approach) in enumerate(zip(axes, ("merged", "cart"))):
        matrix = np.full((len(MITIGATION_ORDER), len(NOISE_LEVELS)), np.nan)
        for row, mitigation in enumerate(MITIGATION_ORDER):
            for column, sigma in enumerate(NOISE_LEVELS):
                mean, _, count = _metric_summary(
                    _filter(
                        selected,
                        approach=approach,
                        mitigation=mitigation,
                        sigma=sigma,
                    )
                )
                if count:
                    matrix[row, column] = mean
        image = axis.imshow(
            matrix,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )
        axis.grid(False)
        axis.set_xticks(np.arange(len(NOISE_LEVELS)), [f"{value:.1f}" for value in NOISE_LEVELS])
        axis.set_yticks(
            np.arange(len(MITIGATION_ORDER)),
            [MITIGATION_LABELS[value] for value in MITIGATION_ORDER],
        )
        axis.set_xlabel("Channel noise sigma")
        axis.set_title(f"({chr(97 + panel_index)}) {APPROACH_LABELS[approach]}")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if np.isfinite(matrix[row, column]):
                    color = "white" if matrix[row, column] < 0.48 else "black"
                    axis.text(
                        column,
                        row,
                        f"{matrix[row, column]:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=6.5,
                    )
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("Final test accuracy")
        colorbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    writer.save(figure, _diagnostic_dir(split) / "mitigation_heatmap_avg", selected)


def _plot_seed_spread(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = _filter(records, environment="hidden_noise", mitigation="ss_ebm")
    if not selected:
        return
    figure, axes = _new_figure(columns=2, height=2.9)
    seed_styles = _seed_styles(selected)
    for panel_index, (axis, approach) in enumerate(zip(axes, ("merged", "cart"))):
        subset = _filter(selected, approach=approach)
        for seed in sorted({record.seed for record in subset}):
            seed_records = sorted(
                _filter(subset, seed=seed),
                key=lambda record: record.sigma,
            )
            axis.plot(
                [record.sigma for record in seed_records],
                [record.scalar("final_avg") for record in seed_records],
                color=seed_styles[seed][0],
                marker=seed_styles[seed][1],
                linewidth=1.0,
                label=f"seed={seed}",
            )
        x_values = []
        means = []
        stds = []
        counts = []
        for sigma in NOISE_LEVELS:
            values = [
                record.scalar("final_avg")
                for record in _filter(subset, sigma=sigma)
            ]
            mean, std, count = _mean_std(values)
            if count:
                x_values.append(sigma)
                means.append(mean)
                stds.append(std)
                counts.append(count)
        if x_values:
            axis.errorbar(
                x_values,
                means,
                yerr=stds,
                color="#111111",
                marker="_",
                linestyle="--",
                linewidth=1.2,
                capsize=2.5,
                label="Mean +/- SD (all seeds)",
                zorder=2,
            )
            for sigma, mean, std, count in zip(
                x_values,
                means,
                stds,
                counts,
            ):
                axis.annotate(
                    f"n={count}",
                    xy=(sigma, mean + std),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color="#333333",
                )
        _add_clean_line(axis, records, approach)
        axis.set_title(f"({chr(97 + panel_index)}) {APPROACH_LABELS[approach]}")
        axis.set_xticks(NOISE_LEVELS, [f"{sigma:.1f}" for sigma in NOISE_LEVELS])
        axis.set_xlabel("Channel noise sigma")
        _accuracy_axis(axis)
        if subset:
            _legend_if_handles(axis, frameon=False, loc="best", ncol=2)
        else:
            axis.text(
                0.5,
                0.5,
                "SS+EBM results pending",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#777777",
            )
    writer.save(figure, _diagnostic_dir(split) / "seed_spread_ss_ebm", selected)


def _plot_stability(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = _filter(records, environment="hidden_noise")
    if not selected:
        return
    figure, axes = _new_figure(columns=2, height=3.0)
    for panel_index, (axis, approach) in enumerate(zip(axes, ("merged", "cart"))):
        for mitigation in MITIGATION_ORDER:
            x_values = []
            values = []
            for sigma in NOISE_LEVELS:
                run_stability = []
                for record in _filter(
                    selected,
                    approach=approach,
                    mitigation=mitigation,
                    sigma=sigma,
                ):
                    history = np.asarray(record.metrics.get("avg_history", []), dtype=np.float64)
                    if history.size:
                        run_stability.append(float(np.std(history[-10:])))
                mean, _, count = _mean_std(run_stability)
                if count:
                    x_values.append(sigma)
                    values.append(mean)
            if x_values:
                axis.plot(
                    x_values,
                    values,
                    color=MITIGATION_COLORS[mitigation],
                    marker=APPROACH_MARKERS[approach],
                    linestyle=APPROACH_LINESTYLES[approach],
                    label=MITIGATION_LABELS[mitigation],
                )
        axis.set_title(f"({chr(97 + panel_index)}) {APPROACH_LABELS[approach]}")
        axis.set_xlabel("Channel noise sigma")
        axis.set_ylabel("Last-10-round standard deviation")
        axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        _legend_if_handles(axis, frameon=False, loc="best", ncol=2)
    writer.save(figure, _diagnostic_dir(split) / "final10_stability", selected)


def _plot_confusion(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    representative_sigma = 0.2
    selected = _filter(
        records,
        environment="hidden_noise",
        mitigation="ss_ebm",
        sigma=representative_sigma,
    )
    if not selected:
        return
    figure, axes = _new_figure(columns=2, height=3.2)
    image = None
    plotted = False
    for panel_index, (axis, approach) in enumerate(zip(axes, ("merged", "cart"))):
        matrices = []
        for record in _filter(selected, approach=approach):
            if "confusion" in record.metrics:
                matrices.append(
                    np.sum(np.asarray(record.metrics["confusion"], dtype=np.float64), axis=0)
                )
        if not matrices:
            axis.set_visible(False)
            continue
        matrix = np.sum(np.stack(matrices), axis=0)
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_sum,
            out=np.zeros_like(matrix),
            where=row_sum > 0,
        )
        image = axis.imshow(
            normalized,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            interpolation="nearest",
        )
        axis.grid(False)
        axis.set_xticks(range(10), range(10))
        axis.set_yticks(range(10), range(10))
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(f"({chr(97 + panel_index)}) {APPROACH_LABELS[approach]}")
        plotted = True
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("Row-normalized frequency")
    if plotted:
        writer.save(figure, _diagnostic_dir(split) / "confusion_ss_ebm_sigma_0_2", selected)


def _plot_cart_telemetry(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    representative_sigma = 0.2
    selected = _filter(
        records,
        approach="cart",
        environment="hidden_noise",
        mitigation="ss_ebm",
        sigma=representative_sigma,
    )
    if not selected:
        return
    figure, axes = _new_figure(columns=2, height=2.9)
    mu, _ = _history_summary(selected, "mu_history")
    coverage, _ = _history_summary(selected, "registry_coverage")
    accepted, _ = _history_summary(selected, "accepted_claims")
    rejected, _ = _history_summary(selected, "rejected_claims")
    if mu.size:
        axes[0].plot(np.arange(1, len(mu) + 1), mu, color="#CC79A7", label="Proximal mu")
    if coverage.size:
        axes[0].plot(
            np.arange(1, len(coverage) + 1),
            coverage,
            color="#009E73",
            linestyle="--",
            label="Registry coverage",
        )
    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("CART signal")
    axes[0].set_title("(a) Proximal strength and coverage")
    axes[0].legend(frameon=False, loc="best")

    if accepted.size:
        axes[1].plot(
            np.arange(1, len(accepted) + 1),
            accepted,
            color="#0072B2",
            label="Accepted",
        )
    if rejected.size:
        axes[1].plot(
            np.arange(1, len(rejected) + 1),
            rejected,
            color="#D55E00",
            linestyle="--",
            label="Rejected",
        )
    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Registry claims")
    axes[1].set_title("(b) Registry verification")
    axes[1].legend(frameon=False, loc="best")
    writer.save(figure, _diagnostic_dir(split) / "cart_registry_telemetry", selected)


def _plot_selection_behavior(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    representative_sigma = 0.4
    selected = _filter(
        records,
        environment="hidden_noise",
        mitigation="ss_ebm",
        sigma=representative_sigma,
    )
    if not selected:
        return
    figure, axes = _new_figure(columns=2, height=3.0)
    plotted = False
    for panel_index, (axis, approach) in enumerate(zip(axes, ("merged", "cart"))):
        rate_sets = {"self": [], "honest": [], "attacker": []}
        subset = _filter(selected, approach=approach)
        for record in subset:
            if "selected_sources" not in record.metrics:
                continue
            sources = np.asarray(record.metrics["selected_sources"], dtype=np.int32)
            if sources.ndim != 2 or sources.size == 0:
                continue
            receivers = np.arange(sources.shape[1], dtype=np.int32)[None, :]
            self_mask = sources == receivers
            attackers = {
                int(value)
                for value in str(record.config.get("attackerIds", "")).split(",")
                if value.strip()
            }
            external_attacker = np.isin(sources, sorted(attackers)) & ~self_mask
            honest_neighbor = (sources >= 0) & ~self_mask & ~external_attacker
            rate_sets["self"].append(np.mean(self_mask, axis=1))
            rate_sets["honest"].append(np.mean(honest_neighbor, axis=1))
            rate_sets["attacker"].append(np.mean(external_attacker, axis=1))
        styles = {
            "self": ("#666666", "-", "Self"),
            "honest": ("#0072B2", "--", "Honest neighbor"),
            "attacker": ("#D55E00", "-.", "External attacker"),
        }
        for key, arrays in rate_sets.items():
            if not arrays:
                continue
            length = min(len(array) for array in arrays)
            mean = np.mean(np.stack([array[:length] for array in arrays]), axis=0)
            color, linestyle, label = styles[key]
            axis.plot(
                np.arange(1, length + 1),
                mean,
                color=color,
                linestyle=linestyle,
                label=label,
            )
            plotted = True
        if subset:
            attack_start = int(subset[0].config.get("attackHiddenStart", 0))
            axis.axvline(
                attack_start + 1,
                color="#B2182B",
                linestyle=":",
                linewidth=0.9,
                label="Attack starts",
            )
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Selection fraction")
        axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        axis.set_title(f"({chr(97 + panel_index)}) {APPROACH_LABELS[approach]}")
        _legend_if_handles(axis, frameon=False, loc="best")
    if plotted:
        writer.save(
            figure,
            _diagnostic_dir(split) / "snapshot_selection_behavior",
            selected,
        )


def _plot_runtime_memory(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = _filter(records, environment="hidden_noise", mitigation="ss_ebm")
    if not selected:
        return
    figure, axes = _new_figure(columns=2, height=2.8)
    labels = []
    runtimes = []
    runtime_errors = []
    memories = []
    memory_errors = []
    plotted_approaches = []
    for approach in ("merged", "cart"):
        subset = _filter(selected, approach=approach)
        runtime, runtime_std, runtime_count = _metric_summary(subset, "runtime_seconds")
        memory, memory_std, memory_count = _metric_summary(subset, "peak_gpu_bytes")
        if runtime_count or memory_count:
            plotted_approaches.append(approach)
            labels.append(APPROACH_LABELS[approach])
            runtimes.append(runtime if runtime_count else 0.0)
            runtime_errors.append(runtime_std if runtime_count else 0.0)
            memories.append(memory / (1024.0 ** 2) if memory_count else 0.0)
            memory_errors.append(memory_std / (1024.0 ** 2) if memory_count else 0.0)
    if not labels:
        return
    positions = np.arange(len(labels))
    colors = [
        "#56B4E9" if approach == "merged" else "#009E73"
        for approach in plotted_approaches
    ]
    hatches = [APPROACH_HATCHES[key] for key in plotted_approaches]
    bars = axes[0].bar(
        positions,
        runtimes,
        yerr=runtime_errors,
        color=colors,
        edgecolor="#222222",
        capsize=3,
        width=0.62,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Runtime (seconds)")
    axes[0].set_title("(a) Wall-clock runtime")

    bars = axes[1].bar(
        positions,
        memories,
        yerr=memory_errors,
        color=colors,
        edgecolor="#222222",
        capsize=3,
        width=0.62,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Peak GPU memory (MiB)")
    axes[1].set_title("(b) TensorFlow peak allocation")
    writer.save(figure, _diagnostic_dir(split) / "runtime_and_memory", selected)


def _plot_class_distribution(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = [record for record in records if "class_counts" in record.metrics]
    if not selected:
        return
    counts = np.asarray(selected[0].metrics["class_counts"], dtype=np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    proportions = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums > 0,
    )
    figure, (axis,) = _new_figure(width=5.6, height=3.0)
    image = axis.imshow(
        proportions,
        cmap="cividis",
        vmin=0.0,
        vmax=max(0.5, float(np.max(proportions))),
        aspect="auto",
        interpolation="nearest",
    )
    axis.grid(False)
    axis.set_xticks(range(10), range(10))
    axis.set_yticks(range(counts.shape[0]), [f"Node {index}" for index in range(counts.shape[0])])
    axis.set_xlabel("CIFAR-10 class")
    axis.set_ylabel("Logical client")
    axis.set_title("Client class distribution")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85, pad=0.03)
    colorbar.set_label("Within-client class proportion")
    writer.save(figure, _diagnostic_dir(split) / "client_class_distribution", selected[:1])


def _plot_auc_profile(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = _filter(records, environment="hidden_noise")
    if not selected:
        return
    figure, axes = _new_figure(columns=2, height=3.0)
    for panel_index, (axis, approach) in enumerate(zip(axes, ("merged", "cart"))):
        for mitigation in MITIGATION_ORDER:
            _errorbar_series(
                axis,
                selected,
                approach=approach,
                environment="hidden_noise",
                mitigation=mitigation,
                metric="learning_curve_auc",
            )
        axis.set_title(f"({chr(97 + panel_index)}) {APPROACH_LABELS[approach]}")
        axis.set_xlabel("Channel noise sigma")
        axis.set_ylabel("Normalized learning-curve AUC")
        axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        axis.set_ylim(0.0, 1.0)
        _legend_if_handles(axis, frameon=False, loc="best", ncol=2)
    writer.save(figure, _diagnostic_dir(split) / "learning_curve_auc", selected)


def _plot_calibration_gamma(
    records: list[RunRecord],
    writer: PlotWriter,
    split: str,
) -> None:
    selected = [
        record
        for record in records
        if record.phase == "calibration" and record.approach == "cart"
    ]
    gammas = sorted(
        {
            float(record.config.get("distillStrength", 0.0))
            for record in selected
        }
    )
    if len(gammas) < 2:
        return
    clean, _, clean_count = _metric_summary(
        _filter(selected, environment="clean", mitigation="none")
    )
    robust_sigmas = sorted(
        {
            record.sigma
            for record in selected
            if record.environment == "hidden_noise"
            and record.mitigation == "ss_ebm"
        }
    )
    sigma_colors = ("#0072B2", "#E69F00", "#009E73")
    figure, (axis,) = _new_figure(width=4.2, height=3.0)
    if clean_count:
        axis.axhline(
            clean,
            color="#111111",
            linewidth=1.5,
            label="Clean control (CART mu=0)",
        )
    for sigma, color in zip(robust_sigmas, sigma_colors):
        robust_values = []
        for gamma in gammas:
            gamma_records = [
                record
                for record in selected
                if abs(
                    float(record.config.get("distillStrength", 0.0)) - gamma
                )
                < 1e-12
            ]
            robust, _, robust_count = _metric_summary(
                _filter(
                    gamma_records,
                    environment="hidden_noise",
                    mitigation="ss_ebm",
                    sigma=sigma,
                )
            )
            robust_values.append(robust if robust_count else math.nan)
        axis.plot(
            gammas,
            robust_values,
            color=color,
            marker="s",
            linestyle="--",
            label=f"SS+EBM sigma={sigma:.1f}",
        )
    axis.set_xlabel("CART gamma")
    _accuracy_axis(axis)
    axis.set_title("CART calibration by noise level")
    axis.legend(frameon=False, loc="best")
    writer.save(figure, _diagnostic_dir(split) / "calibration_gamma", selected)


def _write_csv_atomic(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _summary_rows(records: list[RunRecord]) -> list[list]:
    rows = []
    for record in sorted(
        records,
        key=lambda item: (
            item.phase,
            item.split,
            item.approach,
            item.environment,
            item.sigma,
            item.mitigation,
            item.seed,
        ),
    ):
        history = np.asarray(record.metrics.get("avg_history", []), dtype=np.float64)
        last_ten = history[-10:]
        rows.append(
            [
                record.config.get("runId", ""),
                record.phase,
                record.split,
                record.approach,
                record.environment,
                record.mitigation,
                f"{float(record.config.get('distillStrength', 0.0)):.4f}",
                f"{record.sigma:.1f}",
                record.seed,
                f"{record.scalar('final_avg'):.8f}",
                f"{record.scalar('final_worst'):.8f}",
                f"{np.mean(last_ten):.8f}" if last_ten.size else "",
                f"{np.std(last_ten):.8f}" if last_ten.size else "",
                f"{record.scalar('learning_curve_auc'):.8f}",
                f"{record.scalar('runtime_seconds'):.3f}",
                int(record.scalar("peak_gpu_bytes", 0.0)),
            ]
        )
    return rows


def _paired_rows(records: list[RunRecord]) -> list[list]:
    rows = []

    def gamma(record: RunRecord) -> float:
        return float(record.config.get("distillStrength", 0.0))

    index = {
        (
            record.phase,
            record.split,
            record.approach,
            record.environment,
            record.mitigation,
            record.sigma,
            record.seed,
            gamma(record),
        ): record
        for record in records
    }
    for record in records:
        if record.approach == "cart":
            peer = index.get(
                (
                    record.phase,
                    record.split,
                    "merged",
                    record.environment,
                    record.mitigation,
                    record.sigma,
                    record.seed,
                    0.0,
                )
            )
            if peer is not None:
                rows.append(
                    [
                        "cart_minus_merged",
                        record.phase,
                        record.split,
                        record.environment,
                        record.mitigation,
                        f"{gamma(record):.4f}",
                        f"{record.sigma:.1f}",
                        record.seed,
                        f"{record.scalar('final_avg') - peer.scalar('final_avg'):.8f}",
                        record.config.get("runId", ""),
                        peer.config.get("runId", ""),
                    ]
                )
        if record.mitigation != "none":
            baseline = index.get(
                (
                    record.phase,
                    record.split,
                    record.approach,
                    record.environment,
                    "none",
                    record.sigma,
                    record.seed,
                    gamma(record),
                )
            )
            if baseline is not None:
                rows.append(
                    [
                        "defense_minus_no_mitigation",
                        record.phase,
                        record.split,
                        record.environment,
                        f"{record.approach}:{record.mitigation}",
                        f"{gamma(record):.4f}",
                        f"{record.sigma:.1f}",
                        record.seed,
                        f"{record.scalar('final_avg') - baseline.scalar('final_avg'):.8f}",
                        record.config.get("runId", ""),
                        baseline.config.get("runId", ""),
                    ]
                )
        if record.environment == "hidden_noise" and record.mitigation == "ss_ebm":
            ss = index.get(
                (
                    record.phase,
                    record.split,
                    record.approach,
                    record.environment,
                    "ss",
                    record.sigma,
                    record.seed,
                    gamma(record),
                )
            )
            if ss is not None:
                rows.append(
                    [
                        "ss_ebm_minus_ss",
                        record.phase,
                        record.split,
                        record.environment,
                        record.approach,
                        f"{gamma(record):.4f}",
                        f"{record.sigma:.1f}",
                        record.seed,
                        f"{record.scalar('final_avg') - ss.scalar('final_avg'):.8f}",
                        record.config.get("runId", ""),
                        ss.config.get("runId", ""),
                    ]
                )
    return sorted(rows)


def _write_tables(records: list[RunRecord], plot_root: Path) -> None:
    for split in sorted({record.split for record in records}):
        split_records = [record for record in records if record.split == split]
        table_dir = plot_root / "tables" / split / "cifar10"
        _write_csv_atomic(
            table_dir / "summary.csv",
            [
                "run_id",
                "phase",
                "split",
                "approach",
                "environment",
                "mitigation",
                "cart_gamma",
                "sigma",
                "seed",
                "final_average",
                "final_worst",
                "last10_mean",
                "last10_std",
                "learning_curve_auc",
                "runtime_seconds",
                "peak_gpu_bytes",
            ],
            _summary_rows(split_records),
        )
        _write_csv_atomic(
            table_dir / "paired_differences.csv",
            [
                "comparison",
                "phase",
                "split",
                "environment",
                "method",
                "cart_gamma",
                "sigma",
                "seed",
                "accuracy_difference",
                "left_run_id",
                "right_run_id",
            ],
            _paired_rows(split_records),
        )

        confirmation = [
            record for record in split_records if record.phase == "confirmation"
        ]
        lines = [
            f"# Campaign 3 Results: {split}",
            "",
            f"- Completed records: {len(split_records)}",
            f"- Confirmation records: {len(confirmation)}",
            f"- Calibration records: {len(split_records) - len(confirmation)}",
            "- Accuracy comparisons use paired seeds whenever both arms exist.",
            "- Negative raw effects are retained in `paired_differences.csv`.",
            "- Damage-recovery plots bound the named recovery metric to 0-100%; "
            "they do not replace the raw paired table.",
            "",
            "## SS+EBM Joint Condition",
            "",
            "| Approach | Sigma | Mean final accuracy | SD | Seeds |",
            "|---|---:|---:|---:|---:|",
        ]
        for approach in ("merged", "cart"):
            for sigma in NOISE_LEVELS:
                mean, std, count = _metric_summary(
                    _filter(
                        confirmation,
                        approach=approach,
                        environment="hidden_noise",
                        mitigation="ss_ebm",
                        sigma=sigma,
                    )
                )
                if count:
                    lines.append(
                        f"| {APPROACH_LABELS[approach]} | {sigma:.1f} | "
                        f"{mean:.4f} | {std:.4f} | {count} |"
                    )
        report_path = table_dir / "results_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(f".{report_path.name}.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)


def _generate_campaign3_plots_unlocked(
    *,
    mode: str = "both",
    split: str | None = None,
    result_root: Path | str = RESULT_ROOT,
    plot_root: Path | str = PLOT_ROOT,
    only_changed: bool = True,
    formats: Iterable[str] = FORMATS,
) -> dict:
    """Generate campaign-three figures and machine-readable summary tables."""
    if mode not in ("paper", "diagnostics", "both"):
        raise ValueError("mode must be 'paper', 'diagnostics', or 'both'")

    all_records = load_records(result_root, split=split)
    writer = PlotWriter(
        plot_root,
        only_changed=only_changed,
        formats=formats,
    )
    errors: list[str] = []
    if not all_records:
        writer.finish()
        return {
            "records": 0,
            "generated": [],
            "skipped": [],
            "errors": [],
            "plotRoot": str(plot_root),
        }

    _write_tables(all_records, Path(plot_root))
    splits = sorted({record.split for record in all_records})
    paper_functions: tuple[Callable, ...] = (
        _plot_component_validation,
        _plot_joint_profile,
        _plot_defense_composition,
        _plot_cart_lift,
        _plot_damage_recovery,
        _plot_convergence,
        _plot_average_vs_worst,
        _plot_class_retention,
        _plot_familiar_suite,
    )
    diagnostic_functions: tuple[Callable, ...] = (
        _plot_mitigation_heatmaps,
        _plot_seed_spread,
        _plot_stability,
        _plot_confusion,
        _plot_cart_telemetry,
        _plot_selection_behavior,
        _plot_runtime_memory,
        _plot_class_distribution,
        _plot_auc_profile,
        _plot_calibration_gamma,
    )

    for current_split in splits:
        split_records = [
            record for record in all_records if record.split == current_split
        ]
        confirmation = [
            record for record in split_records if record.phase == "confirmation"
        ]
        if mode in ("paper", "both") and confirmation:
            for function in paper_functions:
                try:
                    function(confirmation, writer, current_split)
                except Exception as error:
                    errors.append(f"{function.__name__} [{current_split}]: {error}")
        if mode in ("diagnostics", "both"):
            diagnostic_source = confirmation or split_records
            for function in diagnostic_functions:
                try:
                    function(diagnostic_source, writer, current_split)
                except Exception as error:
                    errors.append(f"{function.__name__} [{current_split}]: {error}")
            if confirmation:
                try:
                    _plot_calibration_gamma(split_records, writer, current_split)
                except Exception as error:
                    errors.append(
                        f"_plot_calibration_gamma [{current_split}]: {error}"
                    )
    writer.finish()
    return {
        "records": len(all_records),
        "generated": writer.generated,
        "skipped": writer.skipped,
        "errors": errors,
        "plotRoot": str(plot_root),
    }


def generate_campaign3_plots(
    *,
    mode: str = "both",
    split: str | None = None,
    result_root: Path | str = RESULT_ROOT,
    plot_root: Path | str = PLOT_ROOT,
    only_changed: bool = True,
    formats: Iterable[str] = FORMATS,
) -> dict:
    """Serialize plot writes so live refreshes cannot race the final pass."""
    with _PLOT_GENERATION_LOCK:
        return _generate_campaign3_plots_unlocked(
            mode=mode,
            split=split,
            result_root=result_root,
            plot_root=plot_root,
            only_changed=only_changed,
            formats=formats,
        )


def generate_campaign3_live_plots(
    *,
    split: str | None = None,
    result_root: Path | str = RESULT_ROOT,
    plot_root: Path | str = PLOT_ROOT,
) -> dict:
    """Refresh changed PNG previews after one completed experiment."""
    return generate_campaign3_plots(
        mode="both",
        split=split,
        result_root=result_root,
        plot_root=plot_root,
        only_changed=True,
        formats=("png",),
    )


if __name__ == "__main__":
    result = generate_campaign3_plots(mode="both", only_changed=False)
    print(json.dumps(result, indent=2))
