"""Incremental publication and diagnostic figures for Campaign 4 only."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

try:
    from gui.adaptive_study import (
        CAMPAIGN_ID,
        PLOT_ROOT,
        RESULT_ROOT,
        build_main_confirmation,
        config_hash,
        write_json_atomic,
    )
except ImportError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    from gui.adaptive_study import (
        CAMPAIGN_ID,
        PLOT_ROOT,
        RESULT_ROOT,
        build_main_confirmation,
        config_hash,
        write_json_atomic,
    )


PLOT_SCHEMA_VERSION = 3
HIDDEN_ATTACK_START_ROUND = 20
FORMATS = ("png", "pdf", "eps")
NOISE_LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6)
MITIGATIONS = ("none", "ss", "ss_wd", "ebm", "ss_ebm", "ss_ebm_wd")
MITIGATION_LABELS = {
    "none": "No mitigation",
    "ss": "SS",
    "ss_wd": "SS+WD",
    "ebm": "EBM",
    "ebm_wd": "Enhanced EBM",
    "ss_ebm": "SS+EBM",
    "ss_ebm_wd": "SS+Enhanced EBM",
}
COLORS = {
    "clean": "#000000",
    "none": "#6B6B6B",
    "ss": "#0072B2",
    "ss_wd": "#56B4E9",
    "ebm": "#E69F00",
    "ebm_wd": "#CC79A7",
    "ss_ebm": "#009E73",
    "ss_ebm_wd": "#F0E442",
    "merged": "#0072B2",
    "cart": "#D55E00",
    "adaptive": "#009E73",
    "static": "#CC79A7",
    "worst": "#D55E00",
}
MARKERS = {"none": "o", "ss": "s", "ss_wd": "P", "ebm": "^", "ebm_wd": "v", "ss_ebm": "D", "ss_ebm_wd": "*"}
HATCHES = ("", "///", "\\\\", "xx", "..", "++", "oo", "**")
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
SIGMA_COLORS = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00")
STAGE_LABELS = ("SS", "CART", "Training", "Transmit", "Evaluation")
SCIENTIFIC_PHASES = {
    "diagnostic",
    "static_control",
    "confirmation",
}


@dataclass(frozen=True)
class RunRecord:
    config: dict
    metrics: dict[str, np.ndarray]
    telemetry: dict[str, np.ndarray]
    run_path: Path
    metrics_path: Path
    telemetry_path: Path
    completed_at: str = ""

    @property
    def split(self):
        return str(self.config.get("split", "nonIID"))

    @property
    def condition_id(self):
        return str(self.config.get("conditionId", ""))

    @property
    def approach(self):
        return str(self.config.get("approach", "merged"))

    @property
    def environment(self):
        return str(self.config.get("environment", "unknown"))

    @property
    def mitigation(self):
        return str(self.config.get("mitigation", "none"))

    @property
    def effective_mitigation(self):
        """Mitigation label used for plotting, distinguishing weight-decay tiers.

        weightDecayCoefficient/adaptiveWeightDecayMode are orthogonal to
        `mitigation` (see the Campaign 4 research contract), so a run can report
        mitigation="ss" or "ss_ebm" while also applying weight decay - the
        accuracy-ordering matrix's "ss" tier already has adaptive weight
        decay engaged (it pairs SS for the attack with weight decay for the
        noise), so mitigation="ss" alone is not an accurate label. The
        adaptive-weight-decay controller starts every run's coefficient at
        0.0 and adjusts it online, so a static weightDecayCoefficient check
        alone misses adaptive runs entirely - both signals must be checked.
        Without this split, weight-decay runs would silently average into
        the plain SS / SS+EBM bars and lines under a misleading label.
        """
        weight_decay_active = (
            float(self.config.get("weightDecayCoefficient", 0.0)) > 0.0
            or str(self.config.get("adaptiveWeightDecayMode", "none")) != "none"
        )
        if weight_decay_active and self.mitigation in ("ss", "ss_ebm", "ebm"):
            return f"{self.mitigation}_wd"
        return self.mitigation

    @property
    def wd_active(self):
        """True when adaptive weight decay is actually engaged on this run.

        Distinct from a static calibration-sweep coefficient (see
        `is_wd_sweep_point`): this only flags the real adaptive-WD combo
        runs, so it can be folded into rerun-dedup keys alongside
        `condition_id` - otherwise an EBM-only run and an EBM+WD run that
        happen to share a conditionId collapse into a single record and
        silently drop one of them instead of appearing as distinct bars.
        """
        return str(self.config.get("adaptiveWeightDecayMode", "none")) == "adaptive"

    @property
    def is_wd_sweep_point(self):
        """True for single-seed static-weight-decay calibration points.

        These fix a specific `weightDecayCoefficient` (not adaptive) to
        sweep the coefficient itself, and were never meant to sit in the
        per-seed condition-comparison matrix alongside the real mitigation
        tiers - they inflate the bar count and pull the average toward
        whatever coefficients happened to be swept.
        """
        return (
            float(self.config.get("weightDecayCoefficient", 0.0)) > 0.0
            and not self.wd_active
        )

    @property
    def sigma(self):
        return float(self.config.get("channelNoiseSigma", 0.0))

    @property
    def seed(self):
        return int(self.config.get("seed", 0))

    @property
    def phase(self):
        return str(self.config.get("phase", "confirmation"))

    @property
    def ebm_mode(self):
        return str(self.config.get("ebmMode", "none"))

    @property
    def execution_signature(self):
        """Fields that must agree before runs are pooled statistically."""
        return (
            str(self.config.get("protocolRevision", "unknown")),
            int(self.config.get("internalMicroBatchSize", self.config.get("batchSize", 0))),
            str(self.config.get("precisionProfile", "float32")),
            bool(self.config.get("jitCompile", False)),
            str(self.config.get("optimizerStateMode", "unknown")),
            float(self.config.get("weightDecayCoefficient", 0.0)),
        )

    def scalar(self, name, default=math.nan):
        value = self.metrics.get(name)
        if value is None:
            return float(default)
        return float(np.asarray(value).reshape(-1)[0])


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


_CACHE_LOCK = threading.RLock()
_CACHE = {}
_PLOT_LOCK = threading.RLock()


def clear_record_cache():
    with _CACHE_LOCK:
        _CACHE.clear()


def _record_signature(paths):
    values = []
    for path in paths:
        stat = path.stat()
        values.extend((int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(values)


def _validated_record(run_path):
    metrics_path = run_path.with_name("metrics.npz")
    telemetry_path = run_path.with_name("telemetry.npz")
    if not metrics_path.exists() or not telemetry_path.exists():
        return None
    signature = _record_signature((run_path, metrics_path, telemetry_path))
    key = run_path.resolve()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
    record = None
    try:
        metadata = json.loads(run_path.read_text(encoding="utf-8"))
        config = metadata["config"]
        if metadata.get("status") != "completed":
            return None
        if metadata.get("campaignId") != CAMPAIGN_ID:
            return None
        if metadata.get("runId") != config.get("runId"):
            return None
        if metadata.get("configHash") != config_hash(config):
            return None
        phase = str(config.get("phase", "confirmation"))
        if phase in SCIENTIFIC_PHASES and int(config.get("nRounds", 0)) != 100:
            return None
        metrics = _load_npz(metrics_path)
        telemetry = _load_npz(telemetry_path)
        if phase in SCIENTIFIC_PHASES:
            for values in (
                metrics.get("avg_history"),
                metrics.get("worst_history"),
                telemetry.get("stress_ema"),
            ):
                if values is None or np.asarray(values).shape[0] != 100:
                    return None
        record = RunRecord(
            config=config,
            metrics=metrics,
            telemetry=telemetry,
            run_path=run_path,
            metrics_path=metrics_path,
            telemetry_path=telemetry_path,
            completed_at=str(metadata.get("completedAt", "")),
        )
        return record
    except (OSError, ValueError, KeyError):
        return None
    finally:
        with _CACHE_LOCK:
            _CACHE[key] = (signature, record)


def _dedupe_reruns(records):
    """Keep only the most-recently-completed run per (split, approach, seed,
    conditionId, wd_active).

    Rerunning a condition (e.g. after a fix) leaves the old result folder in
    place unless someone manually moves it to superseded/, so both the stale
    and the fresh run land in the active tree. Without this, every plot that
    groups by condition draws one bar/line per duplicate and every average
    is pulled toward whichever stale runs happen to be present.

    wd_active is part of the key because weightDecayCoefficient/
    adaptiveWeightDecayMode are orthogonal to conditionId (see
    RunRecord.effective_mitigation) - an EBM-only run and an EBM+WD combo
    run can share the exact same conditionId/seed/approach/split, and
    without this they'd collapse into one record, silently dropping
    whichever one is older instead of surfacing as two distinct bars.
    """
    best = {}
    for record in records:
        key = (record.split, record.approach, record.seed, record.condition_id, record.wd_active)
        current = best.get(key)
        if current is None:
            best[key] = record
            continue
        current_stamp = (current.completed_at, current.run_path.stat().st_mtime_ns)
        candidate_stamp = (record.completed_at, record.run_path.stat().st_mtime_ns)
        if candidate_stamp > current_stamp:
            best[key] = record
    return list(best.values())


def load_records(result_root=RESULT_ROOT, split=None):
    records = []
    for run_path in Path(result_root).glob("**/run.json"):
        record = _validated_record(run_path)
        if record is not None and (split is None or record.split == split):
            records.append(record)
    records = _dedupe_reruns(records)
    return sorted(
        records,
        key=lambda item: (
            item.split,
            item.approach,
            item.phase,
            item.environment,
            item.sigma,
            item.mitigation,
            item.ebm_mode,
            item.seed,
        ),
    )


def _paper_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def _mean_error(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, 0
    mean = float(np.mean(values))
    error = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, error, len(values)


def _select(records, **criteria):
    return [
        record
        for record in records
        if all(getattr(record, key) == value for key, value in criteria.items())
    ]


def _finals(records):
    return [record.scalar("final_avg") for record in records]


def _profile_slug(signature):
    revision, microbatch, precision, jit_compile, optimizer, weight_decay = signature
    revision_tail = str(revision).rsplit("-", 1)[-1]
    decay_label = f"_wd{weight_decay:g}" if weight_decay > 0.0 else ""
    raw = (
        f"{revision_tail}_mb{microbatch}_{precision}_"
        f"{'xla' if jit_compile else 'no_xla'}_{optimizer}{decay_label}"
    )
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in raw)


def _group_by_execution_profile(records):
    groups = {}
    for record in records:
        groups.setdefault(record.execution_signature, []).append(record)
    return groups


def _source_fingerprint(key, records):
    digest = hashlib.sha256()
    digest.update(str(PLOT_SCHEMA_VERSION).encode("ascii"))
    digest.update(key.encode("utf-8"))
    for record in sorted(records, key=lambda value: value.config["runId"]):
        digest.update(record.config["runId"].encode("ascii"))
        for path in (record.run_path, record.metrics_path, record.telemetry_path):
            stat = path.stat()
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _atomic_save_figure(figure, path, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=f".{fmt}")
    os.close(fd)
    temp = Path(temp_name)
    try:
        FigureCanvasAgg(figure)
        if fmt == "eps":
            # PostScript has no transparency channel; alpha-blended artists
            # (e.g. confidence-interval bands) render opaque instead, which
            # matplotlib logs every time via logging (not warnings.warn, so
            # a warnings filter can't catch it). Expected and harmless for
            # this format only -- other formats keep the logger untouched.
            ps_logger = logging.getLogger("matplotlib.backends.backend_ps")
            previous_level = ps_logger.level
            ps_logger.setLevel(logging.ERROR)
            try:
                figure.savefig(temp, format=fmt, bbox_inches="tight", dpi=300)
            finally:
                ps_logger.setLevel(previous_level)
        else:
            figure.savefig(temp, format=fmt, bbox_inches="tight", dpi=300)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class _Writer:
    def __init__(self, root, only_changed, formats):
        self.root = Path(root)
        self.only_changed = bool(only_changed)
        self.formats = tuple(formats)
        self.manifest_path = self.root / ".plot_manifest.json"
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.manifest = {"schemaVersion": PLOT_SCHEMA_VERSION, "figures": {}}
        self.generated = []
        self.skipped = []

    def figure(self, key, records, relative_base, factory):
        fingerprint = _source_fingerprint(key, records)
        paths = [self.root / f"{relative_base}.{fmt}" for fmt in self.formats]
        previous = self.manifest.get("figures", {}).get(key, {})
        if (
            self.only_changed
            and previous.get("fingerprint") == fingerprint
            and all(path.exists() for path in paths)
        ):
            self.skipped.extend(str(path) for path in paths)
            return
        figure = factory()
        if figure is None:
            return
        for path, fmt in zip(paths, self.formats):
            _atomic_save_figure(figure, path, fmt)
            self.generated.append(str(path))
        self.manifest.setdefault("figures", {})[key] = {
            "fingerprint": fingerprint,
            "sources": [record.config["runId"] for record in records],
            "paths": [str(path) for path in paths],
        }

    def finish(self):
        self.manifest["schemaVersion"] = PLOT_SCHEMA_VERSION
        write_json_atomic(self.manifest_path, self.manifest)


def _finalize_axis(axis, title, ylabel="Final average accuracy"):
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, 1.0)
    _legend_if_any(axis, ncol=2)


def _legend_if_any(axis, **kwargs):
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, frameon=False, **kwargs)


def _evidence_hierarchy(records, approach, split):
    categories = [
        ("Clean", _select(records, environment="clean", effective_mitigation="none")),
        ("Hidden\n+ SS", _select(records, environment="hidden", effective_mitigation="ss")),
        ("Noise\n+ EBM", _select(records, environment="noise", effective_mitigation="ebm")),
        ("Joint\n+ SS+EBM", _select(records, environment="hidden_noise", effective_mitigation="ss_ebm")),
        ("Joint\n+ SS+EBM+WD", _select(records, environment="hidden_noise", effective_mitigation="ss_ebm_wd")),
        ("Joint\n+ SS", _select(records, environment="hidden_noise", effective_mitigation="ss")),
        ("Joint\n+ SS+WD", _select(records, environment="hidden_noise", effective_mitigation="ss_wd")),
        ("Joint\n+ EBM", _select(records, environment="hidden_noise", effective_mitigation="ebm")),
        ("Joint\n+ none", _select(records, environment="hidden_noise", effective_mitigation="none")),
    ]
    present = [(label, group) for label, group in categories if group]
    if not present:
        return None
    means, errors = zip(*[_mean_error(_finals(group))[:2] for _, group in present])
    fig = Figure(figsize=(7.1, 3.2))
    ax = fig.subplots()
    positions = np.arange(len(present))
    bars = ax.bar(
        positions,
        means,
        yerr=errors,
        capsize=3,
        color=[COLORS["clean"]] + [COLORS["ss_ebm"]] * (len(present) - 1),
        edgecolor="black",
        linewidth=0.7,
    )
    for index, bar in enumerate(bars):
        bar.set_hatch(HATCHES[index % len(HATCHES)])
    ax.set_xticks(positions, [label for label, _ in present])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final average accuracy")
    ax.set_title(f"Evidence hierarchy: {approach.upper()} ({split})")
    ax.text(
        0.99,
        0.02,
        "Noisy categories summarize sigma=0.2-0.6; error bars are SD across available runs.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
    )
    return fig


def _line_by_sigma(axis, records, environment, mitigation, label, color, marker):
    xs, means, errors = [], [], []
    for sigma in NOISE_LEVELS:
        group = [
            record
            for record in records
            if record.environment == environment
            and record.effective_mitigation == mitigation
            and abs(record.sigma - sigma) < 1e-9
        ]
        if group:
            mean, error, _ = _mean_error(_finals(group))
            xs.append(sigma)
            means.append(mean)
            errors.append(error)
    if xs:
        axis.errorbar(
            xs,
            means,
            yerr=errors,
            label=label,
            color=color,
            marker=marker,
            capsize=3,
        )


def _joint_challenge(records, approach, split):
    if not records:
        return None
    fig = Figure(figsize=(5.6, 3.5))
    ax = fig.subplots()
    _line_by_sigma(ax, records, "noise", "ebm", "Noise only + EBM", COLORS["ebm"], "^")
    _line_by_sigma(
        ax, records, "hidden_noise", "ss_ebm", "Joint + SS+EBM", COLORS["ss_ebm"], "D"
    )
    hidden = _select(records, environment="hidden", mitigation="ss")
    clean = _select(records, environment="clean", mitigation="none")
    if hidden:
        ax.axhline(np.mean(_finals(hidden)), color=COLORS["ss"], linestyle="--", label="Hidden only + SS")
    if clean:
        ax.axhline(np.mean(_finals(clean)), color="black", linestyle=":", label="Clean reference")
    ax.set_xlabel("Channel-noise sigma")
    ax.set_xticks(NOISE_LEVELS)
    _finalize_axis(ax, f"Joint challenge profile: {approach.upper()} ({split})")
    return fig


def _defense_composition(records, approach, split):
    joint = [record for record in records if record.environment == "hidden_noise"]
    if not joint:
        return None
    fig = Figure(figsize=(5.6, 3.5))
    ax = fig.subplots()
    for mitigation in MITIGATIONS:
        _line_by_sigma(
            ax,
            joint,
            "hidden_noise",
            mitigation,
            MITIGATION_LABELS[mitigation],
            COLORS[mitigation],
            MARKERS[mitigation],
        )
    clean = _select(records, environment="clean", mitigation="none")
    if clean:
        ax.axhline(np.mean(_finals(clean)), color="black", linestyle=":", label="Clean reference")
    ax.set_xlabel("Channel-noise sigma")
    ax.set_xticks(NOISE_LEVELS)
    _finalize_axis(ax, f"Joint defense composition: {approach.upper()} ({split})")
    return fig


def _avg_worst(records, approach, split):
    joint = [record for record in records if record.environment == "hidden_noise" and record.effective_mitigation == "ss_ebm"]
    if not joint:
        return None
    fig = Figure(figsize=(5.6, 3.5))
    ax = fig.subplots()
    for metric, label, color, marker in (
        ("final_avg", "Ring average", COLORS["ss_ebm"], "D"),
        ("final_worst", "Worst node", COLORS["worst"], "v"),
    ):
        xs, means, errors = [], [], []
        for sigma in NOISE_LEVELS:
            group = [record for record in joint if abs(record.sigma - sigma) < 1e-9]
            if group:
                mean, error, _ = _mean_error([record.scalar(metric) for record in group])
                xs.append(sigma)
                means.append(mean)
                errors.append(error)
        ax.errorbar(xs, means, yerr=errors, label=label, color=color, marker=marker, capsize=3)
    ax.set_xlabel("Channel-noise sigma")
    ax.set_xticks(NOISE_LEVELS)
    _finalize_axis(ax, f"Joint SS+EBM average versus worst node: {approach.upper()} ({split})")
    return fig


def _seed_profiles(records, approach, split):
    joint = [record for record in records if record.environment == "hidden_noise" and record.effective_mitigation == "ss_ebm"]
    if not joint:
        return None
    fig = Figure(figsize=(5.6, 3.5))
    ax = fig.subplots()
    for index, seed in enumerate(sorted({record.seed for record in joint})):
        values = sorted((record.sigma, record.scalar("final_avg")) for record in joint if record.seed == seed)
        ax.plot(
            [value[0] for value in values],
            [value[1] for value in values],
            marker=("o", "s", "^")[index % 3],
            label=f"seed {seed}",
        )
    ax.set_xlabel("Channel-noise sigma")
    ax.set_xticks(NOISE_LEVELS)
    _finalize_axis(ax, f"Joint SS+EBM seed profiles: {approach.upper()} ({split})")
    return fig


def _mean_history(group, metric="avg_history"):
    histories = [
        np.asarray(record.metrics[metric], dtype=np.float64)
        for record in group
        if metric in record.metrics
    ]
    if not histories:
        return None, None
    length = min(len(history) for history in histories)
    matrix = np.stack([history[:length] for history in histories])
    mean = np.nanmean(matrix, axis=0)
    spread = (
        np.nanstd(matrix, axis=0, ddof=1)
        if len(matrix) > 1
        else np.zeros(length, dtype=np.float64)
    )
    return mean, spread


def _learning_curves(records, approach, split):
    lines = [
        ("Clean", _select(records, environment="clean", mitigation="none"), "#000000", "-"),
        ("Hidden + SS", _select(records, environment="hidden", mitigation="ss"), COLORS["ss"], "--"),
    ]
    for index, sigma in enumerate(NOISE_LEVELS):
        lines.append(
            (
                f"Joint SS+Enhanced EBM, sigma={sigma:.1f}",
                [
                    record
                    for record in records
                    if record.environment == "hidden_noise"
                    and record.effective_mitigation == "ss_ebm_wd"
                    and abs(record.sigma - sigma) < 1e-9
                ],
                SIGMA_COLORS[index],
                "-",
            )
        )
    present = []
    for label, group, color, style in lines:
        mean, spread = _mean_history(group)
        if mean is not None:
            present.append((label, mean, spread, color, style, len(group)))
    if not present:
        return None
    fig = Figure(figsize=(7.1, 3.7))
    ax = fig.subplots()
    for label, mean, spread, color, style, count in present:
        rounds = np.arange(1, len(mean) + 1)
        ax.plot(rounds, mean, color=color, linestyle=style, label=label)
        if count > 1:
            ax.fill_between(
                rounds,
                np.clip(mean - spread, 0.0, 1.0),
                np.clip(mean + spread, 0.0, 1.0),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
    ax.axvline(20, color="#777777", linestyle=":", linewidth=1.0, label="Hidden attack starts")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Average ring accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Learning curves: {approach.upper()} ({split})")
    ax.legend(frameon=False, ncol=2)
    return fig


def _noise_robustness(records, approach, split):
    relevant = [
        record
        for record in records
        if (record.environment, record.effective_mitigation)
        in (("noise", "ebm"), ("hidden_noise", "ss_ebm"))
    ]
    if not relevant:
        return None
    fig = Figure(figsize=(7.1, 3.3))
    axes = fig.subplots(1, 2)
    paths = (
        ("noise", "ebm", "Noise + EBM", COLORS["ebm"], "^"),
        ("hidden_noise", "ss_ebm", "Joint + SS+EBM", COLORS["ss_ebm"], "D"),
    )
    for axis, metric, ylabel in (
        (axes[0], "final_avg", "Final average accuracy"),
        (axes[1], "learning_curve_auc", "Learning-curve AUC"),
    ):
        for environment, mitigation, label, color, marker in paths:
            xs, means, errors = [], [], []
            for sigma in NOISE_LEVELS:
                group = [
                    record
                    for record in relevant
                    if record.environment == environment
                    and record.effective_mitigation == mitigation
                    and abs(record.sigma - sigma) < 1e-9
                ]
                if group:
                    mean, error, _ = _mean_error(
                        [record.scalar(metric) for record in group]
                    )
                    xs.append(sigma)
                    means.append(mean)
                    errors.append(error)
            if xs:
                axis.errorbar(
                    xs,
                    means,
                    yerr=errors,
                    color=color,
                    marker=marker,
                    capsize=3,
                    label=label,
                )
        axis.set_xticks(NOISE_LEVELS)
        axis.set_xlabel("Channel-noise sigma")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Noise robustness: {approach.upper()} ({split})", fontsize=10)
    fig.tight_layout()
    return fig


def _class_retention(records, approach, split):
    joint = [record for record in records if record.environment == "hidden_noise" and record.effective_mitigation == "ss_ebm"]
    if not joint:
        return None
    matrix = np.full((len(NOISE_LEVELS), 10), np.nan, dtype=np.float64)
    for row, sigma in enumerate(NOISE_LEVELS):
        values = []
        for record in joint:
            if abs(record.sigma - sigma) < 1e-9 and "final_class_accuracy" in record.metrics:
                values.append(np.nanmean(record.metrics["final_class_accuracy"], axis=0))
        if values:
            matrix[row] = np.nanmean(values, axis=0)
    if not np.any(np.isfinite(matrix)):
        return None
    fig = Figure(figsize=(7.1, 3.3))
    ax = fig.subplots()
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(NOISE_LEVELS)), [f"{value:.1f}" for value in NOISE_LEVELS])
    ax.set_xticks(range(10), CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Channel-noise sigma")
    ax.set_title(f"Joint SS+EBM class retention: {approach.upper()} ({split})")
    fig.colorbar(image, ax=ax, label="Mean per-class accuracy", fraction=0.03, pad=0.02)
    return fig


def _paired_delta(records, left_approach, right_approach, environment, mitigation):
    by_key = {}
    for record in records:
        if record.environment != environment or record.effective_mitigation != mitigation:
            continue
        key = (
            record.seed,
            record.sigma,
            record.ebm_mode,
            record.phase,
            record.execution_signature,
        )
        by_key.setdefault(key, {})[record.approach] = record.scalar("final_avg")
    return [
        (key, values[right_approach] - values[left_approach])
        for key, values in by_key.items()
        if left_approach in values and right_approach in values
    ]


def _cart_lift(records, split):
    pairs = _paired_delta(records, "merged", "cart", "hidden_noise", "ss_ebm")
    if not pairs:
        return None
    fig = Figure(figsize=(5.6, 3.5))
    ax = fig.subplots()
    for seed in sorted({key[0] for key, _ in pairs}):
        values = sorted((key[1], delta) for key, delta in pairs if key[0] == seed)
        ax.plot([v[0] for v in values], [v[1] for v in values], marker="o", label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Channel-noise sigma")
    ax.set_ylabel("CART - Merged final accuracy")
    ax.set_xticks(NOISE_LEVELS)
    ax.set_title(f"Paired CART lift for joint SS+EBM ({split})")
    ax.legend(frameon=False)
    return fig


def _static_adaptive(records, approach, split):
    relevant = [
        record
        for record in records
        if record.approach == approach
        and record.environment in ("noise", "hidden_noise")
        and record.mitigation in ("ebm", "ss_ebm")
        and record.ebm_mode in ("static", "adaptive")
    ]
    pairs = {}
    for record in relevant:
        key = (
            record.environment,
            record.mitigation,
            record.sigma,
            record.seed,
            record.execution_signature,
        )
        pairs.setdefault(key, {})[record.ebm_mode] = record.scalar("final_avg")
    paired = [(key, value["adaptive"] - value["static"]) for key, value in pairs.items() if {"static", "adaptive"} <= set(value)]
    if not paired:
        return None
    fig = Figure(figsize=(6.5, 3.5))
    ax = fig.subplots()
    for environment, mitigation, label, color, marker in (
        ("noise", "ebm", "Noise + EBM", COLORS["ebm"], "^"),
        ("hidden_noise", "ss_ebm", "Joint + SS+EBM", COLORS["ss_ebm"], "D"),
    ):
        xs, means, errors = [], [], []
        for sigma in NOISE_LEVELS:
            values = [delta for key, delta in paired if key[0] == environment and key[1] == mitigation and abs(key[2] - sigma) < 1e-9]
            if values:
                mean, error, _ = _mean_error(values)
                xs.append(sigma)
                means.append(mean)
                errors.append(error)
        if xs:
            ax.errorbar(xs, means, yerr=errors, color=color, marker=marker, capsize=3, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(NOISE_LEVELS)
    ax.set_xlabel("Channel-noise sigma")
    ax.set_ylabel("Adaptive - static final accuracy")
    ax.set_title(f"Paired adaptive EBM effect: {approach.upper()} ({split})")
    ax.legend(frameon=False)
    return fig


def _telemetry_panel(records, approach, split):
    relevant = [
        record
        for record in records
        if record.approach == approach
        and record.environment == "hidden_noise"
        and record.mitigation == "ss_ebm"
        and record.ebm_mode == "adaptive"
    ]
    if not relevant:
        return None
    fig = Figure(figsize=(7.1, 8.0))
    axes = fig.subplots(3, 3)
    panels = (
        ("model_norm", "Model norm"),
        ("outgoing_noise_norm", "Measured link-noise norm"),
        ("outgoing_relative_noise", "Measured relative link noise"),
        ("consensus_innovation", "Consensus innovation"),
        ("stress_ema", "Adaptive stress EMA"),
        ("ebm_requested_coefficient", "Requested EBM coefficient"),
        ("ebm_coefficient", "Applied EBM coefficient"),
        ("active_ebm_ratio", "Applied EBM/base ratio"),
        ("clip_fraction", "Gradient clipping fraction"),
    )
    for axis, (metric, title) in zip(axes.flat, panels):
        for sigma in sorted({record.sigma for record in relevant}):
            runs = [record for record in relevant if abs(record.sigma - sigma) < 1e-9 and metric in record.telemetry]
            if not runs:
                continue
            histories = []
            for record in runs:
                values = np.asarray(record.telemetry[metric], dtype=np.float64)
                while values.ndim > 1:
                    values = np.nanmean(values, axis=-1)
                histories.append(values)
            length = min(len(value) for value in histories)
            axis.plot(np.arange(1, length + 1), np.nanmean([value[:length] for value in histories], axis=0), label=f"sigma={sigma:.1f}")
        axis.set_title(title)
        axis.set_xlabel("Round")
        _legend_if_any(axis, ncol=2)
    fig.suptitle(f"Campaign 4 telemetry: {approach.upper()} ({split})", fontsize=10)
    fig.tight_layout()
    return fig


def _gradient_panel(records, approach, split):
    relevant = [
        record
        for record in records
        if record.approach == approach
        and record.environment == "hidden_noise"
        and record.mitigation == "ss_ebm"
        and record.ebm_mode == "adaptive"
    ]
    if not relevant:
        return None
    fig = Figure(figsize=(7.1, 3.1))
    axes = fig.subplots(1, 3)
    for axis, metric, title in (
        (axes[0], "base_gradient_norm", "Base gradient norm"),
        (axes[1], "regularizer_gradient_norm", "EBM regularizer gradient norm"),
        (axes[2], "preclip_gradient_norm", "Combined pre-clip norm"),
    ):
        for sigma in sorted({record.sigma for record in relevant}):
            runs = [
                record
                for record in relevant
                if abs(record.sigma - sigma) < 1e-9 and metric in record.telemetry
            ]
            histories = []
            for record in runs:
                values = np.asarray(record.telemetry[metric], dtype=np.float64)
                while values.ndim > 1:
                    values = np.nanmean(values, axis=-1)
                histories.append(values)
            if histories:
                length = min(len(value) for value in histories)
                axis.plot(
                    np.arange(1, length + 1),
                    np.nanmean([value[:length] for value in histories], axis=0),
                    label=f"sigma={sigma:.1f}",
                )
        axis.set_title(title)
        axis.set_xlabel("Round")
        _legend_if_any(axis, fontsize=6)
    fig.suptitle(f"EBM gradient diagnostics: {approach.upper()} ({split})", fontsize=10)
    fig.tight_layout()
    return fig


def _cart_panel(records, approach, split):
    if approach != "cart":
        return None
    relevant = [
        record
        for record in records
        if record.approach == "cart"
        and record.environment == "hidden_noise"
        and record.mitigation == "ss_ebm"
        and record.ebm_mode == "adaptive"
    ]
    if not relevant:
        return None
    fig = Figure(figsize=(7.1, 5.2))
    axes = fig.subplots(2, 2)
    for axis, metric, title in (
        (axes[0, 0], "cart_gap", "Verified class gap"),
        (axes[0, 1], "cart_mu", "Applied CART mu"),
        (axes[1, 0], "registry_coverage_node", "Registry coverage"),
    ):
        for sigma in sorted({record.sigma for record in relevant}):
            runs = [
                record
                for record in relevant
                if abs(record.sigma - sigma) < 1e-9 and metric in record.telemetry
            ]
            histories = []
            for record in runs:
                values = np.asarray(record.telemetry[metric], dtype=np.float64)
                while values.ndim > 1:
                    values = np.nanmean(values, axis=-1)
                histories.append(values)
            if histories:
                length = min(len(value) for value in histories)
                axis.plot(
                    np.arange(1, length + 1),
                    np.nanmean([value[:length] for value in histories], axis=0),
                    label=f"sigma={sigma:.1f}",
                )
        axis.set_title(title)
        axis.set_xlabel("Round")
        _legend_if_any(axis, fontsize=6)
    claims_axis = axes[1, 1]
    for metric, label, color in (
        ("registry_accepted", "Accepted", COLORS["ss_ebm"]),
        ("registry_rejected", "Rejected", COLORS["worst"]),
    ):
        histories = []
        for record in relevant:
            if metric in record.telemetry:
                values = np.asarray(record.telemetry[metric], dtype=np.float64)
                while values.ndim > 1:
                    values = np.nansum(values, axis=-1)
                histories.append(values)
        if histories:
            length = min(len(value) for value in histories)
            claims_axis.plot(
                np.arange(1, length + 1),
                np.nanmean([value[:length] for value in histories], axis=0),
                color=color,
                label=label,
            )
    claims_axis.set_title("Registry verification activity")
    claims_axis.set_xlabel("Round")
    _legend_if_any(claims_axis)
    fig.suptitle(f"CART diagnostics ({split})", fontsize=10)
    fig.tight_layout()
    return fig


def _selection_panel(records, approach, split):
    relevant = [record for record in records if record.approach == approach and record.config.get("snapshotSelection")]
    if not relevant:
        return None
    values = []
    for record in relevant:
        selected = record.telemetry.get("selected_sources")
        if selected is None:
            continue
        attacker_ids = {int(value) for value in str(record.config.get("attackerIds", "")).split(",") if value}
        active = record.telemetry.get("attack_active", np.zeros_like(selected, dtype=bool))
        active_rounds = np.any(active, axis=1, keepdims=True)
        active_visits = np.broadcast_to(active_rounds, selected.shape)
        denominator = max(1, int(np.sum(active_visits)))
        selected_attackers = np.isin(selected, list(attacker_ids)) & active_visits
        fallback = np.asarray(
            record.telemetry.get("selection_fallback", np.zeros_like(selected)),
            dtype=bool,
        )
        values.append(
            (
                record.sigma,
                float(np.sum(selected_attackers) / denominator),
                float(np.sum(fallback & active_visits) / denominator),
            )
        )
    if not values:
        return None
    fig = Figure(figsize=(5.6, 3.5))
    ax = fig.subplots()
    for index, label in ((1, "Attacker selected"), (2, "SS fallback")):
        xs, ys = [], []
        for sigma in sorted({value[0] for value in values}):
            xs.append(sigma)
            ys.append(np.mean([value[index] for value in values if value[0] == sigma]))
        ax.plot(xs, ys, marker="o" if index == 1 else "s", label=label)
    ax.set_xlabel("Channel-noise sigma (0 = attack only)")
    ax.set_ylabel("Fraction of node visits")
    ax.set_ylim(0, 1)
    ax.set_title(f"Snapshot Selection diagnostics: {approach.upper()} ({split})")
    ax.legend(frameon=False)
    return fig


def _runtime_panel(records, approach, split):
    relevant = [record for record in records if record.approach == approach]
    if not relevant:
        return None
    fig = Figure(figsize=(7.1, 5.5))
    axes = fig.subplots(2, 2)
    paths = (
        ("noise", "ebm", "Noise + EBM", COLORS["ebm"], "^"),
        ("hidden_noise", "ss_ebm", "Joint + SS+EBM", COLORS["ss_ebm"], "D"),
    )
    for axis, metric, scale, ylabel in (
        (axes[0, 0], "runtime_seconds", 1.0 / 60.0, "Runtime (minutes)"),
        (axes[0, 1], "peak_gpu_bytes", 1.0 / (1024.0**3), "Peak GPU allocation (GiB)"),
    ):
        for environment, mitigation, label, color, marker in paths:
            xs, means = [], []
            for sigma in NOISE_LEVELS:
                group = [
                    record
                    for record in relevant
                    if record.environment == environment
                    and record.mitigation == mitigation
                    and record.ebm_mode == "adaptive"
                    and abs(record.sigma - sigma) < 1e-9
                ]
                values = [record.scalar(metric) * scale for record in group]
                values = [value for value in values if np.isfinite(value)]
                if values:
                    xs.append(sigma)
                    means.append(float(np.mean(values)))
            if xs:
                axis.plot(xs, means, color=color, marker=marker, label=label)
        axis.set_xticks(NOISE_LEVELS)
        axis.set_xlabel("Channel-noise sigma")
        axis.set_ylabel(ylabel)
        _legend_if_any(axis)

    staged = [
        record
        for record in relevant
        if "stage_runtime_seconds" in record.metrics
        and record.environment == "hidden_noise"
        and record.mitigation == "ss_ebm"
        and record.ebm_mode == "adaptive"
    ]
    stage_axis = axes[1, 0]
    if staged:
        means = []
        labels = []
        for sigma in NOISE_LEVELS:
            values = [
                np.sum(np.asarray(record.metrics["stage_runtime_seconds"], dtype=np.float64), axis=0)
                for record in staged
                if abs(record.sigma - sigma) < 1e-9
            ]
            if values:
                means.append(np.mean(values, axis=0) / 60.0)
                labels.append(f"{sigma:.1f}")
        if means:
            matrix = np.asarray(means)
            bottoms = np.zeros(len(matrix))
            positions = np.arange(len(matrix))
            for index, stage in enumerate(STAGE_LABELS):
                stage_axis.bar(
                    positions,
                    matrix[:, index],
                    bottom=bottoms,
                    label=stage,
                    color=("#56B4E9", "#CC79A7", "#009E73", "#E69F00", "#6B6B6B")[index],
                    edgecolor="black",
                    linewidth=0.4,
                )
                bottoms += matrix[:, index]
            stage_axis.set_xticks(positions, labels)
            stage_axis.set_xlabel("Channel-noise sigma")
            stage_axis.set_ylabel("Summed stage time (minutes)")
            stage_axis.legend(frameon=False, fontsize=6, ncol=2)

    clip_axis = axes[1, 1]
    for environment, mitigation, label, color, marker in paths:
        xs, means = [], []
        for sigma in NOISE_LEVELS:
            values = []
            for record in relevant:
                if (
                    record.environment == environment
                    and record.mitigation == mitigation
                    and record.ebm_mode == "adaptive"
                    and abs(record.sigma - sigma) < 1e-9
                    and "clip_fraction" in record.telemetry
                ):
                    values.append(float(np.nanmean(record.telemetry["clip_fraction"])))
            if values:
                xs.append(sigma)
                means.append(float(np.mean(values)))
        if xs:
            clip_axis.plot(xs, means, color=color, marker=marker, label=label)
    clip_axis.set_xticks(NOISE_LEVELS)
    clip_axis.set_xlabel("Channel-noise sigma")
    clip_axis.set_ylabel("Gradient clip fraction")
    clip_axis.set_ylim(0, 1)
    _legend_if_any(clip_axis)
    fig.suptitle(f"Runtime and resource diagnostics: {approach.upper()} ({split})", fontsize=10)
    fig.tight_layout()
    return fig


def _completeness(records, split):
    expected = [config for config in build_main_confirmation() if config["split"] == split]
    completed = {
        (record.approach, record.config.get("conditionId"), record.seed)
        for record in records
        if record.phase == "confirmation"
        and not record.config.get("performanceProfileFallback", False)
    }
    condition_keys = sorted({(config["approach"], config["conditionId"]) for config in expected})
    seeds = sorted({config["seed"] for config in expected})
    matrix = np.zeros((len(condition_keys), len(seeds)), dtype=np.float32)
    lookup = {(config["approach"], config["conditionId"], config["seed"]): config for config in expected}
    for row, (approach, condition) in enumerate(condition_keys):
        for column, seed in enumerate(seeds):
            matrix[row, column] = (approach, condition, seed) in completed
    fig = Figure(figsize=(6.2, max(4.0, 0.16 * len(condition_keys))))
    ax = fig.subplots()
    ax.imshow(matrix, vmin=0, vmax=1, cmap=mpl.colors.ListedColormap(["#D55E00", "#009E73"]), aspect="auto")
    ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
    ax.set_yticks(range(len(condition_keys)), [f"{approach}: {condition}" for approach, condition in condition_keys], fontsize=5)
    ax.set_xlabel("Seed")
    ax.set_title(f"Campaign 4 confirmation completeness ({split})")
    return fig


def _write_summary_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".csv")
    os.close(fd)
    try:
        with open(temp_name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "run_id",
                    "phase",
                    "split",
                    "approach",
                    "environment",
                    "mitigation",
                    "ebm_mode",
                    "protocol_revision",
                    "internal_microbatch",
                    "precision_profile",
                    "jit_compile",
                    "optimizer_state_mode",
                    "sigma",
                    "seed",
                    "final_average_accuracy",
                    "final_worst_accuracy",
                    "learning_curve_auc",
                    "runtime_seconds",
                ]
            )
            for record in records:
                writer.writerow(
                    [
                        record.config["runId"],
                        record.phase,
                        record.split,
                        record.approach,
                        record.environment,
                        record.mitigation,
                        record.ebm_mode,
                        record.execution_signature[0],
                        record.execution_signature[1],
                        record.execution_signature[2],
                        record.execution_signature[3],
                        record.execution_signature[4],
                        f"{record.sigma:.1f}",
                        record.seed,
                        record.scalar("final_avg"),
                        record.scalar("final_worst"),
                        record.scalar("learning_curve_auc"),
                        record.scalar("runtime_seconds"),
                    ]
                )
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _seed_matrix_groups(records):
    """Group completed records by (split, approach, seed) for the per-seed,
    legacy-style comparison figures (every condition for that approach+seed
    bundled into one set of plots, mirroring plots2/plotGui.py's layout).

    Excludes single-seed static-weight-decay calibration sweep points
    (RunRecord.is_wd_sweep_point) - those exist to tune a coefficient, not
    to be compared as a mitigation tier, and pooling them in here inflates
    the bar count and pulls the average toward whichever coefficients
    happened to be swept.
    """
    groups = {}
    for record in records:
        if record.is_wd_sweep_point:
            continue
        groups.setdefault((record.split, record.approach, record.seed), []).append(record)
    return groups


_SEED_ENV_ORDER = {"clean": 0, "hidden": 1, "noise": 2, "hidden_noise": 3}
_SEED_MIT_ORDER = {
    "none": 0,
    "ss": 1,
    "ss_wd": 2,
    "ebm": 3,
    "ebm_wd": 4,
    "ss_ebm": 5,
    "ss_ebm_wd": 6,
}


def _seed_condition_key(record):
    return (
        _SEED_ENV_ORDER.get(record.environment, 9),
        _SEED_MIT_ORDER.get(record.effective_mitigation, 9),
        record.sigma,
    )


def _seed_condition_label(record):
    mitigation_label = MITIGATION_LABELS.get(record.effective_mitigation, record.effective_mitigation)
    if record.environment == "clean":
        return "Clean\n(no attack, no noise)"
    if record.environment == "hidden":
        return f"Hidden Attack\n+ {mitigation_label}"
    if record.environment == "noise":
        # environment=noise + mitigation="none" with adaptive weight decay
        # engaged is the "weight-decay-only" tier: effective_mitigation stays
        # "none" (only ss/ss_ebm gain a _wd suffix), so label it explicitly.
        wd_active = (
            float(record.config.get("weightDecayCoefficient", 0.0)) > 0.0
            or str(record.config.get("adaptiveWeightDecayMode", "none")) != "none"
        )
        label = "WD" if (record.mitigation == "none" and wd_active) else mitigation_label
        return f"Noise σ={record.sigma:g}\n+ {label}"
    return f"Hidden+Noise σ={record.sigma:g}\n+ {mitigation_label}"


def _seed_condition_color(record, palette, index):
    if record.environment == "clean":
        return COLORS["clean"]
    return palette[index % len(palette)]


def _seed_palette(n):
    cmap_name = "tab20" if n <= 20 else "hsv"
    cmap = mpl.colormaps[cmap_name].resampled(max(n, 1))
    return [cmap(i) for i in range(n)]


def _is_core_tier(record):
    """True for a record in one of the manuscript condition families:
    Clean, Attack+SS, Noise+Enhanced EBM, Hidden+Noise+SS,
    Hidden+Noise+Enhanced EBM, Hidden+Noise+SS+Enhanced EBM (each
    noise-bearing tier restricted to sigma in {0.2, 0.4, 0.6}). Used to
    drop every other diagnostic tier (SS+WD, no-mitigation baselines,
    sigma=0.3/0.5, plain EBM without weight decay, ...) from the
    comparison figures (final-accuracy bar, experiments line, grid) so
    only the curated set is shown. Plain-EBM data (noise+ebm,
    hidden_noise+ebm, hidden_noise+ss_ebm) stays on disk and is loadable
    for future plots -- this filter only controls what renders here.
    """
    sigma = round(float(record.sigma), 2)
    if record.environment == "clean":
        return record.effective_mitigation == "none" and not record.is_wd_sweep_point
    if record.environment == "hidden":
        return record.effective_mitigation == "ss"
    if sigma not in (0.2, 0.4, 0.6):
        return False
    if record.environment == "noise":
        return record.effective_mitigation == "ebm_wd"
    if record.environment == "hidden_noise":
        return record.effective_mitigation in ("ss", "ebm_wd", "ss_ebm_wd")
    return False


def _seed_ordered_conditions(records):
    return sorted((record for record in records if _is_core_tier(record)), key=_seed_condition_key)


def _seed_title_fragment(seed):
    """' | seed=2025' for a real per-seed plot; empty for cross-seed
    overall-average plots, which pass an aggregate label string (e.g.
    'avg of 4 seeds') in place of an integer seed -- that label is not
    shown in the title."""
    if isinstance(seed, int):
        return f" | seed={seed}"
    return ""


def _seed_final_accuracy_bar(records, approach, seed, split):
    ordered = _seed_ordered_conditions(records)
    if not ordered:
        return None
    labels = [_seed_condition_label(record) for record in ordered]
    finals = [record.scalar("final_avg") for record in ordered]
    palette = _seed_palette(len(ordered))
    colors = [_seed_condition_color(record, palette, index) for index, record in enumerate(ordered)]

    fig = Figure(figsize=(max(9.0, len(ordered) * 0.85), 5.5))
    ax = fig.subplots()
    bars = ax.bar(range(len(ordered)), finals, color=colors, width=0.6, edgecolor="black", linewidth=0.4)
    for bar, value in zip(bars, finals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )
    ax.set_xticks(range(len(ordered)), labels, rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("Final average accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Final Accuracy | CIFAR-10 | {approach.upper()}{_seed_title_fragment(seed)} ({split})")
    return fig


def _noise_env_tier(record):
    """Tier key for an environment="noise" (no-attack) record.

    `effective_mitigation` only appends a `_wd` suffix for ss/ss_ebm/ebm
    (see its docstring), so a plain mitigation="none" run with weight decay
    engaged still reports effective_mitigation="none" - indistinguishable
    from a true no-mitigation run without checking the weight-decay fields
    directly. Without this split, "no mitigation", "WD only", "EBM alone"
    and "EBM+WD" all silently pooled into one misleadingly-labeled bar.
    """
    weight_decay_active = (
        float(record.config.get("weightDecayCoefficient", 0.0)) > 0.0
        or str(record.config.get("adaptiveWeightDecayMode", "none")) != "none"
    )
    if record.mitigation == "ebm":
        return "ebm_wd" if weight_decay_active else "ebm"
    return "wd_only" if weight_decay_active else "none_noise"


_HIDDEN_NOISE_ABLATION_TIERS = (
    ("ss", "SS only"),
    ("ss_wd", "SS+WD"),
    ("ss_ebm", "SS+EBM"),
    ("ss_ebm_wd", "SS+Enhanced EBM"),
)
_NOISE_ABLATION_TIERS = (
    ("wd_only", "Noise + WD only"),
    ("ebm", "EBM"),
    ("ebm_wd", "Enhanced EBM"),
)


def _ablation_values(records, sigmas, tiers, *, environment):
    values = {}
    for sigma in sigmas:
        if environment == "hidden_noise":
            for tier_key, _ in tiers:
                group = _select(records, environment="hidden_noise", effective_mitigation=tier_key)
                group = [record for record in group if abs(record.sigma - sigma) < 1e-9]
                if group:
                    values[(sigma, tier_key)] = _mean_error(_finals(group))[0]
        else:
            noise_records = [
                record
                for record in records
                if record.environment == "noise" and abs(record.sigma - sigma) < 1e-9
            ]
            for tier_key, _ in tiers:
                group = [record for record in noise_records if _noise_env_tier(record) == tier_key]
                if group:
                    values[(sigma, tier_key)] = _mean_error(_finals(group))[0]
    return values


_ABLATION_CORE_SIGMAS = (0.2, 0.4, 0.6)


def _render_ablation_groups(records, approach, seed, split, *, environment, tiers, subtitle, large=False):
    sigmas = sorted(
        {
            round(record.sigma, 2)
            for record in records
            if record.sigma > 0.0 and round(record.sigma, 2) in _ABLATION_CORE_SIGMAS
        }
    )
    if not sigmas:
        return None
    values = _ablation_values(records, sigmas, tiers, environment=environment)
    bar_keys = [key for key, _ in tiers if any((sigma, key) in values for sigma in sigmas)]
    bar_labels = {key: label for key, label in tiers}
    if not bar_keys:
        return None

    tier_colors = dict(COLORS)
    tier_colors["none_noise"] = COLORS.get("none", "#6B6B6B")
    tier_colors["wd_only"] = "#7c3aed"

    title_fs, axis_fs, tick_fs, legend_fs, value_fs = (15, 13, 11, 11, 10) if large else (9, 8, 7, 6.5, 6.5)
    x = np.arange(len(sigmas))
    width = min(0.18, 0.8 / max(1, len(bar_keys)))
    fig = Figure(figsize=(max(8.0, len(sigmas) * 2.6), 5.5))
    ax = fig.subplots()
    for index, key in enumerate(bar_keys):
        offsets = x + (index - (len(bar_keys) - 1) / 2) * width
        heights = [values.get((sigma, key), 0.0) for sigma in sigmas]
        color = tier_colors.get(key, "#6b7280")
        bars = ax.bar(offsets, heights, width=width, label=bar_labels[key], color=color, edgecolor="black", linewidth=0.4)
        for bar, height in zip(bars, heights):
            if height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.008,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=value_fs,
                rotation=90,
            )
    ax.set_xticks(x, [f"σ={sigma:g}" for sigma in sigmas], fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.set_ylabel("Final average accuracy", fontsize=axis_fs)
    max_value = max(values.values()) if values else 1.0
    ax.set_ylim(0, min(1.0, max_value + 0.15))
    ax.set_title(
        f"Ablation Groups ({subtitle}) | CIFAR-10 | {approach.upper()}{_seed_title_fragment(seed)} ({split})",
        fontsize=title_fs,
    )
    if large:
        ax.legend(frameon=False, fontsize=legend_fs, loc="best")
    else:
        ax.legend(frameon=False, fontsize=legend_fs, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    return fig


def _seed_ablation_attack_groups(records, approach, seed, split, *, large=False):
    """Byzantine-attack ablation only (environment="hidden_noise" tiers)."""
    return _render_ablation_groups(
        records,
        approach,
        seed,
        split,
        environment="hidden_noise",
        tiers=_HIDDEN_NOISE_ABLATION_TIERS,
        subtitle="Byzantine Attack",
        large=large,
    )


def _seed_ablation_noise_groups(records, approach, seed, split, *, large=False):
    """Channel-noise ablation only (environment="noise", no-attack tiers)."""
    return _render_ablation_groups(
        records,
        approach,
        seed,
        split,
        environment="noise",
        tiers=_NOISE_ABLATION_TIERS,
        subtitle="Channel Noise",
        large=large,
    )


def _seed_experiments_line(records, approach, seed, split, *, zoom=False):
    ordered = _seed_ordered_conditions(records)
    if not ordered:
        return None
    palette = _seed_palette(len(ordered))
    fig = Figure(figsize=(11.0, 6.0))
    ax = fig.subplots()
    all_values = []
    for index, record in enumerate(ordered):
        history = np.asarray(record.metrics.get("avg_history"))
        if history.size == 0:
            continue
        rounds = np.arange(len(history))
        all_values.extend(float(value) for value in history)
        color = _seed_condition_color(record, palette, index)
        ax.plot(
            rounds,
            history,
            label=_seed_condition_label(record).replace("\n", " "),
            color=color,
            linewidth=1.8,
            marker=MARKERS.get(record.effective_mitigation, "o"),
            markersize=3,
            markevery=max(1, len(rounds) // 10),
        )
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Average Accuracy")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    if zoom and all_values:
        y_min = max(0.0, min(all_values) - 0.03)
        y_max = min(1.0, max(all_values) + 0.06)
        if y_max - y_min < 0.12:
            center = (y_max + y_min) / 2
            y_min = max(0.0, center - 0.06)
            y_max = min(1.0, center + 0.06)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Zoomed Accuracy | CIFAR-10 | {approach.upper()}{_seed_title_fragment(seed)} ({split})")
    else:
        ax.set_ylim(0, 1)
        ax.set_title(f"CIFAR-10 | {approach.upper()}{_seed_title_fragment(seed)} ({split}) | Average Accuracy")
    return fig


def _seed_grid(records, approach, seed, split):
    ordered = _seed_ordered_conditions(records)
    if len(ordered) <= 1:
        return None
    palette = _seed_palette(len(ordered))
    n_cols = min(len(ordered), 3)
    n_rows = (len(ordered) + n_cols - 1) // n_cols
    fig = Figure(figsize=(4.4 * n_cols, 3.4 * n_rows))
    axes = fig.subplots(n_rows, n_cols, squeeze=False).flatten()
    fig.suptitle(f"CIFAR-10 | {approach.upper()}{_seed_title_fragment(seed)} ({split}) | Average Accuracy", fontweight="bold")
    for index, record in enumerate(ordered):
        axis = axes[index]
        history = np.asarray(record.metrics.get("avg_history"))
        if history.size == 0:
            axis.set_visible(False)
            continue
        rounds = np.arange(len(history))
        color = _seed_condition_color(record, palette, index)
        axis.plot(rounds, history, color=color, linewidth=1.8, marker=MARKERS.get(record.effective_mitigation, "o"), markersize=2.5, markevery=max(1, len(rounds) // 10))
        axis.set_title(_seed_condition_label(record).replace("\n", " "), fontsize=7, fontweight="bold")
        axis.set_xlabel("Round", fontsize=7)
        axis.set_ylabel("Average Accuracy", fontsize=7)
        axis.set_ylim(0, 1)
        if len(history):
            axis.annotate(
                f"{history[-1]:.3f}",
                xy=(len(history) - 1, history[-1]),
                fontsize=7,
                fontweight="bold",
                xytext=(-30, 10),
                textcoords="offset points",
            )
    for index in range(len(ordered), len(axes)):
        axes[index].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def generate_seed_matrix_plots(
    *,
    split=None,
    only_changed=True,
    formats=FORMATS,
    result_root=RESULT_ROOT,
    plot_root=PLOT_ROOT,
):
    """Legacy plotGui.py-style figures (final-accuracy bar, ablation groups,
    learning-curve overlay + zoom, small-multiples grid), bundled per
    (split, approach, seed) across every condition in that seed's
    accuracy-ordering matrix. Written under images/gui/{split}/cifar10/
    hidden/{approach}/seed_{seed}/accuracy_ordering/ -- a location separate
    from generate_campaign4_plots' own figures so this can run standalone
    without touching that pipeline or its manifest.
    """
    _paper_style()
    with _PLOT_LOCK:
        records = load_records(result_root=result_root, split=split)
        writer = _Writer(plot_root, only_changed, formats)
        errors = []
        groups = _seed_matrix_groups(records)
        for (split_name, approach, seed), group_records in sorted(groups.items()):
            if split and split_name != split:
                continue
            output = (
                Path("images")
                / "gui"
                / split_name
                / "cifar10"
                / "hidden"
                / approach
                / f"seed_{seed}"
                / "accuracy_ordering"
            )
            figures = (
                ("final_accuracy_avg", lambda rec=group_records: _seed_final_accuracy_bar(rec, approach, seed, split_name)),
                ("ablation_groups_attack_avg", lambda rec=group_records: _seed_ablation_attack_groups(rec, approach, seed, split_name)),
                ("ablation_groups_noise_avg", lambda rec=group_records: _seed_ablation_noise_groups(rec, approach, seed, split_name)),
                ("experiments_avg", lambda rec=group_records: _seed_experiments_line(rec, approach, seed, split_name, zoom=False)),
                ("experiments_avg_zoom", lambda rec=group_records: _seed_experiments_line(rec, approach, seed, split_name, zoom=True)),
                ("grid_avg", lambda rec=group_records: _seed_grid(rec, approach, seed, split_name)),
            )
            for name, factory in figures:
                try:
                    writer.figure(
                        f"seed_matrix:{split_name}:{approach}:{seed}:{name}",
                        group_records,
                        output / name,
                        factory,
                    )
                except Exception as error:
                    errors.append(f"{split_name}/{approach}/seed_{seed}/{name}: {error}")
        writer.finish()
        return {"generated": writer.generated, "skipped": writer.skipped, "errors": errors}


def _overall_average_conditions(records):
    """Group records by condition (environment, effective_mitigation, sigma)
    across every seed present, and average both the final accuracy and the
    round-by-round learning curve pointwise across whatever seeds have that
    condition. Returns a list of dicts sorted into the standard condition
    order, each carrying a representative record (for labeling/coloring)
    plus the averaged values and the seed count actually averaged."""
    buckets = {}
    for record in records:
        if not _is_core_tier(record):
            continue
        key = (record.environment, record.effective_mitigation, round(float(record.sigma), 3))
        buckets.setdefault(key, []).append(record)
    conditions = []
    for group in buckets.values():
        representative = group[0]
        finals = [record.scalar("final_avg") for record in group]
        histories = [
            history
            for history in (np.asarray(record.metrics.get("avg_history")) for record in group)
            if history.size > 0
        ]
        history_mean = None
        if histories:
            min_len = min(history.shape[0] for history in histories)
            history_mean = np.stack([history[:min_len] for history in histories]).mean(axis=0)
        conditions.append(
            {
                "representative": representative,
                "final_mean": float(np.mean(finals)) if finals else math.nan,
                "history_mean": history_mean,
                "n_seeds": len({record.seed for record in group}),
            }
        )
    conditions.sort(key=lambda item: _seed_condition_key(item["representative"]))
    return conditions


def _overall_final_accuracy_bar(records, approach, split):
    conditions = _overall_average_conditions(records)
    if not conditions:
        return None
    labels = [_seed_condition_label(condition["representative"]) for condition in conditions]
    finals = [condition["final_mean"] for condition in conditions]
    palette = _seed_palette(len(conditions))
    colors = [
        _seed_condition_color(condition["representative"], palette, index)
        for index, condition in enumerate(conditions)
    ]

    fig = Figure(figsize=(max(10.0, len(conditions) * 1.0), 6.0))
    ax = fig.subplots()
    bars = ax.bar(range(len(conditions)), finals, color=colors, width=0.6, edgecolor="black", linewidth=0.4)
    for bar, condition in zip(bars, conditions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{condition['final_mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(range(len(conditions)), labels, rotation=30, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Final average accuracy", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Final Accuracy | CIFAR-10 | {approach.upper()} | Overall Average ({split})", fontsize=15)
    fig.tight_layout()
    return fig


_ZOOM_TIER_KEYS = (
    ("clean", "none", None),
    ("hidden", "ss", None),
    ("noise", "ebm_wd", 0.4),
    ("hidden_noise", "ebm_wd", 0.4),
    ("hidden_noise", "ss", 0.4),
    ("hidden_noise", "ss_ebm_wd", 0.4),
)

_ZOOM_MARKERS = {
    ("clean", "none"): "o",
    ("hidden", "ss"): "s",
    ("noise", "ebm_wd"): "^",
    ("hidden_noise", "ss"): "D",
    ("hidden_noise", "ebm_wd"): "v",
    ("hidden_noise", "ss_ebm_wd"): "*",
}


def _overall_experiments_line(records, approach, split, *, zoom=False):
    conditions = [c for c in _overall_average_conditions(records) if c["history_mean"] is not None]
    if zoom:
        def _zoom_match(rep):
            sigma = round(float(rep.sigma), 2)
            for env, mitigation, want_sigma in _ZOOM_TIER_KEYS:
                if rep.environment == env and rep.effective_mitigation == mitigation:
                    if want_sigma is None or sigma == want_sigma:
                        return True
            return False

        conditions = [c for c in conditions if _zoom_match(c["representative"])]
    if not conditions:
        return None
    palette = _seed_palette(len(conditions))
    fig = Figure(figsize=(10.0, 5.5) if zoom else (12.0, 6.0))
    ax = fig.subplots()
    all_values = []
    max_rounds = 0
    attack_active = False
    for index, condition in enumerate(conditions):
        history = condition["history_mean"]
        rounds = np.arange(len(history))
        max_rounds = max(max_rounds, len(history) - 1)
        all_values.extend(float(value) for value in history)
        representative = condition["representative"]
        if representative.environment in ("hidden", "hidden_noise"):
            attack_active = True
        color = _seed_condition_color(representative, palette, index)
        label = _seed_condition_label(representative).replace("\n", " ")
        if zoom:
            marker = _ZOOM_MARKERS.get(
                (representative.environment, representative.effective_mitigation),
                MARKERS.get(representative.effective_mitigation, "o"),
            )
        else:
            marker = MARKERS.get(representative.effective_mitigation, "o")
        ax.plot(
            rounds,
            history,
            label=label,
            color=color,
            linewidth=2.2,
            marker=marker,
            markersize=7,
            markevery=max(1, len(rounds) // 10),
            markeredgecolor="black",
            markeredgewidth=0.4,
        )
    if attack_active and max_rounds >= HIDDEN_ATTACK_START_ROUND:
        ax.axvline(HIDDEN_ATTACK_START_ROUND, color="#444444", linestyle=":", linewidth=1.2, zorder=0)
        ax.text(
            HIDDEN_ATTACK_START_ROUND,
            0.02,
            f" Attack starts (round {HIDDEN_ATTACK_START_ROUND})",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            color="#444444",
            rotation=90,
            va="bottom",
            ha="left",
        )
    ax.set_xlabel("Training Round", fontsize=13)
    ax.set_ylabel("Average Accuracy", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    ax.set_xlim(0, max_rounds if max_rounds > 0 else 1)
    if zoom and all_values:
        y_min = max(0.0, min(all_values) - 0.03)
        y_max = min(1.0, max(all_values) + 0.06)
        if y_max - y_min < 0.12:
            center = (y_max + y_min) / 2
            y_min = max(0.0, center - 0.06)
            y_max = min(1.0, center + 0.06)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Accuracy | CIFAR-10 | {approach.upper()} | Overall Average ({split})", fontsize=15)
    else:
        ax.set_ylim(0, 1)
        ax.set_title(f"CIFAR-10 | {approach.upper()} | Overall Average ({split}) | Average Accuracy", fontsize=15)
    fig.tight_layout()
    return fig


def _overall_grid(records, approach, split):
    conditions = [c for c in _overall_average_conditions(records) if c["history_mean"] is not None]
    if len(conditions) <= 1:
        return None
    palette = _seed_palette(len(conditions))
    n_cols = min(len(conditions), 3)
    n_rows = (len(conditions) + n_cols - 1) // n_cols
    fig = Figure(figsize=(4.4 * n_cols, 3.4 * n_rows))
    axes = fig.subplots(n_rows, n_cols, squeeze=False).flatten()
    fig.suptitle(
        f"CIFAR-10 | {approach.upper()} | Overall Average ({split}) | Average Accuracy (mean across seeds)",
        fontweight="bold",
        fontsize=13,
    )
    for index, condition in enumerate(conditions):
        axis = axes[index]
        history = condition["history_mean"]
        representative = condition["representative"]
        rounds = np.arange(len(history))
        color = _seed_condition_color(representative, palette, index)
        axis.plot(
            rounds,
            history,
            color=color,
            linewidth=1.8,
            marker=MARKERS.get(representative.effective_mitigation, "o"),
            markersize=2.5,
            markevery=max(1, len(rounds) // 10),
        )
        if (
            representative.environment in ("hidden", "hidden_noise")
            and len(history) - 1 >= HIDDEN_ATTACK_START_ROUND
        ):
            axis.axvline(HIDDEN_ATTACK_START_ROUND, color="#444444", linestyle=":", linewidth=1.0, zorder=0)
        axis.set_title(
            _seed_condition_label(representative).replace("\n", " "),
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xlabel("Round", fontsize=10)
        axis.set_ylabel("Average Accuracy", fontsize=10)
        axis.tick_params(axis="both", labelsize=9)
        axis.set_xlim(0, max(len(history) - 1, 1))
        axis.set_ylim(0, 1)
        if len(history):
            axis.annotate(
                f"{history[-1]:.3f}",
                xy=(len(history) - 1, history[-1]),
                fontsize=9,
                fontweight="bold",
                xytext=(-30, 10),
                textcoords="offset points",
            )
    for index in range(len(conditions), len(axes)):
        axes[index].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def generate_overall_average_plots(
    *,
    only_changed=True,
    formats=("png", "eps"),
    result_root=RESULT_ROOT,
    plot_root=PLOT_ROOT,
):
    """Per-split, per-approach figures with every condition averaged across
    whichever seeds have completed it. Written to
    '{split} Overall Average/{APPROACH}/' directly under plot_root
    (plots4/campaign4/ by default) -- a cross-seed summary view separate
    from the per-seed accuracy_ordering figures. Only png/eps are written
    here (no pdf) per the Overall Average folder's output contract.
    """
    _paper_style()
    with _PLOT_LOCK:
        records = load_records(result_root=result_root, split=None)
        writer = _Writer(plot_root, only_changed, formats)
        errors = []
        groups = {}
        for record in records:
            groups.setdefault((record.split, record.approach), []).append(record)
        for (split_name, approach), group_records in sorted(groups.items()):
            output = Path(f"{split_name} Overall Average") / approach.upper()
            seed_label = f"avg of {len({r.seed for r in group_records})} seeds"
            figures = (
                ("final_accuracy_avg", lambda rec=group_records: _overall_final_accuracy_bar(rec, approach, split_name)),
                ("ablation_groups_attack_avg", lambda rec=group_records, seed_label=seed_label: _seed_ablation_attack_groups(rec, approach, seed_label, split_name, large=True)),
                ("ablation_groups_noise_avg", lambda rec=group_records, seed_label=seed_label: _seed_ablation_noise_groups(rec, approach, seed_label, split_name, large=True)),
                ("experiments_avg", lambda rec=group_records: _overall_experiments_line(rec, approach, split_name, zoom=False)),
                ("experiments_avg_zoom", lambda rec=group_records: _overall_experiments_line(rec, approach, split_name, zoom=True)),
                ("grid_avg", lambda rec=group_records: _overall_grid(rec, approach, split_name)),
            )
            for name, factory in figures:
                try:
                    writer.figure(
                        f"overall_avg:{split_name}:{approach}:{name}",
                        group_records,
                        output / name,
                        factory,
                    )
                except Exception as error:
                    errors.append(f"{split_name}/{approach}/{name}: {error}")
        writer.finish()
        return {"generated": writer.generated, "skipped": writer.skipped, "errors": errors}


def generate_campaign4_plots(
    *,
    mode="both",
    split=None,
    only_changed=True,
    formats=FORMATS,
    result_root=RESULT_ROOT,
    plot_root=PLOT_ROOT,
):
    if mode not in ("paper", "diagnostics", "both"):
        raise ValueError("mode must be paper, diagnostics, or both")
    _paper_style()
    with _PLOT_LOCK:
        records = load_records(result_root=result_root, split=split)
        writer = _Writer(plot_root, only_changed, formats)
        errors = []
        warnings = []
        splits = sorted({record.split for record in records})
        if split and split not in splits:
            splits.append(split)
        for split_name in splits:
            split_records = [record for record in records if record.split == split_name]
            confirmation_all = [
                record for record in split_records if record.phase in SCIENTIFIC_PHASES
            ]
            confirmation_profiles = _group_by_execution_profile(confirmation_all)
            base = Path("images") / "gui" / split_name / "cifar10" / "hidden"
            if mode in ("paper", "both"):
                if len(confirmation_profiles) > 1:
                    warnings.append(
                        f"{split_name}: confirmation records use {len(confirmation_profiles)} "
                        "execution profiles; figures were separated and were not pooled."
                    )
                for signature, confirmation in confirmation_profiles.items():
                    profile_slug = _profile_slug(signature)
                    profile_base = (
                        base
                        if len(confirmation_profiles) == 1
                        else base / "profiles" / profile_slug
                    )
                    all_evidence = [
                        record
                        for record in split_records
                        if record.phase in SCIENTIFIC_PHASES
                        and record.execution_signature == signature
                    ]
                    for approach in ("merged", "cart"):
                        selected = [
                            record
                            for record in confirmation
                            if record.approach == approach
                        ]
                        output = profile_base / approach / "paper"
                        figures = (
                            ("evidence_hierarchy", _evidence_hierarchy),
                            ("joint_challenge_profile", _joint_challenge),
                            ("defense_composition", _defense_composition),
                            ("learning_curves", _learning_curves),
                            ("noise_robustness", _noise_robustness),
                            ("average_vs_worst_node", _avg_worst),
                            ("seed_profiles", _seed_profiles),
                            ("class_retention", _class_retention),
                        )
                        for name, factory in figures:
                            try:
                                writer.figure(
                                    f"paper:{split_name}:{profile_slug}:{approach}:{name}",
                                    selected,
                                    output / name,
                                    lambda factory=factory, selected=selected, approach=approach: factory(
                                        selected, approach, split_name
                                    ),
                                )
                            except Exception as error:
                                errors.append(
                                    f"{split_name}/{profile_slug}/{approach}/{name}: {error}"
                                )
                        try:
                            paired_records = [
                                record
                                for record in all_evidence
                                if record.approach == approach
                            ]
                            writer.figure(
                                f"paper:{split_name}:{profile_slug}:{approach}:static_adaptive",
                                paired_records,
                                output / "static_vs_adaptive_ebm",
                                lambda paired_records=paired_records, approach=approach: _static_adaptive(
                                    paired_records, approach, split_name
                                ),
                            )
                        except Exception as error:
                            errors.append(
                                f"{split_name}/{profile_slug}/{approach}/static-adaptive: {error}"
                            )
                    try:
                        writer.figure(
                            f"paper:{split_name}:{profile_slug}:comparison:cart_lift",
                            confirmation,
                            profile_base
                            / "comparison"
                            / "paper"
                            / "cart_lift_over_merged",
                            lambda confirmation=confirmation: _cart_lift(
                                confirmation, split_name
                            ),
                        )
                    except Exception as error:
                        errors.append(
                            f"{split_name}/{profile_slug}/comparison/cart-lift: {error}"
                        )

            if mode in ("diagnostics", "both"):
                diagnostic = [
                    record
                    for record in split_records
                    if record.phase
                    in ("diagnostic", "confirmation", "static_control")
                ]
                diagnostic_profiles = _group_by_execution_profile(diagnostic)
                if len(diagnostic_profiles) > 1:
                    warnings.append(
                        f"{split_name}: diagnostic records use {len(diagnostic_profiles)} "
                        "execution profiles; diagnostics were separated."
                    )
                for signature, profile_records in diagnostic_profiles.items():
                    profile_slug = _profile_slug(signature)
                    profile_base = (
                        base
                        if len(diagnostic_profiles) == 1
                        else base / "profiles" / profile_slug
                    )
                    for approach in ("merged", "cart"):
                        selected = [
                            record
                            for record in profile_records
                            if record.approach == approach
                        ]
                        output = profile_base / approach / "diagnostics"
                        for name, factory in (
                            ("telemetry_panel", _telemetry_panel),
                            ("gradient_diagnostics", _gradient_panel),
                            ("cart_diagnostics", _cart_panel),
                            ("snapshot_selection", _selection_panel),
                            ("runtime_and_resources", _runtime_panel),
                        ):
                            try:
                                writer.figure(
                                    f"diagnostics:{split_name}:{profile_slug}:{approach}:{name}",
                                    selected,
                                    output / name,
                                    lambda factory=factory, selected=selected, approach=approach: factory(
                                        selected, approach, split_name
                                    ),
                                )
                            except Exception as error:
                                errors.append(
                                    f"{split_name}/{profile_slug}/{approach}/{name}: {error}"
                                )
                try:
                    writer.figure(
                        f"diagnostics:{split_name}:completeness",
                        split_records,
                        base / "comparison" / "diagnostics" / "result_completeness",
                        lambda: _completeness(split_records, split_name),
                    )
                except Exception as error:
                    errors.append(f"{split_name}/completeness: {error}")
            _write_summary_csv(split_records, Path(plot_root) / "tables" / split_name / "run_summary.csv")
        writer.finish()
        return {
            "records": len(records),
            "generated": writer.generated,
            "skipped": writer.skipped,
            "errors": errors,
            "warnings": warnings,
            "plotRoot": str(plot_root),
        }


def generate_campaign4_live_plots(*, split=None):
    result = generate_campaign4_plots(
        mode="both",
        split=split,
        only_changed=True,
        formats=("png",),
    )
    seed_matrix_result = generate_seed_matrix_plots(
        split=split,
        only_changed=True,
        formats=("png",),
    )
    # Overall-average figures pool every seed regardless of which split just
    # changed, but the Writer's fingerprint cache makes an unaffected
    # split/approach group a no-op, so this stays cheap to call every time.
    overall_result = generate_overall_average_plots(
        only_changed=True,
        formats=("png",),
    )
    for extra in (seed_matrix_result, overall_result):
        result["generated"] = result["generated"] + extra["generated"]
        result["skipped"] = result["skipped"] + extra["skipped"]
        result["errors"] = result["errors"] + extra["errors"]
    return result


__all__ = [
    "RunRecord",
    "clear_record_cache",
    "generate_campaign4_live_plots",
    "generate_campaign4_plots",
    "generate_overall_average_plots",
    "generate_seed_matrix_plots",
    "load_records",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("paper", "diagnostics", "both"), default="both")
    parser.add_argument("--split", choices=("IID", "nonIID"), default=None)
    parser.add_argument("--all", action="store_true", help="Regenerate unchanged figures.")
    arguments = parser.parse_args()
    print(
        json.dumps(
            generate_campaign4_plots(
                mode=arguments.mode,
                split=arguments.split,
                only_changed=not arguments.all,
            ),
            indent=2,
        )
    )
