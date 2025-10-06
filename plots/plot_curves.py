#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------- Paths ----------
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]                 # .../PaperMerge
IN_DIR    = REPO_ROOT / "experiments" / "results"
OUT_DIR   = REPO_ROOT / "plots" / "images"

# ---------- Expected files by dataset ----------
MNIST_EBM = {
    "gaussian":  "mnist_ebm_looped__gaussian_avg.npy",
    "sign_flip": "mnist_ebm_looped__sign_flip_avg.npy",
    "hidden":    "mnist_ebm_looped__hidden_avg.npy",
}
MNIST_NOISY = {
    "gaussian":  "mnist_noisy_looped__gaussian_avg.npy",
    "sign_flip": "mnist_noisy_looped__sign_flip_avg.npy",
    "hidden":    "mnist_noisy_looped__hidden_avg.npy",
}
MNIST_CLEAN = "acc_mnist_clean.npy"

CIFAR_EBM = {
    "gaussian":  "cifar10_ebm_looped__gaussian_avg.npy",
    "sign_flip": "cifar10_ebm_looped__sign_flip_avg.npy",
    "hidden":    "cifar10_ebm_looped__hidden_avg.npy",
}
CIFAR_NOISY = {
    "gaussian":  "cifar10_noisy_looped__gaussian_avg.npy",
    "sign_flip": "cifar10_noisy_looped__sign_flip_avg.npy",
    "hidden":    "cifar10_noisy_looped__hidden_avg.npy",
}
CIFAR_CLEAN = "acc_cifar10_clean.npy"

# ---------- Helpers ----------
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_curve(fname: str):
    p = IN_DIR / fname
    if p.exists():
        arr = np.load(p)
        print(f"[found] {fname} -> {p} (shape={arr.shape})")
        return arr
    print(f"[miss ] {fname} (searched {p})")
    return None

def autodetect_dataset():
    """Return 'mnist' or 'cifar10' by checking which expected files exist; default to mnist."""
    mnist_hits = sum((IN_DIR / f).exists() for f in MNIST_EBM.values()) + \
                 sum((IN_DIR / f).exists() for f in MNIST_NOISY.values())
    cifar_hits = sum((IN_DIR / f).exists() for f in CIFAR_EBM.values()) + \
                 sum((IN_DIR / f).exists() for f in CIFAR_NOISY.values())
    print(f"[auto ] mnist hits={mnist_hits}, cifar10 hits={cifar_hits}")
    if cifar_hits > mnist_hits:
        return "cifar10"
    return "mnist"

# ---------- Legend text helpers (verbose) ----------
def method_verbose_name(base: str) -> str:
    """Map 'clean'/'noisy'/'ebm' to verbose legend text."""
    base = (base or "").strip().lower()
    if base == "clean":
        return "Clean — no channel noise"
    if base == "noisy":
        return "Noisy — channel noise only, no EBM"
    if base == "ebm":
        return "EBM — channel noise  + EBM regularizer"
    return base.capitalize()

def pretty_stats_label_verbose(base: str, y):
    """
    Combine verbose method description with simple metrics.
    e.g., "EBM — channel noise (σ>0) + EBM regularizer  (final=0.7600, best=0.7700)"
    """
    base_verbose = method_verbose_name(base)
    if y is None or getattr(y, "size", 0) == 0:
        return base_verbose
    y1d = np.asarray(y).ravel()
    return f"{base_verbose}  (final={float(y1d[-1]):.4f}, best={float(y1d.max()):.4f})"

# ---------- Plotting ----------
def plot_overlay(curves, title: str, outpng: Path, ylim=(0.0, 1.0), legend_loc="lower right"):
    """
    curves: list of tuples (base_label, yarray, style_dict)
            base_label in {"Clean","EBM","Noisy"} to trigger verbose legend text.
    legend_loc: 'lower right' | 'upper left' | 'best' | etc.
    """
    ensure_dir(outpng)
    plt.figure(figsize=(7.5, 4.6))
    plotted = 0

    for base_label, y, style in curves:
        if y is None:
            continue
        y = np.asarray(y).ravel()
        x = np.arange(len(y))
        style = {"marker": "o", "markersize": 3, "linewidth": 1.6, **(style or {})}
        label = pretty_stats_label_verbose(base_label, y)
        plt.plot(x, y, label=label, **style)
        plotted += 1

    plt.grid(True, linestyle=":")
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    if plotted:
        leg = plt.legend(
            loc=legend_loc,  # legend INSIDE the axes
            title="Methods",
            frameon=True, fancybox=True, framealpha=0.85,
            fontsize=9, title_fontsize=10, borderpad=0.6
        )
        leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(outpng.as_posix(), dpi=240, bbox_inches="tight")
    print(f"[saved] {outpng}")
    plt.close()

# ---------- Main ----------
def main():
    ds = autodetect_dataset()
    print(f"[use  ] dataset={ds}")

    if ds == "cifar10":
        EBM_FILES, NOISY_FILES, CLEAN_FILE = CIFAR_EBM, CIFAR_NOISY, CIFAR_CLEAN
        title_prefix = "CIFAR-10"
        out_prefix   = "cifar10"
    else:
        EBM_FILES, NOISY_FILES, CLEAN_FILE = MNIST_EBM, MNIST_NOISY, MNIST_CLEAN
        title_prefix = "MNIST"
        out_prefix   = "mnist"

    # Load curves
    clean = load_curve(CLEAN_FILE)
    ebm   = {k: load_curve(v) for k, v in EBM_FILES.items()}
    noisy = {k: load_curve(v) for k, v in NOISY_FILES.items()}

    attack_labels = [("gaussian", "Gaussian"), ("sign_flip", "Sign-Flip"), ("hidden", "Hidden")]

    for key, pretty in attack_labels:
        curves = []
        if clean is not None:
            curves.append(("Clean", clean, {"linestyle": "--"}))
        curves.append(("EBM",   ebm.get(key),   {}))
        curves.append(("Noisy", noisy.get(key), {}))

        outpng = OUT_DIR / f"{out_prefix}_{key}_ebm_vs_noisy.png"
        plot_overlay(
            curves=curves,
            title=f"{title_prefix} — {pretty} Attack: Expectation Based Model (EBM) vs Noisy",
            outpng=outpng,
            ylim=(0.0, 1.0),
            legend_loc="lower right",  # change to 'best' or any corner if needed
        )

if __name__ == "__main__":
    main()
