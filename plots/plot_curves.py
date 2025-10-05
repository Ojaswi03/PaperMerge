# #!/usr/bin/env python3
# from pathlib import Path
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # ---------- Resolve repo structure robustly ----------
# THIS_FILE = Path(__file__).resolve()
# REPO_ROOT = THIS_FILE.parents[1]        # .../PaperMerge
# DEFAULT_IN_DIR = REPO_ROOT / "experiments" / "results"
# DEFAULT_OUT_DIR = REPO_ROOT / "plots" / "images"

# # ---------- Expected file names ----------
# EBM_FILES = {
#     "gaussian":  "mnist_ebm_looped__gaussian_avg.npy",
#     "sign_flip": "mnist_ebm_looped__sign_flip_avg.npy",
#     "hidden":    "mnist_ebm_looped__hidden_avg.npy",
# }
# NOISY_FILES = {
#     "gaussian":  "mnist_noisy_looped__gaussian_avg.npy",
#     "sign_flip": "mnist_noisy_looped__sign_flip_avg.npy",
#     "hidden":    "mnist_noisy_looped__hidden_avg.npy",
# }
# CLEAN_FILE = "acc_mnist_clean.npy"  # optional baseline

# # ---------- Helpers ----------
# def candidate_paths(fname: str):
#     """Yield plausible absolute paths for fname."""
#     name = Path(fname).name
#     # 1) results dir (absolute)
#     yield DEFAULT_IN_DIR / name
#     # 2) repo root (absolute)
#     yield REPO_ROOT / name
#     # 3) current working dir (where you invoked python)
#     yield Path.cwd() / name
#     # 4) brute-force search across repo (first match wins)
#     for p in REPO_ROOT.rglob(name):
#         yield p

# def load_curve(fname: str):
#     """Load .npy curve by trying multiple locations. Return np.ndarray or None."""
#     for p in candidate_paths(fname):
#         try:
#             if p.exists():
#                 arr = np.load(p)
#                 print(f"[found] {fname} -> {p}")
#                 return arr
#         except Exception as e:
#             print(f"[error] failed to load {p}: {e}")
#             return None
#     print(f"[skip] not found: {fname}")
#     return None

# def ensure_dir(p: Path):
#     p.parent.mkdir(parents=True, exist_ok=True)

# def plot_overlay(curves, title: str, outpng: Path, ylim=(0.0, 1.0)):
#     """
#     curves: list of tuples (label, yarray, style_dict)
#     """
#     ensure_dir(outpng)
#     plt.figure(figsize=(7.0, 4.0))
#     plotted = 0
#     for label, y, style in curves:
#         if y is None:
#             continue
#         x = np.arange(len(y))
#         style = {"marker": "o", "markersize": 3, "linewidth": 1.5, **(style or {})}
#         plt.plot(x, y, label=label, **style)
#         plotted += 1
#     plt.grid(True, linestyle=":")
#     plt.xlabel("Round")
#     plt.ylabel("Average Accuracy")
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     if plotted > 0:
#         plt.legend()
#     plt.tight_layout()
#     plt.savefig(outpng.as_posix(), dpi=220, bbox_inches="tight")
#     print(f"[saved] {outpng}")
#     plt.close()

# # ---------- Main ----------
# def main():
#     clean = load_curve(CLEAN_FILE)

#     ebm = {k: load_curve(v) for k, v in EBM_FILES.items()}
#     noisy = {k: load_curve(v) for k, v in NOISY_FILES.items()}

#     attack_labels = [("gaussian", "Gaussian"), ("sign_flip", "Sign-Flip"), ("hidden", "Hidden")]

#     for key, pretty in attack_labels:
#         curves = []
#         if clean is not None:
#             curves.append(("Clean", clean, {"linestyle": "--"}))
#         curves.append(("EBM", ebm.get(key), {}))
#         curves.append(("Noisy", noisy.get(key), {}))
#         outpng = DEFAULT_OUT_DIR / f"mnist_{key}_ebm_vs_noisy.png"
#         plot_overlay(curves, f"MNIST — {pretty} Attack: EBM vs Noisy", outpng)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Repo root detection (robust) ----------------
THIS = Path(__file__).resolve()

def find_repo_root(start: Path) -> Path:
    # Walk up until we see a folder that looks like the repo (has 'plots' and 'experiments')
    cur = start
    for _ in range(6):  # don't climb forever
        plots = (cur / "plots").exists()
        exps  = (cur / "experiments").exists()
        if plots or exps or (cur / ".git").exists():
            return cur
        cur = cur.parent
    # fallback: plots/.. (what we originally assumed)
    return start.parent

REPO_ROOT = find_repo_root(THIS.parent)
IN_DIR    = REPO_ROOT / "experiments" / "results"
OUT_DIR   = REPO_ROOT / "plots" / "images"

print("[debug] __file__     :", THIS)
print("[debug] CWD          :", Path.cwd())
print("[debug] REPO_ROOT    :", REPO_ROOT)
print("[debug] IN_DIR       :", IN_DIR)
print("[debug] OUT_DIR      :", OUT_DIR)

# ---------------- Expected file names ----------------
EBM_FILES = {
    "gaussian":  "mnist_ebm_looped__gaussian_avg.npy",
    "sign_flip": "mnist_ebm_looped__sign_flip_avg.npy",
    "hidden":    "mnist_ebm_looped__hidden_avg.npy",
}
NOISY_FILES = {
    "gaussian":  "mnist_noisy_looped__gaussian_avg.npy",
    "sign_flip": "mnist_noisy_looped__sign_flip_avg.npy",
    "hidden":    "mnist_noisy_looped__hidden_avg.npy",
}
CLEAN_FILE = "acc_mnist_clean.npy"  # optional baseline

# ---------------- Helpers ----------------
def candidates(fname: str):
    """All plausible places to look."""
    name = Path(fname).name
    yield IN_DIR / name
    yield REPO_ROOT / name
    yield Path.cwd() / name
    # brute force
    for p in REPO_ROOT.rglob(name):
        yield p

def load_curve(fname: str):
    """Load .npy or return None with debug."""
    tried = []
    for p in candidates(fname):
        tried.append(p)
        if p.exists():
            try:
                arr = np.load(p)
                print(f"[found] {fname} -> {p} (shape={arr.shape}, dtype={arr.dtype})")
                if arr.size == 0:
                    print(f"[warn] {p} is empty")
                return arr
            except Exception as e:
                print(f"[error] failed to load {p}: {e}")
                return None
    print(f"[miss ] {fname} not found. Tried:\n  - " + "\n  - ".join(str(x) for x in tried))
    return None

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def plot_overlay(curves, title: str, outpng: Path, ylim=(0.0, 1.0)):
    """
    curves: list of tuples (label, yarray, style_dict)
    """
    ensure_dir(outpng)
    plt.figure(figsize=(7.0, 4.0))
    plotted = 0
    for label, y, style in curves:
        if y is None:
            continue
        if y.ndim != 1:
            print(f"[warn] {label} curve is not 1D (shape={y.shape}); flattening")
            y = y.ravel()
        x = np.arange(len(y))
        style = {"marker": "o", "markersize": 3, "linewidth": 1.5, **(style or {})}
        plt.plot(x, y, label=label, **style)
        plotted += 1
    plt.grid(True, linestyle=":")
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if plotted > 0:
        plt.legend()
    else:
        print(f"[error] No curves plotted for: {outpng.name}")
    plt.tight_layout()
    plt.savefig(outpng.as_posix(), dpi=220, bbox_inches="tight")
    print(f"[saved] {outpng}")
    plt.close()

# ---------------- Main ----------------
def main():
    # Print what files are in IN_DIR for visibility
    if IN_DIR.exists():
        avail = sorted(p.name for p in IN_DIR.glob("*.npy"))
        print(f"[debug] Files in {IN_DIR}: {len(avail)}")
        for n in avail[:20]:
            print("   -", n)
        if len(avail) > 20:
            print("   ...")

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
        outpng = OUT_DIR / f"mnist_{key}_ebm_vs_noisy.png"
        plot_overlay(curves, f"MNIST — {pretty} Attack: EBM vs Noisy", outpng)

if __name__ == "__main__":
    main()
