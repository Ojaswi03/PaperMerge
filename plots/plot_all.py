from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR    = REPO_ROOT / "experiments" / "results"
OUT_DIR   = REPO_ROOT / "plots" / "images"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Label helper ----------
def make_label(fname: str) -> str:
    """
    Convert filename into a human-readable legend label.
    """
    name = fname.replace(".npy", "").lower()

    if "clean" in name:
        return "Clean (no noise, no attack)"

    if "noisy" in name and "ebm" not in name:
        return "Noisy communication only"

    if "ebm" in name:
        label = "Noisy + EBM"
        if "gaussian" in name:
            label += " (Gaussian / Byzantine)"
        elif "sign_flip" in name:
            label += " (Sign-Flip / Byzantine)"
        elif "hidden" in name:
            label += " (Hidden / Malicious)"
        return label

    return name.replace("_", " ").title()

# ---------- Plot helper ----------
def plot_single_curve(npy_path: Path):
    y = np.load(npy_path)
    y = np.asarray(y).ravel()
    x = range(len(y))

    label = make_label(npy_path.name)

    plt.figure(figsize=(7.5, 4.6))
    plt.plot(x, y, label=label)
    plt.grid(True, linestyle=":")
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.title(npy_path.stem.replace("_", " ").title())

    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_png = OUT_DIR / f"{npy_path.stem}.png"
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_png}")

# ---------- Main ----------
def main():
    npy_files = sorted(IN_DIR.glob("*.npy"))

    if not npy_files:
        print(f"[error] No .npy files found in {IN_DIR}")
        return

    print(f"[found] {len(npy_files)} .npy files")
    for f in npy_files:
        plot_single_curve(f)

if __name__ == "__main__":
    main()
