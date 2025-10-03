import os
import numpy as np
import matplotlib.pyplot as plt

IN_DIR = "experiments/results"
OUT_DIR = "plots/images"

def load_curve(path):
    return np.load(path) if os.path.exists(path) else None

def plot_one(x, y, title, outpath):
    plt.figure(figsize=(7,4))
    plt.plot(x, y, marker="o")
    plt.grid(True, linestyle=":")
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.title(title)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.show()

def main():
    files = [
        ("acc_mnist_clean.npy", "MNIST — Basil (clean)", "plots/images/mnist_clean.png"),
        ("acc_mnist_noisy.npy", "MNIST — Basil (noisy)", "plots/images/mnist_noisy.png"),
        ("acc_mnist_ebm.npy",   "MNIST — Basil (EBM)",   "plots/images/mnist_ebm.png"),
    ]
    for fname, title, outpng in files:
        path = os.path.join(IN_DIR, fname)
        arr = load_curve(path)
        if arr is None:
            print(f"[skip] not found: {path}")
            continue
        rounds = list(range(len(arr)))
        plot_one(rounds, arr, title, outpng)

if __name__ == "__main__":
    main()
