# scripts/common.py
import os
import numpy as np

def ensure_dirs():
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("plots/images", exist_ok=True)

def save_curve(arr, out_path):
    arr = np.array(arr, dtype=np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)
    return out_path
