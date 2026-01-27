# scripts/common.py
import os
import numpy as np


def setupGpu():
    """Configure TensorFlow to use GPU with memory growth enabled. Falls back to CPU if GPU fails."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth - don't allocate all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[GPU] Successfully configured {len(gpus)} GPU(s): {[g.name for g in gpus]}")
                return True
            except RuntimeError as e:
                print(f"[GPU] Error configuring GPU: {e}")
                print("[CPU] Falling back to CPU (training will be slower)")
                # Force CPU usage
                tf.config.set_visible_devices([], 'GPU')
                return False
        else:
            print("[CPU] No GPU found, using CPU (training will be slower)")
            return False
    except Exception as e:
        print(f"[CPU] Error during GPU setup: {e}")
        print("[CPU] Using CPU (training will be slower)")
        return False


def ensureDirs():
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("plots/images", exist_ok=True)


def saveCurve(arr, outPath):
    arr = np.array(arr, dtype=np.float32)
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    np.save(outPath, arr)
    return outPath


def handleGpuMemoryError(e):
    """
    Handle GPU out-of-memory errors and provide helpful guidance.

    Args:
        e: The exception that occurred

    Returns:
        True if this was a GPU memory error, False otherwise
    """
    errorMsg = str(e).lower()
    isGpuMemoryError = any(keyword in errorMsg for keyword in [
        'out of memory', 'oom', 'memory', 'cudnn', 'cuda', 'gpu'
    ])

    if isGpuMemoryError:
        print("\n" + "="*80)
        print("GPU MEMORY ERROR DETECTED")
        print("="*80)
        print("The GPU ran out of memory. Try one of these solutions:")
        print("  1. Reduce batch size (currently using batchSize parameter)")
        print("  2. Use fewer nodes (reduce nClients)")
        print("  3. Force CPU usage by setting CUDA_VISIBLE_DEVICES='' in environment")
        print("  4. Close other GPU-using applications")
        print("="*80 + "\n")
        return True

    return False
