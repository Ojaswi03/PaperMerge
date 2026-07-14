# scripts/common.py
import gc
import os
import urllib.request
import numpy as np

# --- ntfy.sh phone notifications ---
# 1. Install the "ntfy" app on your phone (Android/iOS)
# 2. Subscribe to your topic in the app
# 3. Set your topic name below
NTFY_TOPIC = "papermerge-ojaswi"   # <-- change this to your own unique topic


def sendNotification(title, message, priority="default"):
    # send POST to ntfy.sh topic; silently ignore any network errors
    try:
        url = f"https://ntfy.sh/{NTFY_TOPIC}"
        req = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": priority,
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Never crash the experiment if notification fails


def setupGpu():
    # find all physical GPUs and enable memory growth
    # safe to call multiple times — ignores already-initialized errors
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    # already initialized (e.g. called a second time) — GPU is still active
                    pass
            gpuNames = [g.name for g in gpus]
            print(f"[GPU] Successfully configured {len(gpus)} GPU(s): {gpuNames}")
            return {"device": "CUDA", "gpus": gpuNames, "count": len(gpus)}
        else:
            print("[CPU] No GPU found, using CPU (training will be slower)")
            return {"device": "CPU", "gpus": [], "count": 0}
    except Exception as e:
        print(f"[CPU] Error during GPU setup: {e}")
        print("[CPU] Using CPU (training will be slower)")
        return {"device": "CPU", "gpus": [], "count": 0}


def cleanupTensorflowMemory(logger=None, context="cleanup", collectCycles=3):
    """Explicitly release TensorFlow/Keras graph state and run Python GC.

    This is meant for experiment boundaries, after models/loaders/results from
    one run are no longer needed. It should not be called mid-training.
    """
    collected = 0
    tfError = None

    try:
        import tensorflow as tf
        from tensorflow.keras import backend as keras_backend

        try:
            keras_backend.clear_session(free_memory=True)
        except TypeError:
            keras_backend.clear_session()

        for device in tf.config.list_logical_devices("GPU"):
            try:
                tf.config.experimental.reset_memory_stats(device.name)
            except Exception:
                pass
    except Exception as e:
        tfError = e

    for _ in range(max(1, int(collectCycles))):
        collected += gc.collect()

    if logger:
        if tfError is None:
            logger(f"[GC] {context}: cleared TensorFlow/Keras state; collected {collected} Python objects.")
        else:
            logger(f"[GC] {context}: Python GC collected {collected} objects; TensorFlow cleanup warning: {tfError}")

    return collected


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
    # check if the exception looks like a GPU out-of-memory error
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
