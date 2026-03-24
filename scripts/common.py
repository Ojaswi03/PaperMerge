# scripts/common.py
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
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth - don't allocate all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                gpuNames = [g.name for g in gpus]
                print(f"[GPU] Successfully configured {len(gpus)} GPU(s): {gpuNames}")
                return {"device": "CUDA", "gpus": gpuNames, "count": len(gpus)}
            except RuntimeError as e:
                print(f"[GPU] Error configuring GPU: {e}")
                print("[CPU] Falling back to CPU (training will be slower)")
                # Force CPU usage
                tf.config.set_visible_devices([], 'GPU')
                return {"device": "CPU", "gpus": [], "count": 0}
        else:
            print("[CPU] No GPU found, using CPU (training will be slower)")
            return {"device": "CPU", "gpus": [], "count": 0}
    except Exception as e:
        print(f"[CPU] Error during GPU setup: {e}")
        print("[CPU] Using CPU (training will be slower)")
        return {"device": "CPU", "gpus": [], "count": 0}


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
