# #!/usr/bin/env bash
# set -e

# ENV_NAME="basil-noise-env"

# echo "=== Creating virtual environment: $ENV_NAME ==="
# python3 -m venv $ENV_NAME
# source $ENV_NAME/bin/activate

# echo "=== Upgrading pip ==="
# pip install --upgrade pip

# echo "=== Installing core Python deps ==="
# pip install numpy matplotlib pandas scikit-learn

# echo "=== Installing TensorFlow (CPU version by default) ==="
# # CPU-only:
# pip install tensorflow==2.16.1
# # if you have a GPU and CUDA installed, comment the above line and
# # use instead (choose the right CUDA version for your system):
# # pip install tensorflow[and-cuda]==2.16.1

# echo "=== Installing PyTorch (CPU by default) ==="
# # CPU-only:
# #pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# # For GPU with CUDA 11.8:
# # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# # For GPU with CUDA 12.1:
# # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# # install PyTorch with CUDA 12.4 runtime
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# echo "=== Optional extras ==="
# pip install wandb ipython

# echo "=== Verification ==="
# python - <<'EOF'
# import tensorflow as tf, torch, numpy as np
# print("TensorFlow version:", tf.__version__)
# print("GPU available:", tf.config.list_physical_devices('GPU'))
# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("NumPy version:", np.__version__)
# EOF

# echo "=== DONE ==="
# echo "Activate with: source $ENV_NAME/bin/activate"



#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="basil-noise-env"

echo "=== [1/7] Create venv: $ENV_NAME ==="
python3 -m venv "$ENV_NAME"
# shellcheck source=/dev/null
source "$ENV_NAME/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo "=== [2/7] Core deps ==="
pip install --upgrade numpy matplotlib pandas scikit-learn ipython wandb

echo "=== [3/7] Detect GPU in WSL ==="
HAS_NVIDIA="false"
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA="true"
  fi
fi
echo "GPU detected? $HAS_NVIDIA"

echo "=== [4/7] TensorFlow ==="
# TensorFlow CPU fallback is always safe. If GPU present, use the new official extra that
# brings CUDA/cuDNN runtime wheels via pip (no system toolkit needed).
if [[ "$HAS_NVIDIA" == "true" ]]; then
  # GPU build (uses bundled CUDA runtime via pip)
  pip install "tensorflow[and-cuda]==2.17.*"
else
  # CPU-only
  pip install "tensorflow==2.17.*"
fi

echo "=== [5/7] PyTorch ==="
if [[ "$HAS_NVIDIA" == "true" ]]; then
  # Install wheels with CUDA 12.4 runtime (best match for your 581.57 driver that caps at CUDA 13.0)
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
  # CPU wheels
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "=== [6/7] Versions & sanity checks ==="
python - <<'PY'
import sys, platform
print("Python:", sys.version.split()[0], "| Platform:", platform.platform())

# NumPy
import numpy as np
print("NumPy:", np.__version__)

# TensorFlow
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print("TensorFlow:", tf.__version__, "| GPUs:", gpus)
except Exception as e:
    print("TensorFlow import FAILED:", e)

# PyTorch
try:
    import torch
    print("PyTorch:", torch.__version__,
          "| CUDA avail:", torch.cuda.is_available(),
          "| Torch CUDA runtime:", getattr(torch.version, "cuda", None))
    if torch.cuda.is_available():
        print("Torch GPU 0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("PyTorch import FAILED:", e)
PY

echo "=== [7/7] Done. Activate with: source $ENV_NAME/bin/activate ==="
