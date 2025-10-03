#!/usr/bin/env bash
set -e

ENV_NAME="basil-noise-env"

echo "=== Creating virtual environment: $ENV_NAME ==="
python3 -m venv $ENV_NAME
source $ENV_NAME/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing core Python deps ==="
pip install numpy matplotlib pandas scikit-learn

echo "=== Installing TensorFlow (CPU version by default) ==="
# CPU-only:
pip install tensorflow==2.16.1
# if you have a GPU and CUDA installed, comment the above line and
# use instead (choose the right CUDA version for your system):
# pip install tensorflow[and-cuda]==2.16.1

echo "=== Installing PyTorch (CPU by default) ==="
# CPU-only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# For GPU with CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For GPU with CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Optional extras ==="
pip install wandb ipython

echo "=== Verification ==="
python - <<'EOF'
import tensorflow as tf, torch, numpy as np
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("NumPy version:", np.__version__)
EOF

echo "=== DONE ==="
echo "Activate with: source $ENV_NAME/bin/activate"
