#!/usr/bin/env bash
# One exploratory CART trajectory on the local PC, not the full HPC matrix.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source environment/basil-noise-env/bin/activate
RUN="${CART_LOCAL_OUTPUT:-$PWD/.papermerge/local-tests/cart-prior-dirichlet-50r-20261029}"
SESSION=cart-prior-local-50r
export MPLCONFIGDIR="$RUN/matplotlib-cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
COMMON=(--preset confirm --arms prior_cart --rounds 50 --attack-start 20
  --output "$RUN" --profile local --gpus 0 --jobs-per-gpu 1
  --gpu-memory-mib 6144 --gpu-headroom-mib 1024 --context-overhead-mib 1024
  --ram-mib-per-worker 8192 --ram-headroom-mib 4096
  --datasets cifar10 --partitions dirichlet --seeds 2025
  --noise-levels 1 --noise-values 0.01)
if [[ ! -f "$RUN/campaign.json" ]]; then
  python scripts/run_cart_comparison.py --preflight "${COMMON[@]}"
  python scripts/run_cart_comparison.py --verify-gpus "${COMMON[@]}"
fi
python scripts/run_cart_comparison.py --execute --resume --output "$RUN" \
  --tmux "$SESSION" --ntfy-topic papermerge-ojaswi
