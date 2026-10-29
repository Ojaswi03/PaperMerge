#!/usr/bin/env bash
# Isolated mechanism screens, impairment controls or full CART/ACDS comparison.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source .venv-hpc/bin/activate
MODE="${1:-screen}"
case "$MODE" in screen|isolation|confirm|acds) ;; *) echo 'Use screen, isolation, acds or confirm.' >&2; exit 2;; esac
PRESET="$MODE"
if [[ "$MODE" == acds ]]; then PRESET=confirm; fi
RUN="${CART_COMPARISON_OUTPUT:-$PWD/.papermerge/local-tests/cart-comparison-${MODE}-v1}"
SESSION="cart-comparison-${MODE}"
export MPLCONFIGDIR="$RUN/matplotlib-cache"
COMMON=(--preset "$PRESET" --output "$RUN" --profile hpc
  --gpus "${CART_COMPARISON_GPUS:-0,1,2,3}" --jobs-per-gpu 2
  --gpu-memory-mib 8192 --gpu-headroom-mib 8192 --context-overhead-mib 1024
  --ram-mib-per-worker 32768 --ram-headroom-mib 32768
  --datasets cifar10 --partitions iid dirichlet --noise-levels 1 --noise-values 0.01
  --share-fraction 0.05 --acds-batches 5)
if [[ "$MODE" == acds ]]; then
  COMMON+=(--arms acds --reference "$PWD/docs/results/cart-admission-cifar6-100r-20261028")
fi
if [[ -f "$RUN/campaign.json" ]]; then
  python scripts/run_cart_comparison.py --execute --resume --output "$RUN" \
    --tmux "$SESSION" --ntfy-topic papermerge-ojaswi
else
  python scripts/hpc.py gpu-status
  python scripts/run_cart_comparison.py --preflight "${COMMON[@]}"
  python scripts/run_cart_comparison.py --verify-gpus "${COMMON[@]}"
  python scripts/run_cart_comparison.py --execute --resume --output "$RUN" \
    --tmux "$SESSION" --ntfy-topic papermerge-ojaswi
fi
