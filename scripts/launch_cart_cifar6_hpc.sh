#!/usr/bin/env bash
# Fixed six-case CIFAR CART evaluation. See docs/CART_NEXT_HPC_RUN.md.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
GPU_ARGS=()
while (($#)); do
  case "$1" in
    --gpu-count|--gpus|--jobs-per-gpu) GPU_ARGS+=("$1" "${2:?Missing GPU/worker selection}"); shift 2 ;;
    --help|-h) echo 'Usage: bash scripts/launch_cart_cifar6_hpc.sh [--gpu-count 1|2|3|4] [--gpus 0,1] [--jobs-per-gpu 1..8]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -f .venv-hpc/bin/activate ]] || { echo 'Create the server environment first: bash scripts/setup_hpc.sh' >&2; exit 1; }
source .venv-hpc/bin/activate

# Modular noise selection: exactly 1, 2 or 3 values.
# Additional levels need matching controls; see the guide before changing these.
NOISE_LEVELS=1
NOISE_VALUES=(0.01)

RUN="${CART_RUN_OUTPUT:-$PWD/.papermerge/local-tests/cart-admission-cifar6-100r-v1}"
if [[ -f "$RUN/plan.json" ]]; then GPU_ARGS+=(--saved-plan "$RUN/plan.json"); fi
SELECTION="$(python scripts/select_cart_gpus.py "${GPU_ARGS[@]}" --format shell)"
read -r GPU_IDS JOBS_PER_GPU <<< "$SELECTION"
echo "Physical GPUs: $GPU_IDS | Workers per GPU: $JOBS_PER_GPU | Output: $RUN"
LATEST="${CART_LATEST_FIGURES:-$PWD/.papermerge/cart-latest-figures}"
export MPLCONFIGDIR="$RUN/matplotlib-cache"
COMMON=(
  --profile hpc --gpus "$GPU_IDS" --output "$RUN"
  --variant prior_neutral_class
  --jobs-per-gpu "$JOBS_PER_GPU" --gpu-memory-mib 8192
  --gpu-headroom-mib 8192 --context-overhead-mib 1024
  --ram-mib-per-worker 32768 --ram-headroom-mib 32768
  --datasets cifar10 --partitions iid dirichlet --seeds 2025 2026 2027
  --noise-levels "$NOISE_LEVELS" --noise-values "${NOISE_VALUES[@]}"
  --conditions attack_noise_defense_rla_cart
  --nodes 100 --rounds 100 --local-steps 5 --attack-start 20 --alpha 0.5 --model paper
  --reference "$PWD/docs/results/cart-v2-72-100r-20261028"
  --latest-figures "$LATEST"
)
python scripts/hpc.py gpu-status
python scripts/run_cart_revision_study.py --preflight "${COMMON[@]}"
python scripts/run_cart_revision_study.py --verify-gpus "${COMMON[@]}"
python scripts/run_cart_v2_managed.py --execute --scheduler admission \
  "${COMMON[@]}" --figures "$LATEST" \
  --ntfy-topic papermerge-ojaswi --tmux cart-admission-cifar6
