#!/usr/bin/env bash
# Isolated mechanism screens, impairment controls or full CART/ACDS comparison.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
MODE=screen
if (($#)) && [[ "$1" != --* && "$1" != -h ]]; then MODE="$1"; shift; fi
GPU_ARGS=()
GPU_EXPLICIT=0
JOBS_EXPLICIT=0
DRY_RUN=0
AUTO_WORKERS=0
MAX_WORKERS=8
RUN_ARG=""
while (($#)); do
  case "$1" in
    --gpu-count|--gpus) GPU_ARGS+=("$1" "${2:?Missing GPU selection}"); GPU_EXPLICIT=1; shift 2 ;;
    --jobs-per-gpu) GPU_ARGS+=("$1" "${2:?Missing worker count}"); JOBS_EXPLICIT=1; shift 2 ;;
    --auto-workers) AUTO_WORKERS=1; shift ;;
    --max-workers-per-gpu) MAX_WORKERS="${2:?Missing worker ceiling}"; shift 2 ;;
    --output) RUN_ARG="${2:?Missing output directory}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      echo 'Usage: bash scripts/launch_cart_comparison_hpc.sh [screen|isolation|confirm|acds|full] [--gpu-count 1|2|3|4] [--gpus 0,1] [--auto-workers [--max-workers-per-gpu 1..8] | --jobs-per-gpu 1..8] [--output PATH] [--dry-run]'
      echo 'Default: detect up to four GPUs; two workers/GPU. Resume retains saved GPUs and workers.'
      echo '--auto-workers: fill GPU/RAM budgets dynamically; safely hand off an existing supervisor.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$MAX_WORKERS" =~ ^[1-8]$ ]] || { echo 'Worker ceiling must be 1..8.' >&2; exit 2; }
if (( AUTO_WORKERS && JOBS_EXPLICIT )); then echo 'Choose --auto-workers or a fixed --jobs-per-gpu.' >&2; exit 2; fi
case "$MODE" in screen|isolation|confirm|acds|full) ;; *) echo 'Use screen, isolation, confirm, acds or full.' >&2; exit 2;; esac
[[ -f .venv-hpc/bin/activate ]] || { echo 'Create the server environment first: bash scripts/setup_hpc.sh' >&2; exit 1; }
source .venv-hpc/bin/activate
PRESET="$MODE"
if [[ "$MODE" == acds || "$MODE" == full ]]; then PRESET=confirm; fi
DATASETS=(cifar10)
if [[ "$MODE" == full ]]; then DATASETS=(mnist cifar10); fi
RUN="${RUN_ARG:-${CART_COMPARISON_OUTPUT:-$PWD/.papermerge/local-tests/cart-comparison-${MODE}-v1}}"
if (( ! GPU_EXPLICIT )); then
  if [[ -n "${CART_COMPARISON_GPU_COUNT:-}" ]]; then GPU_ARGS+=(--gpu-count "$CART_COMPARISON_GPU_COUNT"); fi
  if [[ -n "${CART_COMPARISON_GPUS:-}" ]]; then GPU_ARGS+=(--gpus "$CART_COMPARISON_GPUS"); fi
fi
if (( ! JOBS_EXPLICIT && ! AUTO_WORKERS )) && [[ -n "${CART_COMPARISON_JOBS_PER_GPU:-}" ]]; then
  GPU_ARGS+=(--jobs-per-gpu "$CART_COMPARISON_JOBS_PER_GPU")
fi
if (( AUTO_WORKERS )) && [[ ! -f "$RUN/plan.json" ]]; then GPU_ARGS+=(--jobs-per-gpu 1); fi
if [[ -f "$RUN/plan.json" ]]; then GPU_ARGS+=(--saved-plan "$RUN/plan.json"); fi
SELECTION="$(python scripts/select_cart_gpus.py "${GPU_ARGS[@]}" --format shell)"
read -r GPU_IDS JOBS_PER_GPU <<< "$SELECTION"
echo "Mode: $MODE | Physical GPUs: $GPU_IDS | Workers per GPU: $JOBS_PER_GPU | Output: $RUN"
if (( AUTO_WORKERS )); then echo "Adaptive scheduling: at most $MAX_WORKERS workers/GPU; live memory checks before every launch."; fi
if (( DRY_RUN )); then echo 'Dry run: no verification or training started.'; exit 0; fi
SESSION="cart-comparison-${MODE}"
export MPLCONFIGDIR="$RUN/matplotlib-cache"
COMMON=(--preset "$PRESET" --output "$RUN" --profile hpc
  --gpus "$GPU_IDS" --jobs-per-gpu "$JOBS_PER_GPU"
  --gpu-memory-mib 8192 --gpu-headroom-mib 8192 --context-overhead-mib 1024
  --ram-mib-per-worker 32768 --ram-headroom-mib 32768
  --datasets "${DATASETS[@]}" --partitions iid dirichlet --noise-levels 1 --noise-values 0.01
  --share-fraction 0.05 --acds-batches 5)
if [[ "$MODE" == acds ]]; then
  COMMON+=(--arms acds --reference "$PWD/docs/results/cart-admission-cifar6-100r-20261028")
fi
if [[ "$MODE" == full ]]; then
  COMMON+=(--arms acds prior_fallback --nodes 100 --rounds 100 --local-steps 5
    --attack-start 20 --seeds 2025 2026 2027)
fi
if [[ -f "$RUN/campaign.json" ]]; then
  if (( AUTO_WORKERS )); then
    python scripts/run_cart_adaptive.py --execute --take-over --output "$RUN" \
      --max-workers-per-gpu "$MAX_WORKERS" --tmux "cart-adaptive-${MODE}" --ntfy-topic papermerge-ojaswi
  else
    python scripts/run_cart_comparison.py --execute --resume --output "$RUN" \
      --tmux "$SESSION" --ntfy-topic papermerge-ojaswi
  fi
else
  python scripts/hpc.py gpu-status
  python scripts/run_cart_comparison.py --preflight "${COMMON[@]}"
  python scripts/run_cart_comparison.py --verify-gpus "${COMMON[@]}"
  if (( AUTO_WORKERS )); then
    python scripts/run_cart_adaptive.py --execute --output "$RUN" \
      --max-workers-per-gpu "$MAX_WORKERS" --tmux "cart-adaptive-${MODE}" --ntfy-topic papermerge-ojaswi
  else
    python scripts/run_cart_comparison.py --execute --resume --output "$RUN" \
      --tmux "$SESSION" --ntfy-topic papermerge-ojaswi
  fi
fi
