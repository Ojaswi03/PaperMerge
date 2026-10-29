# Next HPC run: six revised CIFAR-10 CART configurations

This is an evaluation of a new, explicitly versioned CART architecture. It is
not a claim that CIFAR robustness or novelty is established. Short local screens
show improvement over the collapsed v2 non-IID result; the 100-round three-seed
study is the next test. MNIST and unchanged controls are reused from the published
72-run study only after matching data, initialization, physics and numerical runtime.

## Pull and launch

The fixed default sequence is also packaged as a normal shell script:

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
bash scripts/launch_cart_cifar6_hpc.sh
```

It activates `.venv-hpc`, checks GPUs, prepares and verifies the six-case plan,
then launches the manager in tmux. The expanded equivalent follows for inspection
or customization. Run either sequence once, not both.

Run these commands on HPC. The previous campaign should already be completed;
check GPU status before starting. Do not launch two supervisors for the same output.

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
git pull --ff-only
git log -1 --oneline
python scripts/hpc.py gpu-status

RUN="$PWD/.papermerge/local-tests/cart-admission-cifar6-100r-v1"
LATEST="$PWD/.papermerge/cart-latest-figures"
export MPLCONFIGDIR="$RUN/matplotlib-cache"

# Choose exactly 1, 2 or 3 noise values. The existing controls match 0.01.
NOISE_LEVELS=1
NOISE_VALUES=(0.01)

COMMON=(
  --profile hpc --gpus 0,1,2,3
  --output "$RUN" --variant prior_neutral_class
  --jobs-per-gpu 2 --gpu-memory-mib 8192
  --gpu-headroom-mib 8192 --context-overhead-mib 1024
  --ram-mib-per-worker 32768 --ram-headroom-mib 32768
  --datasets cifar10 --partitions iid dirichlet
  --seeds 2025 2026 2027
  --noise-levels "$NOISE_LEVELS" --noise-values "${NOISE_VALUES[@]}"
  --conditions attack_noise_defense_rla_cart
  --nodes 100 --rounds 100 --local-steps 5
  --attack-start 20 --alpha 0.5 --model paper
  --reference "$PWD/docs/results/cart-v2-72-100r-20261028"
  --latest-figures "$LATEST"
)

python scripts/run_cart_revision_study.py --preflight "${COMMON[@]}" &&
python scripts/run_cart_revision_study.py --verify-gpus "${COMMON[@]}" &&
python scripts/run_cart_v2_managed.py --execute --scheduler admission \
  "${COMMON[@]}" --figures "$LATEST" \
  --ntfy-topic papermerge-ojaswi --tmux cart-admission-cifar6
```

Preflight must print **6 configurations, 100 nodes and 100 rounds**. There are
six independent experiments total, with at most two on one GPU; adding six slots
per GPU cannot increase concurrency beyond six experiments. Memory headroom is
checked before admission. GPU verification checks exact candidate scores and the
fresh/local-carrier objectives on the installed TensorFlow build.

One noise value gives 6 runs; two give 12; three give 18. For example, change the
two noise lines to `NOISE_LEVELS=3` and `NOISE_VALUES=(0.005 0.01 0.02)` **before
creating a fresh plan**. The old 72-run reference contains only 0.01, so additional
levels require matching controls or standalone plots; remove `--reference` for
a standalone study. Do not reuse a plan with changed settings.

## Monitor, disconnect and retry

```bash
python scripts/run_cart_revision_study.py --monitor --output "$RUN"
```

Ctrl+C closes only this monitor. Training, plotting and notifications run inside
tmux after SSH disconnect. ETA is a forecast, initially using workload priors;
measurements replace those priors. A six-run study should be much shorter than
repeating 72 runs, but no fixed speedup is guaranteed before HPC measurements.

A failed worker is recorded in `failed_runs.json` and `worker_failure_events.jsonl`;
its slot proceeds to the next assigned experiment. Healthy workers keep running.
Failed experiments are attempted only once per invocation. A shared filesystem
failure can still affect the supervisor; failures are isolated when the filesystem
can record and launch work normally.

After the supervisor finishes, retry incomplete runs using the frozen plan:

```bash
python scripts/run_cart_v2_managed.py --execute --scheduler admission \
  --resume --output "$RUN" --figures "$LATEST" \
  --ntfy-topic papermerge-ojaswi --tmux cart-admission-cifar6-retry
```

Completed runs are skipped. Valid checkpoints resume from their committed state;
absent checkpoints start from zero. Invalid checkpoints remain preserved and the
configuration is marked failed; they are never silently discarded or mixed with
another revision. A changed source contract requires a fresh output.

## Plots and export

Completion automatically updates these stable visible files:

- `$LATEST/cifar10_sigma_0.01_learning_curves.png`
- `$LATEST/cifar10_sigma_0.01_final_bars.png`
- `$LATEST/cifar10_sigma_0.01_primary_bars.png` (last ten rounds)
- `$LATEST/seed_metrics.csv`, `round_averages.csv` and `summary.json`

PDF and SVG versions are also generated. The latest view includes the unchanged
baseline/MNIST curves and replaces only the completed new CIFAR CART groups.
All three new seeds are required for replacement; two new seeds are never mixed
with one old seed. Previous visible figures are archived beside the latest folder.

To regenerate plots without training, or export compact results for Git:

```bash
python scripts/plot_cart_revision.py --output "$RUN" --figures "$LATEST"
python scripts/export_cart_revision.py --output "$RUN" --figures "$LATEST" \
  --destination "$PWD/docs/results/cart-admission-cifar6-100r-v1"
git add -- docs/results/cart-admission-cifar6-100r-v1
git diff --cached --stat
git commit -m "Publish six-run revised CART CIFAR evaluation"
git push origin main
```

Export includes metrics, provenance, source snapshots and figures, without model
checkpoints. Conflicting existing exports are preserved. A plot/runtime compatibility
failure does not require retraining: correct the reference or generate standalone
plots. Notifications give a short status, counts and run name; detailed errors
remain in local logs.

Independent CPU inference is available while checkpoints remain on HPC, but can
take appreciable time for six full models across 100 simulated clients:

```bash
python scripts/verify_cart_revision_results.py --output "$RUN"
```

This recomputes final TRAIN-holdout accuracy and checks recorded invariants; it
does not independently replay training. Official TEST evaluation and an independent
confirmatory experiment remain separate from development holdout results.
