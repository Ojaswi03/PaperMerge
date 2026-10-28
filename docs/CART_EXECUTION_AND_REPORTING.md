# CART execution, runtime estimates, figures and phone alerts

For new launches use [failure-isolating execution and recovery](CART_FAILURE_RECOVERY.md).
It supersedes the two-slot/35-round launch sequence below with the requested
six-slot/100-round study and continues peers after worker failures. The older
parallel launcher remains available to preserve existing frozen contracts.

These changes preserve the frozen CART v2 training source. They change the
execution and reporting layer: independent worker slots, resource admission,
progress estimation, saved-metric figures and ntfy messages.

## What changes and what stays fixed

Up to two independent configurations can share each selected physical GPU.
Each configuration still runs its logical clients sequentially with the same
datasets, partitions, seeds, model, attack, Gaussian noise, five SGD steps,
selection, RLA and CART equations. No data reduction or disabled evaluation
is used to manufacture a runtime improvement. An individual run can slow down
under sharing; total throughput is the quantity to benchmark.

The [local measured benchmark](validation/CART_V2_CONCURRENCY_BENCHMARK_20261028.json)
ran the same two full-TRAIN CIFAR-10 IID cases with 100 logical nodes, two
rounds, early attack0, both mitigations and CART. On the RTX4070Ti with a
3 GiB cap per worker, one slot took 258.64 seconds and two took 128.60 seconds:
**2.01x throughput**. Final arrays, reference memory, telemetry and round
metrics were identical. This is a bounded runtime/equivalence check; it does
not establish long-horizon efficacy, non-IID performance, or the same speedup
on the HPC's different GPUs. If similar throughput transfers, the earlier
5–7 hour CIFAR-10 estimate becomes roughly 2.5–4 hours; observe live HPC rates
before treating that as a schedule.
The benchmark executor is preserved in commit `b1c0f290`; the subsequent
constant-time deque queue and stronger startup-objective selection change
scheduling housekeeping/verification only, not the frozen training worker.

The [managed-workflow validation](validation/CART_V2_MANAGED_VALIDATION_20261028.json)
also completed 72 small real-data GPU cases across both datasets, splits,
three seeds and all six conditions, followed by automatic figures/CSVs and
independent CPU verification. These used four nodes/two rounds and small
TRAIN subsets: they validate execution, not research efficacy. The twelve
CART cases were rerun serially and matched every final array, reference state,
telemetry entry and round metric, including five accepted CART corrections.
No GPU worker processes remained after validation. The full suite passed
1,265 tests and 644 subtests, with four skips; later targeted checks also pass.

The supervisor checks the sum of all admitted GPU memory caps plus headroom.
With two 8,192 MiB workers and 2,048 MiB headroom, it requires at least
18,432 MiB free per selected GPU before execution. A coordinated parent holds
the existing exclusive physical-GPU lease; only its own admitted children
inherit that lease. Unrelated PaperMerge workers remain excluded. This is not
a reservation against other users on the shared machine.

Every configuration has a fresh interpreter and its own CUDA/TensorFlow
context. Exiting releases its GPU allocations and compiled graph caches.
Do not clear the model or compiled graphs during its activations: that would
discard useful state or force expensive recompilation. Do not drop operating
system caches. Heavy libraries remain lazy in orchestration; chart generation
loads NumPy/Matplotlib only after training finishes.
Pending queues use constant-time deque insertion/removal; worker/slot lookup
uses dictionaries. These small orchestration savings do not replace the
necessary model evaluations.
Persistent CUDA compilation caches are retained so Blackwell does not repeat
avoidable PTX compilation for every configuration.

Exact recovery skips completed configurations and resumes interrupted ones
from committed checkpoints. Identical source, configurations and execution
profiles are required. A method change requires a fresh version/output and a
review of which earlier comparisons remain scientifically compatible.

The underlying window/risk evaluation cannot become O(1) in the number of
candidates while preserving its exact decision. This implementation already
keeps only current sender versions and reconstructs deterministic link noise
lazily; no unbounded model history is added as a cache.

## ETA interpretation

The monitor always provides a numerical forecast for an active prepared plan:

1. Use the current run's measured seconds per activation after a full ring.
2. Otherwise use matching measured workloads, pooling seed/noise-specific run
   IDs correctly and preferring the same execution slot.
3. Otherwise use matching completed campaigns supplied by `--timing-from`.
   Training source, host, profile and concurrency must match.
4. Otherwise use an explicitly labeled workload prior from observed HPC pilot
   timings, with a broad planning range. Two-slot priors include a contention
   buffer until live measurements replace them.

The campaign forecast is the maximum remaining sequential work over its
execution slots, reflecting parallel completion. GPU slots are distinct even
when they share one physical GPU. Displayed finish times default to
America/New_York. The range is a planning range, not a statistical confidence
interval or a promised deadline. Setup/JIT, shared hardware and changing work
can move the estimate. A stopped campaign is labeled not running rather than
assigned a fictional completion time. The forecast covers training; figure
export follows completion.

## Automatic figures

On successful completion, the manager writes `RUN/figures/`:

- Learning curves: mean node holdout accuracy within each seed, then mean and
  sample standard deviation across three seeds at each round.
- Final-round average bars and separate configured-last-ten-round average
  bars, with across-seed sample SD error bars.
- IID and non-IID panels side by side, separately for each dataset and sigma.
- PNG, PDF and SVG versions; `seed_metrics.csv`, `round_averages.csv`,
  `summary.json` and an artifact/input-hash manifest.

No best-round selection or averaging across noise levels is performed.
Negative and chance-level results remain visible. A clean baseline can appear
in multiple sigma panels as the same reference, without counting it as another
replication. All three selected seeds and matching CART controls must complete
before final figures are exported. Different horizons/onsets are not pooled.
JSON/CSV accuracies are fractions; plotted axes are percentages. Figures use
saved TRAIN-holdout metrics and do not load official TEST data. Independent
inference verification remains a separate command and can take extra time.

## Phone notifications

The manager publishes a completion alert only after training and chart export
succeed. A training, interruption or chart-export error produces a different
alert and leaves the error in `managed_status.json`. Delivery failures are
retried and recorded; they do not erase successful training. An accepted HTTP
response establishes server acceptance, not guaranteed phone delivery.

The implementation follows the [official ntfy JSON publishing API](https://docs.ntfy.sh/publish/#publish-as-json).
Pass the server/topic explicitly. Authentication, if needed, is read from
`--ntfy-token-file`; tokens are not stored in manifests or command text.

## Complete HPC sequence: 72 focused runs

Stop the previous campaign gracefully if it is still using these GPUs. Use
the session name you started (`cart-v2-pilot` or `cart-v2-focused`), preserve
all results, and repeat the GPU check until its processes have exited:

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
tmux ls
# Run the appropriate one only if that session is still running:
# tmux send-keys -t cart-v2-pilot C-c
# tmux send-keys -t cart-v2-focused C-c
python scripts/hpc.py gpu-status
git pull --ff-only

python scripts/run_cart_v2_managed.py --test-notification \
  --ntfy-server https://ntfy.sh --ntfy-topic papermerge-ojaswi

RUN="$PWD/.papermerge/local-tests/cart-v2-focused-72-35r-managed-v1"
COMMON=(
  --profile hpc --gpus 0,1,2,3
  --output "$RUN" --jobs-per-gpu 2 --gpu-memory-mib 8192
  --datasets mnist cifar10 --partitions iid dirichlet
  --seeds 2025 2026 2027 --sigmas 0.005 0.01 0.02 --only-sigma 0.01
  --conditions attack_noise_defense_rla attack_noise_defense_rla_cart
    clean attack_noise attack_noise_defense attack_noise_rla
  --nodes 100 --rounds 35 --local-steps 5
  --attack-start 20 --alpha 0.5 --model paper
)

python scripts/run_cart_v2_parallel.py --preflight "${COMMON[@]}" &&
python scripts/run_cart_v2_parallel.py --verify-gpus "${COMMON[@]}" &&
python scripts/run_cart_v2_managed.py --execute "${COMMON[@]}" \
  --ntfy-topic papermerge-ojaswi --tmux cart-v2-managed

python scripts/monitor_sharp_study.py --output "$RUN" \
  --timezone America/New_York
```

Preflight must report 72 configurations. Eight slots are permitted on four
GPUs after memory/lease checks. If a short local benchmark does not transfer
to HPC throughput, stop and use one slot in a fresh output; do not claim a
fixed speedup from memory capacity alone. The actual HPC monitor updates the
forecast using its observed sharing overhead.

The tmux manager continues after SSH disconnect. Ctrl+C in the monitor closes
only the monitor. After completion the phone alert includes the figure path.
To restart a shell and inspect artifacts:

```bash
RUN="$PWD/.papermerge/local-tests/cart-v2-focused-72-35r-managed-v1"
ls "$RUN/figures"
cat "$RUN/managed_status.json"
python scripts/verify_sharp_results.py --require-complete --output "$RUN"
```

If training was already started through the original launcher and should
remain unchanged, an independent watcher can add chart export and notification:

```bash
python scripts/run_cart_v2_managed.py --watch --output "$RUN" \
  --ntfy-topic papermerge-ojaswi --tmux cart-v2-notify
```

To retry chart export/notification for a completed campaign:

```bash
python scripts/run_cart_v2_managed.py --postprocess-only --output "$RUN" \
  --ntfy-topic papermerge-ojaswi
```

Valid existing artifacts are checked and reused. If figures need to be
regenerated, use `--figures` with a fresh directory. Do not start both a watcher
and a managed executor for the same output. Postprocessing cannot invent
missing results or an efficacy gain for CART.
