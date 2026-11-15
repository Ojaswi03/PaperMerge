# Noise-only controls on themachine

The [automatic server launcher](FEDSHARP_NOISE_ONLY_AUTO.md) selects this
configuration when the actual hostname is themachine.

Use this launcher when running the eighteen prepared noise-only experiments on
themachine. It prioritizes GPUs 1 and 3. GPUs 0 and 2 can join later for pending
configurations. Hal9k does not need to be reachable for this campaign to run.

The scientific matrix is unchanged: twelve FedSharp runs across MNIST/CIFAR-10
and IID/non-IID, plus six non-IID FedSharp+ runs, with seeds 2025/2026/2027.
All start from initialization, train 500 rounds with 100 clients and five local
SGD steps, and use sigma 0.01 with zero attackers. The defensive window,
proximal mu, learning-rate schedule and all method thresholds remain those
described in [the original noise-only guide](FEDSHARP_NOISE_ONLY_500R.md).
Total: 900,000 activations and 4,500,000 optimizer steps.

This is a separate server launch of the same experiments, not an additional
set of independent seeds. Do not launch the old hal9k campaign as well. This
launcher cannot inspect an unreachable hal9k or stop work there. If the old
campaign was already started, keep its results separate and reconcile duplicate
configurations when access returns. The two launchers do not share a cross-host
claim queue. Their distinct output directories prevent storage collisions.

The original 87 producer files and the hal9k launcher remain unchanged. This
launcher pins three additional operational files, producing a 90-file source
contract. Existing hal9k checkpoints are not resumed on themachine or relabeled.

## Launch once on themachine

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

noise_run="/var/tmp/papermerge-$(id -u)/themachine_fedsharp-noise-only-500r-$(date +%Y%m%d-%H%M%S)"

python scripts/run_fedsharp_noise_only_themachine.py \
  --output "$noise_run" --describe \
  --write-configs "$noise_run/configs.json" &&
python scripts/run_fedsharp_noise_only_themachine.py \
  --output "$noise_run" --execute --background \
  --wait-for-resources --ntfy-topic papermerge-ojaswi
```

Keep the exact output path printed by the supervisor. It starts the browser
publisher after preparing the plan and sends the link through ntfy. Startup
loads and checks the fitting archives; no official test images are used.

Monitor from the same terminal:

```bash
python scripts/run_fedsharp_noise_only_themachine.py \
  --output "$noise_run" --monitor
```

Ctrl+C closes only the monitor. Inspect preparation and device checks with:

```bash
tail -n 50 "$noise_run/noise_only_supervisor.log"
tail -n 30 "$noise_run/gpu_1_gate.log" "$noise_run/gpu_3_gate.log"
```

The gate logs appear when those device checks start. Each GPU must pass its own
source/runtime, selection/SGD and informative FedSharp+ checks before receiving
training work. Verification is performed one device at a time. A device check
can temporarily delay further dispatch, while active training continues.
Busy optional devices do not have to pass a gate before GPUs 1 and 3 can start.
A failed gate sets aside that device; other usable devices can continue.

## How devices join

- GPUs 1 and 3 are preferred and use the existing live RAM/VRAM admission checks.
- GPUs 0 and 2 additionally require utilization samples at or below 35% over
  at least 30 seconds. A busy or unknown sample resets that interval.
- Every admission preserves the existing 8 GiB GPU headroom, 3 GiB TensorFlow
  limit plus 1 GiB context allowance per worker, 8 GiB host RAM per worker and
  16 GiB host RAM headroom. Full unallocated budgets of starting workers are
  reserved. Memory is sampled again after each device check and each launch.
- Workers are balanced by active count, with available memory also considered.
  A newly available GPU receives pending work. Already running CUDA workers
  are not migrated, paused or restarted to rebalance them.
- Other users' processes are not signaled, evicted or modified. Device locks
  coordinate this account's compatible PaperMerge launchers, not other users.
  Their future allocations can still change resource availability.

The 35% and 30-second rules are conservative operational scheduling choices
for sharing optional GPUs. They are not scientific model hyperparameters,
accuracy-based tuning or a guarantee of exclusive GPU capacity. If all eighteen
configurations are already active, there is no queued work for later devices.

## Resume the same campaign

In a new terminal, set `noise_run` to the exact directory printed at launch.
Do not generate a new timestamp or pass new scientific/resource overrides.

```bash
python scripts/run_fedsharp_noise_only_themachine.py \
  --output "$noise_run" --resume --execute --background \
  --wait-for-resources --ntfy-topic papermerge-ojaswi
```

The scheduler adopts recognized active workers without restarting them.
Checkpoint recovery retains the original GPU and runtime contract. Failed
training runs are set aside for diagnosis rather than silently restarted.

## Verify and export after all runs complete

```bash
python scripts/run_fedsharp_noise_only_themachine.py \
  --output "$noise_run" --role export \
  --destination docs/results/themachine_fedsharp-noise-only-500r-20261114
```

Use this entrypoint for themachine export so fresh CPU verification loads the
matching 90-file source contract. The export requires the complete eighteen-case
matrix, zero-attack evidence throughout training, original stable measurements,
and fresh development-holdout inference for all 1,800 final client models.
Checkpoints and logs remain private. The checksum manifest covers the exported
metrics, round timings, optimization receipts, sources and verification receipts.

After successful verification:

```bash
git add docs/results/themachine_fedsharp-noise-only-500r-20261114
python3 scripts/check_publication_policy.py &&
git commit -m "Add verified themachine noise-only 500-round results" &&
git push origin main
```

Training completion also writes `noise_only_training_summary.json` locally,
reporting final-round and rounds-491-to-500 development means per run. It does
not run the historical joint-only plot generator. Fresh inference verification
is still required before publication. Official test evaluation remains a
separate post-training step; see the original guide for its scope and limits.

Historical training and evaluation times are in the
[runtime report](runtime-timings/20261114/report.md). They do not establish the
runtime of this new campaign on themachine's current shared load.
