# FedSharp and FedSharp+ noise-only controls

For execution on themachine instead, use the
[themachine launcher and commands](FEDSHARP_NOISE_ONLY_THEMACHINE.md).
It prioritizes GPUs 1 and 3 and can admit GPUs 0 and 2 later. Do not launch both
server campaigns independently as if they were different configurations.

Prepared for hal9k, GPUs 0 and 1. These are new experiments awaiting execution,
not additional completed results. Their purpose is to measure channel noise
without Byzantine attacks, using the existing impaired-environment algorithms.

| Method | Datasets | Partitions | Seeds | Runs |
| --- | --- | --- | --- | ---: |
| FedSharp | MNIST, CIFAR-10 | IID, non-IID | 2025, 2026, 2027 | 12 |
| FedSharp+ | MNIST, CIFAR-10 | non-IID | 2025, 2026, 2027 | 6 |

Every run starts from initialization and trains for 500 rounds with 100 clients
and five local SGD steps per activation. Total: 900,000 activations and 4,500,000
optimizer steps. No attack-trained checkpoint is reused.

Physical channel sigma is 0.01. Actual attackers are zero and the attack switch
is false. The defensive design budget remains 33 and the candidate window 66.
Proximal mu remains 1 in both partitions, matching the existing noisy FedSharp
runs. The existing learning-rate schedule, data partitions, development holdout,
model initialization, probes, 19 calibration draws, thresholds and FedSharp+
repair/admission/commit rules are retained. No equations are changed.

The adapter stores the original joint-case configuration as reference metadata
inside `comparisonCase.protocolConfig`. The effective checkpoint configuration
explicitly declares zero attackers, noise only and the new study revision.
Reference metadata is not the physical configuration executed by the worker.
Each new run has its own ID, output and frozen source snapshot.

The existing checked execution optimization is enabled: cached fixed probes,
incremental summaries and cached checkpoint serialization. Its zero-coefficient
training branch requires exact comparisons against the original branch on the
actual device; it keeps the original branch if those comparisons fail. It does
not reduce calibration draws or change acceptance thresholds. Each run saves
`optimization.json` and per-round wall intervals in `round_timings.jsonl`.
Resume segments and partial rounds are marked separately in the timing file.

## Launch on hal9k

Run this block once. The timestamp creates a fresh experiment directory.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
noise_run="/var/tmp/papermerge-$(id -u)/hal9k_fedsharp-noise-only-500r-$(date +%Y%m%d-%H%M%S)"

python scripts/run_fedsharp_noise_only.py \
  --output "$noise_run" --describe \
  --write-configs "$noise_run/configs.json" &&
python scripts/run_fedsharp_noise_only.py \
  --output "$noise_run" --execute --background \
  --wait-for-resources --ntfy-topic papermerge-ojaswi
```

The supervisor prints the exact output directory and writes
`noise_only_supervisor.log` there. Keep that path. The browser publisher starts
after the plan exists and sends its link through ntfy. Both GPU gates must pass
before training begins. Local CPU checks do not replace those device checks.

The shared scheduler balances admissions across both GPUs according to live
free VRAM and host RAM. It retains 8 GiB host RAM per worker, 16 GiB host RAM
headroom, 3 GiB TensorFlow VRAM plus context allowance per worker, and GPU
headroom. The configured maximum of 24 workers per GPU is a ceiling, not a
promise of capacity. Hal9k's system RAM can limit concurrency well before its
GPU memory fills. Other users' processes and existing experiments are not
stopped or modified.

Monitor from the same terminal:

```bash
python scripts/run_fedsharp_noise_only.py --output "$noise_run" --monitor
```

Ctrl+C closes only this monitor. To inspect startup:

```bash
tail -n 50 "$noise_run/noise_only_supervisor.log"
```

In a new terminal, set `noise_run` to the exact existing path printed at launch.
Do not create another timestamp. If the supervisor stops, resume that output:

```bash
python scripts/run_fedsharp_noise_only.py \
  --output "$noise_run" --resume --execute --background \
  --wait-for-resources --ntfy-topic papermerge-ojaswi
```

## Verify and export after completion

```bash
python scripts/export_fedsharp_noise_only.py \
  --output "$noise_run" \
  --destination docs/results/hal9k_fedsharp-noise-only-500r-20261114
```

The exporter requires all 18 complete runs, frozen sources and measurements,
zero-attack evidence for every recorded activation, and fresh CPU development
holdout inference of all 1,800 retained final client models. It exports metrics,
timings, sources and receipts with checksums. Checkpoints, logs and private
runtime state remain outside Git. A failed verification must be diagnosed;
do not publish a partial export as a verified result.

After a successful export, publish from this shared checkout once:

```bash
git add docs/results/hal9k_fedsharp-noise-only-500r-20261114
python3 scripts/check_publication_policy.py &&
git commit -m "Add verified FedSharp noise-only 500-round controls" &&
git push origin main
```

Official test images are not used during training or device gates. For an
additional post-training accuracy/recall check after freezing the models:

```bash
python scripts/run_fedsharp_noise_only.py \
  --output "$noise_run" --role verify --official-test
```

This optional check is separate from the export's development verification.
Do not tune on its test results and later describe the same test set as untouched.
The existing full-test evaluation utility is needed for saved per-image
predictions and test-loss reporting; this verifier alone reports accuracy and
recall, not that full evaluation package.

## Timing and interpretation

[Recorded training and evaluation runtimes](runtime-timings/20261114/report.md)
provides per-round averages and 500-round totals for clean, BASIL, BASIL + ACDS,
FedSharp and FedSharp+. Those historical mixed-host, mixed-load measurements
are not a controlled algorithm-speed benchmark or a measured ETA for this new
campaign. Its measured ETA becomes useful after actual workers make progress.

Only BASIL MNIST noise-only controls existed previously. This campaign adds
FedSharp and FedSharp+ controls; it does not create CIFAR-10 BASIL noise-only or
BASIL + ACDS noise-only results. Keep that comparison gap explicit in the thesis.
Report three independent seeds per new curve. Do not pool repeated controls
with older runs as extra seeds or claim significance from a curve alone.
