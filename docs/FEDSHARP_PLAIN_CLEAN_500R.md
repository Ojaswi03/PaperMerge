# Plain clean ring controls for FedSharp

This separate experiment measures the clean pass-through baseline requested on
2026-11-04. It does not overwrite either completed clean-control study.

There are twelve fresh runs: MNIST and CIFAR-10, IID and nonIID, and seeds
2025, 2026 and 2027. Each run has 100 clients, 500 rounds and five local SGD
steps per activation: 600,000 activations and 3,000,000 optimizer steps total.

Every receiver copies its immediately preceding node's latest outgoing model,
without scoring a candidate window, peak screening, ranking or averaging.
There are zero attackers, zero channel noise and no class-aware extension.
Both IID and nonIID use mu=0, with no noise-dependent gradient regularization.
Training uses ordinary cross-entropy SGD, with the previous data partitions,
holdout, model initialization, batch sizes and learning-rate schedule.

Activations run in the existing sequential order, nodes 0 through 99 per
round. Node 0 at the first activation uses the common initialized model because
no predecessor has trained yet. Every subsequent activation uses exactly the
preceding activation's trained model, including node 99 to node 0 at the round
boundary. A missing or stale predecessor fails validation; it does not trigger
selection of another peer. Final client models and development-holdout metrics
are still saved for all 100 nodes.

This is a defense-disabled ring control, not evidence that the full FedSharp
algorithm is identical to a plain ring. The earlier defense-enabled clean
controls remain separately identified. Higher accuracy is a hypothesis to
test, not a guaranteed outcome. NonIID also changes mu from one to zero, so a
comparison with its prior clean curve does not isolate selection alone.

The inner reference prototype retains its archived design-budget and nominal
sigma metadata. The new top-level view and effective checkpoint configuration
explicitly declare window=1, Byzantine budget=0, sigma=0, mu=0, and all defense
switches disabled. The reference prototype is never executed directly.
Original producer files and old run directories stay unchanged. The new study
is `fedsharp_plain_clean_ring_20261104_v1`.

## Run on themachine with four GPUs

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
plain_run="/var/tmp/papermerge-$(id -u)/themachine_fedsharp-plain-clean-500r-$(date +%Y%m%d-%H%M%S)"

python scripts/run_fedsharp_plain_clean.py \
  --output "$plain_run" --describe --write-configs "$plain_run/configs.json" &&
python scripts/run_fedsharp_plain_clean.py \
  --output "$plain_run" --execute --background \
  --wait-for-resources --ntfy-topic papermerge-ojaswi
```

Execution is restricted to themachine. The default preset selects GPUs 0–3.
Each physical GPU checks direct predecessor transfer and all five local
updates against an independently computed ordinary SGD reference before it
runs experiments. These checks use synthetic examples and make no accuracy
claim. A mismatch stops startup.

The shared single-server queue balances runs across eligible verified GPUs,
rechecking live VRAM and host RAM before each launch. Budgets remain 3 GiB
TensorFlow plus 1 GiB context per worker, 8 GiB GPU headroom, 8 GiB host RAM
per worker and 16 GiB host headroom. Other users' allocations remain in the
admission calculation. No other user's processes or reservations are modified.

The detached supervisor survives logout and writes
`$plain_run/plain_clean_supervisor.log`. Save the exact output path. After
preparing plan.json, the supervisor starts the existing permanent browser
publisher and sends its link through ntfy. This ordering avoids publishing
before a campaign plan exists. Publisher startup failure does not stop training.

```bash
python scripts/run_fedsharp_plain_clean.py --output "$plain_run" --monitor
```

Ctrl+C closes only the monitor. In another shell, first restore `plain_run` to
the exact printed directory. Do not create another timestamped output to
resume an existing campaign. Inspect any stopped supervisor before resuming:

```bash
python scripts/run_fedsharp_plain_clean.py --output "$plain_run" \
  --resume --execute --background --wait-for-resources --ntfy-topic papermerge-ojaswi
```

After all twelve runs complete, fresh final-model CPU inference uses the
matching study adapter:

```bash
python scripts/run_fedsharp_plain_clean.py --role verify --output "$plain_run"
```

Do not add the host-local output directory or checkpoints to Git. Exporting
and comparing these results must retain the distinct plain-ring identity.
