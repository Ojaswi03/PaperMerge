# Matched BASIL clean ring controls on themachine

These are twelve fresh BASIL clean runs: MNIST and CIFAR-10, IID and nonIID,
and seeds 2025, 2026 and 2027. Each run uses 100 clients, 500 rounds and five
local SGD steps: 600,000 activations and 3,000,000 optimizer steps total.

Every receiver accepts its immediate predecessor's latest outgoing model.
There are no candidate loss comparisons, peak screening, ranking or averaging.
Both splits use mu=0, zero attackers, zero channel noise and ordinary
cross-entropy SGD. ACDS, class adjustment and all other defenses are disabled.
Only node 0 at the very first activation uses the common initialized model;
every later activation receives the preceding activation's trained model,
including the node 99 to node 0 round boundary. Missing or stale predecessor
versions fail validation rather than selecting another sender.

The data partition, holdout, initialization, keyed batches, learning-rate
schedule and SGD budget match the completed direct-transfer FedSharp controls.
The two clean procedures are numerically equivalent under the same execution.
Each GPU checks exact predecessor transfer, all five updates against an
independent ordinary-SGD calculation, and equality with the matching FedSharp
five-step update before it starts training. Those synthetic checks do not
establish full-run accuracy. Compare the recorded final runs before claiming
full-run equality, including any hardware/runtime differences.

The new revision is `basil_plain_clean_ring_20261104_v1`. The original BASIL
clean runs with selection among 34 predecessors remain archived. These new
runs must not be pooled with the old runs or with FedSharp as extra independent
seeds. The archived inner reference prototype retains its nominal design
budget and noise metadata; the actual top-level view and effective checkpoint
configuration explicitly use window=1, budget=0, sigma=0 and mu=0.

The new study pins the original 80 files, four existing FedSharp clean-ring
files and four new BASIL files. Existing scientific files and output folders
are not edited. The predecessor selector is shared with the already tested
FedSharp control. The new BASIL identity is preserved in configs, contracts,
saved measurements and monitor labels.

## Launch on themachine, GPUs 0–3

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
basil_run="/var/tmp/papermerge-$(id -u)/themachine_basil-clean-500r-$(date +%Y%m%d-%H%M%S)"

python scripts/run_basil_plain_clean.py --output "$basil_run" \
  --describe --write-configs "$basil_run/configs.json" &&
python scripts/run_basil_plain_clean.py --output "$basil_run" \
  --execute --background --wait-for-resources --ntfy-topic papermerge-ojaswi
```

Execution is restricted to themachine. A shared queue dynamically fills the
four verified GPUs under live memory admission. Existing headroom remains:
3 GiB TensorFlow plus 1 GiB context per worker, 8 GiB GPU headroom,
8 GiB host RAM reserved per worker and 16 GiB host headroom. Other users'
allocations remain accounted for. Their processes and settings are untouched.

The background supervisor writes `$basil_run/basil_clean_supervisor.log` and
survives logout. Save the exact path. After plan.json exists, it starts the
permanent dashboard publisher and sends the link through ntfy. All four GPUs
must pass verification before campaign dispatch starts. Publisher failure
does not stop training.

```bash
python scripts/run_basil_plain_clean.py --output "$basil_run" --monitor
```

Ctrl+C closes only the monitor. In another shell, restore `basil_run` to the
exact existing path. If a stopped supervisor has been inspected, resume that
same frozen plan rather than creating a new timestamp:

```bash
python scripts/run_basil_plain_clean.py --output "$basil_run" \
  --resume --execute --background --wait-for-resources --ntfy-topic papermerge-ojaswi
```

## Verify and export after completion

```bash
python scripts/export_basil_plain_clean.py --output "$basil_run" \
  --destination docs/results/themachine_basil-plain-clean-500r-20261104
```

This audits the recorded predecessor model-hash chain and ordinary-SGD policy
for every activation, performs fresh CPU inference on all 1,200 final client
models, and exports compact measurements, source snapshots and verification
receipts with checksums. Checkpoints and logs stay in the original server
output. After it reports twelve configurations, freshInferenceVerified=true
and plainRingRoutingVerified=true:

```bash
git add docs/results/themachine_basil-plain-clean-500r-20261104
python scripts/check_publication_policy.py &&
git commit -m "Add verified matched BASIL clean ring controls" &&
git push origin main
```
