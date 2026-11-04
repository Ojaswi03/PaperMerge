# Fresh FedSharp clean controls on hal9k

This is a separately versioned follow-up to the completed 150-run study.
Only base FedSharp is included. There is no FedSharp+ clean condition.

| Setting | IID | nonIID |
| --- | --- | --- |
| Datasets | MNIST and CIFAR-10 | MNIST and CIFAR-10 |
| Seeds | 2025, 2026, 2027 | 2025, 2026, 2027 |
| Proximal mu | 0 | 1 |
| Channel sigma / attackers | 0 / 0 | 0 / 0 |
| Noise-dependent objective coefficient | 0 | 0 |
| Class-aware extension / sharing | None | None |
| Clients / rounds / local SGD steps | 100 / 500 / 5 | 100 / 500 / 5 |

There are 12 configurations, 600,000 activations and 3,000,000 local SGD
updates. All start from initialization. The preset's nominal sigma=0.01 is
only inherited inner-protocol metadata; effective channel sigma is explicitly
zero in every case and no channel perturbation is applied. Dirichlet alpha=0.5
defines the nonIID allocation; it is not a noise or loss coefficient.

IID clean corrects the earlier mu=1 control. The nonIID runs repeat the mu=1
objective under the same seeds and are reproducibility controls, not additional
independent replications to pool with the old three seeds. Data partitions,
holdout, model initialization, batch sizes, learning-rate schedule, 66-model
window, screening and checked pooling follow the matched study. Mu=0 does not
make FedSharp identical to BASIL, which still uses a different predecessor
window and selection policy. No previous recorded results are relabeled.

The new revision is `fedsharp_clean_mu_by_partition_20261103_v1`. Each view
records its reference configuration and explicit effective mu; the effective
checkpoint contract records the actual objective. The original 80 producer
files remain untouched. Four new source/preset files extend the pinned source
manifest. There is no import of a round-100 or round-500 trained checkpoint.

## Launch on hal9k

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
clean_run="/var/tmp/papermerge-$(id -u)/hal9k_fedsharp-clean-500r-$(date +%Y%m%d-%H%M%S)"
python scripts/run_fedsharp_clean_controls.py --output "$clean_run" \
  --describe --write-configs "$clean_run/configs.json"
python scripts/run_fedsharp_clean_controls.py --output "$clean_run" \
  --execute --background --wait-for-resources --ntfy-topic papermerge-ojaswi
```

The background process survives normal SSH logout. Its log is
`$clean_run/clean_supervisor.log`. Preserve the printed output directory;
initialize the variable with that exact path in a later shell. Do not create a
new directory when resuming this campaign.

Initial preparation loads and verifies the original TRAIN archives. Each GPU
then passes device-local reference-versus-compiled selection and objective
checks, including the specified mu and all five steps. No training starts on
a GPU before its verification passes. The tests are synthetic execution
checks, not accuracy evidence. The controller then fills both GPU queues under
unchanged 3-GiB TensorFlow caps, 1-GiB context reservations, 8-GiB GPU headroom,
8-GiB host RAM reservations per worker and 16-GiB host headroom. Not all 12
workers necessarily fit at once; live resource admission decides concurrency.
Execution is restricted to hal9k.

## Monitor and verification

```bash
python scripts/run_fedsharp_clean_controls.py --output "$clean_run" --monitor
```

Ctrl+C closes the monitor. To select this new campaign on the permanent browser
feed and send its existing URL through ntfy:

```bash
python scripts/share_cart_monitor.py --output "$clean_run" \
  --background --restart-publisher --interval 1 --ntfy-topic papermerge-ojaswi
```

Once all 12 runs complete, fresh final-model CPU inference is available through
the same scoped study adapter:

```bash
python scripts/run_fedsharp_clean_controls.py --role verify --output "$clean_run"
```

To verify and export all twelve completed configurations in one command:

```bash
python scripts/export_fedsharp_clean_controls.py --output "$clean_run" \
  --destination docs/results/hal9k_fedsharp-clean-500r-20261104
```

This runs fresh CPU inference before exporting measurements, mechanisms,
the frozen plan, source snapshots and the inference receipt with SHA256 hashes.
Checkpoints and logs stay in the original hal9k output. Existing conflicting
export files are refused. After the command reports twelve verified exports:

```bash
git add docs/results/hal9k_fedsharp-clean-500r-20261104
python scripts/check_publication_policy.py &&
git commit -m "Add verified FedSharp clean controls with partition-specific mu" &&
git push origin main
```

After a stopped campaign has been inspected, resume its exact frozen plan with
`--output "$clean_run" --resume --execute --background --wait-for-resources`.
Do not repeat the initial launch with a different timestamp or new settings.
Source checks, existing GPU locks and exclusive supervisors still apply.

Local validation covers the exact 12-case matrix, absence of FedSharp+, altered
mu rejection, nonIID equivalence to the original objective, interruption/resume
equality, five-step selection/objective gates on both models, the scheduler
handoff and monitor labels. Actual hal9k GPU checks remain mandatory at launch.
