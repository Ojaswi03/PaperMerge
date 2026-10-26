# CART: fixed 30-round, three-seed comparison

This is an exploratory TRAIN-holdout comparison, not an approved final thesis
result. The existing 117706-parameter CNN, BASIL handoff/selection, prior-adjusted
CE, full EBM and selected-snapshot anchor remain in place. The CART candidate
adds the class-loss barrier with a detached correction capped at 10% of the
base CE gradient norm. It has not established a novelty or Byzantine robustness
guarantee. Earlier local evidence is mixed; see [the audit](IMPLEMENTATION_AUDIT.md).

## Fixed study

- Seeds: **2025, 2026, 2027**, selected before this comparison's outcomes.
- One fixed balanced 1000-example holdout from TRAIN, using holdout seed 2025;
  the remaining 49000 TRAIN examples enter SGD. Official TEST is never opened.
- Dirichlet alpha 0.2; each seed changes the partition, initialization, attacker
  selection, data order, augmentation and keyed channel/attack randomness.
  Controls and CART share those identities within each seed.
- A/B/C/D, each with prior-only control and capped CART: **24 arms** total.
- 30 ring rounds, 10 nodes, five complete local epochs, batch 512 including the
  last partial batch. No microbatching, precision change or helper data is added.
- Attack start index 20, preserving existing code semantics: 20 warm-up rounds,
  followed by 10 attack rounds for B/C/D. C/D channel noise starts at index 0.
- Primary endpoint: CART minus paired-control mean-node validation accuracy,
  averaged over round indices 20–29. Final mean/worst and all seed outcomes are
  also reported. No best-seed or best-round selection.

One comparison pair runs per GPU: control followed by CART. With GPUs 0,1,2,3,
A/B/C/D are assigned respectively; each GPU processes all three seeds. The
scheduler never runs two of its jobs on the same GPU simultaneously. Each
worker caps its allocation at 8192 MiB under the opt-in HPC profile, with one
CPU thread and no data-loader processes. The local profile retains GPU 0 and
its existing 6144 MiB cap. Check current availability; project locks do not
reserve the university GPUs against other users.

## Before the server commands

**Commit/push the reviewed local source first, then pull on HPC.** The new
scripts must exist in the server checkout. Do not stage `.papermerge/`, imported
results, environments or the existing local GUI queue changes. No commit/push
is performed automatically. Keep scientific source unchanged during a campaign;
source changes invalidate exact recovery and device receipts.

For a fresh clone, after VPN/SSH access is available:

```bash
ssh os675524@themachine.cs.ohio.edu
mkdir -p ~/Masters_Research
cd ~/Masters_Research
git clone https://github.com/Ojaswi03/PaperMerge.git
cd PaperMerge
bash scripts/setup_hpc.sh
source .venv-hpc/bin/activate
python scripts/hpc.py cache-data --download
```

For the existing server checkout:

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
python -m pip check
python scripts/hpc.py gpu-status
```

If the environment is missing, use `bash scripts/setup_hpc.sh` and activate it.
The script uses the committed HPC requirements without root access. Installation
and the new GPU path remain unverified on The Machine until the checks below
actually pass there. Existing CIFAR caches are reused; preflight never deletes
or downloads data.

## Validate, verify all four GPUs, then run the bounded pilot

The following two commands start no study. The first inspects actual source,
TRAIN data, partition identities, resource selection and planned workload. The
second performs real CPU/GPU gradients, CE/EBM/anchor/retention checks, the
correction bound and all three seeded initialization checks on each selected GPU.
It prints a physical-GPU heartbeat every five seconds and starts zero research
runs. Its receipts bind source, software, driver, GPU UUID and resource settings.

```bash
RUN="$PWD/.papermerge/local-tests/cart-replication-hpc-30r-v1"

python scripts/run_cart_replication.py --preflight \
  --profile hpc --gpus 0,1,2,3 --output "$RUN"

python scripts/run_cart_replication.py --verify-gpus \
  --profile hpc --gpus 0,1,2,3 --output "$RUN"
```

Run the controlled real-CIFAR pilot on GPU 0 before starting the parallel study:

```bash
python scripts/run_cart_replication.py --pilot \
  --profile hpc --gpus 0 \
  --output "$PWD/.papermerge/local-tests/cart-replication-hpc-pilot-v1"
```

The pilot uses seed 2026 and D's EBM/noise path for both control and CART.
It launches a fresh Python worker with the same CUDA wheel library environment
used by GPU verification and campaign workers. Setting loader paths inside an
already running Python process is insufficient for the Linux startup loader.
It stops after three activations, restores the checkpoint, performs one more,
and compares the complete saved state against four uninterrupted activations.
Models, BASIL history, CART memory, metrics and telemetry must match exactly.
`PILOT_PASSED` is a recovery/execution check, not a completed study or an accuracy
result. Pilot output is separate and cannot be promoted into the full campaign.
If any check fails, inspect its error/log and fix the cause before proceeding.
After pulling a source fix, rerun `--verify-gpus` to refresh the source-bound
receipts, then rerun the pilot in a fresh output directory. Existing caches and
completed experimental evidence do not need to be deleted.

## Launch and monitor the full comparison

Only after the installation, preflight, four GPU checks and pilot succeed:

```bash
python scripts/run_cart_replication.py --execute \
  --profile hpc --gpus 0,1,2,3 --output "$RUN" \
  --tmux papermerge-cart-30r

python scripts/run_cart_replication.py --monitor --output "$RUN"
```

The monitor refreshes the same terminal screen every five seconds, prioritizes
active arms, and stops when the campaign finishes. ETA is estimated from measured
activation timings; early estimates can move substantially. Mean/worst accuracy
is from the held-out TRAIN subset. Initialization, GPU verification and checkpoint
I/O can leave a GPU temporarily idle. The campaign reports completion only after
all 24 arms and their saved evidence pass verification.

Ctrl+C exits the independent monitor. The experiment supervisor runs inside
tmux, so SSH can disconnect while it continues. To reconnect:

```bash
ssh os675524@themachine.cs.ohio.edu
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
RUN="$PWD/.papermerge/local-tests/cart-replication-hpc-30r-v1"
python scripts/run_cart_replication.py --status --output "$RUN"
python scripts/run_cart_replication.py --monitor --output "$RUN"
```

For supervisor output, `tmux attach-session -t papermerge-cart-30r`; detach with
Ctrl+B then D. Per-pair training logs are in `$RUN/logs/`, for example:

```bash
tail -n 60 "$RUN/logs/seed-2025-D.log"
```

## Stop, recover and review

```bash
python scripts/run_cart_replication.py --stop --output "$RUN"
python scripts/run_cart_replication.py --status --output "$RUN"
```

Wait for `STOPPED` and the workers to exit. Eligible work resumes from full
activation-boundary checkpoints; an interrupted activation repeats from its
start. A failed job also stops new launches and asks the other active jobs to
stop gracefully. Inspect the logs and current GPU processes first. Resume using
the same source, software, GPUs and resource settings:

```bash
python scripts/run_cart_replication.py --execute --recover \
  --profile hpc --gpus 0,1,2,3 --output "$RUN" \
  --tmux papermerge-cart-resume
```

Completed arms are skipped. Exact source/data/runtime mismatches are rejected;
recovery does not silently change the method or start from final weights only.
If source changes are needed, retain the old evidence and create a separate,
explicit new campaign rather than overwriting it.

After completion, verify and generate all seed/pair plots without training:

```bash
python scripts/report_cart_replication.py "$RUN" --plot
```

Top-level `review.json` and `REVIEW.md` include every seed and the predeclared
endpoint. Seed/pair folders contain full curves, sources, configs inside
`run.json`, checkpoints and immutable activation telemetry.

## Retrieve evidence to the local computer

For complete independent re-verification, transfer the full campaign separately
from Git. Run this on the local computer:

```bash
cd /home/ojaswi/PaperMerge
mkdir -p .papermerge/local-tests/cart-replication-hpc-import
rsync -av \
  os675524@themachine.cs.ohio.edu:~/Masters_Research/PaperMerge/.papermerge/local-tests/cart-replication-hpc-30r-v1/ \
  .papermerge/local-tests/cart-replication-hpc-import/
environment/basil-noise-env/bin/python scripts/report_cart_replication.py \
  .papermerge/local-tests/cart-replication-hpc-import --plot
```

Read-only reporting supports relocation of complete evidence. A copied campaign
is not an exact local GPU resume of the server computation. Keep generated
artifacts ignored; do not push datasets or checkpoints to GitHub.
