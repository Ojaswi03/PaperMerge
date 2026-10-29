# Automatic GPU worker admission

Use `--auto-workers` to spread the configurations over the selected GPUs and
launch additional workers whenever GPU memory and host RAM allow. Four cases
on four GPUs use one worker per GPU. The 24-case full comparison can reach six
workers per GPU on the four 96-GB GPUs, subject to current resources. The
ceiling is eight workers per GPU; it is a limit, not a requested fixed count.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

bash scripts/launch_cart_comparison_hpc.sh full \
  --gpu-count 4 --auto-workers \
  --output "$PWD/.papermerge/local-tests/cart-acds-full-100r-20261029"
```

The same command works for a new run or hands off a running fixed-slot
supervisor. It requests a graceful stop, waits for checkpoint saves and GPU
lease release, then resumes unfinished cases. Completed cases are skipped.
Keep the existing output path. There is no need to stop or kill workers by
hand. On a different server, new runs can select `--gpu-count 1`, `2` or `3`;
an existing checkpoint still requires its original verified server and GPUs.

This update adds a separate operational supervisor and does not change the
frozen numerical producer modules. The original `plan.json`, source snapshots,
per-worker runtime profiles, GPU verification receipts and checkpoint contracts
remain unchanged. Pulling this additive update is compatible with the earlier
full comparison source fingerprint; unrelated later producer changes would
still prevent recovery. Copies of previous orchestration records go into
`scheduling_history/`; the new supervisor has its own hashes and source copies
in `adaptive_execution.json` and `scheduler_snapshots/`.

Before **every** launch the supervisor checks:

- Live free memory and the recorded physical GPU UUID.
- Every resident worker's full future allocation, including workers that have
  not initialized CUDA yet. Current allocations are credited only to matching
  worker PIDs, never another user's process.
- Available host RAM and the unallocated portion of every worker's RAM budget.
- The per-GPU worker ceiling and original producer source hashes.

The current full-run profiles cap the TensorFlow allocator at 8192 MiB per
worker and budget another 1024 MiB per CUDA context, with 8192 MiB GPU headroom.
Six workers reserve 63488 MiB per GPU. Host RAM reserves 32768 MiB per worker
plus 32768 MiB headroom: 800 GiB for 24 workers. Budgets are not lowered to fit
more workers. Unavailable allocation telemetry pauses admissions. Memory is
sampled again every three seconds, so queued work starts as capacity frees.
Existing experiments retain their assigned GPU to preserve exact checkpoint
runtime identities.

These checks reduce OOM risk; they cannot reserve memory against other users
or guarantee that a CUDA/context allocation stays within an estimated budget.
The controller does not stop another user's processes or force-kill its own
workers to free memory. GPU utilization alone does not establish that admitting
more workers will improve throughput. Each experiment's ring activations remain
sequential, while separate experiments run concurrently.

Inspect the handoff and launch decisions:

```bash
tail -n 30 .papermerge/local-tests/cart-acds-full-100r-20261029/adaptive_supervisor.log

python scripts/run_cart_comparison.py --monitor \
  --output "$PWD/.papermerge/local-tests/cart-acds-full-100r-20261029"
```

The existing browser share and completion notification continue to use the
same output. The terminal monitor reads the adaptive lanes from the live
campaign record instead of assuming the old two-worker layout. Its ETA remains
an estimate; concurrency changes the observed throughput.

For a read-only capacity check, without stopping or launching anything:

```bash
python scripts/run_cart_adaptive.py \
  --output "$PWD/.papermerge/local-tests/cart-acds-full-100r-20261029"
```

Resuming after an interruption uses the same `--auto-workers` command. Do not
start an additional fixed-slot supervisor alongside the adaptive supervisor.
