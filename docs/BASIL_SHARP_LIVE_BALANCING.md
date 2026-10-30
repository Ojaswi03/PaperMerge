# Enable shared GPU scheduling during the current run

The original scheduler gives each GPU its own fixed queue. When one queue runs
out, that GPU cannot take pending work from the other queue. This operational
adapter uses one queue and prefers the verified healthy GPU with fewer active
workers, then more free VRAM. It changes no scientific producer, frozen plan,
optimizer, seed, data split, memory budget, or checkpoint.

Run on **hal9k**, against the current output. Do not stop training, create a new
plan, or restart the normal launcher.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

python scripts/rebalance_basil_sharp_live.py \
  --output /var/tmp/papermerge-os675524/basil-sharp-thesis-direct-checked-100r-20261030 \
  --execute --background --ntfy-topic papermerge-ojaswi

tail -f /var/tmp/papermerge-os675524/basil-sharp-thesis-direct-checked-100r-20261030/live_scheduler.log
```

Ctrl+C closes only `tail`. The detached scheduler and training survive SSH exit.
The permanent browser monitor uses the same folder and link; leave its publisher
running. Its Running, Pending, Completed and Failed/stopped tabs keep updating.

## Handover without restarting CUDA workers

The existing Python supervisor has no hot-reload hook. The adapter briefly
suspends **dispatch only**, validates the original supervisor and every training
PID using UID, executable, output directory and Linux process identity, and opens
pidfds. Validation failure resumes dispatch and leaves workers untouched.

The old supervisor's SIGTERM handler terminates its workers. To avoid that
handler, the adapter sends SIGKILL **only to the validated supervisor**, then
adopts its existing training processes. The old shell may consequently report
the old supervisor as `Killed`; the training workers are never signalled or
restarted. GPU leases stay held by their inherited worker descriptors. The new
scheduler acquires a device lease before launching anything there, so an occupied
GPU may finish its adopted cohort before accepting additional workers.

Already running workers remain on their GPUs. An untouched configuration may
start on either GPU after that GPU's saved verification is checked. A partial
checkpoint stays on the GPU in its recorded runtime contract. This preserves
checkpoint compatibility and independent inference. If all remaining work is
already running, there is no pending work to move onto an idle GPU.

Operational source snapshots, handover receipts, dispatch records, quarantine
state and notification delivery receipts are recorded under
`operational_adapters/`. Existing compact exports include these recursively.
The scientific source fingerprints and original `plan.json` remain unchanged.

## Capacity and failure handling

Every launch reserves the worker's full future allocation, including startup
workers which have not yet allocated memory. It checks live per-GPU free VRAM,
per-process CUDA allocations, process RSS plus swap and globally available host
RAM. Unknown telemetry blocks new launches and is retried; it does not kill jobs.

This run reserves 3,072 MiB for TensorFlow plus 1,024 MiB for CUDA context per
worker, 8,192 MiB of GPU headroom, 8,192 MiB of host RAM per worker and 16,384 MiB
of global host RAM headroom. A 97,887 MiB GPU has a theoretical ceiling of 21 such
workers before other allocations. The additional hard ceiling is 24 per GPU.
**hal9k's 93 GiB host RAM limits the practical cohort to about nine total**, rather
than 21 on each GPU. New workers normally spread about 4/5 across its two GPUs.
Observed low MNIST RSS is not permission to remove CIFAR's future RAM reservation.
These are conservative admission checks, not an OS-enforced per-process RAM cap
or a guarantee against driver failure or unexpected memory peaks.

* A failed configuration is recorded and set aside; healthy workers continue.
  ntfy reports the failed configuration and GPU. A queued eligible configuration
  fills the released capacity; failed jobs are not automatically rerun.
* A CUDA OOM/device error or a GPU missing from inventory quarantines that GPU
  from new dispatch. ntfy reports the device pause. Untouched configurations
  can continue on another verified healthy GPU under the same RAM checks.
* Checkpointed work pinned to a quarantined GPU waits, rather than silently
  changing its runtime contract. Quarantine persists across controller recovery;
  a recovered GPU is not automatically trusted after an OOM.
* A global telemetry failure pauses new dispatch and sends an ntfy alert.
  Scheduling retries automatically while training continues. No CPU fallback.
* A controller error or stop request leaves active workers running and records
  `SUPERVISOR_RECOVERY_NEEDED`. Rerunning the same live command adopts them again.
* Completion sends a final ntfy summary. Complete thesis figures are generated
  only if all configurations completed successfully.

To inspect without changing the run, omit `--execute --background`. Never start
a second original supervisor alongside this adapter. The exclusive controller
locks prevent a second live adapter from scheduling the same work.
