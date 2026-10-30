# Resume after a process-memory telemetry race

The supervisor exited while reading `/proc/<pid>/status`: a worker passed
`poll()` but exited before its `VmRSS` field was read. A process-exit race is an
operational scheduling failure, not evidence of an accuracy or optimizer issue.
The supervisor's exception handler terminates and waits for its active children;
the displayed worker rows can retain their last saved RUNNING states.

Use the recovery launcher below against the original plan and output directory.
It leaves all frozen producer files unchanged and substitutes only a conservative
process-memory reader. When a memory field or process disappears during the
read, the full per-worker memory reservation is kept until the normal scheduler
reaps that worker. Resource budgets, mathematical training, and GPU assignment
remain unchanged. Adapter hashes and source snapshots are saved separately under
`operational_adapters/` and included in the compact export.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
thesis_run=/var/tmp/papermerge-os675524/basil-sharp-thesis-direct-checked-100r-20261030

nohup python -u scripts/resume_basil_sharp_resilient.py \
  --resume --execute --wait-for-resources --output "$thesis_run" \
  --ntfy-topic papermerge-ojaswi \
  >> "$thesis_run/adaptive_supervisor.log" 2>&1 < /dev/null &
echo $! > "$thesis_run/supervisor.pid"
```

Do not launch a fresh plan or supply new device/memory/scientific settings.
Completed configurations are skipped. Interrupted configurations load their
saved checkpoints. Existing bound GPU receipts can be reused when the original
plan/source/runtime checks agree. The measured overall-speed baseline restarts
for this supervisor invocation, excluding earlier activations from its rate.

```bash
python scripts/run_cart_comparison.py --monitor \
  --output /var/tmp/papermerge-os675524/basil-sharp-thesis-direct-checked-100r-20261030
```
