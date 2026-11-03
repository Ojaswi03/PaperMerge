# Hal9k execution optimization

This rollout checkpoints and resumes hal9k only. It refuses to run on
themachine. The shared 150-run catalog, scientific configuration identities,
original source files and existing checkpoint contracts remain unchanged.
The extra execution implementation has its own frozen source hashes, saved
source copies, transition receipt and per-run provenance in `run.json`,
`optimization.json` and checkpoint metadata. Final fleet exports include that
additional provenance and source. Treat optimized and original execution
segments as explicitly recorded segments, not as an undisclosed code change.

## Changes

- A first-order training branch for prior-adjusted FedSharp+ when the realized
  noise coefficient is zero. It retains the gradient-norm diagnostic and all
  five SGD steps. Each worker compares original and candidate updates on its
  actual device, including the nonzero branch and a nonuniform class prior.
  Its first real zero-coefficient update is compared too. A mismatch retains
  the original calculation; the result is recorded, not silently ignored.
- Fixed receiver probes are cached by exact label bytes, dtype, seed, receiver
  and probe limit. Returned arrays are copied to protect the cached values.
- Committed telemetry is JSON-encoded once and cached with lightweight
  compression to reduce RAM overhead. Each checkpoint still contains the full
  history in the original loadable layout, with the original fsync and atomic
  replacement. Large checkpoint writes still occur; this version reduces
  repeated encoding, not all disk traffic.
- Round mechanism summaries update incrementally. A resumed process rebuilds
  its summary cache once from the saved history.

The committed telemetry cache relies on the existing protocol's invariant:
records are compacted before checkpointing and are not edited afterward.
Changing that invariant requires changing or invalidating the cache.

The rollout does not remove calibration draws, change candidate selection,
change validation frequency, skip training steps, reduce memory headroom, or
raise worker limits. Existing configurations retain their recorded GPU binding
for checkpoint/runtime compatibility. The fleet can use either GPU for new
unbound work. This is not a migration of active configurations between GPUs.

## Run on hal9k

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
python scripts/optimize_fedsharp_hal9k.py --queue "$shared_run" --execute --background
```

The command prints the transition PID and log path. The background controller
stops only the verified hal9k dispatcher, fences its launch paths, asks only
its identified workers to checkpoint at a safe boundary, and waits for exit.
It never uses SIGKILL. Pre-transition checkpoint inodes are retained through
hard links under each output's `operational_adapters/optimization_boundary/`.
It checks saved contracts and cursors, updates only hal9k's claims, records the
adapter policy, and starts the normal fleet dispatcher with the optimized
worker entrypoint. Large checkpoint verification is performed outside the
global queue lock, so it does not hold up themachine's claims for that work.

The monitor can temporarily show interrupted or recovery-needed rows during
the deliberate checkpoint/restart. Inspect the printed log for checkpoint
completion and the new fleet PID. Compilation and device qualification happen
before steady training resumes. The timeout leaves workers alone rather than
forcing termination; if it expires, inspect the log before rerunning.

```bash
python scripts/monitor_fedsharp_500r.py --queue "$shared_run"
python scripts/diagnose_fedsharp_activity.py --queue "$shared_run" --seconds 10
```

Run these in the same shell as the initialized variable, or initialize it
again. Per-run ETA uses measured activations/time since the optimization
boundary once new progress is available. Total elapsed time remains cumulative.
Early estimates can fluctuate; wait several minutes before comparing rates.

If policy activation succeeded but launching the dispatcher failed, inspect
the log and verify there is no existing dispatcher before using the normal
`run_fedsharp_fleet.py --queue "$shared_run" --execute --background` command.
Repeating the optimization rollout after it is active is refused. Do not
manually clear claims or delete checkpoints.

## Verification and limitations

Local tests compare complete original/optimized trajectories and interrupted
resumes, including exact models, outgoing versions, telemetry and validation
curves. Additional tests cover byte-equivalent history serialization,
incremental summaries, original checkpoint-loader compatibility, mismatch
fallback, per-run ETA, optimized result export, and a real-process transition
that signals only the simulated hal9k processes while the other-host process
continues running. All original 80 frozen scientific file hashes are preserved.

The local environment uses TensorFlow 2.18; actual hal9k uses 2.21. Device-local
comparisons therefore run again inside each actual worker. No HPC speedup is
claimed before the rollout's activity measurements. This first version does
not implement stateless parallel candidate inference or an incremental
checkpoint file format. Those larger changes need separate qualification.

## One retry after a native segmentation fault

`Fleet worker exited -11` reports SIGSEGV; it does not identify its root cause.
For an advanced failed run, use `scripts/retry_fedsharp_checkpoint.py` with
`--queue`, `--run-id` and `--execute`. It validates the complete saved state
with the original strict loader, latest stored model hashes, ordered SGD
history and the original validation-metric prefix. It preserves an immutable
checkpoint link and failed row, then requeues only that host's dead failed
claim on its existing GPU with its existing token. Healthy workers and the
dispatcher receive no signals. The active dispatcher picks it up under normal
resource admission. Omit `--execute` for validation without requeueing.

The receipt reports the saved cursor and any work to replay since that
checkpoint. This does not reset the run to the round-100 seed. It is restricted
to the reported SIGSEGV failure and one audited attempt per run; a repeat
failure requires investigation rather than an automatic retry loop. If no
dispatcher is running, requeueing alone does not launch a worker.
