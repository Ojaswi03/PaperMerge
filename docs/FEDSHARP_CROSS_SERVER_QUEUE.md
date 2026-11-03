# Share waiting 500-round continuations across servers

The fleet queue lets hal9k take untouched work originally assigned to
themachine, and vice versa. It preserves all 150 experiment identities and
their round-100 checkpoints. Existing training processes are adopted with their
original PIDs, numerical profiles, GPU assignments and output directories.
Only the dispatchers are replaced. Active or partially resumed
training never migrates between servers.

The original master plan and both original server plans remain unchanged.
Operational ownership lives in the ignored shared `fleet/` folder. A global
filesystem lock and an atomically created exclusive per-run claim prevent two
servers from running the same configuration. Claims never expire into another
server's queue merely because a heartbeat is late.

## Enable on both servers

Run this once on **themachine**, then once on **hal9k**:

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
python scripts/run_fedsharp_fleet.py --queue "$shared_run" \
  --execute --background --retry-failed-imports
```

The optional `--retry-failed-imports` flag checks the previously failed imports
before any dispatcher handover: both datasets must verify twice and all
available seed copies must match their frozen hashes, original contracts and
round-100 cursor. It permits one retry only for a failed boundary import with
no target checkpoint or continuation receipt. Original failure records remain
in the operational audit. A repeat failure stays failed. It does nothing when
there are no failures or this host has already joined the fleet.

A host's pending jobs become transferable only after its original dispatcher
has exited and the fleet agent holds its exclusive supervisor locks. Therefore
starting only hal9k cannot take jobs from a still-unmodified donor dispatcher.
The launcher prints the new agent PID and its exact log path. An ntfy message
confirms actual adoption, rather than only process launch.

New transferred jobs use a separate host-local directory:
`/var/tmp/papermerge-<uid>/<server>_fedsharp-fleet-500r-<queue-id>/`.
The same immutable full catalog is present on each server, but only claimed
configurations execute. Original local work continues in its original directory.
The shared claim records link every run to its actual execution plan/location.

## Admission and the initial lease wait

RAM/VRAM budgets, headroom, thread counts and worker ceilings are unchanged.
Admission samples all surviving and new workers, including reserved startup
allocations, before every launch. GPUs balance by active process count.
Existing device/memory quarantines remain in force.

Surviving workers retain their existing exclusive physical-GPU leases. A fleet
agent may consequently wait for those workers to finish before it can use that
GPU for new jobs. It reports `waiting for existing GPU lease`. It does not
interrupt a worker or bypass the lease to eliminate this initial wait.

A receiving GPU runs the existing full verification gate against the immutable
fleet execution plan before starting transferred work. Every actual imported
checkpoint then passes the original strict loader, complete-prefix comparison,
runtime checks and fresh boundary inference. Failed verification pauses dispatch
and leaves healthy workers running.

## Dashboard and terminal

After both agents launch, refresh only the permanent browser publisher on
**hal9k**:

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
python scripts/share_cart_monitor.py --output "$shared_run" \
  --background --restart-publisher --interval 1 --ntfy-topic papermerge-ojaswi
python scripts/monitor_fedsharp_500r.py --queue "$shared_run"
```

The existing dashboard URL is retained. Waiting transferable jobs show SERVER
`pool`, CLAIM `WAITING`. A successful claim changes SERVER to the actual server
and CLAIM to `TAKEN`; completion shows `DONE`. The monitor counts every original
configuration exactly once, including completed legacy runs. Overall speed
excludes inherited round-100 work. ETA covers runnable work and labels any failed
or stopped configurations excluded from the forecast. Ctrl+C closes only the
terminal monitor.

The original launcher's frozen `--monitor` view does not understand transferred
ownership; use the operational monitor above. Capacity checks also understand
both original and transferred workers:

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
python scripts/check_fedsharp_capacity.py --queue "$shared_run"
```

## Recovery and export

If an agent exits, rerun the fleet launch on that same server. Its durable claims
remain owned there. It adopts surviving original/transferred workers and resumes
incomplete local checkpoints under their original GPU/runtime contracts. The
old fixed-server dispatcher is fenced from starting after fleet activation.
Never delete claims, edit the frozen plans or start an old dispatcher to recover
a fleet host. Agent termination does not signal training workers.

When all 150 runs finish successfully, export one compact fragment **on each
server**, choosing a fresh destination for each:

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
export_root="docs/results/$(hostname -s)_fedsharp-fleet-500r-$(date +%Y%m%d-%H%M%S)"
python scripts/export_fedsharp_fleet.py --queue "$shared_run" \
  --destination "$export_root"
```

This validates saved metrics and preserved prefixes, copies the actual local
run records, includes immutable execution plans and sanitized transfer
provenance, and writes a checksum manifest. It excludes full checkpoints,
live claims, queue state and process identities. Full checkpoints remain at
their recorded host-local execution locations. Fresh independent final-model
verification is a separate required review before publication. Combine disjoint
host fragments before three-seed comparisons. New exports require explicit
review/allowlisting before Git publication.

Local validation covers concurrent-process claims, interrupted claim commits,
legacy dispatcher fencing, active/partial-run pinning, live-worker preservation,
unique transferred output locations, compact export provenance, and real-model
CPU continuation from an unchanged parent into a receiving namespace. Actual
NFS operation and destination GPU compatibility are checked during rollout;
local tests do not establish their live state.

## Diagnose performance while training continues

The following tools do not signal, attach a debugger to, restart or reconfigure
training processes. Run on each server so its own kernel lease holders and CPU
counters can be verified locally:

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
python scripts/diagnose_fedsharp_activity.py --queue "$shared_run" --seconds 10
nice -n 10 python scripts/benchmark_fedsharp_cpu.py --dataset cifar10 --iterations 3
```

The activity report reads both agents' shared blocking reasons, progress and
job ETAs. For the local host it checks process identities and kernel `fdinfo`
against the actual GPU-lease inode, and samples CPU time and disk-I/O counters.
Remote lease ownership is explicitly unverified until the tool runs there.
Status updates are batched, so a short activation-rate sample may be uneven.

The separate synthetic benchmark runs the unchanged link-noise reconstruction
and screening functions at the actual model-vector dimensions using one CPU
thread, with lower process priority. It does not import TensorFlow, initialize
GPU contexts, train a model, read datasets/checkpoints or write experiment
files. Its timings cover those CPU components only; they do not establish a
complete activation bottleneck. Candidate scoring, training and validation
still require a separately validated representative benchmark before changing
their implementation.

The browser and terminal ETA also account for the longest currently running
job forecast, rather than assuming aggregate throughput alone determines the
finish. These remain forecasts: startup and changing contention can move the
estimates. A monitor-only update needs a new terminal monitor process or a
browser-publisher refresh as described above; fleet agents and training workers
keep running unchanged.
