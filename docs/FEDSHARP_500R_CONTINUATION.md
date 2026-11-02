# Continue all 150 runs from round 100 to round 500

The preset `configs/research/all_500r_from_100r.json` references the three
immutable, independently verified exports: 84 Hidden/clean runs, 60 Gaussian
runs and six MNIST BASIL noise-only runs. It includes all 12 FedSharp+ runs.
There are 78 MNIST and 72 CIFAR-10 configurations.

Each continuation imports its complete round-100 state and begins at displayed
round 101. This adds 6,000,000 client activations and 30,000,000 SGD steps.
The new histories contain all 500 rounds, including the unchanged first 100.
Seeds, attacks, attack onset, partitions, data, model, local steps, learning-rate
schedule and method settings are retained. The learning rate does not restart.

The original plans, checkpoint contracts and results remain unchanged.
New configurations have new horizon-dependent identities and explicit parent
links. The strict original checkpoint loader is unchanged: an import first
loads the original contract, validates the complete boundary, then writes a
separate checkpoint under the new contract. An import receipt records the
original checkpoint hash and any host/GPU transition.

## Server ownership and dynamic GPU use

Run the resource probe on each server. Initialization assigns whole three-seed
conditions to one server, weighted by the measured available worker capacity.
The assignment is then frozen. Each server uses one shared queue across its
verified GPUs, preferring fewer active workers and then more usable VRAM.
Available GPUs fill dynamically; configurations are not pinned to a static
per-GPU queue before starting.

Server ownership, an atomic output-directory binding, exclusive supervisor
locks, validated process adoption and GPU leases prevent a second server or
dispatcher from running the same assignment. Shared status shows SERVER, GPU,
CLAIM and STATUS. CLAIM progresses from WAITING to TAKEN to DONE. Failed jobs
are set aside rather than silently rerun. A recovery checks real process
identities and adopts surviving workers; partial checkpoints retain their
already-bound GPU/runtime. Server assignments do not move automatically after
initialization.

Admission checks run before every worker launch. Each worker reserves 3,072 MiB
for TensorFlow plus 1,024 MiB for CUDA context, and 8,192 MiB of host RAM. Each
GPU keeps 8,192 MiB of headroom, and the host keeps 16,384 MiB of RAM headroom.
Reservations include startup workers that have not allocated their memory yet.
The hard ceiling is 24 workers per GPU; live VRAM and RAM usually impose lower
limits. Other users' allocations count against free memory. These checks reduce
OOM risk; they cannot prevent exceptional peaks or unrelated allocations after
a check. No other user's processes are stopped.

## 1. Describe locally or on either server

```bash
python scripts/run_fedsharp_500r.py --describe
```

Expected: 150 configurations, source round 100, target round 500, next displayed
round 101 and `trainingStarted: false`. To save the fully expanded configuration
catalog without training or touching GPUs:

```bash
python scripts/run_fedsharp_500r.py --describe \
  --write-configs .papermerge/prepared-continuations/all_500r_configs.json
```

## 2. Probe both servers

Run this once on **hal9k** and once on **themachine**. The path is identical
because the university hosts share the home filesystem.

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
python scripts/run_fedsharp_500r.py --queue "$shared_run" --probe
```

The probe reads actual free GPU memory and MemAvailable, and writes a private
per-server resource snapshot. It does not train. Run both probes shortly before
initialization; initialization refuses probes older than 30 minutes.

## 3. Initialize once on hal9k

This must run on the producing host because the original checkpoints are in
hal9k's local `/var/tmp`. Paths are read from the original frozen plans rather
than guessed. It checks that all 150 originals exist before copying anything,
checks storage headroom, copies checkpoint seeds to the ignored shared folder,
verifies their hashes, and freezes one server assignment for every config.
Copying full checkpoints over shared storage can take time. Training starts
only in the next step.

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
python scripts/run_fedsharp_500r.py --queue "$shared_run" --initialize
```

The final message gives the configuration count assigned to each server and
`trainingStarted: false`. Do not initialize the same assignment twice. If only
hal9k will be used, initialize a fresh shared namespace with `--hosts hal9k`.

## 4. Start once on each server

After initialization finishes, run this on **both servers**:

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
python scripts/run_fedsharp_500r.py --queue "$shared_run" --execute --background
```

The launcher prints its server-specific local output and log paths. It starts
the supervisor in a separate session, so closing a log viewer or SSH terminal
does not send it terminal interrupts. GPU execution gates run before dispatch.
Each worker then checks its original checkpoint's contract, all prefix metric
arrays, model hashes, outgoing attack transformation and initialization/data
identities. Fresh inference on the assigned device must reproduce the saved
round-100 accuracy within the existing one-example tolerance before round 101.
Encoder, classifier, outbound versions, sender rounds, references and telemetry
all carry forward.

hal9k also starts or retargets the permanent browser publisher and sends the
existing dashboard link through ntfy. The page combines both servers, showing
SERVER, CLAIM, ATTACK, GPU, STATUS and the usual progress/ETA columns. No API
credentials or private paths are uploaded. Existing public schema and permanent
feed URL are retained.

The first 100 rounds already account for 20% of each 500-round history. ETA
uses only newly executed activations; the inherited work is not counted as
instantaneous throughput. Source copies and shared status stay in the ignored
home folder. Growing checkpoints and worker logs stay in unique server-local
`/var/tmp/papermerge-<uid>/<server>_fedsharp-all-500r-<queue-id>/` directories.

## Monitor, recover, and finish

Read-only terminal monitor, on either server:

```bash
cd ~/Masters_Research/PaperMerge
source .venv-hpc/bin/activate
shared_run="$HOME/Masters_Research/PaperMerge/.papermerge/continuations-500r-20261102"
python scripts/run_fedsharp_500r.py --queue "$shared_run" --monitor
```

`--status` prints one snapshot. The shared folder also contains
`configuration_status.csv`. Ctrl+C closes only the monitor. A stale server
report is shown as recovery needed, not permission for another server to
duplicate its work. On the affected server, rerun the same
`--execute --background` command after its supervisor exits. It adopts live
workers and resumes checkpointed interrupted work under its original profile.
Do not pull producer changes or edit a frozen plan during execution.

After completion, run `scripts/verify_basil_sharp_thesis_results.py --output`
against each printed server-local output. The compact exporter
`scripts/export_basil_sharp_thesis.py` supports the continuation histories and
retains each `continuation.json` receipt. Each export is a server fragment;
combine disjoint fragments before final three-seed comparisons. Full
checkpoints, shared seed copies and private status remain off GitHub. Review
and allowlist new compact exports separately after independent verification.

## Validation scope

Local real-model CPU tests cover BASIL clean/noise-only, Hidden/Gaussian attacks,
ACDS, FedSharp, and FedSharp+. Small checkpoint forks reproduce uninterrupted
longer runs exactly, including retained models, outgoing models, histories and
classifier reference arrays. Further tests cover runtime and checksum rejection,
disjoint ownership, host output fencing, resource reservations, status/ETA and
browser columns. These tests do not claim that the actual HPC checkpoints or
cross-server GPU execution have already been tested. Initialization and the
per-device boundary gates check those on the servers before continuation.
