# Hidden versus Gaussian attack comparison on hal9k

The new campaign repeats the **60 attacked configurations** from the completed
84-run study with Gaussian attack. It uses the same MNIST and CIFAR-10 datasets,
IID and nonIID partitions, seeds 2025/2026/2027, 100 clients, 100 full ring rounds,
and five local mini-batch SGD steps. The 24 clean controls already exist and are
not queued again. No new results are claimed until this campaign completes.

| Dataset | IID configurations | nonIID configurations | Total |
| --- | ---: | ---: | ---: |
| MNIST | 12 | 18 | 30 |
| CIFAR-10 | 12 | 18 | 30 |
| Total | 24 | 36 | 60 |

IID repeats BASIL and FedSharp, each under attack only and attack plus channel
noise. nonIID adds BASIL + ACDS and FedSharp+ under the joint condition. There
are 600,000 client activations and 3,000,000 local optimizer steps in total.

## Attack and channel definitions

`--attack-type hidden` and `--attack-type gaussian` choose the Byzantine
outgoing-model transformation. They do not change the channel-noise standard
deviation. There are 33 configured attackers out of 100 clients; attacks begin
at zero-based round 20, displayed as round 21. Before that point the attack
transformation is inactive. Both attacks preserve the declared local training
budget and corrupt only the transmitted outgoing state.

The Gaussian attack follows BASIL Section VI-A, Byzantine Attacks, on page 11 of
the [reference PDF](../papers/001-Basil%20A%20Fast%20and%20Byzantine-Resilient%20Approach%20for%20Decentralized%20Training.pdf).
It replaces every outgoing model parameter with an independent zero-mean,
unit-standard-deviation Gaussian draw. For a trained weight tensor `w`:

```text
attack_std = 1.0  # --gaussian-std, separate from channel --sigma
w_out = float32(Normal(0, attack_std^2, size=shape(w)))
```

The original trained values supply only tensor shapes. There is no weighted
blend, scaling by a layer's standard deviation, sign flip or Hidden component
in the Gaussian replacement. The earlier blended-corruption diagnostic remains
explicitly named `gaussian_blended`; the research launcher does not select it.
The Hidden attack remains:

```text
s = max(std(w), 0.01)
z ~ Normal(0, (0.05*s)^2)
w_out = float32(0.45*w + 0.55*(-1.4*w + z))
```

Each attack uses an activation-keyed generator based on the seed, round and
sender. It does not depend on global NumPy RNG state or worker launch order.
The final outgoing states are checked during independent result verification.

Channel noise is separately reconstructed for each sender version and receiver:
independent zero-mean Gaussian parameter-coordinate noise with absolute standard
deviation `--sigma 0.01` in joint cases, and zero in attack-only cases. This is
not relative 1% noise. `--alpha 0.5` controls nonIID class allocation and is not
a noise or attack parameter.

## Pull, preview and start

On `os675524@hal9k.cs.ohio.edu`:

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate

python scripts/run_basil_sharp_thesis.py \
  --config configs/research/hal9k_gaussian_100r.json \
  --output /tmp/fedsharp-gaussian-description --describe
```

The description must report `configurations: 60`, `attackType: gaussian`,
`gaussianAttackMode: pure_model_replacement`, `gaussianAttackStd: 1.0`,
`requestedGpuCount: 2`, `scheduler: shared`, and `trainingStarted: false`.
Description does not require GPU or dataset access and writes no output plan.

The launch below uses the server's local disk and survives SSH logout. It first
checks both original training datasets. GPU execution and attack-formula checks
then run on both selected GPUs before training starts.

```bash
paper_run="/var/tmp/papermerge-$(id -u)/hal9k_gaussian-attack-100r-$(date +%Y%m%d-%H%M%S)"

if python scripts/check_basil_sharp_inputs.py
then
  mkdir -p "$paper_run"
  nohup python -u scripts/run_basil_sharp_thesis.py \
    --config configs/research/hal9k_gaussian_100r.json \
    --output "$paper_run" --execute --wait-for-resources \
    --ntfy-topic papermerge-ojaswi \
    > "$paper_run/adaptive_supervisor.log" 2>&1 < /dev/null &
  echo $! > "$paper_run/supervisor.pid"
  printf 'Run directory: %s\n' "$paper_run"
fi
```

Use one supervisor for this output directory. There is no need to start the
separate live-balancing launcher for a fresh preset run. The shared queue checks
current memory before every launch and prefers the GPU with fewer active
workers, subject to available memory and the configured maximum. Host RAM is a
global constraint: 8 GiB reserved per worker and 16 GiB headroom normally allow
about nine workers across both GPUs, rather than nine on each GPU. Reservations
reduce OOM risk but cannot guarantee that other users or exceptional allocations
will never exhaust memory.

Failed configurations are set aside and reported through ntfy; other work
continues. On a CUDA memory/device failure, the GPU is quarantined for new
launches, and untouched queued configurations can use the other verified GPU.
Already running CUDA processes do not migrate between GPUs. A resumed checkpoint
retains its saved device/runtime contract.

A deliberately early-attack stress test on four synthetic CIFAR-10 clients,
attacked from displayed round 2 with uninformative class probes, exposes a
FedSharp+ numerical overflow after Gaussian corruption propagates. The finite
objective checks detect it; it must be recorded as a worker failure, not a valid
accuracy result. This is retained as an adverse regression test, not hidden by
weakening the Gaussian attack or changing admission rules. The actual research
preset starts attacks at displayed round 21. Before training, the GPU gates
also exercise an informative receiver against pure Gaussian peer models. These
checks do not guarantee that every full-length configuration will succeed;
any numerical failures in the campaign remain reportable outcomes.

## Switch the attack or select fewer cases

Explicit terminal flags override the preset. Use a **new output directory**
when changing the attack, channel sigma, seed matrix or any training parameter.

```bash
# Same 60-case selection, with Hidden attack instead of Gaussian:
python scripts/run_basil_sharp_thesis.py \
  --config configs/research/hal9k_gaussian_100r.json \
  --attack-type hidden --output /tmp/fedsharp-hidden-description --describe

# Equivalent dedicated Hidden preset:
python scripts/run_basil_sharp_thesis.py \
  --config configs/research/hal9k_hidden_100r.json \
  --output /tmp/fedsharp-hidden-description --describe

# Optional smaller selection, without starting training:
python scripts/run_basil_sharp_thesis.py \
  --config configs/research/hal9k_gaussian_100r.json \
  --datasets cifar10 --partitions nonIID --seeds 2025 \
  --output /tmp/fedsharp-small-description --describe
```

The existing Hidden results remain the comparison reference. Neither preset
automatically launches when read, resumes a different attack under an old
checkpoint, or overwrites a completed campaign. Clean cases can be explicitly
included with `--environments clean attack_only joint`, giving 84 runs instead
of 60; they are unnecessary for this matched additional study.

## Progress and permanent browser monitor

```bash
tail -f "$paper_run/adaptive_supervisor.log"
```

Expect verification on GPU 0 and GPU 1, followed by `GPU 0: started ...` and
`GPU 1: started ...`. Ctrl+C closes this log view without stopping training.
After the plan has been written, use:

```bash
python scripts/run_cart_comparison.py --monitor --output "$paper_run"

# Publish this campaign to the existing permanent dashboard:
python scripts/share_cart_monitor.py \
  --output "$paper_run" --background --interval 1 \
  --ntfy-topic papermerge-ojaswi
python scripts/share_cart_monitor.py --print-link
```

The permanent link remains:
<https://ojaswi03.github.io/PaperMerge/?feed=c3e99a29-732b-4bcf-a42c-a4a19410d4ee>.
The terminal monitor uses measured total activation throughput for the ETA and
estimated time of day. Workload changes can change the forecast. You can exit
SSH after the detached supervisor and publisher have started.

If a supervisor stops, inspect the log and live workers first. A normal resume
uses the frozen plan, with no preset or attack/resource overrides:

```bash
nohup python -u scripts/run_basil_sharp_thesis.py \
  --resume --execute --wait-for-resources --output "$paper_run" \
  --ntfy-topic papermerge-ojaswi \
  >> "$paper_run/adaptive_supervisor.log" 2>&1 < /dev/null &
echo $! > "$paper_run/supervisor.pid"
```

If workers remain active after a supervisor error, use the existing live
handover command to adopt them instead of starting another supervisor:

```bash
python scripts/rebalance_basil_sharp_live.py \
  --output "$paper_run" --execute --background --ntfy-topic papermerge-ojaswi
```

## Completion, verification and compact export

The automatic report supports the declared 60-case matrix and creates PNG/SVG
figures only. Those figures contain the four IID and six nonIID Gaussian-attack
curves per dataset; they do not silently fabricate clean curves or combine old
and new evidence. Final reviewed comparison figures can combine the newly
verified Gaussian curves with the existing Hidden results and shared clean
controls after the run finishes.

```bash
python scripts/verify_basil_sharp_thesis_results.py --output "$paper_run" && \
python scripts/export_basil_sharp_thesis.py \
  --output "$paper_run" \
  --destination "$PWD/docs/results/hal9k_gaussian-attack-100r-20261031"
```

Independent CPU inference verifies all saved final client models and the final
seeded outbound attacks. The compact export includes configurations, producer
snapshots, metric arrays, checksums and figures. Checkpoints, local logs, private
monitor credentials and data caches stay off GitHub. Review completion and
verification before staging this explicitly allowlisted new results folder.

The existing completed study's run records, source snapshots and measurements
remain unchanged. Its Hidden configuration identities are covered by a
regression check. New training must use a fresh plan with the updated producer;
do not pull producer changes during an active frozen campaign.
