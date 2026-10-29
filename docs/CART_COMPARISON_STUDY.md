# CART mechanism and ACDS comparison

The existing 72-run archive and six-run revision remain unchanged. This runner
creates a separate experiment identity and compact routing/commit receipts.
It does not claim a new method has already improved the completed results.

## Questions and arms

| Arm | Change relative to the revised CART comparison |
| --- | --- |
| `sharp` | Existing guarded SHARP, no CART |
| `cart` | Existing MCC admission, original head repair and full-update commitment |
| `macro` | Remove the qualified-class penalty and its routing calibration |
| `ingress_only` | Retain admission and repair; remove full-update commitment |
| `commit_only` | Keep SHARP's pooled ingress; construct private class evidence only for commitment |
| `local_only` | Train the receiver-owned model; consume no peer models or fresh link noise |
| `fresh_peer` | Take the newest/nearest screened coherent peer; no CART or head repair |
| `no_calibration` | Remove the routing noise allowance; keep admission and commitment |
| `no_repair` | Check the complete SGD update without classifier repair or new head-reference admission |
| `acds` | SHARP with 5% raw fitting-data sharing; no CART |

The local-only arm is a communication-ablation baseline, not a claim that it
still executes both defenses. It computes the same local SGD step budget and
has no fresh-link regularizer. All arms retain the original model, learning-rate
schedule, data ownership, holdout, attack generator, and five-step budget.

Environment isolation explicitly holds defense/optimization switches fixed
while removing physical attack and/or channel noise. Thus a `clean` environment
in this study means the defended algorithm without physical impairments, not
the legacy untreated `clean` condition. Never silently exchange these controls.

## ACDS attribution and information assumptions

[BASIL, Algorithm 2, Section IV-A](https://arxiv.org/abs/2109.07706) selects a
fraction of each owner's non-sensitive data, circulates shuffled batches within
groups, and shares the collected data across groups. The reference study reports
5% sharing. The added `cyclic_training_share` in `basil_core/acds.py` simulates
the resulting training-data availability using a single logical ring and a drain
round, with floor rounding for unequal owner sizes. It retains all owner data,
adds every other owner's selected examples exactly once, and uses no evaluation
indices. Its empty final dummy batch delivers remaining real examples without
introducing public training examples.

This is an **ACDS training-data-outcome baseline**, not a secure network transport
or a demonstration of cryptographic anonymity. In this image-data simulation,
all fitting examples are declared non-sensitive. The API can instead accept an
explicit eligibility set and rejects insufficient eligible examples.

Data sharing happens before parameter attacks. Every owner contributes by the
same rule; attacker labels do not filter shared data. Shared examples are not
data-poisoned in this threat model. ACDS has additional raw-data access and
storage/distribution cost; this is not an identical-information comparison with
local-data-only CART. Optimizer steps, not unique-example access, are matched.
The audit records original/expanded class histograms, selected counts, index
hashes and transmitted example copies. Protocol training elapsed time excludes
data preparation; supervisor wall time includes worker initialization.

The historical `acds_share` helper is retained for compatibility. It is not used
by the comparison: it partitions a global remainder and does not reproduce
per-owner fractional sharing. The new entry point is separately tested.

## Staged HPC execution

Activate the existing HPC environment through the launcher. Each launcher checks
GPU resources and numerical equivalence before starting a detached supervisor.
Failed workers are recorded while healthy workers continue. Resume reads the
original frozen plan and committed checkpoints. Source/configuration changes
require a fresh output directory. Project GPU locks do not reserve against other
users.

```bash
git pull --ff-only
bash scripts/launch_cart_comparison_hpc.sh screen --gpu-count 2
```

The launcher accepts `--gpu-count 1`, `2`, `3`, or `4`. Without the option it
detects the installed GPU indices, up to four. Examples:

```bash
bash scripts/launch_cart_comparison_hpc.sh isolation --gpu-count 1
bash scripts/launch_cart_comparison_hpc.sh confirm --gpu-count 2
bash scripts/launch_cart_comparison_hpc.sh acds --gpu-count 4
```

To choose particular devices, use `--gpus 1,3`. If both a count and IDs are
provided, they must agree. CLI selection overrides `CART_COMPARISON_GPUS` and
`CART_COMPARISON_GPU_COUNT`; those environment settings remain supported when
no CLI GPU selection is supplied. Two workers per GPU is the default; use
`--jobs-per-gpu 1` to reduce GPU/host-memory demand. Memory and GPU verification
checks still run independently for each selected physical device. Automatic
detection does not mean a device is unused or allocated exclusively.

`--dry-run` prints the resolved mode, GPUs, worker count and output path without
starting verification or training. `--output PATH` or `CART_COMPARISON_OUTPUT`
selects a fresh output namespace. A saved plan retains its physical GPU UUIDs
and worker count. A different server, GPU layout or worker count requires a new
output; the launcher does not silently migrate an existing checkpoint contract.
The older `launch_cart_cifar6_hpc.sh` also accepts GPU count, IDs and worker count.

For the two-GPU `hal9k` server, connect with `ssh os675524@hal9k.cs.ohio.edu`,
clone or pull the repository, and create its own environment using
`bash scripts/setup_hpc.sh` if `.venv-hpc` is absent. The setup script requires
Python 3.12. Do not copy a virtual environment from another host. The GPU gate
must pass on this server before experiments start, regardless of its driver
version or successful execution on the other server.

The default **screen** is 20 configurations: ten arms, two CIFAR partitions, seed
2025, ten rounds, attack beginning at zero-based round 3. It is a diagnostic,
not a full-horizon performance result. Inspect the routing/commit evidence before
choosing the next stage.

```bash
python scripts/run_cart_comparison.py --monitor \
  --output "$PWD/.papermerge/local-tests/cart-comparison-screen-v1"

python scripts/inspect_cart_mechanisms.py \
  --output "$PWD/.papermerge/local-tests/cart-comparison-screen-v1"
```

The monitor refreshes the terminal in place, uses a boxed table with arm labels,
and closes independently of training.

The **isolation** preset is 32 configurations: SHARP, CART, macro-only and ACDS,
two partitions, four physical environments, one seed, ten rounds. Run after the
mechanism screen is interpreted:

```bash
bash scripts/launch_cart_comparison_hpc.sh isolation
```

The **confirm** preset is 12 full configurations: CART and ACDS, two partitions,
three matched seeds, 100 rounds and original attack onset. It reruns both arms
rather than mixing a partially replaced group. It is not an independent new-seed
confirmation: these are the previously used seeds.

```bash
bash scripts/launch_cart_comparison_hpc.sh confirm
```

For only the six ACDS configurations, reusing the complete saved CART groups:

```bash
bash scripts/launch_cart_comparison_hpc.sh acds
```

This executes `--preset confirm --arms acds` and supplies the saved six-run CART
export as the plot reference. If runtime/data checks reject that reference,
training remains completed and recorded; fix reporting or run the fresh paired
`confirm` preset rather than weakening the checks. Full-horizon comparisons check
data, initializations and numerical runtime; a ten-round screen cannot be
overlaid as a 100-round endpoint. Additional noise levels use the existing
`--noise-levels 1/2/3 --noise-values ...` interface.

## Verification and portable export

```bash
python scripts/verify_cart_comparison_results.py \
  --output "$PWD/.papermerge/local-tests/cart-comparison-screen-v1"

python scripts/export_cart_comparison.py \
  --output "$PWD/.papermerge/local-tests/cart-comparison-screen-v1" \
  --destination "$PWD/docs/results/cart-comparison-screen-20261029"
```

CPU inference checks every final model against its saved validation metrics.
It does not replay optimization. The exporter includes `mechanisms.json`, so the
next repository review can inspect routing and commitment without transferring
large checkpoints. After the method has been frozen, the inference script also
supports `--official-test`; do not use that endpoint to select methods.

Successful execution, numerical equivalence and independent inference do not
establish efficacy or novelty. The existing short development checkpoint shows
82 zero-sized commits among 95 checked updates in its last round. That motivates
the repair/commit ablations; it is not proof of the cause of the full HPC plateau.
