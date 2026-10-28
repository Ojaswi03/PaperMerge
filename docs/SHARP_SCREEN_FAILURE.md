# SHARP v1: reproduced noiseless Hidden-attack screening failure

The HPC monitor reported IID MNIST clean accuracy 97.85% after 100 rounds,
unmitigated attack accuracy 10%, and attack-defense accuracy 10% by round34.
The latter is a scientific effectiveness failure even though no process crashed.
CART is **off** in that attack-defense condition; this observation does not
measure the pending CART comparisons.

## Completed local reproduction

A fresh explicit-CPU run used seed2025, IID MNIST, 100 nodes, five SGD steps,
the `paper` model tag, attack onset20, sigma0, and the frozen SHARP v1 source.
Only the horizon was shortened to 35 rounds. Initialization, data partition,
batch streams and learning-rate schedule do not depend on the final horizon.
This is a local reproduction, not recomputation of the HPC checkpoint.

The experiment completed and passed independent final-node inference and
telemetry verification. Its round2 mean was 78.14%, matching the displayed
HPC value. Mean accuracy was 87.05% at displayed round20, 61.66% at round21,
10.63% at round29, and 10% from round30 through35. The
[machine-readable receipt](validation/SHARP_IID_HIDDEN_SCREEN_FAILURE_20261027.json)
preserves the complete round audit, identities and verification result.

Reproduce in a fresh output directory, then independently verify:

```bash
environment/basil-noise-env/bin/python scripts/run_sharp_cpu_case.py \
  --dataset mnist --partition iid --seed 2025 --condition attack_defense \
  --sigma 0 --nodes 100 --rounds 35 --model paper \
  --output .papermerge/local-tests/sharp-screen-failure-replay-new
environment/basil-noise-env/bin/python scripts/verify_sharp_results.py \
  --require-complete --output .papermerge/local-tests/sharp-screen-failure-replay-new
```

The first directly attacked model was admitted at **activation2004,
receiver3, zero-based round20 (displayed round21)**:

| Stage | Unattacked copies | Directly attacked copies |
| --- | ---: | ---: |
| Available window | 65 | 1 |
| Retained by screen | 0 | 1 |
| Selected | 0 | 1 |

The target count was33 but only one copy survived. No honest candidate remained
for the ranker to choose. Before attack onset, the screen rejected every copy
at all100 receivers in many rounds; the specified all-fail fallback then kept
all copies. A dense attacked copy passing the screen can disable that fallback
while honest copies remain rejected. Subsequent mixtures propagate the effect.

## Why the ratio can behave this way

The screen computes:

$$
R(v)=\frac{\|v\|_\infty}{\|v\|_2/\sqrt d},
\qquad v=\theta_{\mathrm{copy}}-\operatorname{median}(W).
$$

Here v is the difference from the window's coordinate median, d is parameter
count, the infinity norm is the largest absolute coordinate, and the denominator
is the coordinate RMS. Multiplying v by a positive scalar leaves R unchanged.
The statistic measures concentration of a deviation, not how large or harmful
it is. Structured benign training differences need not look like independent
Gaussian coordinates. The project Hidden attack changes tensors broadly and
need not look like the PDF's illustrative single-coordinate spike.

A constructive test uses66 benign vectors with small, separate single-coordinate
changes. They all fail the ratio test, so the all-fail policy retains them.
Replacing one with the existing dense Hidden transformation makes only that
attacked vector pass. This is a counterexample to interpreting the screen as a
general Byzantine detector. It does not establish a universally safe substitute.
Test: `test_v1_counterexample_sparse_benign_drift_can_lose_to_dense_hidden_attack`
in [the audit tests](../tests/test_sharp_selection_audit.py).

## Audit the actual HPC checkpoint without stopping training

The new audit script is outside the frozen training fingerprint. Pulling this
audit-only update leaves the current worker and runner source unchanged:

```bash
cd ~/Masters_Research/PaperMerge
git pull --ff-only
source .venv-hpc/bin/activate
python scripts/audit_sharp_selection.py \
  --output "$PWD/.papermerge/local-tests/sharp-hpc-100nodes-v1" \
  --dataset mnist --partition iid --seed 2025 --condition attack_defense
```

It reads the last atomic checkpoint's metadata only, performs no training,
imports no TensorFlow, and writes no experiment files. Its counts distinguish
an actually corrupted sender version from an attacker whose previous version
predates onset. Ancestry means exposure, not proof of permanent damage. The
first-admission details and printed table can confirm whether the actual HPC
trajectory follows the local failure. A checkpoint can lag live progress by
one round.

No repaired algorithm is claimed here. Changing the peak rule, fallback or
selection policy requires an explicitly versioned, matched follow-up. Preserve
the current frozen results; do not silently retune an in-progress experiment.
