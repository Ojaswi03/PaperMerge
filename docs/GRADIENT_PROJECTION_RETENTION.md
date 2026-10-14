# Gradient-projection retention: clean learning gate

This is a **separate experimental method**, not a change to the completed IID,
CE, uniform-anchor or class-aware-anchor studies. No 50/50 model averaging is
introduced. No threat matrix or production queue is started.

The separately authorized [three-round warmed-memory follow-up](GRADIENT_PROJECTION_WARM_DIAGNOSTIC.md)
starts both controls fresh and assesses rounds 1–2, without changing this
implementation or overwriting the one-traversal evidence below.

## Source and adaptation

[Saha, Garg and Roy, *Gradient Projection Memory for Continual Learning*,
Sections 5–6](https://arxiv.org/html/2103.09762v1) constructs layer-input
subspaces using SVD, removes gradient components in those subspaces, and
expands memory with residual representations after learning each task.

| Source method | Our explicit adaptation |
|---|---|
| Layer-input subspaces, residual expansion, orthogonal updates | Same mathematical principle, with covariance eigendecomposition equivalent to left singular vectors |
| No bias units in the reported architectures | Existing CNN biases retained and protected through affine input augmentation |
| Multi-head CIFAR experiments | One shared ten-logit CIFAR-10 head; no inference-time class/node hint |
| Sequential tasks | Fixed logical ring and unchanged receiver-local BASIL selection |
| Task memory | Basis state travels with exactly the selected received snapshot |

The source results do **not** establish that this adaptation will solve
single-class training with a common classifier. It is not exact paper GPM,
exact EWC, a new EBM loss, or a claimed novel algorithm.

## Exact mathematics

For layer l, flatten its kernel into Keras's input-by-output matrix and append
the bias as its final row. Append a constant 1 to each layer-input vector.
Let G be the corresponding ordinary CE gradient and Q the frozen orthonormal
input basis received with the selected snapshot:

```text
Q.T @ Q = I
G_applied = G - Q @ (Q.T @ G)
theta_next = theta - eta_round * G_applied
eta_round = 0.05 / (1 + 0.05 * round)
```

This direction solves the Euclidean projection problem
`min_U ||U-G||²/2 subject to Q.T @ U = 0`.
It is **constrained CE optimization**, not `CE + a quadratic penalty`.
In particular, it is not the previously tested diagonal-importance anchor.

For fixed represented inputs in the protected span, the affine preactivation
change from this step is approximately zero, up to floating-point error.
That does not guarantee whole-network predictions remain unchanged: earlier
layer inputs evolve, representation sampling is approximate, and unrepresented
directions remain free. Bias updates cannot bypass the projection.

After five full epochs, 32 deterministic receiver-training images supply
unaugmented layer-input representations R. For convolutions, use flattened
patches and at most 2,048 deterministic patch visits per layer. Expand Q using
the leading eigenvectors of the residual covariance:

```text
R_residual = R - Q @ (Q.T @ R)
captured_energy + added_residual_energy >= 0.99 * ||R||_F²
Q_new = [Q, orthonormal_new_directions]
```

The 99% threshold and sample bounds are declared before the gate. Old
directions are never dropped to make room for new ones. Rank is bounded by
the existing architecture, not by a tuned accuracy target.

For the approved CNN the affine input dimensions are `28, 257, 65, 385, 193`.
Even at full rank, one basis bundle occupies about 0.98 MiB in float32. Basis
metadata adds communication/storage; it has **no formal privacy or Byzantine
authentication guarantee**. No raw images are transmitted.

## Data/state flow

```text
five predecessor snapshots + their matching basis states
                         |
         minimum CE on ONE receiver-local training batch
                         |
           select ONE complete parameter snapshot
                         |
         load all kernels/biases; freeze incoming Q
                         |
       five full epochs: ordinary CE -> projected gradient
                         |
     build residual bases using this receiver's training data
                         |
  deliver complete parameters + bases to next five memories
                         |
                only the next node trains
```

BASIL does not select using validation accuracy, basis rank, class coverage,
sender-reported metrics or the CIFAR test set. Auxiliary states from other
candidate models are not merged. A newly selected initialization snapshot may
therefore discard a previously trained model's representation history. That
behavior must be reported, not silently overridden.

## Paired gate versus full-data protocol

| Setting | This clean gate | Original full-data protocol |
|---|---:|---:|
| Logical nodes / class assignment | 10 / node i owns class i | Same |
| Horizon | One traversal per paired control | Separately authorized horizon |
| SGD examples/node | 4,500 | 5,000 |
| Reserved training-validation examples/class | 500 | No reservation in completed runs |
| Local epochs / batch | 5 / 512 | 5 / 512 |
| Batches/epoch | 9 | 10 |
| Final partial batch | 404 | 392 |
| SGD steps/activation | 45 | 50 |
| Model / initial LR / schedule / reset SGD | Unchanged | Unchanged |
| BASIL / memory / complete handoff | Unchanged / S=5 | Same |
| Attacks / noise / EBM / anchoring | All off | Defined by separate condition |
| CIFAR test set | **Not evaluated or used** | Evaluation only |
| W&B | Disabled | Independent study setting |
| `researchValid` | **false** | Requires full protocol verification |

The gate's validation images are disjoint from SGD, Snapshot Selection and
basis collection. All 50,000 training examples are accounted for; none of the
10,000 test images enters this gate. Five epochs are never shortened.

The clean controls share initialization, split, training/augmentation seeds,
ring and BASIL selection semantics. Their model trajectories and selected
sources may legitimately diverge after projected updates.

### Predeclared learning-trend screen

1. Each of the first four nodes improves its own validation-class accuracy
   by at least 5 percentage points relative to its received starting model.
2. After Node 3, at least two earlier classes exceed 5% validation accuracy.
3. Their mean exceeds the paired CE control by at least 5 percentage points.

These are modest **validation trend screens**, not software-correctness
assertions or the 50% research target. Preserving a constant-class predictor
while preventing new learning is not success. No coefficients are adjusted
after observing the screen. The cold-start/selection qualification is important:
an empty selected basis means the update is exactly CE, so a failed screen
cannot automatically be attributed to projection in isolation.

If the gate fails, stop and report it; do not run 30/500 rounds unchanged.
If it passes, a full-data 30-round diagnostic also requires completed research
checkpoint integration before it is launched. This gate deliberately cannot
start a production queue.

## Commands

```bash
# Mathematical device check: synthetic data only.
environment/basil-noise-env/bin/python scripts/run_gradient_projection_gate.py --device GPU --verify-device

# Starts the separately labelled clean paired training-validation gate.
environment/basil-noise-env/bin/python scripts/run_gradient_projection_gate.py --device GPU --execute
```

The runner refuses an existing output directory rather than overwriting or
guessing a continuation. It uses the existing single-process GPU lease,
6-GiB allocator cap, 1-GiB startup headroom, float32 numerical policy and
finite checks. A failure records the condition and batch coordinates, losses,
gradient statistics, parameters and bases. No model/epoch/batch tuning occurs
to avoid instability. GPU checks establish tolerances, not bitwise CPU/GPU
equivalence or multiprocessing correctness.

Gate files live under:

- `newResults/nonIID/gradient_projection/clean_training_validation_gate_v1/`
- `newPlots/nonIID/gradient_projection/clean_training_validation_gate_v1/`

Every gate activation round-trips all five predecessor memories, parameter
hashes and orthonormal basis hashes through a lossless NPZ memory checkpoint.
Those memory files are **not full research-run resume checkpoints**: no
cursor/optimizer/history recovery is claimed. Interrupted gates stop rather
than silently loading final weights and pretending to continue.

## Implementation boundaries

| File | Responsibility |
|---|---|
| `basil_core/gradient_projection_retention.py` | Bias-aware projection, frozen state, training-only basis expansion and memory serialization |
| `scripts/run_gradient_projection_gate.py` | Paired clean gate, disjoint validation split, actual device verification, finite audits, gate decision and saved plots/report |
| `scripts/report_gradient_projection_gate.py` | Saved-only contract verification, measured retention/gradient summaries and protected-history confusion plot; never trains |
| `tests/test_gradient_projection_retention.py` | Analytic projection, empty/full basis, SVD-equivalent energy, finite rejection, Round-11 memory, five epochs, no lambda, disjoint split and honest gate decision |

The completed scientific source files are not edited, and historical result
manifests are not rewritten to accommodate this experiment.

## Completed gate result

Both paired controls completed their one traversal on the RTX 4070 Ti:
20 activations, 900 SGD updates, and 8.10 minutes total. All five epochs
executed per activation. **The predeclared learning screen did not pass.**
The mathematical/software contracts and the learning screen are separate.

| Held-out training-validation metric | CE | Projection |
|---|---:|---:|
| Mean across ten trained node states | 10.000% | 10.974% |
| End-of-traversal Node 9 | 10.000% | 12.900% |
| Own-class mean | 100.000% | 58.600% |
| Other-class mean | 0.000% | 5.682% |
| Constant-class trained states | 10/10 | 7/10 |

These are **not full CIFAR test results**. Both methods selected initialization
snapshots at Nodes 0–4, so projection was inactive there and their output
hashes matched CE exactly. This limits what the cold-start screen says about
projection in isolation. Where protected history was present, mean removed
gradient norm/base-gradient norm was 0.999676, and mean applied/base norm was
0.024005. Projection was active, but often suppressed learning the new class.

There was limited positive retention: Node 7 reached 86.0% horse and 52.4%
automobile accuracy. Node 9 retained 78.0% horse and 51.0% automobile accuracy,
but had zero accuracy on the other eight classes, including its own truck
class. Nodes 5, 6, 8 and 9 failed to learn their own class in this gate.
This is not a solution to ten-class learning and is not a reason to silently
retune the projection threshold or change BASIL's selection rule.

All **28 saved-result checks passed**. Synthetic CPU/GPU gradients differed by
relative L2 2.2755e-7 (tolerance 0.01); the maximum SGD update difference was
1.4901e-8 (tolerance 1e-4). GPU batches of 512 and 392 were explicitly checked.
The actual gate peaked at 269.04 MiB with zero memory-pressure events and no
recorded numerical failure/OOM. Multiprocessing remains unverified.

No 30-round run, attack/noise matrix, W&B run or queue modification followed.
The result is a negative gate outcome with some late two-class retention,
not proof that GPM cannot work in every setting.

[Measured gate review, per-node tables and plots](../newResults/nonIID/gradient_projection/clean_training_validation_gate_v1/GRADIENT_PROJECTION_GATE_REVIEW.md)

Saved-only verification/report command:

```bash
environment/basil-noise-env/bin/python scripts/report_gradient_projection_gate.py
```

## Final verification

| Verification | Actual result |
|---|---|
| Compileall | Exit 0 |
| Focused new/existing regression tests | 218 passed, 1 display-dependent skip, 2 dependency warnings; 25.48 seconds |
| Saved gate contracts | 28/28 passed |
| Actual CPU/GPU synthetic checks | Passed declared gradient/update tolerances |
| Historical evidence and paper SHA-256 | 23,651 before / after / matched; changed 0 |
| Learning screen | Did not pass; no longer run started |

The SHA-256 baseline is `gui/worker_state/gradient_projection_before.json`.
Only the new `gradient_projection/` result/plot namespaces are excluded from
the after comparison; no pre-existing file is excluded. The initial manifest
was recorded before those namespaces were created. Existing scientific source
hashes also matched the completed gate's manifest. The known unrelated legacy
full-suite collection problem was not fixed or claimed fixed; these are
focused regression results.

Actual compile/test commands:

```bash
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests

DISPLAY= MPLCONFIGDIR=/tmp/papermerge-projection-test-mpl PAPERMERGE_LIVE_TRACKING=disabled WANDB_MODE=disabled PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_gradient_projection_retention.py tests/test_class_aware_retention.py tests/test_retention_diagnostic.py tests/test_noniid_study.py tests/test_iid_study.py tests/test_research_protocol.py tests/test_protocol_recovery.py tests/test_iid_500r.py tests/test_memory_safety.py tests/test_research_audit.py tests/test_iid_gpu.py tests/test_iid_campaign.py
```

The saved-report generator's NumPy-count JSON serialization defect was fixed
and regression-tested after training. It did not affect SGD, projections,
validation measurements or recorded raw activations; no retraining occurred.
