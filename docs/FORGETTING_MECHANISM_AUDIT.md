# One-class forgetting: four follow-up actions

**The bounded audit is complete. No longer campaign or new retention method was started.**

The [detailed measured report](../newResults/nonIID/forgetting_audit/warmed_node0_head_backbone_v1/FORGETTING_MECHANISM_REVIEW.md)
contains before/after-each-epoch tables, inference interventions, registry
measurements and figures. The prior [three-round study](STRICT_CART_PROTOCOL.md)
and every saved IID/nonIID run are preserved.

## What we did

| Requested follow-up | Action and outcome |
|---|---|
| Locate forgetting in the classifier or features | One warmed Node-0 activation per prior CE/fixed-anchor/CART state; save weights before and after every epoch and independently swap head/backbone for inference only |
| Check CART's missing retention signal | Confirm class-0-only control support; the local gap remained zero while the previously predicted class was lost |
| Choose the next bounded adaptation | Recommend a separately revisioned distributed training-side class-retention measurement; **not integrated into training or launched** |
| Keep EBM separate | Rerun full-objective/Hessian/no-lambda/anchor contracts; this clean audit runs without noise or EBM, so it makes no noise-mitigation claim |

“Four actions” does not mean launching four new A/B/C/D experiments.

## Diagnostic scope

```text
Preserved final R2 memory for each previous clean control
             |
Node 0: candidates from senders 5–9, each from R2
             |
Select by the same receiver-local 512-image CE batch
             |
Complete selected model load; frozen anchor; reset SGD
             |
Five full epochs over all 5,000 airplane training images
             |
Before + after each epoch:
save parameters, analyze features, compare inference-only heads
             |
STOP after one activation per control — no ring continuation
```

| Setting | Executed value |
|---|---|
| Device | RTX 4070 Ti, deterministic single process, float32/TF32 off |
| Number of activations | Three total: CE, fixed μ=0.01, current CART |
| Training per activation | 5,000 class-0 images, five epochs, 50 updates |
| Batches per epoch | Nine of 512 plus final 392 |
| Learning rate | Approved round decay at diagnostic R3: `0.05 / 1.15` |
| Optimizer | Reset SGD; momentum 0, no clipping or weight decay |
| BASIL | S=5, min receiver-local CE, nearest predecessor tie-break, no averaging |
| Attack / noise / EBM / W&B | All off |
| Test set | Not loaded or evaluated in this audit |
| Analysis probe | 64 training examples/class, 640 total; disjoint from selection |
| Research-valid status | False: a bounded mechanism audit, not a complete research run |
| Runtime | 93.78 seconds total |
| GPU tracked allocator peak | 272.41 MiB under the existing 6,144-MiB cap |
| Detected numerical failures / OOM | None in this diagnostic |

The balanced probe is centralized **analysis only**. It does not provide a node
with other classes for SGD, BASIL selection, CART control or tuning. Those
examples were not reserved from the original training: the scores are neither
held-out validation nor global test accuracy. Incoming states differ across
controls; do not treat this as a fresh common-initialization comparison.

## Main finding

| Final inference intervention | CE | Fixed anchor | CART |
|---|---|---|---|
| Actual trained model | Every probe image → airplane | Every probe image → airplane | Every probe image → airplane |
| Trained backbone + selected head | Every image → truck | Every image → dog | Every image → truck |
| Selected backbone + trained head | Every image → airplane | Every image → airplane | Every image → airplane |
| Trained model with output bias zeroed | Every image → airplane | Every image → airplane | Every image → airplane |

All are 10% balanced training-probe accuracy. CE/CART switch from truck to
airplane by epoch 2; fixed anchoring switches from dog by epoch 1.
Restoring the selected head restores its **old constant-class collapse**, not
useful cumulative learning. The trained head is sufficient to switch the
constant prediction on both backbones; output bias alone is not responsible.
Backbones also drift. These interventions do not prove that ten-class feature
knowledge is present, absent, or recoverable by a better classifier.

### Why the gradient does this

For a class-i minibatch, logits `z`, hidden features `h`, final kernel `W` and
bias `b`:

```text
p = softmax(z)                       # explanatory only; training uses logits CE
dCE/dz_c = p_c - 1[c=i]
dCE/db_c = batch_mean(p_c - 1[c=i])
dCE/dW[:,c] = batch_mean(h * (p_c - 1[c=i]))
```

These provide direct gradient pressure toward the active class and away from
non-target classes. Actual nonlinear updates need not move every logit
monotonically. There is no supervised previous-class term in the local batch.
The analytical head-gradient unit test checks these equations against actual
TensorFlow autodiff; raw single-class probe gradients are saved per epoch.

## CART's blind spot and the next decision

| Existing signal | What it can establish | What it cannot establish |
|---|---|---|
| Local class-0 probe | Current receiver-class loss/accuracy | Current retention of classes 1–9 |
| Carried class-best registry | Historical source-reported scores with support/round | That this exact current model still reproduces them |
| Frozen selected snapshot | A parameter reference | That its predictions are globally useful; these incoming states were already collapsed |
| Parameter-distance penalty | Discourage weight drift | Guarantee preservation of class decision boundaries |

**Selected next investigation: distributed training-side retention verification.**
Each class owner would evaluate the same current model on a deterministic local
probe and return `(model hash, owner, class, count, correct count, round, epoch)`.
Images would remain at their owner. Fresh hash-bound results would expose
which classes were lost, unlike historical maxima. BASIL's selection must
remain receiver-local and independent of these audit results.

This is a design decision, not an implemented or validated new algorithm.
It adds probe communication and needs explicit approval as a separate protocol
revision. Honest-owner scores could provide current evidence; a model hash
does **not** authenticate a Byzantine owner's accuracy claim. Scores also do
not provide missing-class gradients. Such a measurement system alone cannot
be advertised as a solution to forgetting.

Before applying it to a controller:

1. Test model-hash binding, support counts, staleness and absent-class handling.
2. In a clean bounded diagnostic, confirm it detects known losses while local
   images stay private and no test data enters control.
3. Reserve training-side validation **before** any new method's training or
   hyperparameter choice; disclose the resulting reduced training counts.
4. Predeclare the controller and assess both old-class retention and new-class
   learning. Stop a method that merely preserves an old constant predictor.

No replay, ACDS, public reference data, larger μ, head freezing, head merging,
new loss, remote controller or production queue was enabled here. In particular,
distilling a collapsed teacher on only the new class's inputs is not assumed
to recover previously unseen classes.

## EBM remains the noise mechanism

```text
L = CE + sigma_e² ||gradient CE||² + (mu/2) ||theta - theta_selected||²
       |                               |
       source-motivated EBM             snapshot-drift anchor

applied gradient = g + 2 sigma_e² H g + mu (theta - theta_selected)
```

Nested tapes compute the Hessian-gradient correction without materializing the
CNN Hessian. The anchor is outside the inner CE gradient norm; legacy lambda
does not affect this mode. With absolute Gaussian communication, coordinate
standard deviation is `sigma_e`, variance is `sigma_e²`, independent of model
norm. The relative-L2 branch remains separate and unchanged.

The mathematical contracts passed, but they do not show noise mitigation or
learning success. A constant-class predictor can have near-zero CE and CE
gradient on its own class, so a local gradient-norm term need not penalize its
failure on other classes. First obtain interpretable clean retention evidence;
then examine EBM under matched noisy conditions without changing the baseline.

## Files and reproduction

| File | Purpose |
|---|---|
| `basil_core/forgetting_audit.py` | Lazy analysis helpers: stable logits CE, head/backbone copies, drift and feature statistics |
| `scripts/run_forgetting_mechanism_audit.py` | Explicitly gated three-activation GPU/CPU audit; no queue or test loading; refuses overwrites |
| `scripts/report_forgetting_mechanism_audit.py` | Saved-only verification, CSV, epoch/confusion/intervention plots and measured review |
| `tests/test_forgetting_audit.py` | Exact split, mathematical gradients, finite checks, probe separation, callback non-mutation and incomplete-report rejection |

```bash
# Default is config-only: no TensorFlow, data loading or training.
environment/basil-noise-env/bin/python scripts/run_forgetting_mechanism_audit.py

# Regenerate the completed audit's report and figures; zero training.
environment/basil-noise-env/bin/python scripts/report_forgetting_mechanism_audit.py
```

The `--device GPU --execute` command was used once for this explicitly
authorized diagnostic. It refuses the existing output directory; do not delete
the evidence to trigger an automatic retry. Saved memory state is an evidence
source, not a complete production-resume state. Original three-round epoch
weights were unavailable, so no original intermediate weights were fabricated.

No mathematical training code or GUI/queue behavior was changed. New scripts
also keep heavy imports lazy. Existing results/plots/papers were SHA-256
inventoried before work; final verification is recorded in the new audit's
`evidence_integrity.json`.

## Verification actually executed

| Check | Result |
|---|---|
| Full suite, CPU/headless | **449 passed, 2 skipped, 644 subtests passed**; two dependency deprecation warnings; 69.95 seconds |
| Focused initial audit/EBM/BASIL/lazy suite | 125 passed; later tests are included in the full-suite total, not added to it |
| Compile all application/research/test sources | Passed |
| Lazy-import inventory | 204 Python files; 0 movable/retained direct heavy imports at module scope; no rewrite performed |
| Saved audit contracts | 15 groups passed; 90 stage/intervention CSV rows, 18 hashed epoch checkpoints |
| Executed GPU audit | All 150 optimizer steps completed; no detected non-finite update/evaluation or OOM |
| Existing evidence/papers SHA-256 | **23,751 before, 23,751 after, 23,751 matched, 0 changed**; exclude only newly created audit output directories |

The two suite skips require a GUI display. This turn did not execute those
widget checks in a virtual display. No manual legacy convergence experiment,
long research run, noise robustness study or new multiprocessing check is
claimed as passed. Existing synthetic GPU verification was checked against
the actual device/executed-source fingerprints before the audit; CPU/GPU
bitwise equivalence is not claimed.

```bash
environment/basil-noise-env/bin/python -m compileall -q basil_core gui noise_comm reporting scripts tests run_gui.py

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-forgetting-audit PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests

environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py

# Explicitly authorized bounded execution, already completed once:
environment/basil-noise-env/bin/python scripts/run_forgetting_mechanism_audit.py --device GPU --execute

# Saved-only checks, CSV/report and figures:
environment/basil-noise-env/bin/python scripts/report_forgetting_mechanism_audit.py
```

**Conclusion:** the four follow-up actions produced an audited diagnosis and
a bounded next-design recommendation, not a new working retention algorithm.
The clean collapse still blocks promotion to a longer non-IID campaign.
