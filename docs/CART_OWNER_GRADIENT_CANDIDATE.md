# CART retention: failure diagnosis and owner-gradient learning diagnostic

## Measured outcome, not an accuracy target

Both clean controls completed **30/30 rounds** on the RTX 4070 Ti:
600 activations, 30,000 optimizer updates and **5.01 hours** total.
Saved-result verification passed **4,260 checks**, with 3,600 epoch rows.
No numerical failure or OOM was recorded. Five full epochs and strict handoff
worked; the **learning mechanism did not retain other classes**.

| Final metric | Feature replay | + Selected-encoder preservation |
|---|---:|---:|
| Node-9 full-test accuracy | 10.000% | 10.000% |
| Mean-node full-test accuracy | 10.000% | 10.000% |
| Worst-node full-test accuracy | 10.000% | 10.000% |
| Own-class training-probe accuracy | 100.000% | 100.000% |
| Other-class training-probe accuracy | 0.000% | 0.000% |
| Mean old-class squared feature drift | 117.052974 | 0.346721 |

Full-test metrics use the intact 10,000-image CIFAR test set, evaluation only.
The 640-image training probe is neither test accuracy nor held-out validation.
Preservation greatly reduced drift, but did not restore class decisions.
Do not present this as a successful retention result.

Read the [saved-only failure audit](../newResults/nonIID/cart_retention_failure/saved_30r_v1/CART_RETENTION_FAILURE_REVIEW.md)
and the [completed comparison](../newResults/nonIID/cart_representation/clean_30_rounds_fixed_gamma_v1/REPRESENTATION_PRESERVATION_REVIEW.md).

## What the existing objective can and cannot do

For `K` available older classes, the approved feature objective is

$$L=CE_i+\frac{\beta}{K}\sum_{c\in C_{old}}CE_c^{feature}
 +\frac{\mu_e}{2}\|\theta-\theta_{sel}\|^2
 +\gamma\,\mathbb E_{x\in D_i}\|h_\phi(x)-h_{\phi_{sel}}(x)\|^2.$$

| Term | Confirmed behavior | Limitation |
|---|---|---|
| Local CE | Trains encoder and classifier on the current class | Encourages specialization |
| Detached feature replay | Supplies older labels to the classifier | Exactly zero replay gradient for the encoder |
| Parameter anchor | Limits movement from the selected snapshot | Does not supply older-class training evidence |
| Selected-encoder preservation | Constrains features on current-class images | Selected teacher is weak; old images have no direct supervised encoder path |
| CART registry | Carries class/support/history metadata | Not replay data, a gradient or proof of retained knowledge |

### Relative class weighting is not uniform

With beta=1, the current class has CE coefficient **1**, while each of nine
older classes has **1/9**. Equal weighting *among older classes* is not equal
weighting *among all ten classes*. This follows the approved objective; it is
not an implementation error to conceal by rewriting old results.

For an uninformative constant-logit classifier, its stationary probabilities
are `p_current=1/2`, `p_each_old=1/18`. This analytic example is verified by
autodiff. It is **not** proof that imbalance alone explains the CNN result:
feature coordinates, representation quality and the classifier also matter.
Changing beta alone does not create older-class encoder supervision.

At final Node 9, feature replay achieves only **14.583%** on its frozen old
features, and preservation achieves **0%**; neither retains old raw-image
classes. Both selected-encoder/selected-head training-probe states also score
10%. Keeping a weak incoming reference stable is not cumulative learning.

## Separate candidate: class-balanced owner-local gradient rehearsal

Implement this only as a **separate experimental CART method**, not a change
to the approved BASIL baseline:

```text
BASIL memory -> local SS -> load one complete selected snapshot
                                      |
                              active parameters theta_k
                                      |
       +------------------------------+----------------------------+
       |                              |                            |
   current owner                  old owner c                 old owner d
   local batch                    local TRAIN batch           local TRAIN batch
   compute CE gradient            compute CE gradient         compute CE gradient
       |                              |                            |
       +------------------ gradients only; no images ---------------+
                                      |
                         equal-class gradient mean
                                      |
                      ONLY active trainer applies SGD
                                      |
                   usual sequential snapshot handoff
```

Let `C` contain the active class and genuinely available previous owners.
Each owner computes a minibatch mean CE using the **same current parameters**:

$$F_c(\theta_k)=\frac1{|B_c|}\sum_{(x,y)\in B_c}CE(f_{\theta_k}(x),y).$$

The class-balanced supervised component is

$$F_{bal}(\theta_k)=\frac1{|C|}\sum_{c\in C}F_c(\theta_k),
\qquad g_{bal}=\frac1{|C|}\sum_{c\in C}\nabla_\theta F_c(\theta_k).$$

Unlike detached feature replay, older-class gradients reach **both encoder and
classifier**. References/registries and any retained CART anchor must be
labelled as separate components, not folded into this supervised CE gradient.
No EBM or channel noise is part of this initial candidate verification; the
existing source EBM and no-lambda behavior are untouched.

This is **gradient aggregation**, not model averaging. Complete starting-weight
replacement can remain intact, and each owner keeps its own images. However,
this adds older-class supervision to local updates and is **not pure
one-class-only SGD or an exact reproduction of paper BASIL**. Communication
approval was obtained for one clean traversal with 16 samples per older-owner
response. No raw-image sharing,
ACDS, pretrained encoder or silent epoch reduction is proposed.

### Operational requirements before learning

- Recompute owner gradients at every current SGD state. A gradient computed
  once at the selected snapshot is stale after the first update.
- Check state hash, receiver, round, step, class owner and tensor layout.
- Only actual available owners contribute; missing/future classes are not invented.
- Owner helpers do not apply SGD or transmit optimizer state. Only the active
  node performs the fixed local optimizer steps.
- Explicitly choose helper batch size and any retained anchor/preservation
  components before running; do not select them from test accuracy.
- Start with a clean bounded traversal and no W&B. Passing algebra does not
  justify another long campaign or a claimed >50% result.

### Costs and threat limitations

For 117,706 float32 parameters, a full gradient contains 470,824 coordinate
bytes. Nine helpers return **4,237,416 bytes/step**, about **202.06 MiB per
50-step activation**, excluding outgoing current-weight synchronization and
serialization. This is substantially more communication than feature packets.
A practical runner should reuse one helper model rather than allocate models
or trace new graphs per batch.

Gradients can leak information; no privacy guarantee is claimed. Correct hashes
do not prove an owner is honest. BASIL snapshot filtering does **not** automatically
protect a new gradient-response channel against malicious owners. Attack/noise
semantics and defenses require separate work after clean learning is established.
Physical distributed transport is not implemented by the mathematical interface.

## Code and verification scope

| File | Responsibility |
|---|---|
| `basil_core/cart_owner_gradients.py` | Prior diagnosis, immutable owner gradient responses, owner-side CE backprop, exact-state balanced gradient mean; no runner/optimizer/queue wiring |
| `basil_core/cart_owner_gradient_worker.py` | Opt-in active worker: fresh older-owner CE gradients at every current step, equal-class weighting, existing receiver-supported CART anchor, one reusable helper model |
| `scripts/run_cart_owner_gradient_diagnostic.py` | Plan-only default, independent CPU/GPU synthetic validation, approved one-traversal learning runner; rejects overwrites and unverified device proofs |
| `scripts/report_cart_owner_gradient_diagnostic.py` | Saved-data-only full-test/TRAIN-probe review and plots; rejects incomplete evidence |
| `scripts/audit_cart_retention_failure.py` | Saved 30-round failure audit and zero-update actual-CNN derivative proof; plan-only default |
| `tests/test_cart_owner_gradients.py` | Exact objective gradients, encoder supervision, stale/duplicate/nontraining rejection, immutable payloads, lazy imports |

The actual-CNN **CPU** proof uses 16 training examples/class, disjoint from the
existing SS and analysis batches. Sixteen is a derivative-test input size,
**not itself approval of a research helper batch size**. The subsequent explicit
operator approval selected 16 for the bounded learning diagnostic. The original
CPU derivative proof performs **zero SGD updates**
and never loads test images. Both canonical and saved collapsed model states
remain hash-identical after measurement.

| CPU derivative check | Canonical CNN | Saved preservation Node 9 |
|---|---:|---:|
| Relative gradient difference vs direct balanced CE | 2.212164e-6 | 2.286797e-7 |
| Maximum absolute tensor-coordinate difference | 2.112938e-8 | 5.140901e-7 |
| Parameters changed | No | No |
| Older-class encoder gradients nonzero | 9/9 | 9/9 |

Tolerance: relative difference ≤1e-5 and maximum absolute difference ≤1e-6.
Proof: `gui/worker_state/CART_OWNER_GRADIENT_CPU_VERIFICATION.json`.
The new active worker subsequently passed an independent full-CNN objective
check on CPU and GPU, with both 512- and 392-example active batches and 16-example
owner batches. CPU/GPU relative gradient differences were 2.749174e-7 and
1.507779e-7; maximum update differences were 9.313226e-9 and 7.450581e-9.
It also passed exact same-device no-lambda checks. This is representative numerical
validation, not a claim of bit-identical CPU/GPU training or Byzantine safety.
Proof: `gui/worker_state/CART_OWNER_WORKER_GPU_VERIFICATION.json`.

### Authorized learning setup

The new diagnostic retains the existing CART registry and receiver-local
epoch-wise anchor, capped at 0.01. It **does not** use the failed detached feature
replay or selected-encoder preservation terms. Its objective is

$$L_k=\frac{1}{|C_k|}\sum_{c\in C_k}CE_c(\theta_k)
 +\frac{\mu_e}{2}\|\theta_k-\theta_{selected}\|^2.$$

For the first traversal, `C_k={0,...,i}` while Node `i` trains: no future classes
are requested. The active node still processes all 5,000 local examples in
each of five epochs. Helpers use 16 TRAIN samples at each active SGD step,
with separate keyed sampling/augmentation; they perform no optimizer updates.
Their gradients use minibatch **means** and equal **class** weights, even though
the active batch has 512/392 examples and a helper batch has 16.

Starting snapshot selection remains receiver-local and uses the same 512 own-class
TRAIN images for all candidates. Each selected snapshot fully replaces the
active parameters and stays immutable as the anchor. No attack, channel noise,
EBM, lambda, clipping, momentum, W&B or queue launch is enabled.

One active model plus one serially reused helper model is used under the verified
6,144-MiB GPU cap. TensorFlow retains bounded reusable allocator blocks; no unsafe
session/VRAM resets are performed during SGD. Finite-value failures save the
actual pending active weights/batch and per-owner indices/key/context rather
than replaying an unrelated CE-only graph. Resume is not automatically supported.

Training-side class-retention/confusion metrics are recorded before training
and after every epoch using the same 640 TRAIN analysis images as earlier
diagnostics. They are not held out from ordinary SGD, and never control SS or
the anchor. Helpers exclude both this analysis probe and the SS batch. The full
10,000-image test set is evaluated after each activation only; no test values
control training, coefficients or stopping. Results are diagnostic, not
production-valid promotion evidence.

| Previous interface-stage engineering check | Actual result |
|---|---|
| Candidate focused contracts | **13 passed** in 2.27 seconds |
| Complete CPU/headless suite | **527 passed, 2 skipped, 644 subtests passed** in 71.23 seconds |
| Warnings | Two dependency deprecations and one Keras input-structure warning in a synthetic fixture; no failed tests |
| Display-dependent exclusions | Two GUI tests skipped; no new manual GUI validation claimed |
| Compilation | `gui`, `basil_core`, `reporting`, `scripts`, `tests` and `run_gui.py` passed |
| Lazy-import scan | 217 Python sources; zero direct heavy module-scope imports |
| Default CLI plan | Zero NumPy/TensorFlow/W&B/matplotlib imports and zero training |
| Existing evidence integrity | **23,889 saved result/plot/paper files matched SHA-256; zero changed**; only the new saved-only failure-audit directory excluded |

The integrity baseline is
`gui/worker_state/cart_retention_solution_before.json`. It includes the now
completed 30-round comparison, not just older IID evidence. The new analysis
does not regenerate its plots, edit its manifests or alter its mathematical
source files.

```bash
# Plan only: zero ML imports or training.
environment/basil-noise-env/bin/python scripts/audit_cart_retention_failure.py

# Completed saved audit and CPU proof; refuse output overwrite:
environment/basil-noise-env/bin/python scripts/audit_cart_retention_failure.py --audit
environment/basil-noise-env/bin/python scripts/audit_cart_retention_failure.py --verify-owner-objective

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-cart-owner-tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_cart_owner_gradients.py

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-cart-owner-full-tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests

environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests run_gui.py
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py
```

## Reproduction commands

```bash
# Plan only; no heavy imports, data loading or training.
environment/basil-noise-env/bin/python scripts/run_cart_owner_gradient_diagnostic.py

# Completed synthetic verification. Refuses to overwrite its existing proof.
environment/basil-noise-env/bin/python scripts/run_cart_owner_gradient_diagnostic.py --device GPU --verify-device

# Authorized bounded learning run; refuses existing result/plot directories.
environment/basil-noise-env/bin/python scripts/run_cart_owner_gradient_diagnostic.py --device GPU --execute

# Saved results only, no retraining; refuses existing review output.
environment/basil-noise-env/bin/python scripts/report_cart_owner_gradient_diagnostic.py
```

Do not rerun the unchanged failed loss expecting more rounds or stronger
preservation alone to supply missing class supervision. This candidate changes
the auxiliary supervision protocol, not the approved BASIL baseline.

## Completed one-traversal learning result

The operator-approved GPU run completed **10/10 activations**, **500 active SGD
updates**, and **zero helper SGD updates** in **342.37 seconds (5.71 minutes)**.
All ten active nodes used their full 5,000-example local dataset for five epochs:
ten batches per epoch, with the final batch of 392 retained. There were 2,250
older-owner responses, each using 16 training images. No attack, noise, EBM,
W&B, research queue, 30-round follow-up or production promotion was started.

| Node after its five epochs | Full-test accuracy | Own-class TRAIN probe | Older-class TRAIN probe mean |
|---|---:|---:|---:|
| 0 | 10.00% | 100.00% | Unavailable |
| 1 | 10.00% | 100.00% | 0.00% |
| 2 | 10.00% | 0.00% | 50.00% |
| 3 | 11.84% | 0.00% | 39.58% |
| 4 | 10.00% | 0.00% | 25.00% |
| 5 | 10.00% | 0.00% | 20.00% |
| 6 | 10.00% | 0.00% | 16.67% |
| 7 | 10.02% | 0.00% | 14.51% |
| 8 | 10.17% | 0.00% | 13.09% |
| 9 | **12.50%** | **0.00%** | **13.72%** |

| End-of-traversal metric | Result |
|---|---:|
| Node-9 full-test accuracy | 12.50% |
| Mean full-test accuracy across the ten trained node states | 10.453% |
| Worst-node full-test accuracy | 10.00% |
| Node-9 balanced TRAIN analysis-probe accuracy | 12.344% |
| Node-9 full-test CE | 2.308092 |
| Finite batch traces with matching current/owner hashes | 500/500 |
| Numerical failures / OOM / memory-pressure events | 0 / 0 / 0 |
| Peak TensorFlow GPU allocation / configured cap | 270.63 / 6,144 MiB |
| Older-owner response coordinate bytes | 1,059,354,000 (~1,010.28 MiB) |

Response traffic is a **simulated coordinate payload count**, not measured
network traffic. It excludes current-parameter requests and serialization.
The extra channel is therefore costly even though this bounded single-process
implementation uses little VRAM. Clearing a live CUDA allocator was unnecessary.

### What improved, and what did not

Node 9 no longer predicts exactly one class. Its full-test class accuracies are:

| Class | Accuracy | Class | Accuracy |
|---|---:|---|---:|
| airplane | 71.20% | dog | 0.00% |
| automobile | 49.90% | frog | 0.00% |
| bird | 0.70% | horse | 0.00% |
| cat | 3.20% | ship | 0.00% |
| deer | 0.00% | truck | 0.00% |

**This is partial old-class discrimination, not successful all-class retention.**
The end model predicts only classes 0–3: 5,863 airplane, 3,890 automobile, 49 bird,
and 198 cat predictions over the 10,000 test images. The tiny change from the
previous 10% result must not hide failure to learn six classes or its own class.
Nodes 2–9 also score 0% on their own class on the training analysis probe.

The canonical initialization hash, partition hash and all 640 analysis image IDs
match both the earlier one-traversal feature comparison and completed 30-round
representation comparison. The latter is nevertheless a **different objective
and horizon**, not a paired 30-round ablation of this new method. No statistical
significance or converged performance is claimed from one traversal/seed.

### Evidence explaining the limited learning

| Observation | Evidence | Interpretation / scope |
|---|---|---|
| Old-class encoder supervision exists | Mean old-owner encoder-gradient norm at Node 9: 1.025867 | The missing encoder path was fixed; zero replay encoder gradients are no longer the explanation |
| Initial snapshots are repeatedly preferred | Nodes 0–4 select initial sender-9 snapshots at source round -1 | Receiver-local one-class SS can reject previous specialized states; correct BASIL selection does not guarantee globally cumulative knowledge |
| Balanced updates differ from own-class fine-tuning | Node-9 mean own-class CE gradient norm 2.410335; applied norm 0.139910 | Class averaging/cancellation changes update size/direction; this is not clipping or a forced multiplier |
| Snapshot anchor is small here | Node-9 mean anchor/own-CE gradient norm ratio 0.0002173 | There is no evidence that the anchor dominates this failure |
| End model remains weak | Full-test CE 2.308092; only four predicted classes | The candidate is not ready for attacks/noise or a long production queue |

Only five epochs on each active node and 16-sample older-owner responses were
approved. More repetitions might change learning, but that is **unmeasured**.
Do not silently alter SS, the CNN, epochs, LR, helper sample count or add a longer
run to chase the test accuracy. Any follow-up must be explicitly scoped and
declared before training; test data remain evaluation-only.

For compatibility, inherited telemetry names `ebmObjective` and
`robustObjectiveGradientNorm` store the **complete applied objective** and
**applied gradient norm**, even though EBM is disabled. `baseCrossEntropy` and
`baseGradientNorm` describe the **current owner's** ordinary CE, not balanced
CE. New `balancedClassCrossEntropy`, `ownerResponseBytes`, and
`olderEncoderGradientNorm` fields separate the auxiliary mechanism. Every
recorded EBM coefficient/penalty is exactly zero.

### Current engineering and integrity evidence

| Check | Actual outcome |
|---|---|
| Focused owner/worker/CART/protocol/lazy suites | **126 passed** in 10.23 s |
| Full CPU/headless suite | **535 passed, 2 skipped, 644 subtests passed** in 69.86 s |
| Warnings | Two dependency deprecations and one synthetic Keras input-structure warning |
| Compile checks | Passed for GUI, core, reporting, scripts and tests |
| Lazy-import scan | 221 Python files; no movable/direct heavy module-scope imports |
| GPU objective checks | Passed independent balanced-CE-plus-anchor gradient and SGD checks for batches 512 and 392; exact no-lambda same-device repeat |
| Existing evidence/paper SHA-256 | **23,891/23,891 matched; zero changed** |
| Multiprocessing / RPC / malicious helpers | Unverified; not used |

Integrity baseline:
`gui/worker_state/cart_owner_gradient_learning_before.json`; comparison record:
`gui/worker_state/cart_owner_gradient_learning_integrity.json`. Only newly created
files under this specific diagnostic's result/plot directories are allowed additions.
The completed 30-round comparison, IID outputs and papers remain unchanged.

Read the [measured review](../newResults/nonIID/cart_owner_gradients/clean_one_traversal_16_v1/OWNER_GRADIENT_LEARNING_REVIEW.md)
and inspect the [full-test node/class heatmap](../newPlots/nonIID/cart_owner_gradients/clean_one_traversal_16_v1/node_class_accuracy.png)
and [within-epoch TRAIN retention heatmap](../newPlots/nonIID/cart_owner_gradients/clean_one_traversal_16_v1/five_epoch_class_retention.png).

```bash
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-owner-worker-tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_cart_owner_gradients.py tests/test_cart_owner_gradient_worker.py tests/test_strict_cart.py tests/test_research_protocol.py tests/test_lazy_imports.py

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-owner-full-tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests

environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py
```
