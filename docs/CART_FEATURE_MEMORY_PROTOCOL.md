# Experimental CART feature memory

**Implemented separately; clean bounded validation only.** Existing IID,
one-class baselines, legacy CART, GUI queues and saved results are unchanged.
Revision: `strict_cart_activation_feature_replay_v1`.

## Completed outcome

Both clean controls completed one traversal on GPU. **This version did not
improve raw-image class retention.** These are balanced, non-held-out TRAINING
probe scores, not global test accuracy:

| Final metric across node states | Current CART | CART + feature replay, β=1 |
|---|---:|---:|
| Mean balanced training-probe accuracy | 10.000% | 10.000% |
| Mean own-class accuracy | 100.000% | 100.000% |
| Mean other-class accuracy | 0.000% | 0.000% |
| Constant own-class predictors on the entire 640-image probe | 10/10 | 10/10 |

Replay is active: parameters differ at Nodes 1–9 and replay gradients are
nonzero. At Node 1, the classifier achieves 100% accuracy on the frozen old
class-0 features while current class-0 image-probe accuracy is 0%. The encoder
changes during every activation. This is consistent with the representation
drift limitation; the two sample sets differ, so it is not isolated causal
proof. See the [measured review](../newResults/nonIID/cart_feature_memory/clean_one_traversal_beta1_v1/CART_FEATURE_REVIEW.md)
for all nodes, component ratios, selections and epoch plots.

Execution: **20 activations, 1,000 updates, 548.33 seconds**, RTX 4070 Ti;
tracked allocator peak **272.57 MiB**, no detected non-finite update/evaluation
or OOM. Saved-result verification passed **130 checks**, producing 120 epoch
CSV rows. No long queue, attacks, noise, W&B or beta retuning followed.

**Not ready for a larger feature-memory campaign on this evidence.** Passing
math and execution contracts is different from passing a learning gate.

## What changed

| Component | Implementation | Status / scientific meaning |
|---|---|---|
| Class owner packets | Immutable bounded embeddings, class mean, support, owner, model/encoder hashes and activation identity | Training-side features; not images, privacy proof or trusted Byzantine evidence |
| Memory admission | Reject wrong model, encoder, receiver, round, duplicate owner, unsupported class or non-finite data | Provenance/compatibility checks, not a defense against a dishonest class owner |
| Representation refresh | Previously visited owners encode a deterministic local subset using the exact BASIL-selected encoder | Explicit additional communication, simulated with logical owners in one process |
| Replay objective | Equal-weight old classes; detached feature CE trains the final classifier | Additional missing-class supervision; no direct feature-replay encoder gradient |
| Current CART | Existing registry/controller and immutable selected-snapshot anchor inherited unchanged | Historical scores alone do not establish current class retention |
| Source EBM | Existing nested-tape local-CE gradient-norm objective retained as a separate optional component | No extra lambda; off in the clean learning diagnostic |
| Execution | Explicit one-traversal runner with current-source device verification | No automatic A/B/C/D campaign, promotion, W&B or production queue |

All numerical, plotting and ML libraries remain lazy. Imports alone do not
load TensorFlow, NumPy, matplotlib or W&B. Standard-library imports remain
ordinary imports.

## Complete flow

```text
Five latest predecessor snapshots
             |
Same receiver-local SS batch -> select ONE minimum-CE candidate
             |
Completely load selected weights and biases, freeze anchor, reset SGD
             |
NEW auxiliary owner requests:
send selected encoder to previously visited class owners
             |
owners encode their TRAINING images locally; return bounded embeddings only
             |
check selected model/encoder/activation identities; freeze feature memory
             |
five complete epochs: CE + current anchor + beta * old-class feature CE
             |
trained parameters -> original five successor memories
only the immediate clockwise successor trains next
```

The auxiliary requests are **not** BASIL's ordinary five-link parameter
broadcast: as many as nine previous owners may be contacted. This increases
communication and changes the information-sharing protocol, explicitly. BASIL
memory S=5, selection and the order of actual node training remain unchanged.
No receiver is supplied another owner's raw images for its local SGD.

At Node 0 of the first traversal, memory is empty. At Node 1 only previously
visited class 0 can contribute; at Node 9 the visited classes are 0–8. Never
manufacture future-class entries or initialize memory with test data.

## Loss and gradients

Let `theta=(phi,psi)`, where `phi` is the feature extractor and `psi` is the
final linear classifier. `theta_sel` is exactly the received selected snapshot.

$$
L_i = CE_i(\theta)
 + \underbrace{\sigma_e^2\|\nabla_\theta CE_i\|_2^2}_{\text{EBM}}
 + \underbrace{\frac{\mu_e}{2}\|\theta-\theta_{sel}\|_2^2}_{\text{CART-controlled snapshot anchor}}
 + \underbrace{\beta L_{replay}(\psi;M_i)}_{\text{new CART feature memory}}.
$$

$$
L_{replay} = \frac1{|C_{old}|}\sum_{c\in C_{old}}
             \frac1{|M_{i,c}|}\sum_{z\in M_{i,c}}CE(a_\psi(\operatorname{stop\_gradient}(z)),c).
$$

Empty memory means replay loss/gradient zero, not an invented sample. Each
available old class receives equal total weight, independent of packet size.
The applied gradient is:

$$
g+2\sigma_e^2H_{CE}g+\mu_e(\theta-\theta_{sel})
 +\beta\begin{bmatrix}0_\phi\\\nabla_\psi L_{replay}\end{bmatrix}.
$$

EBM still uses **local CE only**, not CE plus replay or anchoring. References
and replay features are detached. The temporary selected encoder is not
frozen for training: local CE/EBM/anchoring still update all model parameters.

| Coefficient | Meaning | This diagnostic |
|---|---|---|
| `sigma_e²` | Actual absolute channel coordinate variance, not an adjustable gradient multiplier | 0; no channel noise or EBM |
| `mu_e` | Existing CART confidence/gap-controlled anchor, fixed within each epoch | Unchanged controller, cap 0.01 |
| `beta` | Older-class feature-supervision weight, unrelated to EBM variance or legacy lambda | **1.0, explicitly approved before execution** |

No field called `legacyEbmLambda` affects the new path. β=0 reproduces the
existing CE/EBM/anchor gradient exactly in the mathematical contract tests.

## Memory bounds and feature drift

Each owner encodes up to 512 normalized, unaugmented local training images.
The packet contains their mean embedding plus at most 16 actual embeddings,
in deterministic order. The approved CNN has a 192-dimensional penultimate
representation. This is not FeTrIL/FedProK feature translation.

At most nine old classes give 144 replay rows and nine means: **117,504 bytes
(114.75 KiB)** of float32-equivalent coordinate payload per activation. Python
tuples, mean precision, serialization/JSON and fixed padded TensorFlow buffers add overhead. This is
not a bound on the entire application or TensorFlow allocator. The existing
6,144-MiB GPU cap/headroom check remains active.

Refreshing with the selected encoder solves the **initial-coordinate** mismatch
only. Features remain fixed during five epochs while the encoder changes.
The worker records selected/final encoder hashes; epoch analysis compares
frozen-feature classification with predictions on current raw-image embeddings.
A difference is a concern to inspect, not proof of a single cause.

## Validation scope

| Setting | Bounded learning comparison |
|---|---|
| Controls | Current CART versus current CART + feature memory |
| Data | Complete 50,000-image training cache, ten class owners, all 5,000 samples/node remain in SGD |
| Horizon | One complete ring traversal per control; 20 activations total |
| Epochs / batches | Five epochs, batch 512; 9×512 plus 392; 50 updates/activation |
| Initial model / seed | Same canonical CPU-created 117,706-parameter CNN / seed 2025 |
| Optimizer | Reset SGD, LR 0.05 at R0, momentum/weight decay/clipping off |
| Selection | Original receiver-local min CE; fixed selection batch; S=5; no averaging |
| Analysis | 64 training examples/class, balanced 640-image centralized analysis probe |
| Probe separation | Probe excluded from SS and feature packets, but **not** ordinary owner SGD |
| Test set | Not loaded; no test accuracy or generalization claim |
| Attacks / noise / EBM / W&B | Off in this learning comparison |
| Research validity | False; bounded mechanism diagnostic, not a production research run |

The probe is **not held-out validation**. It cannot be used for tuning β, μ,
sigma, early stopping or method selection. Its measurements are analysis only.
A single traversal also does not answer long-run convergence or resilience.

## Source credit and distinctions

* **PASS**: related feature-space rehearsal for old-class decisions.
* **FedProto**: class-mean embedding construction and prototype exchange.
* **FedProK / FeTrIL**: related retention through feature knowledge; this path
  does **not** implement their translation or frozen-extractor assumption.
* **PearFL**: decentralized prototype exchange precedent; its model averaging
  is **not** adopted.
* **BASIL**: ring, predecessor memory, receiver-local selection and complete handoff.
* **Ang et al.**: gradient-norm noise-aware objective concept, adapted to CNN CE.
* **FedProx**: established proximal regularization; our reference is a selected snapshot.

See [primary-source review and downloads](CART_LITERATURE_AND_ATTRIBUTION.md)
for exact citations, the source Eq. 14 squared-loss qualification, recent work
and privacy limitations. This is a source-grounded experimental integration,
not a novelty or exact-paper-reproduction claim.

## Files and safe commands

| File | Responsibility |
|---|---|
| `basil_core/cart_feature_memory.py` | Immutable packets, activation memory, owner refresh, exact composite gradient, opt-in worker |
| `scripts/run_cart_feature_diagnostic.py` | Plan-only default; synthetic CPU/GPU verification; explicitly gated bounded clean comparison |
| `scripts/report_cart_feature_diagnostic.py` | Saved-result contracts, epoch CSV, tables and figures; cannot train |
| `tests/test_cart_feature_memory.py` | Mathematical, provenance, bounds, no-lambda, frozen memory/reference and model compatibility contracts |

```bash
# Zero training, zero downloads, no ML imports:
environment/basil-noise-env/bin/python scripts/run_cart_feature_diagnostic.py

# Synthetic device contract, not a CIFAR research trajectory:
environment/basil-noise-env/bin/python scripts/run_cart_feature_diagnostic.py --device GPU --verify-device

# One bounded clean comparison, ONLY after matching device proof:
environment/basil-noise-env/bin/python scripts/run_cart_feature_diagnostic.py --device GPU --execute

# Read saved completed evidence and generate report; zero training:
environment/basil-noise-env/bin/python scripts/report_cart_feature_diagnostic.py
```

The learning runner refuses existing evidence paths: no overwrite, automatic
rerun or guessed production resume. On failure, saved batch audit identifies
round/node/epoch/batch and retains the failing state. On interruption the
diagnostic stays incomplete; it is not silently extended or promoted.

## Limitations before any larger study

No Byzantine feature-owner validation or privacy guarantee is implemented.
The extra owner-response channel is clean in this experiment; noise on CNN
parameter links does not automatically describe noise on feature packets.
Extending attacks/noise requires explicit threat semantics and tests first.
Feature drift and head-only supervision may still prevent useful retention.
Do not infer success from finite gradients or accuracy on frozen features.

The completed measured report, when available, is
[`CART_FEATURE_REVIEW.md`](../newResults/nonIID/cart_feature_memory/clean_one_traversal_beta1_v1/CART_FEATURE_REVIEW.md).
No 30-/50-/500-round or A/B/C/D queue is prepared automatically.

## Verification actually executed

| Check | Result |
|---|---|
| Full CPU/headless test suite | **490 passed, 2 skipped, 644 subtests passed** in 68.76 seconds on the final rerun; two existing dependency deprecation warnings |
| Display exclusions | Two GUI widget checks need a display; not claimed as manually verified |
| Compile | `gui`, `basil_core`, `reporting`, `scripts`, `tests`, `run_gui.py` compiled successfully |
| Lazy import inventory | 210 Python sources; no direct heavy module-scope imports moved or retained |
| Download integrity | All 12 downloaded paper PDFs matched recorded SHA-256 and byte counts |
| Pre-existing evidence integrity | 23,812 saved result, plot and paper files before/after; 23,812 SHA-256 matches, zero changed. Only the new diagnostic result/plot directories were excluded from the comparison. |
| Synthetic GPU CE + anchor + replay | CPU/GPU gradient relative difference 2.6543e-7; maximum SGD update difference 1.4902e-8 |
| Synthetic GPU EBM + anchor + replay | CPU/GPU gradient relative difference 1.8658e-7; maximum SGD update difference 7.4506e-9 |
| Device contract scope | Approved CNN, 512/392 batches, finite gradients and lambda independence; not bitwise CPU/GPU equivalence or a multiprocessing check |

```bash
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests run_gui.py

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-cart-feature-final PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests

environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py
```

Source/hash-bound device proof is recorded in
`gui/worker_state/CART_FEATURE_GPU_VERIFICATION.json`. The bounded comparison's
saved manifest and measured review are the evidence for its actual outcome;
mathematical test success alone is not a retention claim.
