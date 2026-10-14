# CART: same-image diagnosis and selected-encoder preservation

**Experimental opt-in, not a replacement for the approved baseline.** This
addresses the missing *encoder* preservation path in detached feature replay.
It does not claim that all scientific problems are solved.

## What the saved evidence establishes

The original feature-memory comparison finished five full epochs at every node,
but both controls retained only their own class on the balanced training probe.
Frozen replay features and image predictions previously used different sample
sets. The new audit removes that confound:

```text
The SAME training images (fixed identities, pixels and labels)
        |
        +--> selected encoder --> selected head   [incoming reference]
        +--> selected encoder --> current head    [classifier change]
        +--> current encoder  --> selected head   [encoder change]
        +--> current encoder  --> current head    [actual current model]
```

The saved audit reconstructs the exact selected snapshot from the canonical
initialization or an already-saved trained predecessor. All input/output hashes
must match. It performs **zero optimizer steps** and never changes live model
weights. Its 40 observations cover two controls × ten nodes × before/after.
Intermediate epochs 1–4 were not saved in that older run and are not fabricated.

On identical images, restoring selected-encoder features raises visited-class
accuracy in **8 of 18** final observations with a nonempty visited-class set.
That supports investigating encoder preservation, but head changes and weak
initial representations remain relevant. This is not a proof of one sole cause.

Read the [same-image measured table and figure](../newResults/nonIID/cart_representation/same_image_saved_v1/SAME_IMAGE_REPRESENTATION_AUDIT.md).
All scores are from a **640-image TRAINING probe**, not held-out validation or
CIFAR test accuracy. The probe is excluded from SS, feature packet construction
and gamma calibration, but remains part of ordinary owner SGD.

## The optional loss

Write parameters as `theta=(phi,psi)`, where `phi` is the encoder and `psi` the
final linear classifier. The selected snapshot is `theta_sel`. The teacher is
an independent, frozen copy of its encoder. The student encoder still trains.

$$
\begin{aligned}
L_i(\theta)={}&\underbrace{CE_i(\theta;B_i)}_{\text{local learning}}\\
 &+\underbrace{\sigma_e^2\|\nabla_\theta CE_i\|_2^2}_{\text{source-grounded EBM; optional}}\\
 &+\underbrace{\frac{\mu_e}{2}\|\theta-\theta_{sel}\|_2^2}_{\text{existing CART-controlled snapshot anchor}}\\
 &+\underbrace{\beta L_{replay}(\psi;M_i)}_{\text{bounded old-class feature replay}}\\
 &+\underbrace{\gamma\frac1{|B_i|}\sum_{x\in B_i}
 \|h_\phi(x)-\operatorname{stopgrad}(h_{\phi_{sel}}(x))\|_2^2}_{\text{new receiver-local representation preservation}}.
\end{aligned}
$$

| Symbol | Meaning | Boundary |
|---|---|---|
| `B_i` | Receiver's own training minibatch | No other owner's raw images |
| `h_phi` | Current 192-dimensional encoder | Trainable, not frozen |
| `h_phi_sel` | Exact BASIL-selected encoder | Detached, immutable throughout five epochs |
| `sigma_e²` | Absolute Gaussian channel coordinate variance | No extra lambda; not model-L2 normalized |
| `mu_e` | Existing receiver-supported CART controller | Unchanged cap 0.01; fixed per epoch |
| `beta` | Old-class head replay strength | Previously approved 1.0 |
| `gamma` | Encoder feature-preservation strength | New, explicit; not beta, mu or EBM lambda |

Both encoder paths see **exactly the same augmented local batch**. Preservation
uses a mean over samples of the **sum over feature coordinates**, not a mean
over coordinates. At handoff the preservation penalty is exactly zero.

For feature difference `d(x)=h_phi(x)-h_phi_sel(x)`, the new encoder gradient is

$$
\nabla_\phi L_{preserve}
=\frac{2\gamma}{|B_i|}\sum_x J_{h_\phi}(x)^T d(x).
$$

The head contribution of this term is zero by construction. Conversely, replay
has a head gradient but no encoder gradient. Neither term enters the inner CE
gradient norm of EBM. Its second-order contribution remains `2 sigma_e² H_CE g`.
There is **no legacy lambda** anywhere in this new objective.

## Approved, training-only gamma calibration

The runner has **no implicit gamma default**. `--execute` without a fixed gamma
or an explicit calibration rule fails before importing TensorFlow.

The operator approved this training-only rule before the new learning runs:

1. Use one canonical Node-0 training minibatch, disjoint from SS/analysis.
2. Freeze the canonical selected encoder.
3. Temporarily form one ordinary CE SGD update with LR 0.05.
4. At that temporary state measure CE gradient norm `G` and unit-coefficient
   preservation gradient norm `P`.
5. Set `gamma = rho G/P` once; restore all weights exactly.
6. Keep gamma fixed for every subsequent activation.

Fixed `rho=0.10` is a declared diagnostic scale, **not a paper-derived value**
or an accuracy-optimized setting. Zero/non-finite norms abort calibration;
there is no guessed fallback or coefficient clamp. A 10% ratio at the virtual
calibration step does not guarantee a 10% contribution later; actual ratios
are logged and never control updates. The measured calibration gives
**gamma = 0.7140174035127002**, from CE gradient norm 1.1252317428588867
and unit preservation gradient norm 0.15759164094924927. These are measured
at one temporary training-only CE update, not chosen from an accuracy sweep.
The temporary weights are restored exactly before either control starts.

## Authorized one-traversal check and 30-round comparison

The requested comparison runs **30 complete rounds per control**, not 30
activations. Both controls start fresh and execute one-class-per-node training
with 5,000 images/node, five full epochs, batch 512 and 50 SGD updates per
activation. The controls are:

| Control | Local learning | Existing CART anchor | Feature replay | Selected-encoder preservation |
|---|---|---|---|---|
| `feature_replay` | CE | Receiver-local controller, cap 0.01 | beta = 1.0 | Off |
| `feature_replay_preserve` | Same CE | Same controller/cap | Same beta | Fixed calibrated gamma |

Both use the approved CNN, strict selected-model handoff, rolling BASIL S=5,
receiver-local minimum-loss SS, reset SGD and LR `0.05/(1+0.05*round)`.
There is no attack, channel noise, EBM, clipping, momentum, weight decay or
W&B. This is an experimental clean retention diagnostic, `researchValid=false`,
not the A/B/C/D threat study or a 500-round queue.

The one-traversal execution check must finish with finite values and correct
protocol counts before starting 30 rounds. **Low accuracy does not fail this
execution gate.** Neither gamma nor the horizon is changed after seeing scores.

For 30 rounds, the complete 10,000-image CIFAR test set is evaluated after
each activation, evaluation only. Round CSVs distinguish end-of-ring Node-9
accuracy/loss from the mean and minimum of ten trained node states. They are
cross-checked against activation JSON before plotting. A separate balanced
640-image TRAINING probe supplies before-training/epoch-1-through-5 class
accuracy and same-image encoder/head interventions; it is not a test score
or held-out validation result.

Each round boundary saves node parameters, five-predecessor memories, sender
rounds, registries, configuration and source hashes. These are evidence
checkpoints: automatic exact resume is **not** implemented for this isolated
runner. Runs refuse output overwrite/retry; a non-finite operation aborts and
saves its diagnostic evidence without fabricating later rounds.

| Horizon | Result directory | Purpose |
|---|---|---|
| 1 round/control | `newResults/nonIID/cart_representation/clean_one_traversal_fixed_gamma_v1/` | Finite execution/protocol gate; no test data |
| 30 rounds/control | `newResults/nonIID/cart_representation/clean_30_rounds_fixed_gamma_v1/` | Authorized full-data comparison; test set evaluation only |

Plots use matching paths under `newPlots/nonIID/cart_representation/`.
The completed 30-round comparison contains **600 activations and 30,000
optimizer updates**, confirmed by `run.json`.

### Execution check and 30-round comparison completed

The one-traversal GPU execution check completed in **567.40 seconds**:
20 activations, 1,000 optimizer updates, 140 saved-evidence checks and
120 epoch-metric rows. No numerical failure, OOM or memory-pressure event
was observed; tracked TensorFlow peak GPU memory was **284.20 MiB**, within
the 6,144 MiB allocator cap.

| One-traversal metric (TRAINING probe only) | Feature replay | + Encoder preservation |
|---|---:|---:|
| Balanced probe accuracy | 10.000% | 10.000% |
| Own-class accuracy | 100.000% | 100.000% |
| Other-class accuracy | 0.000% | 0.000% |
| Mean old-class squared feature drift | 124.867839 | 0.193411 |

Preservation reduced measured drift but **did not improve class retention in
this traversal**. This is not a final 30-round result or a global test score.
The execution gate passed, so the explicitly requested 30-round comparison
ran on the RTX 4070 Ti using the same fixed gamma and canonical setup.
Its status is authoritative in
`newResults/nonIID/cart_representation/clean_30_rounds_fixed_gamma_v1/run.json`.

The first detached shell launch exited before creating any result directory
or training update. The actual job uses a persistent managed execution session
with hangup protection and a saved log; no experimental condition was retried
after observing a result. Both controls run sequentially in one process. A
saved-only report is invoked automatically **only if training exits successfully**.
Both controls completed: **18,039.54 seconds (5.01 hours)** total. Saved-result
verification passed **4,260 checks**, producing 3,600 epoch CSV rows. Both
finished at **10% Node-9/mean-node/worst-node full-test accuracy**, 100% own-class
and 0% other-class training-probe accuracy. Mean old-class squared feature
drift was 117.052974 versus 0.346721. **Lower drift did not improve retention.**
No numerical failure or OOM was recorded. The [completed review](../newResults/nonIID/cart_representation/clean_30_rounds_fixed_gamma_v1/REPRESENTATION_PRESERVATION_REVIEW.md)
preserves all measured evidence. See the [failure diagnosis and separate candidate](CART_OWNER_GRADIENT_CANDIDATE.md)
for the next information-path study, not a silent retuning of this method.

```bash
# Read-only progress; does not allocate GPU memory or change the job:
tail -n 20 gui/worker_state/cart_representation_30r.log
```

Based on the one-traversal GPU timing, allow roughly **4–6 hours** for the
two-control comparison, including full-test evaluation and evidence writing.
This is an estimate, not a completion promise. Do not launch the same command
again while it is active: the GPU lease/output guard will reject duplicates.

## Verification and execution safeguards

| Issue | Addressed in code | What is not claimed |
|---|---|---|
| Different-image audit confound | Same-image identity/pixel/label hashes across all interventions | Causal proof from all contexts |
| Replay reaches head only | Separate trainable-encoder preservation gradient | Guaranteed old-class retention from one-class inputs |
| Teacher drift | Independent reusable clone; checks exact selected-encoder hash | CPU/GPU bit identity |
| Coefficient ambiguity | Required fixed gamma or explicit training-only calibration | Test-score tuning |
| Objective confusion | Separate CE, EBM, parameter anchor, replay and preservation metrics | EBM solves forgetting |
| Memory growth | One reusable teacher, bounded replay, read-only cache, GPU cap/guard | Absolute immunity to OOM |
| Historical evidence | New output paths; original artifacts read only | Rewriting failed diagnostic findings |
| Unverified threat extension | Initial study is clean/no-noise only | Byzantine-safe owner feature exchange |

Tests cover exact composite gradients, finite differences, gamma-zero
equivalence, teacher/reference freezing, deterministic calibration/restoration,
no-lambda loss/gradient/update identity, same-image audit isolation, lazy imports
and zero-training default. The bounded learning runner inherits BASIL S=5,
receiver-local SS, complete selected-weight replacement, five epochs, 5,000
samples/node, batch 512 with final 392, reset SGD, no clipping/WD/momentum.

The authorized diagnostic compares feature replay
against feature replay plus preservation. Both start fresh from the same
canonical initialization and partition; attacks, noise, EBM and W&B remain off.
It records before training and after each of five epochs, including identical
image interventions, class accuracies, confusion matrices and component norms.
It refuses output overwrite/resume and never adds a research queue.

## Files and commands

| File | Responsibility |
|---|---|
| `basil_core/cart_representation.py` | Frozen selected teacher, exact preservation gradient, opt-in worker, same-image audit, one-time calibration |
| `scripts/run_cart_representation_diagnostic.py` | Zero-training plan, saved-checkpoint audit, synthetic device proof, explicitly gated learning |
| `scripts/report_cart_representation_diagnostic.py` | Saved-only contract checks, CSVs, tables and plots; cannot train |
| `tests/test_cart_representation.py` | Mathematical, isolation, coefficient, frozen-state and execution-gate tests |

```bash
# Plan only: no ML imports, downloads or training.
environment/basil-noise-env/bin/python scripts/run_cart_representation_diagnostic.py

# Completed saved-checkpoint audit: zero training, refuses overwriting evidence.
environment/basil-noise-env/bin/python scripts/run_cart_representation_diagnostic.py --device GPU --audit-saved

# Synthetic device verification, not a CIFAR learning trajectory.
environment/basil-noise-env/bin/python scripts/run_cart_representation_diagnostic.py --device GPU --verify-device

# Saved audit reporting: no training.
MPLCONFIGDIR=/tmp/papermerge-cart-representation-report environment/basil-noise-env/bin/python scripts/report_cart_representation_diagnostic.py

# Authorized one-traversal finite execution check:
environment/basil-noise-env/bin/python scripts/run_cart_representation_diagnostic.py --device GPU --execute --calibrate-ratio 0.10 --rounds 1

# ONLY after that diagnostic completes:
environment/basil-noise-env/bin/python scripts/report_cart_representation_diagnostic.py --learning

# Authorized 30-round comparison, after the execution check passes:
environment/basil-noise-env/bin/python scripts/run_cart_representation_diagnostic.py --device GPU --execute --calibrate-ratio 0.10 --rounds 30

# Report only after BOTH 30-round controls complete; performs zero training:
environment/basil-noise-env/bin/python scripts/report_cart_representation_diagnostic.py --learning --rounds 30
```

## Source attribution and remaining scientific questions

Knowledge distillation on new/local data has established precedent in
[Learning without Forgetting](https://arxiv.org/abs/1606.09282). Feature replay
and representation stability are related to [FeTrIL](https://arxiv.org/abs/2211.13131)
and [FedProK](https://arxiv.org/abs/2405.02685), which use assumptions that are
not reproduced here. The exact selected-encoder feature-MSE integration above
is an experimental adaptation; no novelty claim is made.

Preserving features on only the receiver's class may not preserve old classes.
An initially untrained/specialized selected model may contain little useful
knowledge. BASIL may legitimately select an initialized snapshot in the first
traversal. These are research limitations, not resolved by passing tests.
Do not silently freeze the student, share raw images, increase coefficients,
pretrain on other classes or extend the horizon to manufacture a better score.

## Verification actually completed

| Check | Actual result |
|---|---|
| Baseline suite before changes | 490 passed, 2 skipped, 644 subtests passed in 70.47 seconds |
| Earlier representation CPU/headless suite | **509 passed, 2 skipped, 644 subtests passed in 69.23 seconds**; two dependency deprecation warnings |
| 30-round wiring full CPU/headless suite | **512 passed, 2 skipped, 644 subtests passed in 71.58 seconds**; two dependency deprecation warnings; before the additional CSV regression test |
| Final 30-round CPU/headless suite, including CSV/float32 gamma regressions | **514 passed, 2 skipped, 644 subtests passed in 72.12 seconds**; two dependency deprecation warnings |
| Final focused representation/feature/CART/lazy suite | **114 passed** in 7.72 seconds |
| Display-dependent exclusions | Two GUI widget tests skipped; no manual display validation claimed |
| Compilation | All requested source/test trees and GUI entry point compiled successfully |
| Lazy-import audit | 214 Python sources; zero direct heavy module-scope imports moved/retained |
| Saved same-image audit | 40 observations, unchanged image identities, zero optimizer steps |
| GPU synthetic CE composite objective | CPU/GPU gradient relative difference 4.4381e-6; maximum update difference 1.3411e-7 |
| GPU synthetic full EBM composite objective | CPU/GPU gradient relative difference 4.4240e-6; maximum update difference 1.3411e-7 |
| Device test scope | Approved CNN; batch sizes 512/392; fixed test-only gamma 0.01; teacher frozen; lambda 1 vs 999999 identical |
| Device tolerance | Gradient relative difference ≤0.01; maximum update difference ≤1e-4; not bit identity or multiprocessing validation |
| Evidence integrity | **23,832 pre-existing result/plot/paper files matched SHA-256; zero changed**; only new representation-audit trees excluded |
| Current 30-round preparation integrity | **23,836 pre-existing result/plot/paper files matched SHA-256; zero changed**; only the two new learning-output trees excluded |
| New CIFAR learning comparison | **Both 30-round GPU controls completed**; no class-retention improvement, no numerical failure/OOM; 4,260 saved checks passed |
| Queue / W&B / long campaigns | Not started or modified |

Exact compile/test commands:

```bash
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests run_gui.py

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-cart-representation-focused PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_cart_representation.py tests/test_cart_feature_memory.py tests/test_strict_cart.py tests/test_lazy_imports.py

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-cart-representation-final PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests

# Current 30-round focused/final test commands:
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-cart-representation-focused-30r PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_cart_representation.py tests/test_cart_feature_memory.py tests/test_strict_cart.py tests/test_lazy_imports.py

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-cart-representation-30r-final-tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests

environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py
```

The GPU proof is recorded in
`gui/worker_state/CART_REPRESENTATION_GPU_VERIFICATION.json` and matches the
current numerical source hashes. Saved-audit results and CSVs are under
`newResults/nonIID/cart_representation/same_image_saved_v1/`; its figure is under
the matching `newPlots/nonIID/` tree. The evidence baseline manifest is
`gui/worker_state/cart_representation_before.json`.
The current learning-task baseline is
`gui/worker_state/cart_representation_30r_before.json`; it includes the earlier
same-image audit unchanged as well as the IID/non-IID evidence and papers.

The first focused test pass exposed a synthetic zero-gradient fixture that
actually had near-zero float32 cancellation, not an exactly zero gradient.
The fixture was corrected to have a mathematically zero encoder gradient;
the calibration formula was not altered to hide that test failure.

The learning preparation also exposed a reporting-only precision mismatch:
calibration JSON stores a Python double, but the applied TensorFlow gamma is
float32 (`0.714017391204834`). The reporter now checks exact equality to the
configured coefficient's float32 representation, not its double representation.
A one-ULP change still fails the check. No optimizer/objective update changed.

**Status: implementation and mathematical/device checks pass, but the completed
learning comparison did not improve retention. Not ready for a larger
representation-preservation campaign with this unchanged loss.**
