# Strict-handoff CART with source-grounded EBM

This is an **explicitly revised CART adaptation**, not an exact reproduction of
the legacy CART path. Legacy `basil_core/cart.py` remains available with its
mathematics unchanged; its dependency imports were subsequently made lazy.
The revision is `strict_cart_receiver_supported_confidence_gap_v1`.

Follow-up: the [head/backbone mechanism audit](FORGETTING_MECHANISM_AUDIT.md)
completed one warmed activation per preserved control. It confirms the local
gap blind spot and output-head involvement without changing this controller
or claiming that a new retention solution has succeeded.

## Framework, not just a loss coefficient

```text
Latest snapshot from each of five predecessor senders
                   |
      BASIL minimum receiver-local CE selection
                   |
       Selected model + carried class registry
                   |
     Complete weight/bias replacement; NO averaging
                   |
 Freeze selected weights as reference for five full epochs
                   |
 Locally verify supported registry claims / mark absent classes unavailable
                   |
   Local confidence and supported gaps -> epoch-wise anchor strength
                   |
  Five complete SGD epochs: CE + optional true EBM + frozen-reference anchor
                   |
 Update local class evidence and carried historical registry
                   |
 Optional configured outbound attack -> independent per-link channel noise
                   |
  Refresh next five memories; only the immediate successor trains next
```

Registry fields are accuracy, source, measurement round and sample support per
class. These are **historical performance records**, not raw examples or a
certificate that the transmitted model still knows those classes. Carried
remote claims and receiver-verified evidence are separate objects. A receiver
owning only class `i` cannot verify the other nine classes. Their missing local
scores are unavailable, not observed 0% accuracies. No cross-class test scores
enter verification, selection, anchoring, SGD or tuning.

## Loss: three distinct mechanisms

```text
L(theta) = CE(theta)
           + sigma_e^2 * ||gradient CE(theta)||^2
           + (mu_e / 2) * ||theta - theta_selected||^2
              |                    |
        CART controller      frozen BASIL-selected snapshot

CE term:      local supervised learning
EBM term:     communication-noise sensitivity adaptation
Anchor term:  investigate destructive parameter drift / forgetting
```

| Symbol | Meaning |
|---|---|
| `theta` | All current trainable CNN weights and biases |
| `theta_selected` | Exactly the received snapshot selected by BASIL; immutable during all five epochs |
| `CE` | Stable ten-class sparse cross-entropy from logits on local training minibatches |
| `sigma_e` | Fixed coordinate standard deviation for the absolute Gaussian channel |
| `mu_e` | Nonnegative scalar held constant within local epoch `e` |

The full applied direction is:

```text
g = gradient CE
H = Hessian of CE (never explicitly materialized)
applied gradient = g + 2*sigma_e^2*H*g + mu_e*(theta-theta_selected)
```

Nested TensorFlow tapes compute full EBM through
`gradient_norm_objective_gradients`; the anchor gradient is added exactly.
The anchor is NOT inside the inner CE gradient norm. Legacy lambda has zero
effect. There is no scalar-gradient EBM shortcut or gradient clipping.
`sigma_e=0.010` means standard deviation 0.010, variance 0.0001—not 1% noise.
Actual channel noise is a separate operation after outbound attack; optimizing
EBM does not replace the channel.

The gradient-norm CE term is the existing multiclass adaptation of the
noise-aware source concept. The registry/controller and proximal anchor are
our revised CART adaptation, not a claim that the communication paper proposed
this combined framework or that this solves forgetting.

## Receiver-supported controller: exact rule

The legacy gap becomes zero if a newly loaded model is compared with itself.
This revision therefore explicitly includes a positive training-side confidence
term. It does **not** silently change legacy CART.

```text
L_SS = selected snapshot CE on the receiver's fixed local selection batch
q = 1 / (1 + L_SS)

target[c] = min(historical supported registry accuracy[c],
                selected snapshot accuracy verified locally[c])

gap_e = mean(max(0, target[c] - current local probe accuracy[c]))
        over classes with actual receiver-local support only

mu_e = mu_max * min(1, q + gap_e)
mu_max = 0.01 in this diagnostic, fixed before seeing test results
```

The receiver records its measured selected-reference accuracy in its verified
local registry before setting the target. Consequently the target cannot exceed
the selected reference's locally measured capability. At handoff `gap=0`, but
`q>0`; later local degradation can increase the coefficient. A one-class node
usually improves its own class, so this gap may remain zero. This limitation is
important: a registry cannot generate supervision for absent classes.

The same unaugmented 512-example local TRAINING batch evaluates every BASIL
candidate and the epoch controller. It is fixed per receiver with keyed RNG.
An epoch's coefficient is set before that epoch; after epoch five the recorded
next coefficient is telemetry only. Global evaluations are calculated only
after the local controller update and cannot feed it. Confidence is not a
guarantee of global usefulness.

## Protocol retained

| Setting | Value |
|---|---|
| Data | Full CIFAR-10: 50000 training, 10000 evaluation-only test images |
| Partition | Node `i` owns all 5000 training images of class `i` |
| Model | Existing BASIL CNN, 117706 parameters |
| Ring / memory | Fixed 0→…→9→0; latest five distinct predecessors, S=5 |
| Selection | Minimum receiver-local CE, same batch; ties nearest counterclockwise |
| Initialization / handoff | Identical canonical initialization; full selected weight/bias replacement |
| Training | Five complete local epochs; 512 batch; 9×512 +392 per epoch; 50 SGD updates |
| Optimizer | Fresh SGD per activation; LR `0.05/(1+0.05*round)` |
| Momentum / decay / clipping | Zero / zero / disabled |
| Augmentation | Existing keyed flip/pad/crop, training only |

Before Node 0 of round 11: `5:R10, 6:R10, 7:R10, 8:R10, 9:R10`.
Before Node 1 after Node 0 sends: `6:R10, 7:R10, 8:R10, 9:R10, 0:R11`.
Registry metadata accompanies each selected snapshot; it is not a sixth model
or an average of the five models.

## Bounded diagnostic and controls

Exactly three fresh controls, each **three full-data rounds**, are run in order:

| Control | Objective | Anchor coefficient |
|---|---|---|
| CE | CE only | 0 |
| Fixed anchor | CE + frozen-reference proximal penalty | 0.01 |
| Strict CART | CE + frozen-reference proximal penalty with registry framework | Receiver-supported controller above, cap 0.01 |

All are clean: no actual attackers, no channel noise, no EBM, no GPM, no W&B.
Assumed Byzantine bound 4 and BASIL S=5 remain unchanged. These are diagnostic
outputs (`researchValid=false`), not a production campaign. Test evaluation
before training and after each of five epochs measures all ten class accuracies
and full confusion matrices. No accuracy target is a software correctness test;
no coefficient is chosen by these test outcomes.

This isolates forgetting. It does not test whether EBM mitigates real channel
noise. That requires a later authorized paired noisy CART-without-EBM versus
noisy CART-with-EBM comparison. No 30/50/500-round continuation is automatic.

## Files and reproducibility

| File | Responsibility |
|---|---|
| `basil_core/strict_cart.py` | Registry verification, local controller, strict frozen-reference CE/full-EBM worker |
| `basil_core/class_registry.py` | Existing registry schema/support/provenance, mathematically unchanged; lazy dependency loading |
| `basil_core/noniid_study.py` | Existing exact CE/EBM + anchor gradients, mathematically unchanged; lazy dependency loading |
| `scripts/run_strict_cart_diagnostic.py` | Serial GPU diagnostic, full-data audits, per-step finite checks, model/registry evidence snapshots |
| `scripts/report_strict_cart_diagnostic.py` | Saved-data verification, measured tables, trajectories and confusion figures |
| `tests/test_strict_cart.py` | Hessian/anchor, no-lambda, local-support, frozen-reference, epoch/memory/config contracts |

```bash
environment/basil-noise-env/bin/python scripts/run_strict_cart_diagnostic.py --device GPU --verify-device
environment/basil-noise-env/bin/python scripts/run_strict_cart_diagnostic.py --device GPU --execute
# Saved results only; zero training:
environment/basil-noise-env/bin/python scripts/run_strict_cart_diagnostic.py --report
```

Execution refuses existing output directories—no silent overwrite or guessed
resume. Memory files contain the fifty current model snapshots and registries;
they are **evidence checkpoints, not complete resumable research state**. The
GUI/production queue remains unchanged until a separately verified integration.
GPU uses the existing 6 GiB cap, 1 GiB minimum free headroom, float32/TF32-off
policy, GPU lease and memory guard. GPU access/verification must succeed; there
is no silent CPU fallback. CPU/GPU synthetic tolerances are 1% relative gradient
L2 and 1e-4 maximum parameter-update difference—not bitwise equivalence.

Outputs: `newResults/nonIID/strict_cart/clean_full_data_three_round_v1/` and
matching `newPlots/nonIID/strict_cart/clean_full_data_three_round_v1/`.
The completed review and CSVs report measured outcomes, including failure to
retain classes if that is what happens. Existing IID/non-IID results and papers
are SHA-256 checked, not regenerated.

## Completed diagnostic: no improvement in retention

All controls completed three full-data GPU rounds (90 activations, 4500 SGD
updates total) in **2474.877 seconds / 41.25 minutes**. Nineteen saved-evidence
contract groups passed, including paired actual input/augmentation hashes.

| Control | Final mean / worst test accuracy | Own-class accuracy | Other-class accuracy | Final mean drift² |
|---|---:|---:|---:|---:|
| CE | 10% / 10% | 100% | 0% | 1.95522 |
| Fixed anchor μ=0.01 | 10% / 10% | 100% | 0% | 1.79511 |
| Strict CART | 10% / 10% | 100% | 0% | 1.94117 |

All ten final node models in every control predicted their own class for all
10000 test images. This shows constant-class prediction collapse and loss of
previously predicted classes; it is not a fabricated software failure or a
reason to silently reduce the required five epochs.

CART applied μ from **0.000783366 to 0.00303435**; the supported gap was zero
for all 150 applied epoch controls. The mean anchor/CE gradient norm ratio
was **0.0156438**. A one-class receiver improves its own supported class and
cannot verify losses on the other nine classes. The registry is historical
evidence, not a mechanism that supplies absent-class supervision. Parameter
drift was slightly reduced without preserving predictions. These data do not
justify a longer production continuation or claim that CART solves this split.

Peak tracked TensorFlow GPU allocation was **272.424 MiB**, below the existing
6144 MiB cap; no pressure/OOM event was recorded. Eleven figures include class
trajectories, final matrices and before/after-five-epoch confusion panels.

[Measured tables, framework explanation and figures](../newResults/nonIID/strict_cart/clean_full_data_three_round_v1/STRICT_CART_REVIEW.md)

After completion, [lazy-import refactoring](LAZY_IMPORTS.md) was applied across
the repository. The executed sources remain archived with their original
hashes. Exact deterministic synthetic before/after equality and repeated
CPU/GPU gradient checks passed; the original training manifest was not rewritten.
All **23708** existing result/plot/paper files matched the before-work SHA-256
inventory; **zero changed**. No larger research or W&B run was launched.

## Executed verification commands

```bash
environment/basil-noise-env/bin/python -m compileall -q basil_core gui reporting scripts tests

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
MPLCONFIGDIR=/tmp/papermerge-cart-test-mpl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
environment/basil-noise-env/bin/python -m pytest -q \
  tests/test_strict_cart.py tests/test_gradient_projection_warm_gate.py \
  tests/test_gradient_projection_retention.py tests/test_class_aware_retention.py \
  tests/test_retention_diagnostic.py tests/test_noniid_study.py \
  tests/test_iid_study.py tests/test_research_protocol.py \
  tests/test_protocol_recovery.py tests/test_iid_500r.py \
  tests/test_memory_safety.py tests/test_research_audit.py \
  tests/test_iid_gpu.py tests/test_iid_campaign.py
```

Compile passed. Final focused suite: **240 passed, 1 display-dependent skipped,
2 dependency deprecation warnings** (24.56 seconds). An earlier run with the
inherited `DISPLAY=:0` reported 237 passed and one unavailable-display error;
the explicit headless rerun above is the final result. No GUI defect was fixed
or claimed. This is not a claim that the unrelated legacy full-suite collection
problem is resolved.

Synthetic CPU/GPU verification passed on RTX 4070 Ti:

| Objective | Relative gradient L2 difference | Maximum update difference |
|---|---:|---:|
| CE + anchor | 2.268286465e-7 | 7.450580597e-9 |
| Full EBM + anchor | 1.724218066e-7 | 7.450580597e-9 |

Both 512- and 392-image gradient checks were finite. Within-device losses,
gradients and SGD updates were identical for legacy lambda 1 versus 999999.
The tiny-model contract also compared the full gradient with an explicitly
calculated Hessian reference at 1e-7 absolute / 1e-6 relative tolerance.
All 16 strict-CART tests passed, including a compiled-gradient coefficient
update check, evaluation-score noninterference, remote-claim/local-verification
separation, and rejection of partial reports.

After the repository-wide import refactor, the complete suite passed
**436 tests and 644 subtests**, with two headless display skips and two
dependency warnings (71.07 s). Both skipped GUI checks were then exercised
under a virtual display and passed **2/2**, without a worker launch. All
42 lazy-import/startup contracts passed, and the guarded manual legacy MNIST
convergence experiment was not executed. See [exact verification commands and
before/after equivalence](LAZY_IMPORTS.md).
