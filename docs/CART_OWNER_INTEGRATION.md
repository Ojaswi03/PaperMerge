# CART owner-gradient integration: selection, EBM and threat boundaries

## Status — contracts verified; clean selection improved, full threat study blocked

Revision: `cart_owner_balanced_selection_local_ebm_v1`.
This is an **isolated, opt-in CART adaptation**, not a replacement for paper
BASIL or any completed result. No A/B/C/D queue or W&B run was started.
Synthetic derivative tests and single-update device checks passed, and the
separate [fixed-anchor clean selection comparison](CART_BALANCED_SELECTION_DIAGNOSTIC.md)
completed with saved-evidence checks passing. Its Node-9 accuracy improved from
14.25% to 23.18%, but cat/deer learning remains weak. That bounded comparison
does not enable EBM or threats and does not validate all-class convergence.

| Question | Current evidence / status |
|---|---|
| Is receiver-local BASIL selection implemented correctly? | Yes; unchanged and retained |
| Can balanced training-side owner selection help? | In the completed fixed-μ clean pair: Node 9 23.18% vs 14.25%, mean 22.55% vs 12.18%, worst 17.36% vs 10.38%; not solved |
| Does the new owner objective contain true local-CE EBM? | Yes; independent nested-tape, analytic Hessian and actual CNN/worker checks passed |
| Can legacy lambda affect it? | No; losses, gradients and one-step updates were identical for 1 versus 999999 |
| Is snapshot attack followed by independent absolute Gaussian noise? | Yes; exact keyed-output tests passed |
| Are hash-bound helper gradients Byzantine-safe? | **No**; a finite poisoning fixture passes hash checks and changes the update |
| Are full-node B/C/D research runs ready? | **No**; the helper defense and approved experimental scope remain unresolved |
| Is the new revision GPU-checked? | Representative synthetic worker check passed; not a long-run equivalence guarantee |

The latest **learning** evidence is the
[fixed-anchor balanced-selection pair](CART_BALANCED_SELECTION_DIAGNOSTIC.md).
The earlier, separately configured three-round pair remains preserved:
[SS versus direct handoff](../newResults/nonIID/cart_owner_gradients/clean_three_round_selection_pair_v1/OWNER_SELECTION_REVIEW.md).
Direct handoff reached 25.35% Node-9 test accuracy, but cat remained 0.8%.
Neither the learning results nor contract checks establish satisfactory all-class learning,
convergence, privacy or Byzantine robustness.

## 1. New selection rule — explicitly not BASIL Eq. (3)

Paper BASIL continues to minimize the receiver's own training-batch loss.
The new optional rule evaluates each candidate on the same per-owner training
batches, with equal class weights:

$$S(\theta_j)=\frac{1}{|C|}\sum_{c\in C}CE_c(\theta_j;B_c^{selection}),
\qquad \theta_{input}=\arg\min_{\theta_j\in M_i}S(\theta_j).$$

| Symbol | Meaning |
|---|---|
| `M_i` | Exactly five snapshots from distinct counterclockwise predecessors |
| `C` | Genuinely available class owners, including the active receiver |
| `B_c^selection` | Immutable, unaugmented TRAIN indices for owner/class `c`; identical for every candidate |
| `CE_c` | Locally computed mean ten-class cross-entropy, not sender-reported loss |
| `theta_input` | One selected complete model, never an average |

The selection API validates the complete 50,000 training-label provenance,
class ownership, index bounds/duplicates, exact requested owner coverage,
five distinct predecessors, finite losses and unchanged candidate hashes.
Test/validation inputs are rejected by the declared provenance interface.
Nearest-counterclockwise ties use the existing 1e-8 loss tolerance.

The API receives only TRAIN indices and an owner evaluator; it has no global
test-accuracy criterion. This is an interface/data-boundary contract, not
cryptographic verification of arbitrary caller-provided arrays. The bounded runner
derives probes from the same actual visited-owner set as training: only
already visited owners in Round 0, all other nine owners after warm-up. The
existing worker's owner-availability guard remains unchanged.

The selector also returns the chosen candidate's **receiver-local CE**, distinct
from its balanced score. Pass that local value to the existing CART controller;
do not silently replace its confidence input with a global/balanced metric.
The original receiver-local selector and direct-handoff diagnostic remain
available. The completed clean comparison fixes μ=0.01 in both arms;
it intentionally does not use the adaptive controller, to isolate selection.

## 2. Exact training objective and derivative

Define `F_c(theta)` as each available owner's minibatch mean CE at the same
current parameters. Define `g_i = gradient F_i` for the active receiver only.

$$L(\theta)=
\underbrace{\frac{1}{|C|}\sum_{c\in C}F_c(\theta)}_{\text{balanced class supervision}}
+\underbrace{\sigma_e^2\|g_i(\theta)\|_2^2}_{\text{local-CE source EBM; D only}}
+\underbrace{\frac{\mu_e}{2}\|\theta-\theta_{input}\|_2^2}_{\text{frozen snapshot anchor}}.$$

$$\nabla L=\frac{1}{|C|}\sum_{c\in C}\nabla F_c
+2\sigma_e^2H_i g_i
+\mu_e(\theta-\theta_{input}).$$

| Component | Implementation and qualification |
|---|---|
| Balanced CE | Fresh full encoder/classifier gradients; equal **class**, not sample, weighting |
| Local EBM | Nested TensorFlow tapes compute the derivative of the squared **receiver-local CE gradient** |
| Coefficient | Exactly absolute coordinate variance `sigma_e²`; no extra lambda |
| Snapshot anchor | Detached input reference, unchanged throughout the activation; existing receiver-supported epoch controller, cap 0.01 |
| Complete gradient | Sum of balanced CE gradient, actual local Hessian-gradient correction and anchor gradient |
| Optimizer | Existing SGD path; no optimizer at helper owners, clipping, momentum, weight decay or model averaging added |

For D, `sigma_e=0.010` gives coefficient `0.0001`. C can have the same actual
channel while its EBM coefficient and correction remain **zero**. There is no
relative-L2 calibration in this revision.

**Important distinction:** the inner EBM gradient is not the gradient of
`balanced CE + anchor`. Moving it to that expression would introduce different
Hessian/cross terms. The local EBM penalty is not divided by `|C|` in this
explicit objective. This is a new owner-gradient CART adaptation, not a claim
that Ang et al. proposed this entire objective or helper architecture.

The epoch controller can give different numerical anchor strengths when input
snapshots differ. A future selection-specific comparison should separately
control that effect; these unit tests prove the gradient for a supplied frozen
coefficient, not an empirical causal selection effect.

## 3. Threat boundary — the unresolved research decision

```text
five cached candidates
       |
same TRAIN probes at available owners -> balanced candidate CE
       |
select ONE complete input -> freeze anchor -> reset active SGD
       |
five FULL local epochs: fresh owner class mean + local EBM + anchor
       |
honest trained model
       |
Hidden outbound parameter transformation at R20, if configured
       |
independent absolute Gaussian perturbation for EACH directed snapshot link
       |
five clockwise memories; only immediate successor activates next

SEPARATE AUXILIARY CHANNEL:
current model -> owner-local CE gradient -> active trainer
This channel needs its own explicit trust, adversary and noise definition.
```

With one owner per class, an attacker may be the sole source of that class's
helper evidence. A correct state hash proves which parameters a response
claims to describe, **not whether the response is honest**. The regression
test replaces a valid helper gradient by `-100 * gradient`, retaining its hash,
context and finite values. Ordinary aggregation accepts it and changes the
update. This is deliberate evidence against claiming a defense from hashes.

| Scope identifier | Meaning | Execution guard |
|---|---|---|
| `full_node_threats` | Byzantine owners may also poison helper gradients/evaluation claims | Any configured attacker blocks the helper factory and balanced selector until a defense is specified |
| `honest_auxiliary_snapshot_only` | Helpers are a separately assumed honest, clean service; attacks/noise affect snapshots only | May construct components explicitly; still **not research-ready**, and cannot be described as full-node Byzantine safety |

The guard does not remove the known attacker IDs from aggregation or selection.
Such simulator-oracle filtering would be a false defense. No gradient clipping,
median/trimming rule, new helper Gaussian variance or privacy guarantee was
silently introduced. Snapshot parameter noise `sigma_e=0.010` does **not** define
a helper-gradient noise budget.

Actual project Hidden attack defaults and keyed streams are reused without
tuning. It is `delayed_hidden_parameter_attack_v1`, not `gaussianAttack` and
not an asserted exact reproduction of another paper's gradient attack.
Rounds 0–19 have no attack; Round 20 is active. Honest trained parameters stay
unchanged. Each link starts from the attacked outbound parameters and receives
its own keyed absolute noise. Keyed C/D draws match when upstream weights match.

## 4. Prepared zero-training plan, not an executable queue

| Condition | Snapshot attackers | Snapshot channel | Local EBM |
|---|---|---|---|
| A | None | Clean | Off |
| B | [0, 1, 5, 7], Hidden R20 | Clean | Off |
| C | Same as B | Absolute sigma 0.010 | Off |
| D | Same as B | Same as C | Local CE gradient norm, coefficient 0.0001 |

All plans retain ten one-class owners, 5,000 local samples, five full epochs,
batch 512 with partial batch 392, SGD reset per activation, round-decayed
initial LR 0.05, 117,706 parameters, strict replacement and S=5. W&B is disabled.
No horizon is silently selected: `nRounds=null`. All plans are
`researchValid=false` and `queueStarted=false`, with explicit blockers.
C/D configuration checks differ only in the condition label and EBM mode.

The plan is separate from GUI saved configurations and is not accepted as a
paper-BASIL configuration. Balanced selection requires its own approved runner;
the generic historical validator continues to reject changes to the paper rule.
No existing GUI queue, engine or completed configuration was rewritten.

This provisional plan still names the adaptive controller; the completed
fixed-μ selection pair is not silently substituted as validation of an
adaptive-anchor/noisy/Byzantine trajectory. Its weak class results and the
unresolved helper threat boundary preclude a production-ready recommendation.

## 5. Files and mathematical responsibilities

| File | What was added |
|---|---|
| `basil_core/cart_owner_protocol.py` | Immutable TRAIN probes, balanced candidate selection, helper threat gate, attack-then-noise snapshot links and non-executable A/B/C/D plan |
| `basil_core/cart_owner_ebm.py` | Nested local-CE Hessian-gradient term, exact combination with balanced owner gradients/frozen anchor, lazy worker factory using the existing five-epoch/controller path |
| `scripts/check_cart_owner_integration.py` | Default zero-training plan; opt-in synthetic CPU/GPU derivative and single-update checks; no research runner |
| `tests/test_cart_owner_integration.py` | Training provenance/selection/ties, malicious finite response demonstration, attack order/link RNG, independent objective/no-lambda tests, actual CNN and worker-path proofs |

Paper PDFs and recorded results were not modified. The later clean comparison
adds an overridable anchor-policy hook and explicit runner hooks; historical
default mathematics is unchanged. Heavy numerical/plotting/cloud imports remain lazy.

## 6. Verification evidence

| Check | Result |
|---|---|
| Focused integration/owner/selection/CART/protocol/lazy suites | **157 passed**, 3 warnings, 9.09 s |
| Full suite before the clean-comparison addition | **566 passed**, 2 skipped, 644 subtests, 3 warnings, 77.91 s |
| Full suite including clean-comparison addition | **570 passed**, 2 skipped, 644 subtests, 3 warnings, 82.73 s |
| New integration coverage | 25 tests, including two worker-path CE/EBM cases and the approved CNN |
| Independent tiny-model derivative | Matches class-mean CE + local EBM + frozen anchor within rtol 2e-5 / atol 1e-7 |
| Analytic softmax Hessian | Matches `2 sigma_e² H g`; detects a simple scalar-gradient substitution |
| No-lambda | Exact same losses, gradients and one-step SGD weights at 1 versus 999999 |
| CPU/GPU synthetic actual worker | Passed all ten classes for both 512 and 392 active batches; 16 samples/helper |
| Maximum CPU/GPU relative gradient difference | 3.19007195e-6; predeclared limit 0.01 |
| Maximum CPU/GPU update difference | 7.45058060e-9; predeclared limit 1e-4 |
| Synthetic optimizer updates | 12 total; reset canonical inputs each case; **zero full local epochs/research runs** |
| GPU | RTX 4070 Ti, TF 2.18, deterministic float32, TF32 off, 6,144-MiB cap |
| Peak TensorFlow GPU allocator | 355.17 MiB; not a measurement of all system GPU allocations |
| Imports | 231 Python sources audited after comparison addition; no movable/direct heavy imports |

The synthetic cases named `noisy_ce` and `noisy_ebm` check local objective
configuration at sigma 0.010; they are **not actual noisy-channel research
trajectories**. Snapshot noise/order are verified independently by unit tests.
No OOM or non-finite update occurred in the device proof. CPU/GPU agreement
within tolerance does not establish bit-identical trajectories or long-run
stability. Source-bound proof:
`gui/worker_state/CART_OWNER_INTEGRATION_GPU_VERIFICATION.json`.

That EBM proof records the source revision before the subsequent runner/policy
hook refactor. It remains historical and is not relabelled as current-source
proof. The new clean comparison has its own source-matched GPU proof:
`gui/worker_state/CART_BALANCED_SELECTION_FIXED_ANCHOR_3R_GPU_VERIFICATION.json`.
The two skipped full-suite tests require a display. Before the clean run,
all **23,934** retained result/plot/paper files matched the recorded SHA-256 baseline.

Commands actually executed:

```bash
DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-owner-integration-tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_cart_owner_integration.py tests/test_cart_owner_selection.py tests/test_cart_owner_gradient_worker.py tests/test_cart_owner_gradients.py tests/test_strict_cart.py tests/test_research_protocol.py tests/test_lazy_imports.py

DISPLAY= CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-owner-integration-final PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 environment/basil-noise-env/bin/python -m pytest -q tests

environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests run_gui.py
environment/basil-noise-env/bin/python scripts/audit_lazy_imports.py
environment/basil-noise-env/bin/python scripts/check_cart_owner_integration.py

MPLCONFIGDIR=/tmp/papermerge-owner-integration-proof environment/basil-noise-env/bin/python scripts/check_cart_owner_integration.py --verify-device GPU
```

The verifier refuses to overwrite an existing device proof. The default plan
loads no TensorFlow, NumPy, matplotlib or W&B. Old source-bound proofs stay
historical; they are not relabelled as proof of this new objective.

## 7. Remaining gates before a full threat study

1. Choose the helper threat boundary explicitly. Full-node B/C/D remain blocked
   without a defined, tested helper defense; honest auxiliary helpers give only
   a narrower snapshot-attack experiment.
2. The authorized fixed-μ comparison is complete and shows early improvement,
   but not reliable all-class retention. Preserve that finding without tuning
   against the final test set or automatically extending the run.
3. Specify any helper transport noise/adversary mechanism independently of
   snapshot noise; do not borrow a parameter sigma for gradient coordinates.
4. Only after those choices and checks, consider matched research A/B/C/D.
   No 30-/50-/500-round experiment is automatically authorized by unit tests.

Test data must stay evaluation-only, including for coefficient choice, snapshot
selection, early stopping and hyperparameter selection. This document does
not claim the non-IID learning problem is resolved.
