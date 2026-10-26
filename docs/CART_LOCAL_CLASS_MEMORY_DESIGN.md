# CART with receiver-local class memory

Status: implemented as an isolated, bounded diagnostic, 25 October 2026. The
user selected Dirichlet alpha 0.2 to match the completed non-IID baseline.
The numerical settings below were fixed before the local tests. They are not
an HPC campaign approval or evidence of an established accuracy improvement.

Revision: `cart_local_class_memory_v1`.

## Scope and motivation

Keep the completed Dirichlet baseline's partition, initialization, CNN, local
prior-adjusted objective, selected-snapshot handoff, SGD schedule and channel.
Add a differentiable, class-balanced retention term using each receiver's own
TRAIN data and locally measured historical predictions. The class registry now
stores the targets needed by that loss, rather than only performance scores.

The old strict-CART controller has a different objective and a one-class split.
Its supported target simplifies to the selected snapshot's accuracy, regardless
of the historical maximum; see [the exact rule](STRICT_CART_PROTOCOL.md).
It remains an archived baseline. This proposal does not silently repair that
revision or reuse its readiness receipt.

This proposal does not add helper gradients, exchange images, replicate other
nodes' TRAIN data, or trust a sender's reported class scores. There is no central
retention teacher. Data-local operation is not a differential privacy guarantee.

## Unchanged experimental foundation

| Component | Proposed setting |
|---|---|
| Data | Existing CIFAR-10 TRAIN cache, Dirichlet alpha 0.2, seed 2025, ten nodes |
| Model | Existing 117706-parameter BASIL CNN |
| Ring | Fixed sequential node order; five predecessor memories |
| Selection | Existing minimum raw receiver-local TRAIN CE; existing tie-break |
| Handoff | Complete selected weight/bias replacement; no parameter averaging |
| Local training | Five complete epochs; batch 512; retain partial batches |
| Optimizer | Reset SGD; LR 0.05/(1+0.05*round); zero momentum/weight decay/clipping |
| Base loss | Existing local TRAIN-prior adjusted CE, tau 1, count floor 1 |
| Snapshot anchor | Fixed mu 0.01, as in the completed Dirichlet baseline |
| Evaluation | Raw logits; TEST is reporting only |
| Attack/channel | Existing configuration, activation schedule and independent per-link noise |

The old strict-CART adaptive scalar is not stacked onto this baseline. That
would add a second optimization change and obscure the memory comparison.

## Local class memory

For receiver i, define C_i as classes having at least one local TRAIN example.
Select a fixed probe P_ic of min(16, n_ic) examples per supported class using
a separate keyed RNG stream. Reuse exactly these indices throughout the run.
The stream must not consume existing partition, selection, augmentation,
attack or channel RNG state. Probe examples remain in ordinary five-epoch SGD;
these are training-side control data, not held-out validation.

For each supported class, keep one locally evaluated teacher record:

- Cache indices, partition/data hashes and class support.
- Ten-way raw-logit log-softmax targets for every probe example.
- Mean raw class-probe CE, teacher model hash and activation identity.

Admit records only at defined activation boundaries: evaluate the fully loaded
selected snapshot before training and the trained snapshot after five epochs.
An empty entry accepts the first finite candidate. Replace an existing entry
only for strictly smaller class-probe CE on the identical indices; ties retain
the existing entry. Both comparisons use actual receiver-local labels. A model
hash binds provenance; it does not certify an honest sender or generalization.
The targets remain frozen during the activation's optimizer updates.

Each class can retain targets from a different historical model. No weights
from those historical models are averaged or loaded for local SGD. Missing
classes have no targets and are not assigned zero accuracy or synthetic data.
Records are receiver-local and are not attached to transmitted model packets.

At most 160 ten-way float32 targets require 6400 bytes per node, plus indices
and metadata. These are additional targets; existing local images are reused.
This is not a bound on total process memory, checkpoints or TensorFlow buffers.

## Objective and applied gradient

Let z_theta(x) be raw CNN logits and p_theta(x)=softmax(z_theta(x)). The existing
local-prior base loss on an SGD minibatch B is

\[
\pi_{ic}=\frac{\max(n_{ic},1)}{\sum_k\max(n_{ik},1)},\qquad
F_i(\theta;B)=\frac1{|B|}\sum_{(x,y)\in B}
 CE\bigl(y,z_\theta(x)+\log\pi_i\bigr).
\]

For each cached class, write its frozen teacher predictions as q_ic(x) and
its stored raw class-probe CE as ell_ic^mem. At the start of epoch e, evaluate
current raw class-probe CE ell_ic(theta_e), then freeze the regression weight
for the entire epoch:

\[
w_{ic,e}=\operatorname{clip}\left(
 \frac{\ell_{ic}(\theta_e)-\ell_{ic}^{mem}}
      {\ell_{ic}(\theta_e)+10^{-8}},0,1\right).
\]

The score compares losses on exactly the same labeled examples. A positive
weight means that current class-probe CE is worse than its saved teacher's;
it is not a statistical confidence or Byzantine certification. No gradients
flow through the weight, admission rule, teacher or stored targets.

Use all cached probe examples for the retention calculation at each optimizer
step; equal weight is assigned to each cached class, independent of its count:

\[
R_i(\theta;e)=\frac1{|C_i^{mem}|}\sum_{c\in C_i^{mem}}
 w_{ic,e}\frac1{|P_{ic}|}\sum_{x\in P_{ic}}
 D_{KL}\bigl(q_{ic}(x)\Vert p_\theta(x)\bigr).
\]

Empty memory gives R_i=0. Use unaugmented probes, raw logits, temperature 1
and stable log-softmax arithmetic. The regularizer differentiates through the
current complete CNN, including its encoder. It reevaluates raw local images;
there is no detached feature-only classifier replay. Ordinary minibatch
augmentation is unchanged.

The proposed complete loss is

\[
L_i=F_i+c\|\nabla_\theta F_i\|_2^2
       +\frac{\mu}{2}\|\theta-\theta_{sel}\|_2^2
       +\beta R_i.
\]

For g_i=grad F_i and H_i=Hessian F_i, its applied gradient during epoch e is

\[
\nabla_\theta L_i=g_i+2cH_ig_i
             +\mu(\theta-\theta_{sel})+\beta\nabla_\theta R_i.
\]

EBM regularizes the prior-adjusted base F_i only. Retention and the anchor
remain outside its inner gradient norm. The anchor is the exact selected
received snapshot, including any channel perturbation already present.

Predeclared local-diagnostic values: beta=1, mu=0.01, 16 probe examples/class,
temperature 1, epsilon=1e-8. Beta is a new retention coefficient, not the legacy
EBM lambda. These choices have no demonstrated optimality. Any later changes
need a recorded revision and TRAIN-only validation; never select them using
global TEST trajectories. Extra probe forwards/backwards are additional work
and must be counted and timed separately from the original SGD minibatches.

## Example

A receiver's saved truck-class probe CE is 0.6 and its current CE is 1.0.
The epoch's truck retention weight is approximately (1.0-0.6)/1.0=0.4.
The retention term encourages the current predictions on those same truck
images toward the frozen historical predictions. A class currently doing at
least as well as its teacher has weight zero for that epoch. If a supported
class degrades during the epoch, the weight responds at the next epoch
boundary. The final post-epoch measurement is telemetry, not another update.

This can discourage measured local forgetting. It cannot recreate absent-class
examples, repair a bad teacher by itself, guarantee test accuracy or eliminate
the stability/plasticity tradeoff. Repeated selection of a teacher using the
same TRAIN probes can overfit them; report this limitation.

## A/B/C/D and the evidence needed

All four non-IID conditions use the same CART memory mechanism and coefficients.

| Condition | Attack | Absolute Gaussian channel | EBM coefficient c |
|---|---|---|---|
| A | None | Off | 0 |
| B | Existing configured attack | Off | 0 |
| C | Same attack | Existing sigma 0.01 | 0 |
| D | Same attack | Same sigma 0.01 | sigma squared = 0.0001 |

IID studies retain their separately declared protocol; CART is not inserted
into IID solely to pursue a target accuracy. The new non-IID revision uses
fresh output paths and source-bound readiness evidence.

## Implemented local validation profiles

The implementation is in `basil_core/cart_local_memory.py`, with explicit
configuration validation in `basil_core/cart_memory_study.py`. The shared ring
engine requires the matching worker and saves/restores all receiver caches in
its atomic activation checkpoint. Other objectives retain their existing paths.

Two separate three-round profiles are available. Both are diagnostic-only and
cannot be promoted to production simply by changing a round count or flag:

| Profile | Controls | TRAIN | Evaluation | Attack onset |
|---|---|---|---|---|
| `smoke` | A beta=0, A, B, C, D | Original full partition, then at most 1024 examples/node; five epochs | Fixed 100 TEST examples/class | Round 1, explicitly shortened to exercise attacks |
| `clean_learning` | A beta=0, A | All 50000 TRAIN examples, original alpha-0.2 partition; five epochs | All 10000 TEST examples | Original round 20; no attacks in the three clean rounds |

The reduced smoke subsets change empirical local priors and are not an exact
reproduction of the completed full-data study. The subsequent clean profile
keeps the original data and fixed coefficients. Neither profile selects
hyperparameters, stops early at a favorable metric, or enables cloud tracking.

The runner defaults to inspection only. Local commands, after activating the
existing local environment, are:

```bash
python scripts/run_cart_memory_smoke.py
python scripts/run_cart_memory_smoke.py --execute \
  --output-root .papermerge/local-tests/cart-memory-smoke
python scripts/run_cart_memory_smoke.py --execute --profile clean_learning \
  --output-root .papermerge/local-tests/cart-memory-clean
python scripts/report_cart_memory_smoke.py \
  .papermerge/local-tests/cart-memory-clean --plot
```

Use fresh output directories. `--recover` requires the same configuration,
data, numerical environment and source identity; completed conditions are
skipped. SIGINT/SIGTERM requests a stop between optimizer steps. The incomplete
activation's class-memory changes roll back before the prior boundary is
checkpointed. The runner prints physical GPU, activation count, elapsed time
and measured ETA, with a five-second diagnostic heartbeat. Final completion
ends the heartbeat. Existing research evidence and data caches are retained.

The tests in `tests/test_cart_local_memory.py` check the independent combined
gradient against an explicit Hessian and closed-form KL derivative, beta=0
reduction, encoder/classifier corrections, class balance, absent support,
teacher integrity, TEST-report noninterference, source-bound recovery and
bitwise-identical resumed weights/metrics/telemetry on a tiny ring.

## First measured local results

RTX 4070 Ti, float32, TF32 disabled, existing 6144-MiB allocator cap. Both
profiles completed all their fixed three rounds without coefficient changes.
All five smoke conditions performed 300 optimizer updates each; the full-data
clean pair performed 1530 each, for 4560 updates and 210 activations total.

| Profile / condition | Round 1 mean | Round 2 mean | Round 3 mean | Final worst |
|---|---:|---:|---:|---:|
| Reduced smoke A beta=0 | 10.030% | 10.520% | 10.850% | 10.600% |
| Reduced smoke A | 10.030% | 10.520% | 10.850% | 10.600% |
| Reduced smoke B | 10.030% | 10.200% | 10.210% | 10.000% |
| Reduced smoke C | 10.000% | 10.350% | 10.290% | 9.600% |
| Reduced smoke D | 10.000% | 10.350% | 10.310% | 9.700% |
| Full-data clean A beta=0 | 11.953% | 11.703% | 13.052% | 11.100% |
| Full-data clean A | 11.951% | 11.707% | 13.079% | 11.090% |

The full-data final mean difference is only +0.027 percentage points; final
worst-node accuracy differs by -0.010 points. This is not convincing evidence
of a retention/generalization benefit. Smoke A/control predictions had the
same reported accuracies despite different trained parameters. The noisy D/C
smoke difference is also too small and early to support an EBM benefit claim.

The full-data CART correction was nonzero in 1428/1530 steps, with mean
retention/base gradient norm ratio 0.001258 (about 0.126%). The loss is active
but weak relative to the supervised gradient in this early diagnostic. This
observation does not authorize tuning beta from TEST scores. The fixed horizon
was not extended until a favorable result appeared.

The smoke B/C/D runs exercised eight outbound attack activations each. C/D
applied channel noise at all 30 activations. The full clean pair exercised no
attack or channel noise. These retain the existing outbound-attack simulation;
they do not establish protection against arbitrary malicious local optimizers,
fabricated local caches, or all Byzantine behaviors.

Measured condition runtimes were approximately 24-26 seconds each for smoke,
95.38 seconds for full-data A beta=0 and 97.20 seconds for full-data CART A.
The largest tracked TensorFlow allocation across these runs was 408.79 MiB;
this excludes driver/context and reserved allocator memory. No OOM was observed.

Artifacts are isolated under `.papermerge/local-tests/cart-memory-v1-20261025/`
and `.papermerge/local-tests/cart-memory-full-clean-20261025/`, including exact
executed source archives, checkpoints, complete telemetry, metrics, independent
review JSON/CSV and trajectory plots. Subsequent study-label and progress-label
corrections do not alter the objective or these archived executions. They are
not production readiness receipts, and historical evidence is not rewritten.

After those runner safeguards were finalized, a fresh A beta=0/A/B/C/D smoke
repetition reproduced all 95 metric arrays and all 150 activation telemetry
records exactly. Its highest tracked allocation was 413.89 MiB. A completed
`--recover` invocation skipped every activation and preserved all scientific
artifacts, retaining the new synthetic verification as a separate attempt.

Final headless repository suite: 1022 tests and 644 subtests passed, four
skipped and three dependency/model-input warnings. The independent mathematical
checks include CPU/GPU CNN gradient agreement within 3e-7 relative L2 for this
environment. All 879 protected pre-existing evidence/local-state files in the
before-work inventory remained byte-identical. These checks do not verify
execution of this new revision on the university HPC.

The four conditions isolate attack, channel and EBM effects within CART. They
do not alone show CART is better than no CART. A matched beta=0 non-IID control,
with identical partition, prior, anchor, compute accounting and seed, is needed
to attribute a difference to retention. That extra comparison is recommended,
not automatically added as another long run. Historical completed results are
context, not a replacement for a current matched implementation ablation.

Before HPC: verify analytic gradients, beta=0 reduction, frozen teachers,
encoder gradients, class-balanced weighting, invalid/absent data, deterministic
probe selection, unchanged old RNG streams and TEST noninterference. Then run
bounded local learning checks for all four conditions. Software tests alone do
not demonstrate a beneficial learning trend. Report mean/worst/per-class raw
test accuracy, TRAIN-probe retention, base/retention/EBM/anchor gradient norms,
admission counts, cache ages and actual added runtime.

Each receiver's cache, targets, scores, indices and provenance must be part of
the same atomic activation-boundary checkpoint as model/memory/RNG state.
Resume must restore the cache rather than rebuild it from later predictions.
Interrupted activations repeat from their last committed start. Cross-revision
resume from old strict-CART or Dirichlet checkpoints must be rejected.

## Contribution and attribution

This is a proposed CART integration of receiver-local, class-specific historical
targets and loss-regression weights with strict selected-snapshot ring training
under the declared attack/channel model. That is a research hypothesis, not an
established novelty or Byzantine robustness result.

Knowledge distillation and historical prediction replay are established:
[Learning without Forgetting](https://arxiv.org/abs/1606.09282) and
[Dark Experience Replay, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)
are relevant prior work. DER uses stored historical logits; this proposal uses
class-specific locally selected probability targets and epoch-frozen regression
weights. This difference alone does not establish novelty. BASIL, prior-logit
adjustment, proximal regularization and EBM retain their component attributions.
Compare against [the broader CART review](CART_LITERATURE_AND_ATTRIBUTION.md)
before making thesis or publication claims.
