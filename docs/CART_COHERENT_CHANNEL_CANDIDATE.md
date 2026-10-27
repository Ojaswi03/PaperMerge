# CART candidate: coherent local reference and channel-calibrated class budgets

Status: isolated research candidate, updated 27 October 2026. Numerical primitives,
checkpoint recovery, CPU workers and bounded GPU execution are checked. Full-data predictive benefit and
algorithmic novelty have not been established. This is a separate revision from
both the manuscript's ClassRegistry-controlled proximal method and the later
historical class-minimum barrier.

27 October pilot: three full-data rounds, Dirichlet seed 2025, sigma 0.01,
joint Hidden/BASIL/EBM. Final heldout means were 11.29% without CART, 12.79%
with the old barrier, and 11.21% with coherent CART. The coherent guard
accepted zero movement at 39.12% of protected steps, admitted no new reference
after initialization, and cost 1.94 times the control runtime. This failed
pilot is not evidence of predictive benefit.

The separately named `coherent_refresh` diagnostic permits a whole reference
to replace its predecessor when its equal-class mean CE improves by more than
1e-4 and every class still meets the PREVIOUS calibrated envelope. It relaxes
classwise dominance, not the measured feasibility checks. Individual reference
class losses can increase within that envelope; no monotone per-class or
original-reference bound across repeated refreshes is claimed. Each admitted
reference is independently recalibrated before later use.

An explicit `local_prior` training-objective ablation uses established
training-time logit adjustment, logits + log(floor-one local TRAIN priors),
with tau=1 and no proximal anchor. The same objective is used in its paired
no-CART control; raw logits still define selection, probes and evaluation.
EBM differentiates the adjusted CE in this declared ablation. Logit adjustment
is attributed to Menon et al.; any resulting baseline gain is not CART's gain.
The original raw-CE configurations remain separately identified and available.

The completed prior-adjusted seed-2025 pair also failed: control 37.59%,
per-step `coherent_refresh` 21.57% after three rounds. It admitted 27 references
but rejected 22.01% of protected steps and used 1.75 times the control runtime.
The remaining development batch was stopped; this is not a three-seed result.

Two explicitly separate commit-only variants are now diagnostic candidates:
`coherent_commit_ce` and `coherent_commit_error`. Both load the complete selected
BASIL model, perform the same five local SGD/EBM epochs as their control, and
then test a finite segment from the receiver's stored reference to the trained
proposal. Only the committed outgoing honest model must meet the envelope;
intermediate SGD iterates are unconstrained. There is no input restoration,
per-step guard or momentum scaling in these variants.

The CE variant retains the definitions above. The error variant instead uses
the empirical supported-class zero-one risk

```
r_ic(theta) = mean_{(x,c) in P_ic} 1[argmax_k z_theta,k(x) != c].
```

The same 19-draw joint-score construction calibrates bounds on this bounded
class risk, using `max(r_ic(reference), 0.1)` as its declared positive scale.
The rank argument remains valid with discrete scores and conservative ties.
Error-risk references refresh when their equal-class mean error decreases
within the previous envelope; equal-error candidates need lower mean raw CE.
This does not turn TRAIN-probe accuracy into a population-confidence bound.
The finite-grid test is not the constrained optimum, and reference refresh
does not provide a bound relative to the initial model for all future rounds.
Model interpolation and class-risk constraints themselves are established;
channel-calibrated, receiver-local commit control is the proposed distinction.
Neither commit variant yet has established full-data predictive benefit.

The completed seed-2025 commit-only development comparison gave 27.27% (CE)
and 29.10% (classification error), versus 37.59% for their joint-condition
control and 55.58% for the matched clean reference. Remaining development jobs
were stopped. At initialization, six of ten reference models recalled only
one or two supported classes above uniform-guess recall. The controller was
protecting early class specialization rather than established broad knowledge.

The separate `coherent_qualified_error` candidate therefore waits for an
empirically broader anchor. With supported-class set C_i and measured error
rates r_ic, its admission qualification is

```
Q_i(theta) = 1[ mean_c r_ic(theta) < 0.9 AND
                number of c in C_i with r_ic(theta) < 0.9
                >= max(2, ceil(|C_i|/2)) ].
```

Before a qualified anchor exists, ordinary local training proceeds without a
retention constraint. A qualified candidate can replace an unqualified anchor
when its mean class error is no worse; unqualified anchors can also improve
while local training remains unconstrained.
After qualification, reference updates must remain qualified and follow the
previous risk-envelope/mean-error rule. Protected committed outputs still meet
their stored class-risk envelope. Qualification itself is a local empirical
warm-up/admission criterion, not a statistical above-chance test or population
certificate. Warm-up is established prior practice, including in GEC; the
criterion is not presented as a new generic theory. This candidate has not yet
demonstrated predictive improvement.

Its completed ten-round seed-2025 development pair was also negative:
46.74% control versus 43.34% qualified scalar CART (minus 3.40 percentage
points). The remaining seeds were interrupted rather than represented as a
completed replication. Qualification reduced the earlier initialization problem
but did not establish an accuracy advantage.

## Latest candidate: repair the activation direction

The separately fingerprinted `coherent_projected_error` variant retains the
same qualified whole reference and calibrated error budgets. It changes only
the commit repair geometry. Complete BASIL handoff and all five local epochs
still precede repair; no intermediate SGD iterate is constrained.

For the completed activation proposal, let `d = theta_proposal - theta_ref`.
Identify classes whose actual TRAIN-probe error exceeds the stored bound.
Compute those classes' raw CE gradients at both the reference and proposal.
Their normalized rows form G. Use the established GEM/GEC projection idea:

```
v = argmin_u 0.5 ||u-d||_2^2   subject to G u <= 0.
```

The small-Gram dual solver checks feasibility and complementarity; a numerical
failure explicitly returns the zero direction. The gradient cone is a surrogate
and does not certify nonlinear class error. Therefore the code checks actual
class error along the finite grid from the reference to `theta_ref + v`.
It separately computes the original scalar-grid repair from the reference to
the unmodified proposal. Among the two measured-feasible outputs, choose the
one nearest the original trained proposal in squared parameter distance.

For the **same** reference and proposal this supplies a narrow property: the
selected output meets the stored probe-error bounds and is no farther from the
proposal than the scalar counterfactual. It does not imply higher accuracy or
global optimality. The projected ray's scale is not the fraction of the raw
SGD update retained. Report proposal fidelity
`1 - ||theta_commit-theta_proposal|| / ||d||`, scalar counterfactual scale,
projection selections, solver fallbacks, qualifications and runtime.

Projection itself is prior art. The proposed contribution remains the
receiver-local/channel-calibrated commit mechanism, subject to literature and
matched effectiveness comparisons. Needed ablations include a fixed budget,
scalar repair, and a matched GEM/GEC-style control. Do not assign model or
local-prior baseline improvements to CART.

At 07:25 EDT, the four-arm IID/Dirichlet GPU smoke completed and its saved
artifacts verified; no qualified protection was active in that small-data
smoke. The complete CPU suite passed 1,164 tests plus 644 subtests.

At 07:41 EDT, the first full-data three-round pair in
`.papermerge/local-tests/cart-projected-prior-three-seed-v1` completed and
verified: control **37.59%**, projected CART **33.61%** (minus 3.98 percentage
points). Fourteen activations were protected; six selected the projected
family, with zero measured class-risk violations or numerical fallbacks.
Mean proposal fidelity was .6676 versus the scalar counterfactual's .5179.
Thus the geometric property was exercised on CIFAR, but did not yield an
accuracy improvement. Remaining clean, seed and partition checks are still
running/pending; no three-seed conclusion is available. See
[AI_HANDOFF.md](AI_HANDOFF.md) for execution paths and continuation rules.

## Problem being changed

Historical per-class minima can come from different parameter vectors. Their
coordinate-wise minimum does not provide a single known model satisfying every
target simultaneously. A scalar proximal penalty also bounds neither actual
class CE nor class accuracy. GEM already addresses remembered-loss constraints;
it is essential prior work, not an invention claimed here.

The initial per-step candidate keeps one whole receiver-owned reference, calibrates a loss budget
to the explicitly modeled parameter channel, and verifies nonlinear probe losses
after receiving a model and after optimizer proposals. It uses no global
evaluation feedback, trusted remote accuracy claims, public helper data, or
labels absent from the receiver. The reference is an additional local model
state; no class-specific model parameters or raw examples are transmitted.

## Definitions

At honest receiver i, P_ic contains up to 16 fixed local TRAIN examples of each
supported class c. Define the unaugmented, raw-logit probe loss

```
ell_ic(theta) = mean_{(x,c) in P_ic} -log softmax(z_theta(x))[c].
```

The reference theta_i^ref is one complete locally trained model. All reference
losses ell_ic^ref are measured at that SAME model. A new whole reference is
admitted only when every supported class CE is no larger and their equal-weight
mean improves by more than 1e-4. Initial admission follows the first completed
local activation. These are local empirical rules, not poisoning certificates.

## Modeled channel allowance

Freeze the reference and probes before calibration. Generate m=19 independent
corruptions xi_j from the configured absolute-coordinate Gaussian channel:

```
theta_j = theta_i^ref + xi_j,       xi_j ~ N(0, sigma^2 I)
s_c     = max(ell_ic^ref, 0.1)
S_j     = max_c max(0, (ell_ic(theta_j) - ell_ic^ref) / s_c)
q_i     = max_{j=1..19} S_j
b_ic    = ell_ic^ref + q_i s_c.
```

The class maximum is taken BEFORE calibration, so this is one joint score, not
separate unjustified 95% statements for ten classes. The positive scale floor is
a fixed numerical/modeling choice, not an optimized parameter or novelty claim.
With sigma=0 the allowance is zero.

For a fresh independent corruption of this fixed reference under the SAME
channel, exchangeability of its score with the 19 calibration scores gives

```
P( all supported c: ell_ic(theta_i^ref + xi_new) <= b_ic ) >= 19/20.
```

For distinct scores, the new score has each of the twenty ranks with equal
probability; only the largest rank exceeds the maximum calibration score. Ties
with an inclusive comparison are conservative. This is a standard rank argument
related to conformal calibration, not an original probability theorem.

**Scope:** the probability is marginal over calibration and a fresh corruption,
conditional on the previously fixed reference/probes. It is not 95% test
accuracy, a conditional guarantee for one realized budget, a guarantee across
all future transmissions, a statement about a selected peer model with a
different underlying parameter vector, or Byzantine robustness. Unknown,
correlated or shifted channels require separate modeling and validation. The
current code knows sigma from the experiment; a deployment would need a justified
estimate or measurement. Models with untransmitted non-trainable state are
excluded from this candidate's worker.

## Measured feasibility restoration

At reception, the selected BASIL snapshot is the proposal and the coherent
local reference is a feasible anchor. Evaluate actual class CE on a fixed grid

```
theta(alpha) = anchor + alpha (proposal - anchor)
alpha in {1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 0}.
```

Keep the first point satisfying every ell_ic(theta(alpha)) <= b_ic + 1e-6.
The zero endpoint supplies a feasible witness. No convexity of CE is assumed;
each candidate is measured. This is a finite-grid restoration, not the nearest
Euclidean projection or an optimizer of the constrained problem. Conditional
input restoration is explicitly an additional CART operation after BASIL
selection; it is not claimed to be unchanged BASIL handoff or FedAvg.

During local SGD, compute the same CE/full EBM and common kernel-L2 proposal as
the paired control. The previously accepted parameters are now the feasible
anchor. Accept the largest measured-feasible grid step. With SGD momentum,
scale the proposed momentum buffers by the accepted alpha; a zero step clears
them. Optimizer state still resets at each activation. Measure and report the
accepted-step distribution, rejected steps, probe work and time overhead.

**Empirical invariant:** once initialized, each committed protected optimizer
boundary satisfies its stored supported-class loss envelope, provided its
anchor is feasible, inference state is fixed, and evaluations are finite. The
proof is the explicit feasibility test and inclusion of the anchor. This
invariant is independent of the channel probability interpretation. It is
restricted to stored TRAIN probes. It does not guarantee generalization,
convergence, useful plasticity, or preservation of absent classes. Repeated
zero/very small steps and a stale reference are failure modes to measure.

On those probes only, a conservative accuracy interpretation follows because a
misclassified example has raw CE at least log(2): the class error fraction is
at most min(1,(b_ic+1e-6)/log(2)). This bound is often vacuous and is not a test
accuracy claim.

## Required comparisons and claim discipline

Use identical partitions, initial models, training recipe, attacks, channel
draws and horizons for: no CART, the existing bounded class-minimum barrier,
and this coherent candidate. Both IID and Dirichlet alpha 0.2 are required.
Primary development levels are sigma 0.005, 0.01 and 0.02 with seeds 2025–2027.
Begin with a bounded matched subset; do not present pending full-grid results.

Separate two ablations before attributing a gain: a coherent reference with a
fixed allowance, and the channel-calibrated allowance with coherent references.
Measure memory-loss violations and plasticity as well as heldout accuracy.
The desired outcome—strongest impaired condition and a small gap to clean—is a
hypothesis, not a prescribed ranking. Neither a passing invariant test nor a
new combination of established components establishes algorithmic novelty.

Relevant primary sources, downloaded under
`papers/cart_novelty_audit_20261026/`, include GEM and Shafer–Vovk's conformal
prediction tutorial. GEM uses earlier-task episodic losses and a first-order
gradient projection. The proposed distinctions here are the single coherent
receiver reference, channel-calibrated joint class budget, and checks on actual
nonlinear losses at reception and local updates. A targeted literature review
has not established that this precise combination is absent from prior work.

A particularly close recent comparator is Lai et al., **Gradient-Guided
Epsilon Constraint Method for Online Continual Learning**, NeurIPS 2025
(downloaded official PDF). It already uses remembered-loss slack, per-task
constraints, dynamically updated reference losses and a stability/plasticity
controller. Its Sections 3–5 and Appendix D preclude claiming those generic
ideas as new. Its gradient-guided QP/barrier update differs from this candidate's
parameter-channel calibration and explicit finite-grid checks of nonlinear
class losses at reception and optimizer boundaries. This is a proposed narrow
distinction requiring baseline comparisons, not a certified novelty claim.
