# Diagnosis of the completed Dirichlet C/D decline

Saved-evidence analysis, 2026-10-17, against baseline checkout
`4c4910610915e6d2861358437994e1dc6f426556`, followed by the software repairs
documented below. No research condition,
continuation, parameter change, queue preparation, or cloud run was executed.
The completed baseline and its reports remain unchanged.

The strongest explanation supported by the saved evidence is **persistent
channel perturbations accumulating along selected model ancestry, with little
net radial correction by local training**. Declining SGD step sizes, a
potentially noisy anchor, local-score/global-quality disagreement, and the
limited target of the EBM surrogate explain why mitigation is not assured.
Their individual causal contributions to accuracy are not identified.

## Verified scope and results

The local HEAD and local `origin/main` tracking ref both resolve to the commit
above; this is not an independent live GitHub fetch. The three pre-existing
modified files are the `wandb`/`wb` environment launchers and
`gui/queue_state.json`. All eight persisted queue entries are completed.

Re-reading the eight manifests, CSVs and NPZs reproduces the existing audit's
completion, endpoint and peak-round values. All four 100-round metric and
selection arrays exactly equal the first 100 entries of the corresponding
fresh 500-round runs. These prefixes are not independent replications.

| Mean-node full-test accuracy | A | B | C | D |
|---|---:|---:|---:|---:|
| R99 | 53.973% | 53.558% | 46.511% | 46.894% |
| R499 | 60.166% | 59.726% | 35.172% | 35.828% |

These are ten-model arithmetic averages, not Node-9 `full_test_accuracy`.
D−C is +0.656 percentage points at R499, but only about +0.146 pp over
R490–499. Both noisy conditions peak at R48. This is inadequate sustained
noise mitigation, despite correct recorded execution.

The analyzer rechecks 20,000 ordered activations and 1,020,000 recorded
optimizer steps using the existing contract checker, including five complete
epochs, unequal node sizes, partial batches, rolling memory, minimum raw-local
CE selection, anchor/input identity and finite telemetry. Each condition's
22 current executed-source fingerprints matches its manifest; all archived
source hashes match. A separate read-only replay again verifies all 200 final
directed transmissions and saved memories, including attack-before-noise order.
This is not a new exhaustive replay of all historical optimizer updates.

The actual PDF directory is lowercase `papers/`, not `Papers/`; it is preserved.
The loss guide's stale 30-round scope has been corrected to cover the completed
30/100/500-round families. Its mathematics is unchanged.

## 1. Parameter growth is predominantly received-link energy

For activation t, let p(t) be the selected source activation, u its outbound
parameters after any attack, v the received selected parameters, and w the
honestly trained output. Define squared-norm changes:

```
attack:   ||u||² − ||w_parent||²
channel:  ||v||² − ||u||²
training: ||w||² − ||v||²
```

Following the selected parent pointers telescopes these into
`||w_final||² − ||w_initial||²`. The new analyzer uses saved received and
trained norms and verifies the selected link's transmitted/input hash at
every noninitial parent edge. Accounting residuals are below 1e-7.

| Final Node-9 ancestry | C | D |
|---|---:|---:|
| Selected noisy links inherited | 1,530 | 1,527 |
| Initial squared norm | 227.388 | 227.388 |
| Sum of channel squared-norm changes | +17,929.826 | +18,163.294 |
| Sum of local-training squared-norm changes | −210.830 | −214.360 |
| Sum of attack squared-norm changes | 0 | 0 |
| Final trained norm | 133.964 | 134.820 |

Local training offsets only about **1.18%** of this net channel energy along
each Node-9 ancestry. Across all ten final models, mean channel/training energy
changes are +17,911.551/−210.706 in C and +18,148.396/−214.177 in D.
Their paths overlap heavily; these are not ten independent observations.
Clean A/B have zero channel contribution and final norms near 19.79/19.99.

For an unselected fixed outbound vector with independent Gaussian noise,
`E[||u+xi||² − ||u||²] = d*sigma² = 11.7706` here. Observed Node-9 ancestry
averages are 11.719/11.895 per selected link, close to that scale. The measured
channel term includes `2*u.dot(xi)` and selection conditions on xi; it is not
an application of an independence theorem to the selected trajectory.

This strengthens the accumulation explanation for **parameter norm growth**.
It does not establish what fraction of accuracy loss that growth causes:
parameter norms are not function-space distances, and some perturbation
directions may be functionally harmless. The channel remains absolute
sigma=0.010 throughout. There is no relative-L2 noise amplification rule.
Nor does one model inherit all 25,000 delivered links.

## 2. Decaying step sizes face persistent perturbations

For a complete activation, SGD's parameter displacement is bounded by its
path length `U = eta_r * sum_b ||applied_gradient_b||`. This includes D's
Hessian-gradient term and the anchor. It is an upper bound, not net movement
or a measurement of movement specifically opposite channel noise.

| Saved activation averages | C, R0–19 | C, R480–499 | D, R0–19 | D, R480–499 |
|---|---:|---:|---:|---:|
| Per-minibatch applied step norm | 0.04984 | 0.01879 | 0.05097 | 0.01752 |
| Complete-activation path-length bound U | 2.5417 | 0.9585 | 2.5994 | 0.8936 |

Noninitial selected links have noise norm about 3.43. Gradient norms rise late,
partly compensating for the approximately 26-fold LR decay; actual step norms
therefore do not fall 26-fold. Nevertheless, late training moves on a much
smaller scale than an incoming perturbation in Euclidean parameter space.
Recovering classification need not require removing every noise coordinate.

A stable scalar quadratic illustrates the concern: with K local SGD steps
and curvature h, `a=(1−eta*h)^K`, an isolated receive/train recurrence is
`x_next=a*(x_parent+xi)`. Its fixed-eta noise variance is
`a²*sigma²/(1−a²)`, approximately `sigma²/(2*K*eta*h)` when K*eta*h is small.
Less contraction allows more retained noise. This toy calculation is not a
CNN convergence theorem: BASIL selection, nonconvexity, node changes and
time-varying eta violate its simplified assumptions.

The saved trajectories support this mechanism, but do not prove LR decay is
the unique cause or justify changing the approved schedule.

## 3. Raw-CE selection and adjusted training judge different things

Training uses `CE(y, z+log(pi_i))`; selection uses `CE(y,z)` on one fixed,
unaugmented receiver-local TRAIN sample of up to 512 images. Both are
implemented as declared. The TRAIN-prior sign, floor-one counts and raw-logit
inference agree with the documented adaptation of
[Menon et al., Eq. (10)](https://arxiv.org/abs/2007.07314).
The cached full paper was also consulted. Receiver-local priors, absent-class
floors and sequential handoff are adaptations, not guarantees from that paper.

Different scoring rules can reverse candidate rankings even on identical
labels: for a 90/10 two-class sample, raw CE prefers a 90/10 predictor to a
50/50 predictor, while prior-adjusted CE prefers the 50/50 raw predictor.
A tiny test confirms the reversal with the actual ten-class offset function.
This establishes possibility, not its quantitative effect in these runs.
Because A/B use the same mismatch, it cannot alone explain C/D's specific
decline; interaction with noisy candidate quality is a hypothesis.

Late admissions are direct evidence that low local loss can disagree with
global quality. C admits 38 actually corrupted snapshots, starting R287;
D admits 19, starting R412. Node 0 has 2,938 examples, 1,765 trucks and
1,147 cats. The first attacked winners have lower local CE than every honest
alternative yet only 10% selected-model global accuracy. Their five-epoch
outputs reach 9.87%/10.95%. Those before/after numbers are not causal drops
relative to the receiver's prior unselected model.

Before any corrupted selection, C is already at **40.829% mean at R286**
and D at **37.898% at R411**, down from their roughly 47.5% peaks.
Moreover, **none of the ten final C/D selected ancestries includes an
actually attacked outbound link**. Late failures need not persist on the
final retained path. This does not rule out transient/indirect effects or
identify the counterfactual outcome without attackers; C/B remains a
noise effect within the attacked environment.

The local BASIL PDF's Eq. (3) supports minimum receiver-local loss selection.
Its theoretical section assumes IID data, and its non-IID extension uses ACDS
data sharing, absent here. S=b+1 guarantees candidate connectivity under its
model, not that a skewed local CE oracle always recognizes globally useful
honest models. No source guarantee automatically transfers to this local-only
Dirichlet adaptation. No TEST-based selection or shared examples are proposed.

## 4. EBM regularizes local loss gradients, not accumulated channel error

The executed objective is
`F_i + 0.0001*||grad F_i||² + 0.005*||theta−received_selected||²` in D.
Nested autodiff supplies `g + 0.0002*H*g`; the anchor gradient is added
analytically. Seven existing tiny objective/prior/no-lambda cases pass.
There is no demonstrated implementation defect in this loss path.

The correction/base-gradient norm ratio averages 7.18%, reaches 13.83% late,
and has maximum 46.02%. D's mean applied gradient norm is about 4.9725,
versus base norm 4.5586. A penalty on gradient norm does not mean every
applied step has smaller norm; Hessian directions matter. Activity alone is
not effective mitigation.

For smooth loss and sufficiently small isotropic parameter noise,

```
E[F(theta+xi)] = F(theta) + (sigma²/2)*trace(H_F) + higher-order terms
E[(F(theta+xi)−F(theta))²] ≈ sigma²*||grad F||²   (first-order loss variation)
```

Thus the implemented surrogate has a local sensitivity interpretation but
is not generally expected noisy CE. At a stationary point its penalty and
correction vanish, even if curvature and expected noisy loss are positive.
A tiny Gaussian-quadrature check demonstrates this for empirical binary CE:
at its 90/10 optimum, g=0 and H=0.09; expected excess loss divided by sigma²
is approximately 0.045, while the gradient-norm surrogate is zero.
For the ReLU CNN, a local smooth expansion additionally requires care around
activation boundaries; these formulas are not exact finite-noise guarantees.

The actual local Ang PDF, Eqs. (13)/(22), motivates the squared-gradient
penalty. Its Eq. (14) uses a squared-loss expansion, and Eq. (23)'s simplified
scaling is not the derivative of general nonlinear CNN CE. Its centralized
aggregation and convex assumptions differ from this ring. We preserve the
source-inspired qualification and the distinct historical compatibility mode.

Finally, the code penalizes the gradient of each **augmented minibatch mean**,
not the full-local-data mean. Under an unbiased minibatch model,
`E||g_B||²=||E g_B||²+trace(Cov(g_B))`. A tiny algebra test confirms the
distinction. Batch size, augmentation and sampling therefore affect the
surrogate. Saved D ratios by partial-batch size are confounded with node and
round; they do not show a monotonic small-batch explosion. There is no evidence
here that partial batches are a bug or should be removed.

## 5. Anchoring preserves the received state

At receipt, `theta=theta_selected`, so the proximal gradient is exactly zero.
Noise already in that snapshot is not an anchor residual. Each next activation
re-centers the reference, so there is no cumulative clean-state constraint.
Even a stronger anchor could preserve a noisy or attacked reference; simply
increasing mu would not establish denoising.

Measured anchor/base-gradient ratios average only 0.03065%/0.03185% in C/D.
A tiny actual-objective check confirms zero restoring anchor gradient at a
perturbed received reference. The proximal structure is related to FedProx
(local PDF, Eq. (2)); replacing the server model by a selected received
snapshot is this project's adaptation, not BASIL's or Ang's anchor mechanism.

## Defects, limitations and remaining hypotheses

| Classification | Finding |
|---|---|
| Confirmed scientific implementation defect | None demonstrated by this review. Passing contracts does not establish scientific effectiveness. |
| Reproduced and repaired operational defect | W&B initialization failure previously mislabeled nonzero resumes as zero-work startup failures; retry rejected retained work. The repaired path preserves checkpoints, counters and cloud identity. |
| Repaired documentation defect | The loss guide now covers completed 30/100/500-round studies. |
| Confirmed method limitations | No clean anchor reference; gradient-norm surrogate differs from expected noisy CE; raw local score can admit globally poor attacked snapshots; original BASIL IID assumptions do not apply automatically. |
| Strong measured mechanism evidence | Channel energy dominates final selected-ancestry norm growth; local training offsets little net radial energy; deterioration predates corrupt admissions. |
| Untested causal hypotheses | Smaller late SGD contraction retains more damaging noise; accumulated function-space sensitivity increases; scoring mismatch interacts with noise; local forgetting compounds drift. |

The W&B reproduction executes the runner's real initialization-failure branch
with synthetic checkpoint metadata and a fake offline SDK. Data, checkpoint
decoding, approval and device setup are mocked; entry into the scientific
engine is forbidden. It proves the status-routing problem under those inputs,
not occurrence in these completed jobs or a cause of their accuracy decline.
The repair now labels such attempts `interrupted`, preserves their counters
and cloud identity, and uses verified checkpoint recovery. Older nonzero
`tracking_unavailable` manifests also require a verified checkpoint, including
previous-checkpoint fallback. Fresh startup retries require zero recorded work
and no committed checkpoint or scientific artifacts. A failure never permits
restarting nonzero work from initialization.

Tracking identity is retained before SDK initialization, including when SDK
loading or settings validation fails. Retried SDK initialization reuses the
attempted/saved run ID and last successfully delivered metric cursors; real
cloud contents were not fetched. New regression tests cover repeated failures
with a fake SDK, and separate tests use the real decoder on tiny synthetic
checkpoints to check missing/corrupt evidence, source mismatch and fallback.

The GUI's completed-result display now reads compact completed manifests and
contiguous measured CSV rows without requiring permission to execute current
source. It preserves completed counters on a clone that lacks checkpoints or
machine-local receipts, while launch/resume approval remains separate.

These software edits change executed-source fingerprints. The 22/22 source
match statement above and the saved diagnostic JSON describe the initial
read-only review before these repairs. Running the analyzer now reports
current-source differences while validating the original archived sources.
Executed manifests, checkpoints, approval receipts and measured results are
not rewritten. Source-sensitive launch/resume checks intentionally remain
blocked until a separately reviewed compatible implementation/approval exists.
This is not an automatic source-compatibility migration or HPC run approval.

## Small prospective validation plan — not execution approval

1. **Saved-model sensitivity, zero SGD.** Predeclare final B/C/D Node-0 and
   Node-9 snapshots, the existing receiver-local SS batches, and one fixed
   balanced TRAIN-only audit probe (20 images/class). Apply 16 antithetic
   perturbation pairs at the existing sigma=0.010, in temporary model copies.
   Measure adjusted/raw CE changes, prediction disagreements and per-layer
   perturbation effects; compare measured expected CE changes with Hessian-trace
   and gradient-norm quantities. The balanced probe is offline diagnosis only,
   never a selection or training data-sharing mechanism. No TEST tuning.
   Early-round weights must not be fabricated if not retained.
2. **After explicit approval: twelve independent one-activation diagnostics.**
   Start every copy from the declared saved B500 Node-9 trained state; use local
   TRAIN at receivers 0 and 9. Cross the two existing schedule values eta_0 and
   eta_499 with three regimes: clean CE+anchor, noisy CE+anchor, noisy
   EBM+anchor. Keep five complete epochs, partial batches, sigma and mu fixed;
   EBM c=0.0001 only in its noisy regime. Match data order, augmentation and
   noise draws. This is 660 optimizer batches total, no propagation or queue.
   It deliberately isolates receive/train repair outside BASIL selection and
   is a separately named mechanism diagnostic, not a continuation of baseline.
   Measure noisy-minus-clean output distance, its projection onto injected
   noise, raw/adjusted TRAIN loss and EBM activity. This can test contraction at
   a fixed starting state; it cannot establish 500-round robustness.
3. **Only if warranted, propose a separate algorithm/control study.** Report
   the above mechanisms before choosing any remedy. An attacker-free noisy
   control would separate noise from attack interaction; selection/objective
   alignment, a different robustness objective or a clean-reference mechanism
   would each be a new algorithmic condition requiring separate review and
   explicit approval. No coefficient search, extra lambda, test-round picking
   or automatic long-horizon run follows this report.

## Reproduction and preserved evidence

New read-only tooling:
[`analyze_dirichlet_mechanisms.py`](../scripts/analyze_dirichlet_mechanisms.py).
It emits JSON to stdout and never writes run artifacts, gates or cloud state:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 \
  environment/basil-noise-env/bin/python scripts/analyze_dirichlet_mechanisms.py \
  > /tmp/papermerge-mechanisms.json
```

It requires local activation archives and executed-source snapshots, which a
Git clone alone may lack. Missing evidence produces an error, not replacements.
The compact derived output is
[`DIRICHLET_MECHANISM_ANALYSIS.json`](DIRICHLET_MECHANISM_ANALYSIS.json).
Tests are in [`test_dirichlet_mechanisms.py`](../tests/test_dirichlet_mechanisms.py).

The initial read-only focused CPU invocation passed **13 cases**: six analytical or
mocked-recovery checks and seven existing objective/prior/no-lambda cases.
Two dependency deprecation warnings were emitted. No full-suite rerun is
claimed for that review. Parse, whitespace, lazy-import and derived-JSON
equality checks also passed at that point.

The software repair validation then passed **207 tests**, with one GUI test
skipped because no display was provided and two dependency warnings. It covers
tracking, checkpoint recovery, Dirichlet queue gating, completed-result display,
read-only accuracy reporting, staged queues and lazy imports. Two additional
malformed-identity cases were added afterward; the latest tracking-only rerun
passed **26 tests**. The analyzer also rejects invalid norms, ambiguous selected
candidates and tampered executed-source archives; differences in current source
are reported without rewriting historical provenance. Tiny real-checkpoint
recovery regressions are in
[`test_dirichlet_tracking_recovery.py`](../tests/test_dirichlet_tracking_recovery.py).

The existing completed analysis, audit and supplementary checks were read,
not regenerated. Protected-input hashes cover all 56 selected current100/500
raw files, the three completed reports, three pre-existing modified local files
and all 37 local PDFs: **99 files, zero changes** after this review.

Source attribution was checked against local BASIL (Eq. 3, theoretical
assumptions and ACDS section), Ang (Eqs. 13–15/22–23), FedProx (Eq. 2), and
cached Menon full text with its [primary paper record](https://arxiv.org/abs/2007.07314).
This is a mechanism diagnosis, not a novelty review.
