# IID / Dirichlet development comparison

This opt-in comparison separates the environment from its defenses. It does
not replace the original BASIL reproduction or overwrite completed CART runs.
The existing bounded CART candidate underperformed in the completed 30-round,
three-seed replication; including it here is an ablation, not evidence of a
successful repair. Full comparison execution stays disabled during development.

27 October: [new coherent/channel-calibrated CART candidates](CART_COHERENT_CHANNEL_CANDIDATE.md)
are isolated in `run_cart_coherent_pilot.py`. Their optional, declared local-prior
objective is shared with their no-CART controls. The 144-run definition below
still contains the historical barrier ablation; it has not silently become a
validated new CART benchmark. Promotion requires matched predictive evidence
and the final agreed protocol, including the forthcoming professor pseudocode.

## Conditions

| Condition | Byzantine attack | Channel noise | BASIL selection | Full EBM | CART |
| --- | --- | --- | --- | --- | --- |
| clean | off | off | off | off | off |
| attack_only | on | off | off | off | off |
| attack_basil | on | off | on | off | off |
| noise_only | off | on | off | off | off |
| noise_ebm | off | on | off | on | off |
| attack_noise | on | on | off | off | off |
| attack_noise_basil | on | on | on | off | off |
| attack_noise_ebm | on | on | off | on | off |
| attack_noise_basil_ebm | on | on | on | on | off |
| attack_noise_basil_ebm_cart | on | on | on | on | on |

Use IID and Dirichlet alpha 0.2, seeds 2025/2026/2027, and three common absolute
Gaussian coordinate standard deviations. Each noisy condition runs at all
three sigmas for every seed. The three noise-free conditions run once per seed;
their sigma-zero reference is shared across the three noise comparisons.
There are `2 * 3 * (3 + 7 * 3) = 144` distinct runs. This includes untreated
controls needed to measure the contributions of each defense.

All conditions retain strict sequential parameter handoff, ten nodes, five full
local epochs, fixed ring order, memory size five, seed-keyed batches,
augmentations, attacks and link noise. The Byzantine defense is receiver-local
CE snapshot selection. With it disabled, the immediate predecessor supplies
the next model. Four seeded nodes use the existing delayed hidden parameter
attack. No optimizer or raw examples are communicated.

## Explicit common learning recipe

The original 117,706-parameter CNN and its settings remain available in their
original protocols. This separate development recipe uses a CIFAR residual CNN
with GroupNorm, batch 128, SGD momentum 0.9 reset per activation, cosine round
decay from 0.1 to 0.001, and coupled kernel-only L2 coefficient 0.0005. GroupNorm
has trainable scale/offset parameters and no untransmitted running statistics.
The common recipe applies to every condition, including untreated controls.
Neither logit prior adjustment nor a snapshot anchor is enabled in this study.
These recipe changes must be disclosed when comparing to the paper reproduction.

For ordinary minibatch CE `F(theta)` and absolute channel standard deviation
`sigma`, the full EBM component is

```
F_ebm(theta) = F(theta) + sigma^2 * ||grad F(theta)||^2
grad F_ebm = grad F + 2 * sigma^2 * Hessian(F) * grad F
```

The noise mitigation uses that full derivative, rather than scaling an SGD
gradient. Kernel L2 is common training regularization, separate from the EBM
coefficient. With noise mitigation disabled, its contribution is exactly zero.

The CART ablation retains receiver-local class-loss memory and its existing
0.1 correction/base-CE gradient norm cap. It does not silently alter historical
CART hyperparameters or claim guaranteed robustness. A new CART mechanism must
receive a separate version and paired validation before promotion.

## Data and aggregation

Only the CIFAR TRAIN cache is read. A fixed, stratified 100-per-class TRAIN
holdout leaves 49,000 fitting examples. The official 10,000-image TEST set is
not loaded by development, noise screening, memory admission, or optimization.

Let `a[p,c,s,sigma,r,n]` be the holdout accuracy for partition `p`, condition
`c`, seed `s`, noise `sigma`, round `r`, and node `n`. Report round curves as

```
seed_curve[s, r] = mean over the ten nodes of a[p,c,s,sigma,r,n]
mean_curve[r]    = mean over seeds 2025, 2026, 2027 of seed_curve[s,r]
curve_seed_sd[r] = sample standard deviation over those same three seeds
```

For the planned 100-round comparison, the endpoint is each seed's mean node
accuracy over completed rounds 91–100, then the equal-weight mean and sample
standard deviation of those three seed endpoints. Keep per-seed endpoints and
curves. Never combine different sigmas, partitions, or conditions, never count
nodes/rounds as independent seeds, and never substitute best-round accuracy.
Missing or failed runs prevent the complete-study summary.

Noise screening uses a fixed clean pilot's final node-9 model on a stratified
already-fit TRAIN probe. Three common corruption draws per candidate measure
CE degradation; choose increasing sigmas nearest fixed 5%, 25%, and 75% median
relative CE increases. The candidate grid is
`0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16`.
Defense results and validation/test accuracy do not choose the levels. Levels
are provisional until a completed, adequately learned clean pilot is screened.

## Development stages and acceptance

`scripts/run_factorial_study.py` defaults to a plan without importing TensorFlow.
`smoke` runs two rounds on 640 fitting examples and starts attacks at round index
1 to exercise their branch. Smoke accuracy is not a convergence result.
`pilot` runs ten rounds on the full 49,000 fitting examples. The comparison
definition uses 100 rounds with attacks starting at round index 20.

Local validation must establish finite CE/EBM/CART updates, independent switches,
five full epochs, exact checkpoint recovery, seeded identities, and a useful
clean learning trend. The IID clean target is 90–95% measured accuracy. Recovery
close to clean and a positive paired CART contribution are performance targets,
not imposed rankings or guaranteed outcomes. A short pilot cannot certify them.
The full matrix must not be launched as a validated thesis result until those
performance questions and a final frozen noise/recipe plan are resolved.

Each running config owns one physical GPU. A four-GPU HPC campaign schedules
four independent configs concurrently. It does not parallelize dependent BASIL
node activations or combine VRAM. GPU startup uses a fresh Python process with
CUDA library paths already set. Source, data, config and runtime identities bind
checkpoints; source changes require a fresh output. The terminal monitor clears
on each five-second refresh and exits after completion.

Before any HPC launch, commit the validated source locally, push it, and pull
that commit on HPC. Keep `.papermerge`, checkpoints, data, local environments,
and generated outputs out of commits.
