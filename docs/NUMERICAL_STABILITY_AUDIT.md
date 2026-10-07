# Research numerical stability audit

Verdict: **NOT READY FOR PRODUCTION**. CPU evidence only. No production run
was launched, and no protocol hyperparameter was changed to rescue accuracy.

## Evidence and indexing

Original execution logs and results were read and preserved. New observations
are under `experiments/professor_validation_results/`, not the historical trees.
Rounds, nodes and batches are zero-based; local epochs are numbered 1–5.
Smoke runs use three rounds, five complete epochs, 512 training images/class,
100 evaluation images/class and smoke-only attack start 1. They are not research-valid.

`basil_core/research_audit.py` records batch/augmentation/parameter/gradient
hashes, finite tensor ranges, optimizer scalars, before/after parameters,
epoch confusion matrices and outgoing-link statistics. A failing batch is
saved before replay. TensorFlow graph dumps isolate executed non-finite
arithmetic; full tensor dumps retain the offending values. Replay does not
continue training or choose a new hyperparameter.

Historical logs cannot identify their original first arithmetic operation:
they did not retain batch checkpoints. The following locations are measured
in new reproductions, not invented reconstructions of those old executions.
Graph instrumentation can change fusion/scheduling; operation claims refer
to the saved-batch replay and are corroborated by layer-level eager probes.

## First non-finite operations

| Condition / evidence | Round | Node | Epoch | Batch / stage | First arithmetic failure |
|---|---:|---:|---:|---|---|
| relative 0.4 full EBM, oneDNN reproduction | 2 | 8 | 1 | training batch 0 | `Square` of first convolution's CE gradient: 27 of 432 coordinates overflow |
| relative 0.6 full EBM, final serial CPU | 1 | 5 | 2 | evaluation batch 0 after epoch | output-dense `MatMul`, shape `[512,10]`: 512 positive and 512 negative infinities |
| relative 0.6 legacy, final serial CPU | 1 | 1 | 1 | evaluation batch 0 after epoch | output-dense `MatMul`, shape `[512,10]`: 512 positive and 512 negative infinities |
| relative 0.6 no EBM, final serial CPU | 2 | 1 | 1 | evaluation batch 0 after epoch | 192-unit dense `MatMul`, shape `[512,192]`: 29,184 positive and 34,304 negative infinities |

The relative-0.4 checkpoint is in
`deterministic/direct_rel_04/trace/failure_batch.npz`; its full-tensor replay is
in `deterministic/direct_rel_04/first_tensor_replay/`. CE is approximately
`1.45176e24`, coordinate variance `9.886605e6`, and the first CE gradient has
finite maximum `2.98622e19`. Squaring that exceeds float32's `3.40282e38`
maximum. The gradient-norm penalty becomes infinite; its outer gradient then
contains NaNs. After runtime/precision corrections, this condition completes
in the targeted reference runs, but residual reproducibility is unresolved.

For the three final relative-0.6 cases, use each condition's
`final_cpu/smoke/<condition>_direct/first_tensor_replay/`. The files
`first_nonfinite_tensor.npy`, `first_nonfinite_tensor_statistics.json` and
`graph_nonfinite_operations.json` provide values, indices and operation names.
For example, no-EBM's finite dense-product entries range from `-7.31585e37`
to `2.50218e37` while other entries already overflow. Its final logits subsequently
contain 4,608 NaNs and 512 positive infinities. CE is not the initial failure.

## Arithmetic preceding the failing forward pass

| Mode at relative 0.6 | CE before update | Base gradient L2 | Update gradient L2 | Parameter L2 before → after |
|---|---:|---:|---:|---|
| none | `3.75859e15` | `4.89004e13` | `4.89004e13` | `1.03167e5 → 2.22274e12` |
| legacy scaling | `5.67487e10` | `4.79438e9` | `4.79438e10` | `1.06430e4 → 2.28304e9` |
| full objective | `4.53691e8` | `3.21705e7` | `1.53479e15` | `9.64627e2 → 7.30854e13` |

All these gradients and updated parameters are finite at this step. Their
next forward products exceed float32 range. SGD uses the unchanged round
schedule: `0.047619...` in round 1, `0.045454...` in round 2.
Full EBM's coefficient is `2.8459206`, gradient-norm squared approximately
`1.03494e15`, and complete objective approximately `2.94537e15`.
The correction is genuinely much larger than ordinary CE's gradient here.
Legacy's coefficient is `25 * 0.6² = 9`, i.e. multiplier 10, not a Hessian term.

Authoritative no-EBM/legacy pre-update probes are in `actual_update_replay/`.
Early instrumentation incorrectly computed a counterfactual full-objective
probe for those two modes. That diagnostic bug was fixed, mode-specific tests
were added, and those earlier probes must not be interpreted as executed
training. The original optimizer traces already distinguish the actual modes.

A four-thread relative-0.6/full-EBM reproduction instead first overflowed
gradient squaring at round 1/node 4/epoch 1. Failure location is runtime-sensitive;
there is no justified universal claim that EBM is the sole failing mechanism.

## Relative-noise norm amplification

For independent zero-mean coordinate noise with
`s = sigma_rel * ||w|| / sqrt(d)`,

`E[||w + epsilon||² | w] = (1 + sigma_rel²) * ||w||²`.

Thirty channel-only immediate-predecessor transmissions, with no SGD or attack,
were simulated for 20 independent noise streams using the initialized CNN:

| Relative budget | Predicted RMS norm factor | Observed mean norm factor |
|---:|---:|---:|
| 0.2 | 1.80094 | 1.80209 |
| 0.4 | 9.26552 | 9.27144 |
| 0.6 | 100.71256 | 100.75019 |

Thus relative calibration is not a bound on cumulative parameter magnitude.
The actual huge SGD jumps above greatly exceed channel-only growth: channel
amplification and unstable local optimization interact. Full EBM adds another
feedback path because its coordinate variance grows with the current norm.
The five cache deliveries are independent links, not five extra activations.

## Confirmed software fixes

- NumPy model/noise norms now accumulate squares in float64. Finite float32
  weights such as `1e20` previously generated infinite norm metadata/noise.
- Relative EBM variance accumulates in float64 before a stopped float32
  coefficient is produced. A tested `1e19` parameter example now retains its
  finite `4e36` coefficient instead of overflowing an intermediate sum.
- Evaluation rejects non-finite logits instead of taking `argmax` and reporting
  an apparently valid accuracy from invalid parameters/activations.
- The research CLI selects deterministic operations, oneDNN-off by default,
  and a single CPU lane. These reduce sensitivity, but do not completely fix
  multiprocessing reproducibility; see the separate audit.

Training weights, logits and gradients remain float32. There is no clipping,
weight decay, momentum, epoch reduction, noise reduction or test-set adaptation.
Historical engines and artifacts are not reinterpreted by these changes.

## Noise and objective verification

Both channels remain distinct: paper absolute noise uses coordinate std
`sigma_e`; relative noise uses `sigma_rel * ||w|| / sqrt(d)`. Calibration at
`d=117706`, initial norm `15.0793830` gives relative realized ratios
`0.200003 / 0.400073 / 0.599158`. Absolute `0.005 / 0.01 / 0.02` produce
ratios approximately `0.114 / 0.227 / 0.455`, not percentages equal to their sigmas.
The calibration reports are in `docs/validation_noise_calibration/`.

Finite-difference and explicit Hessian tests verify
`grad(CE + c||grad CE||²) = grad CE + 2c H_CE grad CE`, with stopped coordinate
variance. Legacy scaling is tested separately. Source Eq. 14 approximates a
squared-loss quantity; our unsquared ten-class CE objective is an adaptation
of the proposed noise-aware principle, not that squared-loss expression or a
reproduction of the source's convex SVM convergence guarantee. Source Eq. 23's
scalar simplification is not a general identity for this CNN.

## Targeted final smoke outcomes

`python scripts/run_research_validation.py --suite smoke --output experiments/research_validation_results/new_cpu_check`
completed **9 of 12**, with **3 numerical failures** and exit 1. Relative 0.4
full EBM, relative 0.4 none/legacy, relative 0.2 full EBM, absolute 0.02 full
EBM, clean one-class, hidden+SS, and the two relative-0.4 hidden+SS comparisons
completed. Each completed condition ended with average/worst/best accuracy
10%. The three relative-0.6 noise-only variants failed; no final scores are fabricated.
