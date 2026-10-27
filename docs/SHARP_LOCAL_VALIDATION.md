# SHARP and CART local validation — 27 October 2026

This report separates implementation checks from learning effectiveness.
The [professor's specification](specifications/SHARP_professor_20261027.pdf)
and [implementation/commands](SHARP_IMPLEMENTATION.md) describe the new family.

## Completed implementation evidence

- The focused tests verify the analytic `g + 2 lambda H g + mu(theta-anchor)`
  update against a binary softmax model with an independently computed Hessian.
- A 100-node, 66-sender-window protocol test reproduces uninterrupted parameters,
  telemetry and metrics exactly after saving and recovering a partial run.
- Real TensorFlow CART tests repair a deliberately degraded classifier for
  **both 10 and 100 classes**, preserve the proposal backbone exactly, and
  pass a separate local check. Real multi-round TensorFlow checkpoints reload.
- The tests check channel realization reuse, independent receiver noise, peak
  screening, all-fail fallback, ranking ties, window filling and coefficient
  changes with learning rate and selected count.
- All 432 full-matrix configurations passed TRAIN partition preflight at
  100 nodes. Splits are disjoint and exhaustive and exclude the reporting holdout.
- GPU objective/SGD checks passed on RTX 4070 Ti for MNIST, CIFAR-10 and
  CIFAR-100, including exact same-device repeated updates. The optional
  CIFAR residual model is verified separately in its 100-node pilot.
- The final complete repository suite passed **1,206 tests plus 644 subtests**,
  with four skips and three warnings, in 254 seconds. This includes reporting
  that keeps sigmas separate, a completed monitor that does not refresh, and
  explicit budget-valid or matched-window BASIL comparisons. Local log:
  `/tmp/sharp-full-suite-publication.log`.

The all-finite numerical invariants above are not general Byzantine robustness
or population accuracy guarantees. Peak screening targets concentrated parameter
deviations; broad look-alike corruption may pass. Model initialization, local
training and outbound attack/noise ordering are explicit in the manifests.

## Completed five-round, ten-node learning pilots

These **18 actual GPU runs** use seed 2025, sigma 0.01, five local gradient
steps per activation, 200 fitting samples per class, Dirichlet concentration
0.5, and the paper CNN or documented MNIST MLP. Attack starts at round index 1.
They use a TRAIN holdout, not official TEST. Every final node and class metric
was independently recomputed on CPU from the saved final checkpoint.

| Dataset / split | Clean | SHARP, attack + noise | Same + CART | CART difference |
| --- | ---: | ---: | ---: | ---: |
| MNIST IID | 83.17% | 64.00% | 62.62% | -1.38pp |
| MNIST Dirichlet | 74.92% | 11.86% | 10.51% | -1.35pp |
| CIFAR-10 IID | 10.57% | 10.00% | 10.00% | 0.00pp |
| CIFAR-10 Dirichlet | 10.29% | 10.00% | 10.00% | 0.00pp |
| CIFAR-100 IID | 0.94% | 1.00% | 1.00% | 0.00pp |
| CIFAR-100 Dirichlet | 0.88% | 1.00% | 1.00% | 0.00pp |

These pilots **do not establish a CART accuracy benefit** in the new SHARP
family. CART made nonzero checked corrections in the MNIST and CIFAR-10 runs,
but better local checked CE did not ensure better balanced holdout accuracy.
Near-chance clean CIFAR results also show that these short small-network runs
are insufficient as an effectiveness comparison for that model/training recipe.
They do not justify a claim of closeness to clean or recovery to 90–95%.

The [machine-readable validation receipt](validation/SHARP_INITIAL_5R_20261027.json)
preserves final metrics, correction counts, independent inference differences,
source hashes and dataset identities. Local raw output is
`.papermerge/local-tests/sharp-cart-5r-three-datasets-v4/`.

## Further validation in progress

The next frozen cohort exercises **100 actual logical nodes**, five rounds,
full fitting data and the explicit GroupNorm residual CIFAR model, with clean,
SHARP and SHARP+CART conditions. This retains five steps per activation and
therefore performs 2,500 updates per run, compared with 250 in the ten-node
pilot. It is not pooled with the smaller-model/subset results above.

A separate bounded switch-coverage cohort exercises every requested condition
and all three sigma values on IID and Dirichlet versions of all three datasets.
A three-seed 100-node MNIST cohort follows for a direct paired retention check.
Their completion and results must be read from their frozen run manifests and
added to this report only after validation; no pending value is a result.

## Reproduction and readiness scope

GPU workers are separate processes with their CUDA loader paths and selected
physical-device profile set before startup. Source hashes, TRAIN identities,
partition audits, initialization hash and runtime profile are frozen. Recovery
rejects mismatches. The monitor refreshes every five seconds and exits on
completion. One experiment occupies one GPU, allowing four independent HPC
configurations concurrently.

The commands in [SHARP_IMPLEMENTATION.md](SHARP_IMPLEMENTATION.md) include data
preparation, preflight, per-GPU verification, pilot execution, monitoring and
reporting. GPU verification must be repeated on HPC's Blackwell cards. Local
RTX verification is not a substitute for that hardware check.

The currently supported conclusions concern implementation and the explicitly
reported development experiments. A successful small test does not certify
long-run convergence, arbitrary Byzantine tolerance or CART novelty. Longer
HPC comparisons should be frozen and reported as new experiments, including
negative outcomes, with matched CART/non-CART settings and separate sigma means.
