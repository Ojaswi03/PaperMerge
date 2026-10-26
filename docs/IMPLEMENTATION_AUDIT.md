# Implementation audit

Date: 26 October 2026. This review covers the first-party Python source,
experiment configurations, terminal/GUI execution paths and deployment scripts.
It separates software correctness, the objective actually optimized, and
empirical learning behavior. A passing suite is not proof of convergence,
novelty or arbitrary Byzantine robustness.

## Inspection coverage

The initial inventory parsed all 317 first-party Python files (64161 lines),
including research kernels, historical engines, runners, reporting, GUI and
tests. All parsed successfully. A symbol-table check found one apparent missing
name, `tonic` in N-MNIST; manual inspection confirmed that the optional loader
installs that binding through `globals()` before use. It is not an undefined
name failure. No unguarded module-level experiment/training calls were found
by the targeted entry-point scan.

All 1952 tracked JSON configurations/state files in the source/configuration
trees parsed successfully. All four repository shell scripts passed `bash -n`.
The publication-policy check passed. None of these checks launches a study.

Manual numerical and data-flow review focused on strict BASIL selection and
handoff, keyed RNGs, absolute versus relative channel variance, local-prior
adjustment, full EBM/Hessian gradients, CART variants, helper trust assumptions,
activation-boundary checkpoints, physical GPU identity, terminal supervision,
queue serialization, progress reductions and measured-result reporting.
Legacy adapters remain separate from the strict research implementation.

## Findings and changes

| Finding | Result |
|---|---|
| Standalone TensorFlow EBM helper rejected Keras 3 variables through explicit `tape.watch` | Reproduced the exception; use automatic trainable-variable watching. Tests check the loss and full second-order gradient against a tiny explicit Hessian. |
| Legacy WCM exception handler announced ordinary SGD fallback but skipped the batch | Restore pre-attempt parameters and perform the promised SGD update exactly once. A test deliberately moves parameters and fails inside WCM, then verifies equality with ordinary SGD. |
| Queue loader silently discarded non-object entries | Reject the entire malformed queue, identify the entry, and leave the original persistence file unchanged. Valid legacy/enriched formats remain supported. |
| Parallel terminal queue ETA was always unavailable | Use the slowest measured active ETA when all unfinished conditions are running. Retain unknown ETA when pending work or invalid estimates prevent an honest calculation. |
| Historical Campaign 3 averages microbatch EBM penalties | Preserve historical numerical behavior and correct its description. A numerical counterexample demonstrates that this differs from full-batch EBM. Strict research kernels retain their distinct full-batch objective. |
| Legacy attack module described particular accuracy targets as though implied by its settings | Remove those unsupported claims; attack transformations and strengths are unchanged. |
| Cached CIFAR partitions checked total size but not full disjoint coverage, and non-IID cache identity omitted label order | Both failures reproduced, then fixed after the active execution completed. Cached indices must be integer vectors with complete, disjoint coverage; non-IID keys include the ordered labels' hash. Fresh deterministic partitions retain their numerical allocation rule. |
| Historical registry descriptions implied stronger Byzantine verification than performed | Clarify that receiver-local score checks are consistency heuristics and historical maxima are not current/global knowledge certificates. Numerical registry operations are unchanged. |

The microbatch distinction is mathematical: in general,

\[
\sum_j a_j\|g_j\|^2 \ne \left\|\sum_j a_jg_j\right\|^2.
\]

Consequently, averaging gradients of per-slice EBM objectives cannot be called
an exact full-effective-batch EBM derivative. Historical results retain their
executed sources; this audit does not regenerate them under a different loss.
Likewise, the legacy clipped-gradient scaling branch is explicitly described
as scaling, not as the Hessian correction of the strict EBM objective.

## CART investigation

The initial prediction-memory revision is mathematically operational but has a
weak early signal: in the saved full-data clean diagnostic, the median class
regression weight was about 0.0000461. Multiplying it by a small prediction
discrepancy further suppresses the retention correction. This motivates a
separate candidate; it does not justify changing old results or tuning on TEST.

The new class-loss barrier candidate uses locally verified best class CE b_ic:

\[
R_i=\frac1{|C_i|}\sum_{c\in C_i}
 [\ell_{P_{ic}}(\theta)-b_{ic}-0.001]_+.
\]

The complete objective remains prior-adjusted local CE plus its optional EBM
term, the fixed selected-snapshot anchor, and beta times this new retention
term. Beta is fixed at one for this candidate; supported classes have equal
total weight. Targets are detached. The applied retention derivative is the
mean of the current class CE gradients whose loss ceilings are violated;
there is no second multiplication by the small normalized regression weight.
The original KL revision is retained separately.

`scripts/run_cart_train_validation.py` predeclares ten rounds each for prior
only, original KL retention, and loss-barrier retention. A keyed 100-example
per-class holdout is reserved from the original TRAIN set before partitioning:
49000 examples enter SGD and 1000 disjoint TRAIN examples enter validation.
The original TEST files are not opened. All three arms share data, partition,
initialization, five complete epochs, partial batches, learning-rate schedule
and anchor. Coefficients and horizon remain fixed during execution. These are
exploratory validation results, not an unchanged 50000-example reproduction or
a production queue approval. A bounded, separate learning comparison is needed
before any longer run is justified.

The registry and new losses are proposed integrations, not established novel
primitives or robustness guarantees. Historical teacher performance is local
training evidence, not a certificate of generalization or sender honesty.

The completed ten-round clean validation produced the following mean/worst
held-out TRAIN accuracies: prior only 26.85%/16.90%, KL retention 26.30%/14.60%,
and the unbounded loss barrier 24.14%/20.30%. The barrier improved the worst node
by 3.4 percentage points while reducing the mean by 2.71 points. Its retention
gradient averaged about 1.90 times the base-loss gradient. This is a measured
stability/plasticity tradeoff, not an overall improvement claim.

A separate budgeted follow-up fixes the auxiliary correction limit at 10%:

\[
a_t=\operatorname{stopgrad}\!\left(\min\!\left\{1,
 \frac{0.1\|\nabla F_i\|}{\|\nabla R_i\|+10^{-12}}\right\}\right),
\quad d_t=\nabla F_i+2cH_i\nabla F_i+
\mu(\theta-\theta_{sel})+\beta a_t\nabla R_i.
\]

This scales only the retention correction, not CE, EBM or the anchor. The scale
is detached and fixed for each update. It is an adaptive update policy, not a
claim to differentiate one globally fixed scalar objective with the scale's
derivative omitted. Telemetry reports both the raw and applied correction and
the effective coefficient. Unit tests and CPU/GPU checks enforce the bound.
The coefficient is fixed before this follow-up's validation outcomes are read;
there is no hyperparameter grid or official TEST evaluation.

The follow-up profiles are a ten-round clean prior/budgeted pair and fixed
five-round paired B/C/D stress comparisons, with the same TRAIN holdout. The
stress profile shortens attack onset to round 1 solely to exercise the branch;
it is not a reproduction of the original round-20 production onset.

After the cache-integrity correction, the repeated prior-only control reproduced
all five saved metric arrays of the earlier ten-round control exactly. Historical
source archives and completed results remain intact.

The completed budgeted clean pair reached 27.97% mean and 18.60% worst-node
held-out TRAIN accuracy, versus 26.85% and 16.90% for its prior-only control.
The differences are +1.12 and +1.70 percentage points at the predeclared final
round. Earlier rounds do not consistently favor retention. The applied
correction/base-gradient ratio averaged 0.09883, within the fixed 0.1 cap.
Both conditions completed 5050 SGD updates, with identical initial model,
partition and augmentation keys. This single-seed result motivates replication;
it does not establish a significant gain or meet an 80–90% accuracy target.

The budgeted worker also passed exact interrupted/resumed ring comparisons for
C and D on a small deterministic model. The interruption occurs after attack
onset with populated teacher memory and noise history. Full activation telemetry,
final parameters, selected senders and accuracy arrays match uninterrupted work
exactly. Saved-result review requires every predeclared arm, verifies source
archives, model hashes and checkpoint telemetry, and rejects incomplete comparisons
even if a summary labels itself completed.

All six fixed five-round stress arms completed. Their final held-out TRAIN
means were:

| Condition | Prior-only control | Budgeted CART | Difference |
|---|---:|---:|---:|
| B: attack, clean channel | 10.00% | 11.62% | +1.62 pp |
| C: attack and channel noise | 11.54% | 10.81% | -0.73 pp |
| D: attack, channel noise and EBM | 11.52% | 10.83% | -0.69 pp |

Every stress arm's final worst-node accuracy was 10.00%. Each completed 2525
SGD updates and 50 activations; source archives, full model/checkpoint hashes,
teacher-state hashes, complete epoch coverage and attack/noise/EBM telemetry
were verified. The applied correction stayed below the fixed cap. Peak tracked
TensorFlow allocation was below 413 MiB; this is not total driver-reserved VRAM.

These early-onset stress results are near chance and do not support a robustness
or overall improvement claim. They also do not measure the original round-20
attack schedule. Do not promote this candidate to a long research queue on the
strength of the clean pair alone. The next scientific comparison must be
predeclared, preserve the original schedule where reproduction is claimed, and
replicate across seeds before a method-level conclusion. Official TEST remains
unused throughout these comparisons.

Local evidence is isolated in ignored directories:
`cart-train-validation-20261026` (three-arm clean comparison),
`cart-budgeted-clean-20261026` (capped clean pair), and
`cart-budgeted-stress-20261026` (six stress arms), all beneath
`.papermerge/local-tests/`. Each contains `review.json`, `REVIEW.md`, a validation
trend plot, per-condition sources, metrics, checkpoints and telemetry. These
generated artifacts are not committed.

## Limits and deployment

Runtime tests cover explicit contracts and synthetic/controlled paths. They do
not execute every possible input or launch every historical calibration script.
Display-dependent GUI checks and optional real N-MNIST/Tonic loading require
their own environment. Historical tuning scripts should not be confused with
the fixed scientific profiles or used to select hyperparameters from TEST.

Before each new long experiment, review the code/configuration differences,
their effect on the stated formulas and information boundaries, and the
predeclared comparisons. Fix concrete faults and run affected tests before
starting work. Changes to shared numerical/execution paths warrant the broader
regression suite. Freeze and record the reviewed source/configuration before
reading new learning outcomes; do not modify a running study to obtain a desired
accuracy trend.

The CART TRAIN-validation runner now performs read-only input preflight before
TensorFlow import, GPU lease acquisition or output creation. The same checks
are available through `--preflight`: complete float32 normalized finite TRAIN
images, int32 balanced original labels, disjoint holdout, exact partition
coverage and nonempty nodes, fixed configuration, source/data recovery identity,
and optimizer workload. A missing/corrupt cache or mismatched recovery identity
blocks execution before numerical GPU verification. Preflight never downloads
data or deletes caches/results. It does not replace code review, independent
gradient tests, real device verification or empirical learning checks.

The actual local cache passed this preflight in approximately one second with
TensorFlow absent from the process and no output directory created. The existing
ten-round paired clean profile schedules 5050 SGD updates per arm. The separate
30-round, three-seed profile is now implemented and reviewed as described below;
the earlier smoke profiles retain their original definitions and evidence.

The repository still tracks 49 legacy virtual-environment files. Their existing
local modifications are preserved; this audit does not delete the home setup or
stage environment files. HPC deployment uses its isolated requirements and
repository setup script rather than that copied environment.

The existing local environment's `pip check` reports ten PyTorch/CUDA wheel-pin
conflicts (Torch 2.6.0+cu124 versus the TensorFlow runtime wheels). No packages
were replaced during this audit. This is a real local environment limitation;
successful TensorFlow execution does not make the optional PyTorch environment
consistent. The isolated HPC requirements are separate from this legacy setup.

Before the startup-preflight addition, the inventory parsed 325 Python files
(65072 lines). The full
repository run after numerical/cache fixes passed 1054 tests and 644 subtests,
with four skipped tests and three dependency/model-input warnings. Focused
queue-persistence and ETA checks also passed. After the final runtime-identity
metadata change, all 16 CART barrier/validation tests passed, including the new
hardware-identity contract. The final real RTX 4070 Ti CPU/GPU gradient check
passed with the 6144 MiB cap and started zero research runs. The skipped/optional
paths are not described as runtime-verified by these results.

After adding startup preflight, the refreshed inventory parsed 326 Python files
(65242 lines). The broader CPU run passed 1070 tests and 644 subtests, skipped
four GUI-dependent checks, and caught one import-policy violation in the new
test file. NumPy was moved into the test functions; all 104 affected preflight,
lazy-import, CART-gradient, data-isolation and checkpoint tests then passed.
The actual CLI preflight and changed-source recovery rejection both completed
with TensorFlow absent from the process. All 879 protected files in the saved
pre-test inventory remained unchanged. No further learning run was launched.

Changed scientific sources intentionally invalidate old execution approvals.
Completed artifacts remain readable; starting or resuming new work requires
matching source, data, runtime and checkpoint evidence. This audit and local
GPU tests do not verify the new CART revision on Ohio University's The Machine.
No commit, push or university-server execution is automatic.

The diagnostic runner now records GPU UUID/name/driver, executable, validated
resource profile, memory cap, thread settings, TF32 and deterministic arithmetic
settings in its recovery identity. Live utilization and free memory are excluded
because they change during normal execution. This metadata tightening occurred
after the completed comparisons; their source archives remain unchanged and
readable, but the revised runner correctly refuses to claim exact recovery of
those older source/runtime contracts. Its final synthetic CPU/GPU gradient check
does not relabel older learning results as executions of the new source.

## Fixed 30-round replication package

The [HPC runbook](CART_30R_HPC_REPLICATION.md) specifies seeds 2025/2026/2027,
24 control/CART arms, the unchanged original attack start index 20, a fixed
TRAIN holdout and a predeclared mean-accuracy difference over indices 20–29.
The full experiment seed now reaches partitioning, initialization, attacker
selection, keyed data order/augmentation/noise and receiver memory. Holdout seed
2025 remains fixed. Earlier profiles remain restricted to their original seed.

The opt-in supervisor assigns a complete pair to each physical GPU, validates
per-GPU source/software/UUID/resource receipts, freezes the source and data
contract before launch, and rejects changed inputs in a later child before GPU
initialization. Explicit child profiles override inherited profile environment.
Initialization and partition hashes are checked by the actual engine before
updates. Four-slot scheduling, failure-induced graceful stopping, monitoring
and complete-evidence relocation are covered by tests. Resume requires the same
source, data and runtime, and completed arms are skipped.

Final local validation parsed 329 Python files (66126 lines). The full CPU
suite passed **1091 tests and 644 subtests**, with four GUI-dependent skips and
three dependency/model-input warnings. The actual RTX 4070 Ti passed CPU/GPU
gradient checks and identical CPU/GPU initialization checks for all three seeds;
the initial models differ across seeds. The 6144 MiB local cap was preserved.

The real-CIFAR seed-2026 D pilot checked both control and capped CART. It stopped
after three activations, restored the full checkpoint, completed one additional
activation, and matched the complete state of four uninterrupted activations:
model arrays, BASIL memories, CART targets, metrics and telemetry. The final
`PILOT_PASSED` receipt is isolated under
`.papermerge/local-tests/cart-replication-local-pilot-final-20261026/`.
All 25 executed-source hashes still matched the final verification receipt.
The 879 protected historical/user files remained unchanged.

This prepares the package for server acceptance checks; the new path has not
yet been executed on The Machine. Its full 24-arm study has not completed
locally. The server must pass installation/dependency checks, four real device
verifications and its own bounded pilot before the full launch. No new accuracy,
novelty or final thesis conclusion is inferred from software/pilot success.
