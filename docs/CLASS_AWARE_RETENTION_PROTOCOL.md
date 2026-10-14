# Class-aware retention diagnostic — separate experimental method

## Why this experiment exists

The completed clean 30-round CE and uniform μ=0.01 anchor controls both
specialized almost completely to each active node's own class. Those saved
results are retained unchanged. This diagnostic tests a different mechanism;
it does not turn a failed baseline into a different algorithm under the same name.

| Setting | Predeclared value |
|---|---|
| Revision / objective | `class_gradient_importance_anchor_v1` / `class_gradient_importance_anchor` |
| Data | Full CIFAR-10: 5,000 training images of class i at Node i |
| Evaluation | Full 10,000-image test set, evaluation only |
| Horizon | 30 full ring rounds; no early stopping or extension |
| Training | Five complete epochs, batch 512, last batch 392, 50 SGD updates/activation |
| Model / optimizer | Existing 117,706-parameter CNN; reset SGD, zero momentum |
| Learning rate | Existing `0.05 / (1 + 0.05 round)` schedule |
| BASIL | Strict full-snapshot handoff, five distinct rolling predecessors, minimum receiver-local CE |
| Anchor strength | μ=0.01, retained from the approved diagnostic |
| New importance rule | Training-path squared minibatch CE gradients, normalized and saturated at 100 |
| Threats / EBM / sharing | No attack, noise, EBM, ACDS, raw-data sharing, or W&B |
| Status | Mechanism diagnostic, `researchValid=false`; not a production queue |

## Source basis versus our adaptation

[Kirkpatrick et al., EWC, Section 2, Equation (3)](https://arxiv.org/html/1612.00796v2)
uses a diagonal Fisher-weighted quadratic penalty around previously learned
parameters. That motivates protecting important coordinates differently from
an ordinary uniform proximal penalty.

**This implementation is not exact EWC and does not estimate the true Fisher.**
It uses squared gradients of minibatch-mean CE along the training trajectory,
not averaged squared per-example score gradients at a fixed optimum. It also
anchors to the BASIL-selected received parameters, not separate class optima.
Importance normalization, saturation, class aggregation and transport are our
explicit experimental adaptations. No novelty or guaranteed retention claim is made.

## Exact objective

For each class c carried by a selected snapshot, retain the latest diagonal
summary from that class's local activation:

```text
m[c,q] = sum_batches(batch_size * g_batch[q]^2) / (5 * local_sample_count)
g_batch = gradient of ordinary CE, before the SGD update
I[c,q] = min(100, m[c,q] / mean_over_all_parameter_coordinates(m[c]))
```

If the entire moment vector is zero, its importance is zero. Normalization is
over all 117,706 coordinates, not independently per layer. Neither μ nor the
cap is selected using test accuracy. The cap saturates the **importance
estimate**, not the applied gradient; no gradient clipping is enabled.

For active class i, use the selected snapshot's other-class summaries:

```text
Omega[q] = 1 + mean_c(I[c,q]), c in received_classes excluding i
Omega[q] = 1 if no other class is present

L = CE + (mu/2) sum_q Omega[q] * (theta[q] - theta_selected[q])^2
gradient L = gradient CE + mu * Omega * (theta - theta_selected)
```

The unit baseline preserves the previous uniform anchor. Omega is detached,
frozen for the entire activation, and lies between 1 and 101. The selected
weights are loaded completely and frozen as the reference for all five epochs.
No averaging, optimizer-slot transmission or cross-node image access is added.
There is no EBM coefficient or legacy lambda in this clean diagnostic.

## Data and state flow

```text
five predecessor snapshots + their class-importance summaries
                         |
       lowest CE on the SAME receiver-local training batch
                         |
       select ONE snapshot; do not consult its class coverage
                         |
        load complete parameters + matching importance state
                         |
   freeze selected parameters and other-class importance weights
                         |
      five full local epochs, gather ordinary CE gradient moments
                         |
 replace this class's previous summary; preserve other class summaries
                         |
     send parameters + summaries to the next five receiver memories
                         |
                 only the next node trains
```

The latest summary replaces the same class's previous summary; repeated
activations do not continually add the same class's importance. A snapshot
can lack some classes. Coverage is recorded honestly and **never changes
BASIL's minimum-local-loss selection**. Importance is an auxiliary communicated
quantity: this is an extension beyond the original weights-only baseline.
It may reveal information about training gradients and has no formal privacy guarantee.

## Checkpoint and numerical contracts

Activation-boundary checkpoints contain parameters, all rolling memories,
class summaries, class IDs, sample-visit counts, measured metrics and the
activation cursor. Summaries are stored as deduplicated NumPy arrays, not
large JSON lists. Their hashes, dimensions, bounds and source implementation
are checked during restore. Missing summaries cannot be treated as exact resume.
An interrupted activation is discarded and deterministically rerun from its
last committed boundary; no completed activation is lost.

Finite checks remain active for logits, CE, importance, its accumulators,
penalties, gradients, parameters and evaluation. The existing single-process
GPU lease, 6-GiB allocator cap and memory guard remain active. CUDA caches are
released when the diagnostic process exits; no additional model ensemble is loaded.

## Execution and interpretation

```bash
python scripts/prepare_class_aware_retention.py --prepare
python scripts/run_noniid_basil_study.py --execute --config gui/configs/nonIID/class_aware_retention/clean_class_gradient_importance_anchor_30r_gpu.json
python scripts/report_class_aware_retention.py
```

The second command requires current synthetic GPU verification. The reporting
command reads saved results only. Output is separately routed to
`newResults/nonIID/class_aware_retention/` and `newPlots/nonIID/class_aware_retention/`.
Compare to both preserved 30-round controls, including global/own/other-class
accuracy, forgetting, confusion matrices, importance coverage and gradient
contribution ratios. Test measurements describe behavior; they do not select
coefficients, stopping time or a winning hyperparameter.

Do not add attacks/noise or prepare a 500-round queue just because mathematical
contracts pass. Importance metadata is not Byzantine-authenticated; a future
threat study requires an explicit trust/corruption policy. This clean study
cannot establish Byzantine robustness or show that the method solves forgetting.

## Verification performed before this diagnostic

```bash
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests

DISPLAY= PAPERMERGE_LIVE_TRACKING=disabled PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 environment/basil-noise-env/bin/python -m pytest -q tests/test_class_aware_retention.py tests/test_retention_diagnostic.py tests/test_noniid_study.py tests/test_iid_study.py tests/test_research_protocol.py tests/test_protocol_recovery.py tests/test_iid_500r.py tests/test_memory_safety.py tests/test_research_audit.py tests/test_iid_gpu.py tests/test_iid_campaign.py

PAPERMERGE_LIVE_TRACKING=disabled WANDB_MODE=disabled environment/basil-noise-env/bin/python scripts/verify_iid_gpu.py
PAPERMERGE_LIVE_TRACKING=disabled WANDB_MODE=disabled environment/basil-noise-env/bin/python scripts/verify_class_aware_gpu.py
```

| Check | Observed result |
|---|---|
| Compileall | Passed |
| Focused regression suite | 197 passed, 1 display-dependent skip, 2 dependency warnings; 25.70 seconds |
| Retention-specific subset | 34 passed, 2 dependency warnings; 8.30 seconds (included in the focused suite, not additional) |
| Weighted-anchor mathematics | Analytic loss/gradient agreement; nonuniform importance differs from uniform anchoring |
| State recovery | Synthetic interrupted ring exactly matches uninterrupted parameters, metrics and selected-state hashes |
| Legacy lambda | Values 1 and 999999 give identical loss, gradients and update |
| Class-aware CPU/GPU gradients | Relative L2 difference 1.8028288e-7; declared tolerance 0.01 |
| Class-aware CPU/GPU SGD update | Maximum parameter difference 7.4505806e-9; declared tolerance 1e-4 |
| Frozen reference / batch sizes | Verified; synthetic batches 512 and 392 |
| Synthetic class-aware GPU peak | 286,424,576 bytes, below the 6-GiB cap |

GPU checks ran on the actual RTX 4070 Ti. They establish synthetic numerical
contracts, not identical CPU/GPU trajectories or full training convergence.
Multiprocessing remains unverified and is not used. The known legacy full-suite
collection problem was not repaired or claimed fixed by this focused test run.

## Completed result and decision

The separate GPU diagnostic completed **30 rounds, 300 activations and 15,000
SGD updates** in 1,577.8858 seconds (26.3 minutes). All 31 saved-result checks
passed, including incoming transmission hashes, matching importance-state
hashes, checkpoint restoration of every memory, full epochs, fixed LR decay,
the 117,706-parameter count and CSV/NPZ agreement.

| Clean method | Final mean test accuracy | Own class | Other classes |
|---|---:|---:|---:|
| Preserved CE control | 10.0050% | 99.9800% | 0.0078% |
| Preserved uniform anchor | 10.0000% | 100.0000% | 0.0000% |
| New class-aware importance anchor | 10.0000% | 100.0000% | 0.0000% |

The mean squared parameter drift was 0.329492 versus 1.071725 under uniform
anchoring, but no other-class predictions were retained. The correction/CE
gradient norm ratio averaged 1.374759, so failure was not caused by an absent
regularizer. All final node confusion matrices showed constant own-class
prediction. BASIL chose the fifth stored predecessor in 294/300 decisions,
forming narrow opposite-node model histories in the late rounds; the
receiver-local minimum-loss rule remained correct. Class-importance metadata
alone does not establish that the referenced model still knows those classes.

**NOT READY FOR A 500-ROUND NON-IID STUDY WITH THIS METHOD.** No threat matrix
or larger queue was started. No coefficient, epoch count or selection rule
was changed after observing this negative result.

The final focused suite passed **198 tests**, with **1 display-dependent
skip** and 2 dependency warnings in 24.81 seconds. Compileall passed. Synthetic
GPU checks passed; the completed run recorded no NaN/Inf failure, OOM or
memory-pressure event and peaked at 276.26 MiB in the GPU allocator. This does
not guarantee future OOM immunity or verify multiprocessing.

SHA-256 verification matched **all 23,271 preserved files**: 20,247 IID/paper
files plus 3,024 existing non-IID result/plot files, including the completed
clean controls. Changed files: **0**. The newly generated diagnostic has 48
per-method plots and 6 comparison plots. Existing figures were not regenerated.

[Full three-control report, tables, confusion matrices and source plots](../newResults/nonIID/class_aware_retention/comparison_30r/CLASS_AWARE_RETENTION_REVIEW.md)
