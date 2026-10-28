# PaperMerge

PaperMerge studies Byzantine-resilient decentralized learning under noisy
communication using Python, TensorFlow and a Tkinter experiment workspace.

**27 October 2026 — professor SHARP specification:** the new
[implementation and HPC commands](docs/SHARP_IMPLEMENTATION.md) support
100-node rings, screened/ranked averaging, hop-aware RLA, a frozen proximal
anchor and an independently switched CART classifier correction. MNIST,
CIFAR-10 and CIFAR-100 have explicit IID/Dirichlet configurations with three
seeds and three noise levels. This family uses **five gradient steps**, not
the historical five-epoch protocol. See the supplied
[algorithm PDF](docs/specifications/SHARP_professor_20261027.pdf) and
[local validation evidence](docs/SHARP_LOCAL_VALIDATION.md). Completed checks
include 144 GPU configuration smoke cases, exact fresh-process CPU/GPU recovery,
and 1,212 tests plus 644 subtests. In the 25-round non-IID MNIST CPU study,
CART improves the configured primary endpoint in all nine seed/noise pairs;
the separate five-round early-attack GPU study worsens in all three seeds.
These bounded results do not establish near-clean or general CART superiority.

A later [35-round noiseless Hidden-attack reproduction](docs/SHARP_SCREEN_FAILURE.md)
identified a peak-screen failure in the IID attack-defense condition. The
read-only checkpoint audit documents actual corrupted-copy admission; the
running HPC training source remains frozen.

A separately versioned [survivor-aware selector correction](docs/SHARP_SURVIVOR_GUARD.md)
now has verified 100-node, 35-round MNIST development results across three seeds:
attack-defense averages 89.62% IID and 82.80% non-IID, below clean controls of
96.70% and 96.11%. The correction is separate from CART. In the matched
one-seed joint-noise pilot, unchanged CART **reduces** final accuracy by 1.96
and 4.37 percentage points respectively. All outcomes, validation receipts and
separate HPC pilot commands are documented; no general CART benefit is claimed.

All archived research PDFs and their source/download records are linked in
the [complete paper index](papers/FIRECRAWL_REFERENCE_INDEX.md).

**CART v2 development revision:** [mechanism, mathematics, evidence and HPC
commands](docs/CART_V2_IMPLEMENTATION.md). It qualifies saved class knowledge
on two local probes, fits only correct-reference/incorrect-proposal targets,
and rejects corrections that reduce a checked class's correct count. The
100-node ten-round MNIST pilot gives final CART effects of +0.042 points IID
and −0.148 points non-IID (one seed, sigma0.01); this establishes implementation
and bounded recovery, not consistent superiority. Use its separate launcher
and fresh outputs for the all-288-combination HPC pilot.

[Managed execution and reporting](docs/CART_EXECUTION_AND_REPORTING.md) add
bounded concurrent slots per GPU, coordinated GPU leases, always-present
labeled runtime forecasts, three-seed line/bar charts and CSVs, and ntfy
completion/failure alerts. The local two-case CIFAR-10 benchmark measured
2.01x throughput with exact matching training state; HPC speedup still needs
its own measurements. The frozen CART training source is unchanged.

[Higher-concurrency execution](docs/CART_HIGH_CONCURRENCY.md) adds a separate
one-to-eight-slot launcher with CUDA-context and host-RAM budgets. Six workers
completed a local CIFAR-10 benchmark 2.24x faster than two, with identical
training state. The guide restores the explicit 100-round final-study target;
HPC throughput and CART efficacy still require their own evidence.

[Failure-isolating execution and recovery](docs/CART_FAILURE_RECOVERY.md)
keeps other configurations running when a worker fails, records failed runs
for later retry, and tolerates temporary monitoring filesystem errors.
It includes the current 100-node/100-round/six-workers-per-GPU commands.

[Compiled-transfer checkpoint pickup](docs/CART_COMPILED_TRANSFER.md) provides
an optional faster model-loading path, guarded by exact per-device comparisons.
It resumes the existing study after a graceful restart and records its runtime
provenance separately from the unchanged mathematical/checkpoint contract.

[Live campaign inspection](docs/CART_LIVE_INSPECTION.md) shows all workers,
manager timestamps, ETA assumptions and CART/control accuracy at matching
completed rounds. It can be pulled during training without restarting workers.

[Modular noise selection](docs/CART_NOISE_SELECTION.md) defaults to one level
at sigma 0.01 and supports two or three explicit levels for future plans.
CART-only previews contain 12, 24 or 36 configurations, respectively.

[CIFAR learning-failure evidence](docs/CIFAR_LEARNING_FAILURE_20261028.md)
separates the HPC pre-attack failure from eight paired local handoff diagnostics.
Historical selection remains weak even without noise or proximal anchoring;
these findings motivate method development rather than a CART superiority claim.

**28 October completed HPC campaign:** [saved 72-run metrics and line/bar plots](docs/CART_HPC_72_RESULTS_20261028.md),
[verified metric/provenance receipt](docs/validation/CART_HPC_EXPORTED_METRICS_20261028.json),
and [continuity candidate, equations and novelty requirements](docs/CART_CONTINUITY_CANDIDATE.md).
CART v2 does not improve the non-IID CIFAR-10 endpoint (10%); its IID CIFAR-10
mean gain is 2.583 points with substantial seed variation. Four additional primary
papers are [downloaded and indexed](papers/cart_continuity_novelty_20261028/README.md).
Compact exports use `scripts/export_cart_campaign.py`; reruns reuse matching files
without replacing old results. New candidate screens remain separate from v2.

## Historical A/B/C/D results — non-IID 100 and 500 rounds

[Full results, ten analysis tables, numerical checks and limitations](newResults/nonIID/500round/FINAL_ABCD_RESULTS_ANALYSIS.md).
All eight A/B/C/D queue entries completed. The primary values below are the
**arithmetic mean of all ten node models' full-test accuracies**, not Node 9 alone.

| Condition | 100-round final mean | 500-round final mean | 500-round worst node |
|---|---:|---:|---:|
| A — clean | 53.97% | 60.17% | 59.90% |
| B — retained Hidden attack | 53.56% | 59.73% | 59.32% |
| C — attack + absolute Gaussian noise | 46.51% | 35.17% | 26.90% |
| D — attack + noise + EBM | 46.89% | 35.83% | 30.93% |

The clean-channel conditions improve with longer training; the noisy conditions
peak early and deteriorate. D's final mean exceeds C by only **0.66 percentage
points**. This does not demonstrate satisfactory long-horizon noise mitigation.
C/D selected 38/19 actually corrupted snapshots at 500 rounds; none had been
selected in the 100-round runs. These are single-seed observations, not universal
robustness or statistical-significance claims. Protocol/numerical checks passed;
that is distinct from scientific effectiveness.

[Mechanism diagnosis and recovery fixes](docs/DIRICHLET_MECHANISM_DIAGNOSIS.md)
trace accumulated channel perturbations along selected ancestry, distinguish
method limitations from implementation defects, and document the repaired
W&B initialization/retry path. Completed evidence is preserved; no new
scientific condition is enabled by these software fixes.

[500-round overall accuracy lines](newPlots/nonIID/500round/ABCD_overall_average_accuracy_vs_round.png)
· [Final mean-accuracy bars](newPlots/nonIID/500round/ABCD_final_average_accuracy_bar.png)
· [Mean/worst-node consistency](newPlots/nonIID/500round/ABCD_mean_and_worst_node_accuracy.png).

## Overall A/B/C/D accuracy plots

The simplified figures use **mean-node overall test accuracy**: the arithmetic
mean of all ten node models' accuracies on the full test set, not a per-class
curve and not the Node-9 metric. They include an A/B/C/D line graph, an endpoint
bar chart and a mean-versus-worst-node consistency figure. No per-class plots
are generated by this report.

```text
newPlots/{IID,nonIID}/{30round,100round,500round}/
newResults/{IID,nonIID}/{30round,100round,500round}/
```

Refresh these read-only views without training or disturbing the queue:

```bash
environment/basil-noise-env/bin/python scripts/report_round_accuracy.py
```

Each results folder contains `round_metrics.csv`, `summary.csv` and A/B/C/D
subfolders with their measured rows and source/configuration references.
The original raw result locations stay unchanged for queue validation and
checkpoint/resume compatibility; large checkpoints are not duplicated.
IID 30/100-round views are explicitly labeled **prefixes of the saved 500-round
IID runs**, not independent experiments. In-progress runs show only committed
rounds, with incomplete/missing conditions clearly labeled. Refresh the report
again after more rounds complete; these are snapshots, not a background watcher.

Published evidence includes compact executed configs/manifests, partition audits,
metric arrays, CSV summaries, overall-accuracy figures and the completed analysis.
Large checkpoints, activation archives, source snapshots, local readiness receipts,
credentials, W&B runtime logs and dependency environments remain local. A fresh
Git checkout is therefore not a complete resume bundle or an independently
replayable checkpoint audit. Do not bypass missing evidence gates to launch runs.

[Current: non-IID 100/500-round queues, validation gate, W&B and work countdowns](docs/NONIID_LONG_HORIZON_RUNS.md).
The requested A/B/C/D longer-horizon profiles keep Dirichlet α=0.2, the same
local-prior loss, frozen μ=0.01 anchor and D-only EBM. The 100-round queue is
local-only; the combined eight-entry queue automatically validates all four
100-round runs before advancing to the 500-round W&B stage. A failed gate
pauses the queue; preparing/loading it
never starts training. Individual and whole-queue counters show rounds,
activations and optimizer batches; time estimates remain explicitly labeled.

[Current non-IID formula and A/B/C/D explained](docs/NONIID_LOSS_AND_ABCD_GUIDE.md):
the complete local-prior-adjusted classification loss, frozen snapshot anchor,
and D-only no-lambda EBM, with a condition table and training-flow diagram.
The guide separates ordinary-CE BASIL Snapshot Selection from local training
and explains the retained Hidden attack and absolute Gaussian channel.

[Latest: Dirichlet rerun readiness, results, formulas and launch instructions](docs/DIRICHLET_RERUN_READINESS.md).
The full-data GPU gates completed. Local-prior training reached **32.90%**
Node-9 accuracy versus **19.13%** for the matched original ten-round control.
At 23 rounds, B/C/D reached **41.29% / 45.04% / 46.11%**, with no actually
attack-corrupted snapshot selected in their three attacked rounds. These are
single-seed diagnostics, not solved retention or universal robustness.

The separate **30-round A/B/C/D GPU confirmation completed**: final Node-9
accuracies were **44.35% / 44.69% / 47.14% / 47.75%**. B/C/D each applied
40 attacked activations; C/D each delivered 1,500 noisy links. No actually
corrupted snapshot was selected. This is evidence about the retained project
attack realization, not universal robustness or a guaranteed EBM benefit.
It used local-prior CE, frozen snapshot anchoring and EBM in D only, without
W&B, sharing, helpers or CART. The subsequent **100-round and 500-round** stages
are now complete; the 500-round stage used W&B after the 100-round validation
passed. See the completed report at the top for the longer-horizon findings.

## Earlier studies and investigation notes

The status statements below describe their historical stages, not the current
completed long-horizon study. Obsolete short diagnostic outputs were removed
in the authorized cleanup; retained 30/100/500-round studies remain separate.
[Initial three-round GPU calibration tests](docs/DIRICHLET_CALIBRATION_SMALL_TESTS.md)
record the local-accuracy reporting correction and earlier bounded evidence.

[Current: Dirichlet α=0.2 BASIL protocol, formulas and thesis attribution](docs/DIRICHLET_BASIL_PROTOCOL_AND_ATTRIBUTION.md).
[Diagnosis: clean forgetting and attack-induced feature collapse](docs/DIRICHLET_LEARNING_FAILURE_ANALYSIS.md).
The read-only audit verifies full epochs and local-loss selection, but finds that
corrupted, contracted models can win under label skew. No remedial method or new
research run was enabled by that read-only diagnosis. The subsequent small test
report above records the live-engine telemetry correction and separate learning
extensions; completed 30-round curves are not rewritten.
All four **30-round, local-data-only GPU** conditions completed. They use
receiver-local Snapshot Selection and fixed μ=0.01 anchoring; C/D add absolute
channel σₑ=0.010, and D adds full no-lambda EBM. Unequal partitions retain five
complete epochs and every partial batch. No helper gradients, TRAIN replication
or W&B were enabled. The queue is now idle and contains no pending runs.

| Condition | Final Node-9 full-test accuracy | Final node mean | Final worst node |
|---|---:|---:|---:|
| A — clean, anchor | 24.34% | 22.79% | 16.53% |
| B — Hidden attack, anchor | 10.00% | 10.00% | 10.00% |
| C — Hidden attack + noise, anchor | 10.00% | 10.00% | 10.00% |
| D — Hidden attack + noise + EBM, anchor | 31.56% | 21.16% | 10.00% |

These are one-seed feasibility results, not sufficient learning performance:
B/C collapsed, and D does not preserve adequate performance across all nodes.
Numerical/protocol correctness is not evidence that the learning problem is
solved. Saved results and plots remain local under
`newResults/nonIID/dirichlet_alpha02_local_anchor_30r_v1/` and the matching
`newPlots/nonIID/` namespace; generated outputs are not part of the source commit.

[Previous, separate one-class verified-CART study](docs/NONIID_CART_30_ROUND_STUDY.md)
retains its evidence and documented replicated-TRAIN/trusted-coordinator
assumptions. It is not the new Dirichlet protocol.

[Latest: full-batch EBM/noise and adversarial-helper gate](docs/CART_FULL_BATCH_THREAT_GATE.md#completed-results).
Two matched five-epoch GPU activations completed: all 50 faulty-helper updates
exactly matched the honest reference. Full/partial-batch EBM and no-lambda checks
passed; no non-finite failure or OOM was observed. **This is not a learning or
Byzantine-filtering success:** the next receiver selected a genuinely corrupted
snapshot because its verified balanced TRAIN loss was lowest. No A/B/C/D or W&B
run was started; full research readiness remains false.

[Latest: completed verified-CART learning diagnostic](docs/CART_VERIFIED_LEARNING_DIAGNOSTIC.md#completed-results).
One full clean traversal completed all **500 updates** with five full epochs per
node. Node-9 accuracy was **13.90%**, node mean **10.80%**, worst node **10.00%**.
The saved evidence passed 2,525 certificate / 22,725 signature checks, with no
observed OOM or non-finite failure. This verifies execution, **not solved retention**:
five classes still had zero final Node-9 recall. No A/B/C/D or W&B run was started.

[Latest: CART resource and recovery gates](docs/CART_RESOURCE_AND_RECOVERY_GATES.md).
Kernel RAM/task/CPU limits are enforced; a full TRAIN replica fit below its 2 GiB
cap without OOM. Explicit trainer-refusal recovery matched an honest CPU step,
and restart remained exact. The bounded GPU-coordinator/CPU-helper check now
passes, including exact same-device checkpoint recovery. Razer Axon was stopped
with permission to free VRAM; the checker also needed an explicitly CPU-placed
reference computation. That gate stage did not start learning; the separate
bounded learning result above is now complete. No A/B/C/D execution, queue
changes or W&B run was started.

[Latest: isolated CART runner and recovery checks](docs/CART_ISOLATED_RUNNER_VALIDATION.md).
CPU sandbox processes now validate helper replies and feed the existing CART/EBM
objective. A restarted synthetic step reproduced the uninterrupted result exactly;
poisoned, forged, withheld and terminated-worker responses were exercised.
This earlier CPU-stage report is superseded for GPU/resource status by the
resource and recovery report above. No learning diagnostic or A/B/C/D run was started.

[Latest implementation: replicated TRAIN verification](docs/CART_REPLICATED_TRAIN_VERIFICATION.md).
Independent signed recomputation replaces trust in claimed helper gradients and
scores in an isolated prototype. It explicitly requires multi-class TRAIN-data
replication, so it is a separate information model. CPU certificate checks pass;
GPU checking was blocked at that earlier stage; the bounded integration check
above has since passed. Full-data learning and long-run resource validation
remain outstanding; no A/B/C/D run was started.

[Current blocker: full-node helper threat audit](docs/CART_FULL_NODE_THREAT_READINESS.md).
Both helper-gradient and selection-score paths now have isolated adversarial
diagnostics. Bounded outlier checks pass, but explicit counterexamples show that
trimming can lose an honest class signal and reverse candidate rankings.
**Full-node A/B/C/D tests remain blocked; trusted-helper execution is not enabled.**

[Latest completed: 30-round clean CART validation](docs/CART_30R_CLEAN_VALIDATION.md).
Both full-data GPU controls completed: balanced-owner TRAIN selection reached
**49.10%** Node-9 / **49.77%** mean-node / **49.07%** worst-node accuracy,
versus **46.71% / 46.19% / 44.98%** with receiver-local selection. Cat/deer
recovered to **28.7% / 29.6%** on Node 9. Five full epochs and fixed μ=0.01
were unchanged; no NaN/Inf or OOM occurred. No W&B, attacks, noise or EBM
were enabled. This is positive clean-learning evidence, **not full A/B/C/D
threat validation**; the helper-channel threat model still needs resolution.

[Earlier: three-round balanced-selection CART diagnostic](docs/CART_BALANCED_SELECTION_DIAGNOSTIC.md).
In a matched three-round, full-data GPU pair with fixed μ=0.01 and five full
local epochs, balanced owner TRAIN selection reached **23.18%** Node-9 accuracy,
**22.55%** mean-node and **17.36%** worst-node, versus **14.25%**, **12.18%** and
**10.38%** with receiver-local selection. All executed-step and pairing checks
passed. This is **early improvement, not a solved protocol**: cat and deer remain
very weak. No attacks, channel noise, EBM, W&B or production queue were enabled.

[CART integration safety and exact objective](docs/CART_OWNER_INTEGRATION.md):
an isolated revision adds balanced training-side candidate evaluation and true
local-CE EBM to owner-gradient supervision. Mathematical/device checks passed;
the separate clean comparison above is complete. Full-node Byzantine helper defense remains
unresolved, so the new A/B/C/D research plan stays blocked and outside the GUI queue.

[Earlier: three-round CART selection comparison](docs/CART_OWNER_SELECTION_DIAGNOSTIC.md#completed-results).
Both paired, full-data GPU controls completed with five local epochs unchanged.
Direct predecessor handoff reached **25.35%** Node-9 full-test accuracy,
**24.02%** mean-node and **16.64%** worst-node, versus **13.50%**, **12.28%**
and **10.47%** with receiver-local BASIL Snapshot Selection. This is an early
improvement, **not a solved non-IID problem**: cat accuracy remained 0.8%.
Both use additional clean owner-gradient supervision, not plain BASIL;
the direct-handoff control disables SS explicitly and is not a Byzantine defense.
[Results, per-class tables and comparison plots](newResults/nonIID/cart_owner_gradients/clean_three_round_selection_pair_v1/OWNER_SELECTION_REVIEW.md).
No W&B, attacks, noise, EBM, production queue or longer run was started.

[Earlier: completed owner-gradient CART diagnostic](docs/CART_OWNER_GRADIENT_CANDIDATE.md#completed-one-traversal-learning-result).
The approved clean GPU traversal completed ten full-data activations and 500
SGD updates in **5.71 minutes**. Node-9 full-test accuracy was **12.50%**, mean-node
**10.45%**, and worst-node **10.00%**. Older-owner gradients now supervise the
encoder as well as the head, but six classes still scored 0% at Node 9: this is
**not a solved retention problem**. The new auxiliary gradient channel is an
isolated CART adaptation, not plain BASIL or a Byzantine-safe/privacy guarantee.
[Measured review and plots](newResults/nonIID/cart_owner_gradients/clean_one_traversal_16_v1/OWNER_GRADIENT_LEARNING_REVIEW.md).
No attack/noise/EBM, W&B, GUI queue or longer campaign was started. The earlier
30-round replay controls remain intact at 10% full-test accuracy.

[CART same-image audit and representation preservation](docs/CART_REPRESENTATION_PRESERVATION.md):
an isolated diagnostic separates encoder and classifier changes on identical
training images. A new opt-in loss preserves the selected encoder's features
on receiver-local batches; its coefficient is explicit, not legacy lambda.
The clean **30-round comparison per control completed** using training-only
calibrated, fixed encoder preservation strength. It stayed isolated from the
GUI queue and W&B. Both controls remained at 10% full-test accuracy;
[measured review](newResults/nonIID/cart_representation/clean_30_rounds_fixed_gamma_v1/REPRESENTATION_PRESERVATION_REVIEW.md)
and [protocol](docs/CART_REPRESENTATION_PRESERVATION.md#authorized-one-traversal-check-and-30-round-comparison)
separate successful execution from unsuccessful class retention.

[Experimental CART feature memory](docs/CART_FEATURE_MEMORY_PROTOCOL.md): an
isolated source-attributed module adds older-class feature supervision with
approved β=1.0, activation-bound provenance and lazy imports. Mathematical
tests passed and both bounded clean GPU controls completed, but both remained
at **10% balanced training-probe accuracy, 100% own-class and 0% other-class**.
Replay was active yet did not improve retention; these are not test-set scores.
No longer queue, W&B run or parameter retuning followed.

[CART literature and attribution review](docs/CART_LITERATURE_AND_ATTRIBUTION.md):
prototype sharing and feature-space rehearsal have established precedents.
Twelve primary papers are [downloaded and indexed](papers/prototype_retention/README.md).
The review precedes the isolated implementation linked above; no successful
retention or novelty claim is implied, and existing queue/results are unchanged.

[Latest: head/backbone forgetting audit](docs/FORGETTING_MECHANISM_AUDIT.md).
Three bounded warmed activations completed on GPU. The final classifier head
switches the constant-class prediction even on the selected backbone; removing
output bias alone does not fix collapse. CART's local retention gap remained
zero despite losing the previous class. This is diagnosis, not a successful
new retention method; no longer campaign, queue or W&B run was started.

[Strict-handoff CART adaptation](docs/STRICT_CART_PROTOCOL.md): the registry,
receiver-local controller and frozen snapshot anchor are separated from the
source-grounded EBM term. A bounded full-data CE/fixed-anchor/CART comparison
completed all three rounds per control on GPU. All three ended at **10% global,
100% own-class and 0% other-class accuracy**: this revision did not solve
one-class forgetting. See the [measured review with tables and figures](newResults/nonIID/strict_cart/clean_full_data_three_round_v1/STRICT_CART_REVIEW.md).
No production queue or W&B run was started.

[Lazy dependency loading](docs/LAZY_IMPORTS.md) now covers the core, GUI,
plotting, scripts and tests. Import-only mathematical equivalence and startup
checks are documented separately from learning results.

[Latest: three-round warmed-memory retention diagnostic](docs/GRADIENT_PROJECTION_WARM_DIAGNOSTIC.md).
Both fresh GPU controls completed. Final mean **training-validation** accuracy
was 10.000% for CE versus 13.204% for gradient projection. Projection retained
some older classes but left five nodes at or below 5% own-class accuracy;
the prespecified warmed-memory screen did not pass. No 30-/500-round run or
W&B run followed. See the guide for measured tables and linked plots.

[New clean gradient-projection retention gate](docs/GRADIENT_PROJECTION_RETENTION.md):
a separately labelled, bias-aware GPM-inspired adaptation with training-only
validation. It does not change existing runs, add averaging, or start a
production queue.

The paired gate completed on GPU. Projection showed limited two-class
retention (end-of-traversal validation accuracy 12.9% versus CE's 10%), but
several nodes failed to learn their own class and the predeclared screen did
not pass. These are held-out training-validation results, not test accuracy.
The later three-round diagnostic above preserves this earlier evidence.

## Completed study: four 50-round one-class BASIL validation runs

[Non-IID protocol, exact anchor/EBM equations and continuation review](docs/NONIID_50_ROUND_PROTOCOL.md)

The IID A/B/C/D runs completed all 500 rounds and remain preserved. The new
one-class study uses a fixed snapshot anchor μ=0.01 in A/B/C/D, with EBM added
only in D. All four completed locally on GPU without W&B. Execution and
mathematical checks passed, but mean test accuracy remained approximately
10%, with almost perfect own-class accuracy and nearly zero other-class
retention. **The learning-trend gate failed: no 500-round non-IID queue was
prepared or started.** See the guide for the measured results and report path.

The follow-up **30-round clean CE-versus-anchor controls also completed**.
Final mean accuracy was 10.005% for CE versus 10.000% for μ=0.01 anchoring;
both specialized to their own class. The anchor reduced parameter drift
slightly but did not preserve class knowledge. See the
[completed control comparison](docs/NONIID_50_ROUND_PROTOCOL.md#completed-30-round-clean-controls).

A separately approved [class-aware retention diagnostic](docs/CLASS_AWARE_RETENTION_PROTOCOL.md)
tests training-only per-class gradient importance with the same μ=0.01 and
unchanged five-epoch protocol. It is not exact EWC, does not use ACDS or test
data for its importance estimates, and remains separate from the original
CE/anchor baselines. No non-IID 500-round queue is prepared automatically.

That diagnostic **completed all 30 rounds** and all 31 saved-result contracts
passed, but it also remained at 10% mean test accuracy, 100% own-class and 0%
other-class accuracy. Its stronger parameter-wise anchor did not preserve
class predictions. See the [three-control review with tables and plots](newResults/nonIID/class_aware_retention/comparison_30r/CLASS_AWARE_RETENTION_REVIEW.md).

## Completed study: four paired 500-round IID BASIL runs

[Protocol, exact equations, GPU safety and live W&B setup](docs/IID_500_ROUND_PROTOCOL.md)

All four saved IID conditions are completed at **500/500**. Earlier
interrupted attempts are preserved in an ignored local backup.

| Condition | Attack | Channel | Local objective |
|---|---|---|---|
| A | None | Clean | Cross-entropy |
| B | Delayed Hidden, R20 | Clean | Cross-entropy |
| C | Same Hidden, R20 | Absolute Gaussian, σₑ=0.010 | Cross-entropy |
| D | Same Hidden, R20 | Same Gaussian | CE + 0.0001 ‖∇CE‖² |

All four use `sequential_basil_iid_v1`, ten IID nodes, the approved
117,706-parameter CNN, five complete local epochs, batch 512, reset SGD,
initial LR 0.05 with unchanged round decay, BASIL memory S=5, receiver-local
Snapshot Selection and strict selected-model handoff. No momentum, weight
decay, clipping, anchoring or CART is enabled. B/C/D use attackers `[0,1,5,7]`
from zero-based round 20.

## Open the desktop workspace

```bash
source environment/basil-noise-env/bin/activate
python run_gui.py
```

Open **Results** to inspect the completed IID evidence. Do not reload the
old fresh IID queue just to view results: it is an execution preset, not a
result viewer. The new non-IID 50-round validation runs use the local-only
runner below; their completed review does not justify a 500-round non-IID queue.

Results: `newResults/IID/*_500r_gpu/`; plots: `newPlots/IID/*_500r_gpu/`.
Non-IID evidence is stored separately under `newResults/nonIID/` and
`newPlots/nonIID/`. Checkpoint recovery restores models
and all BASIL memories, not just final weights. No old-result continuation is
configured. Dashboard shows the experiment, timing/ETA and live accuracy;
Network displays candidate-selection/link telemetry; Queue shows done/total.

## Local non-IID validation (no W&B)

The [fixed 30-round, three-seed CART/HPC guide](docs/CART_30R_HPC_REPLICATION.md)
contains the reviewed plan, four-GPU verification, bounded recovery pilot,
parallel execution, five-second monitor, stop/resume and evidence-transfer
commands. It compares 24 control/CART arms using a TRAIN holdout; no official
TEST evaluation or final novelty/robustness claim is implied.

[Receiver-local CART class memory](docs/CART_LOCAL_CLASS_MEMORY_DESIGN.md) is a
separate diagnostic revision for Dirichlet alpha 0.2. It adds class-balanced
historical prediction retention to the existing prior CE/EBM/snapshot anchor.
`python scripts/run_cart_memory_smoke.py` inspects its bounded profiles without
training; execution requires `--execute`. All new test artifacts stay under
ignored `.papermerge/local-tests/`. It does not enable a production HPC queue.

The [implementation audit and CART validation](docs/IMPLEMENTATION_AUDIT.md)
document a separate class-loss barrier and a candidate that caps its auxiliary
gradient at 10% of the base CE gradient. Inspect the fixed TRAIN-only comparisons
with `python scripts/run_cart_train_validation.py --stage budgeted_clean` or
`--stage noisy_attack`; these commands start no training. The execution runner
reserves 1000 examples from TRAIN, fits on the remaining 49000, and does not open
official TEST. These exploratory candidates do not replace historical CART or
approve a long HPC research queue.

Before a new CART validation run, review the changed numerical/execution code
and fixed configuration, run the affected tests, then check the actual inputs:

```bash
python scripts/run_cart_train_validation.py --preflight --stage budgeted_clean \
  --output .papermerge/local-tests/cart-reviewed-clean
```

This read-only check validates the normalized TRAIN cache, disjoint holdout,
nonempty Dirichlet nodes, fixed configuration, source/data recovery identity,
and planned optimizer work. It starts no training, imports no TensorFlow, and
does not open official TEST. Execution repeats these checks before GPU
initialization and still requires a numerical device check. A preflight pass
does not establish learning quality or approve a different horizon/seed.

```bash
python scripts/run_noniid_basil_study.py --prepare --mu 0.01 --device GPU
python scripts/run_noniid_basil_study.py --execute --mu 0.01 --device GPU
```

These conditions are already complete; do not rerun them merely to view results.
The second command starts the four explicitly authorized 50-round conditions
only when no completed evidence exists; it refuses to overwrite evidence.
Use `--execute --recover` only
after investigating an interruption. All selected snapshots, frozen anchor
references and rolling memories are checkpointed/verified. After completion,
`python scripts/report_noniid_study.py` regenerates the comparison without
training. See the [non-IID guide](docs/NONIID_50_ROUND_PROTOCOL.md) for equations,
attack provenance and the continuation gate.

## Shared IID graphs: Weights & Biases Models

The W&B project is **`Research Paper`**; live plots/run labels use **`IID RUN`**.
A/B/C/D log identical metric names and round
axes so their accuracy and loss curves can share the same workspace panels.
During IID training, the active condition updates after each completed round;
finished curves remain visible. **The new non-IID runs do not use W&B.**
Local CSV/JSON/NPZ evidence remains authoritative if uploads fail.

```bash
pip install -r environment/requirements-tracking.txt
wandb login
python scripts/verify_live_tracking.py
```

Never commit or share an API key. Destination settings are machine-local in
`gui/worker_state/live_tracking.json`. Use team membership/invitations to
share a private project, or explicitly change visibility if public sharing
is desired. API access alone does not establish someone else's viewing access.
Filter the completed IID workspace to the appropriate group listed in the
[IID protocol guide](docs/IID_500_ROUND_PROTOCOL.md#fresh-tracking-attempt).

## Run or resume from a remote terminal

For the original local IID500 workflow, the existing runner remains available:

```bash
tmux new-session -A -s papermerge-iid "cd '$PWD' && environment/basil-noise-env/bin/python scripts/run_iid_queue.py --execute"
```

`Ctrl+C` requests a graceful checkpointed stop; repeat the command to resume.
`Ctrl+B`, then `D` detaches and leaves training running. Completed experiments
are skipped; an interrupted activation repeats from its beginning. See the
[protocol guide](docs/IID_500_ROUND_PROTOCOL.md#remote-terminal-no-gui-needed)
for recovery limits. Without `--execute`, the script only validates.

## Ohio University HPC and headless queues

The opt-in terminal profile targets Linux/Python 3.12, including Ohio
University EECS's **The Machine** (`themachine.cs.ohio.edu`). It does not require
Tkinter, X11, Slurm, root access or a CUDA compiler. The legacy local environment
and GUI defaults remain GPU 0, a 6,144 MiB cap and one CPU thread.

After reviewing and pushing source changes locally, connect through BobcatRAS
VPN, then deploy from Git:

```bash
ssh os675524@themachine.cs.ohio.edu
git clone https://github.com/Ojaswi03/PaperMerge.git
cd PaperMerge
# Later deployments: stop workers before updating their source, then git pull --ff-only
bash scripts/setup_hpc.sh
source .venv-hpc/bin/activate
```

`setup_hpc.sh` creates a fresh, isolated `.venv-hpc`, refuses to replace an
unrecognized environment, checks installation errors and records installed
versions inside that environment. `environment/requirements-hpc.txt` pins a
**candidate** TensorFlow 2.21 environment, separate from local TensorFlow 2.18.
Installation and Blackwell kernels must be verified on the actual server;
local tests do not establish HPC compatibility. The driver's CUDA ceiling
does not identify an installed toolkit or prove TensorFlow kernel support.

Select an allocated physical GPU explicitly. The initial HPC policy caps each
worker at **8,192 MiB**, requires **2,048 MiB free headroom**, uses **one CPU
thread** and supervises a **32,768 MiB worker-tree RSS budget**. These limits
are configurable; the deterministic loader currently requires zero loader
processes. RSS supervision requests a checkpointed stop rather than imposing
an OS memory reservation. A project lock does not reserve a GPU against other
users. Obtain administrator permission/resource availability first.

```bash
python scripts/hpc.py gpu-status
python scripts/hpc.py queues
RUN_ROOT="$PWD/.papermerge/runs/blackwell-single"
STATE="$PWD/.papermerge/terminal/blackwell-single"
QUEUE=gui/queues/iid_a_100r_gpu_fresh.json
python scripts/hpc.py doctor --gpus 1 --output-root "$RUN_ROOT" --state-dir "$STATE" --queue "$QUEUE"
python scripts/hpc.py plan --gpus 1 --output-root "$RUN_ROOT" --state-dir "$STATE" --queue "$QUEUE"
```

`doctor` performs real synthetic GPU matmul, CNN forward/gradient/SGD,
second-order EBM, partial-batch and repeatability checks; it launches no research
condition. Failure blocks execution. It reports tools, dependencies, devices,
build CUDA/cuDNN versions in its verification log, memory, writable locations
and selected-queue prerequisites. `--no-gpu-check` is discovery only and cannot
establish readiness. GPU index 1 is mapped by UUID to TensorFlow's `/GPU:0`;
memory checks and locks use the physical device. No requested GPU silently
falls back to CPU.

For the HPC profile, child GPU processes prepend the active environment's
`nvidia/*/lib` directories to their loader path and expose the wheel's `ptxas`
binary. This addresses pip-installed CUDA libraries being present but invisible
to the dynamic loader without changing shared drivers or system settings.
`doctor` probes shared-library loading in a separate process, reports exact
`dlopen` errors and the TensorFlow build, then runs the actual kernel checks.
Library loading alone does not establish Blackwell compatibility. See
[TensorFlow's virtual-environment GPU troubleshooting](https://www.tensorflow.org/install/pip).

`doctor` prints the active physical GPU and check to stderr, with an activity
update every second while a check is running. It shows elapsed time and stage
counts, not simulated kernel or training progress. The first GPU has no ETA;
subsequent GPUs use timings from successful checks in the same command to
estimate their own remaining time. Slow stages show that the estimate is being
recalculated. SSH terminals refresh a compact activity line that fits the
window width, keeping elapsed time visible. Full check names appear at stage
transitions; redirected output keeps full activity lines. Final JSON stays on stdout, so
`python scripts/hpc.py doctor --gpus 0,1,2,3 > /tmp/papermerge-doctor-results.json` still shows
live activity in the terminal. No research queue starts from this command.

The dataset cache and scientific readiness evidence are external prerequisites,
not supplied by a Git clone. Prepare CIFAR through the existing loader with
`python scripts/hpc.py cache-data --download`, or transfer the four existing
`experiments/cache/cifar10/*.npy` files to the same repository-relative cache
path. Transfer approved readiness evidence and any complete recovery bundle
separately, preserving its relative `newResults/` and `gui/worker_state/` paths
under `RUN_ROOT`. A bundle includes manifests, checkpoints, referenced
checkpoint telemetry and required archives, not just weights. Run `plan` after
transfer: transferred local hardware receipts/checkpoints do not authorize
Blackwell execution. Source or runtime changes require appropriate fresh
validation/approval; the CLI never fabricates evidence or weakens a gate.
Staged non-IID500 still requires the full saved non-IID100 audit.

Only after diagnostics, research gates and approval of the selected experiment:

```bash
python scripts/hpc.py run --queue "$QUEUE" --gpus 1 --output-root "$RUN_ROOT" --state-dir "$STATE" --execute --tmux papermerge
tmux attach-session -t papermerge
# Detach: Ctrl+B, then D. From a second SSH terminal in PaperMerge:
source .venv-hpc/bin/activate
python scripts/hpc.py monitor --state-dir .papermerge/terminal/blackwell-single --interval 5 --screen
python scripts/hpc.py status --state-dir .papermerge/terminal/blackwell-single --json
python scripts/hpc.py logs --state-dir .papermerge/terminal/blackwell-single --follow
python scripts/hpc.py logs --state-dir .papermerge/terminal/blackwell-single --experiment 0 --lines 50
python scripts/hpc.py stop --state-dir .papermerge/terminal/blackwell-single --wait 60
python scripts/hpc.py resume --state-dir .papermerge/terminal/blackwell-single --execute --tmux papermerge
```

`run` and `resume` validate without training unless `--execute` is supplied.
Completed items are skipped. Existing output without a manifest, missing or
incompatible checkpoints, changed source and hardware transitions block
execution. Resume retains the saved GPU/resource policy and scientific config;
it never silently restarts at round zero. An interrupted activation may repeat
from its committed boundary. A failed scientific run retains the existing
investigation requirement. Stop uses a queue-specific request file and verified
process identity; it never signals arbitrary stored PIDs.

The independent monitor refreshes every five seconds while worker counters advance
only at genuine emitted boundaries. It distinguishes latest-node accuracy,
round mean, worst node and Node-9 test loss, shows unavailable metrics honestly,
and retains stopped/failed/completed entries in the denominator. Its atomic
snapshot is separate from scientific manifests/checkpoints. SSH rendering uses
an alternate screen and clears each frame; a compact view fits smaller windows
without accumulating dashboard scrollback. `--screen` forces this rendering for
SSH wrappers; `--plain` preserves readable output when terminal controls are
unavailable. Refresh stops automatically on completion, failure, stop or a
blocked queue, and a final completion/status summary remains visible. No dataset,
checkpoint, result or log cache is deleted. Exiting the monitor does not stop
training. Without terminal rendering, use `--plain --once` or
`status --json`. Controller and worker logs rotate at 4 MiB with three backups.
If tmux is missing, request tmux or an administrator-approved scheduler/nohup
workflow; foreground execution requires a persistent terminal. No system
service is installed.

W&B is optional for the dashboard. Local-only conditions remain local-only;
conditions requiring online tracking still require successful initialization.
Authenticate interactively with `wandb login`; never place credentials in a
queue. The CLI reports credential presence without reading or printing its
value, and the dashboard retains tracking mode, identity and upload errors.

Independent multi-GPU scheduling is explicitly gated by a completed compatible
single-GPU IID run in the same namespace on one selected GPU, per-device verification, distinct GPU
locks, and `--parallel --independent-approved --gpus 0,1`. It currently accepts
fresh IID conditions only, with no transition or staged dependencies. The flag
attests resource permission and protocol review; it is not automatic permission.
No distributed model training is introduced. Sequential execution is default.

Generate per-condition plots from completed HPC results without training, then
retrieve results from your local terminal:

```bash
python scripts/hpc.py report --queue "$QUEUE" --output-root "$RUN_ROOT"
# On your local computer (adjust the remote clone path if needed):
rsync -av os675524@themachine.cs.ohio.edu:PaperMerge/.papermerge/runs/blackwell-single/ ./hpc-results/
```

Runtime state, generated HPC results and `.venv-hpc` are ignored by Git.
Use a new output namespace for separately approved fresh studies. The CLI
does not authorize repeating the already completed local A/B/C/D studies.

## Generate completed-study plots without training

```bash
python scripts/report_iid_campaign.py --device GPU --rounds 500
```

Plots require real measured rounds. A/B measures attacks under BASIL; B/C
measures channel noise; C/D is the matched EBM comparison. No ranking or
80–90% accuracy is assumed.

## Validation

```bash
python -m compileall -q gui basil_core reporting scripts tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 python -m pytest -q tests/test_iid_500r.py tests/test_live_tracking.py tests/test_iid_study.py tests/test_research_protocol.py tests/test_protocol_recovery.py
```

`python scripts/verify_iid_gpu.py` performs synthetic GPU checks, not a research
run. The local verification record must match the implementation and device.
Execution uses a **6 GiB GPU cap**, another **1 GiB free headroom** at startup,
one-worker lock, finite-value checks and no silent CPU fallback. CIFAR is reused
through a read-only CPU memory-mapped cache, not repeatedly copied to VRAM.
Checkpoint-boundary cleanup collects unreachable objects, without deleting live
models or clearing TensorFlow's reusable allocator. Incremental telemetry
checkpoints avoid rewriting the entire history on every activation while
retaining exact recovery evidence. Other processes can still cause OOM; no
absolute memory guarantee or whole-run speedup is promised.

## Repository and publication policy

`basil_core/`: research/runtime; `gui/`: state/services/views; `reporting/`:
plots; `scripts/`: explicit entry points; `tests/`: headless contracts.
The `papers/` directory is retained unchanged.

Old outputs, archives and obsolete documentation were removed by explicit user
authorization. The current study must not recreate old output paths. Historical
source/configuration support remains separate from the prepared IID queue.
Handoff documents are excluded by ignore rules, local commit/push hooks and
a publication-policy CI check. Enable hooks in new clones with
`git config core.hooksPath .githooks`. Hooks can be bypassed; previously
committed documents can remain in Git history. No history rewrite or push is
performed automatically.
