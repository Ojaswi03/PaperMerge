# PaperMerge

PaperMerge studies Byzantine-resilient decentralized learning under noisy
communication using Python, TensorFlow and a Tkinter experiment workspace.

[Latest: Dirichlet rerun readiness, results, formulas and launch instructions](docs/DIRICHLET_RERUN_READINESS.md).
The full-data GPU gates completed. Local-prior training reached **32.90%**
Node-9 accuracy versus **19.13%** for the matched original ten-round control.
At 23 rounds, B/C/D reached **41.29% / 45.04% / 46.11%**, with no actually
attack-corrupted snapshot selected in their three attacked rounds. These are
single-seed diagnostics, not solved retention or universal robustness.

A separate **30-round A/B/C/D GPU confirmation preset is ready and idle**:
`gui/queues/noniid_dirichlet_prior_abcd_30r_gpu.json`. Open `python run_gui.py`,
then **Queue → Load Queue**, select that preset, and click **Run Queue** only
when ready. It uses Dirichlet α=0.2, local-prior CE, frozen snapshot anchoring,
and EBM in D only. No W&B, data sharing, helper services or CART is enabled.
The new full queue has **not** started; no 500-round campaign is approved.

## Earlier studies and investigation notes

The status statements below describe their historical stages, not the current
confirmation preset. Earlier outputs are preserved rather than overwritten.
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

For remote terminals, the existing IID runner can start/resume its saved queue
without a GUI. This is not the new non-IID command:

```bash
tmux new-session -A -s papermerge-iid 'cd /home/ojaswi/PaperMerge && environment/basil-noise-env/bin/python scripts/run_iid_queue.py --execute'
```

`Ctrl+C` requests a graceful checkpointed stop; repeat the command to resume.
`Ctrl+B`, then `D` detaches and leaves training running. Completed experiments
are skipped; an interrupted activation repeats from its beginning. See the
[protocol guide](docs/IID_500_ROUND_PROTOCOL.md#remote-terminal-no-gui-needed)
for recovery limits. Without `--execute`, the script only validates.

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
