# Four-condition 100-round IID BASIL preparation

Configuration preparation is complete. **No research runs were started.**
The queue is idle. All four conditions start fresh at Round 0 and end at Round 99.

## Conditions and paired controls

| Condition | Actual attackers | Channel | Local objective | Start |
|---|---|---|---|---|
| A | None | Clean | CE | Fresh R0–R99 |
| B | 0, 1, 5, 7; Hidden from R20 | Clean | CE | Fresh R0–R99 |
| C | Same as B | Absolute Gaussian, coordinate std 0.010 | CE | Fresh R0–R99 |
| D | Same as B | Same as C | CE + 0.0001 × gradient-norm squared | Fresh R0–R99 |

All use `sequential_basil_iid_v1`, seed 2025, ten IID CIFAR-10 nodes,
the approved 117,706-parameter CNN, strict complete snapshot replacement,
five full local epochs, batch 512, reset SGD without momentum, weight decay
or clipping, and the unchanged round schedule `0.05/(1 + 0.05*r)`.
Each epoch visits all 5,000 local samples: nine batches of 512 and one of 392.

The assumed Byzantine bound is four and BASIL memory is five in **every**
condition, including A. Actual attacker count is distinct from this bound.
Snapshot Selection evaluates the five latest distinct counterclockwise
snapshots on the same receiver-local training batch. Minimum local CE wins;
loss ties within 1e-8 select the nearest counterclockwise sender. No sender
telemetry or test accuracy participates. Outbound attack precedes independent
per-link Gaussian noise. No anchor, proximal loss, CART or Monte Carlo loss
is enabled.

D differentiates the complete source-motivated multiclass adaptation:
`CE + sigma_e² ||grad CE||²`, giving `g + 2*sigma_e²*H*g` through nested tapes,
without materializing a Hessian or reading legacy lambda. Here coordinate
standard deviation is 0.010; variance and EBM coefficient are 0.0001.
See [source equations](SOURCE_EQUATIONS.md) for the squared-loss approximation
in Eq. (14) and why Eq. (23)'s scalar simplification is not a general CNN identity.

## Pairing evidence

The existing IID partitioner was reused, not rewritten. Cached full-data labels
and the completed reference's saved partition indices were checked: 50,000
unique/disjoint training assignments, 5,000/node, all ten classes/node;
10,000 evaluation-only test examples, 1,000/class.

- Partition SHA-256: `db79edd6b35d709b0a40fcc9ca5b5253e1da52a414cfba30a87de6aff4339b29`
- Initial-model SHA-256: `7c717dbbd9bd1f2f230b1ff3d288bd8fec33dcd3ec730eb43e8a14e496cdc2df`

Both hashes reproduce the completed 50-round references. All four configurations
declare these identities; the manual execution path checks them again before
training. Initialization, partition, shuffle, augmentation and SS-batch streams
are shared. B/C/D resolve the same attackers. C/D channel draws use the same
keyed stream `(2025, channel_noise, round, sender, receiver)`, with independent
draws for distinct links. Different resulting trajectories are expected.

The exact scientific configuration diffs are:

- A/B: actual attacker count/IDs and Hidden activation only; BASIL stays enabled.
- B/C: channel enablement and its absolute sigma only.
- C/D: `localObjective` and its derived `ebmMode` only.

Per-condition identity/display/output metadata differs intentionally. The
machine-readable audit and per-node class histogram are in
`newResults/IID/ABCD_100r_comparison/preparation.json` and
`iid_partition_audit.json`.

## Resume decision and evidence preservation

This section records the original preparation decision. Subsequent interruptions
are handled by the new [checkpoint/reconstruction workflow](IID_QUEUE_RECOVERY.md).
The original 50-round reference runs are still preserved, not extended or replaced.

At original preparation, exact stateful resume was **not implemented and verified**. The engine
initializes all logical nodes and memories at R0 and has no loader restoring
a completed round's rolling per-receiver snapshots, sender/round identities,
logical states and continuation state. Final node weight files are not an exact
BASIL checkpoint. Keyed RNG alone does not restore the missing memory state.

Therefore B and D are **fresh 100-round reruns**, not model-only continuations.
Their completed 50-round directories, comparison, summary and plots remain
unchanged. Existing outputs cannot be silently overwritten by the GUI runner.
The GUI still refuses weights-only continuation. New activation-boundary
checkpoints restore the complete memory state; older interrupted A/B/C runs
reconstruct it through verified deterministic replay. See the recovery document.

## GUI and manual execution

From the repository root:

```bash
source environment/basil-noise-env/bin/activate
python run_gui.py
```

1. Open **Queue** in the left navigation rail. The saved current queue already
   contains A, B, C, D in that order.
2. If necessary, click **Load Queue**, select `gui/queues/iid_abcd_100r.json`,
   and confirm replacement of the idle queue. Loading never starts execution.
3. Confirm four Pending entries, 100 rounds each, and Fresh R0→R99 labels.
   Select each row to inspect protocol, attackers, channel and objective.
4. Confirm Idle status and Workers: 0. Do not add duplicate preset entries.
5. When ready, click the single **Run Queue** button in the top command bar.
   Runs proceed serially A → B → C → D. Stop or an execution failure prevents
   automatic progression to the next condition.

The Builder preset dropdown also exposes all four configurations. Individual
JSON files are under `gui/configs/IID/sequential_basil_100r/`.
The current queue is `gui/queue_state.json`; the reusable preset is
`gui/queues/iid_abcd_100r.json`.

Execution uses one deterministic CPU research process at a time, not the legacy
multiprocessing worker pool. The GUI's reader forwards structured events to the
Tk main thread. Duplicate starts are rejected. Queue status is persisted;
an interrupted Running entry reloads as Stopped, not automatically running.
The launcher defers code-reload restarts while the queue is active.
GPU behavior and the prior multiprocessing discrepancy remain unverified.

## Outputs and reporting

Results and plots respectively use `newResults/IID/` and `newPlots/IID/`, with
these new directories (old 50-round outputs are not reused):

```text
A_clean_no_attack_ce_100r/
B_attack_clean_ce_100r/
C_attack_noise_ce_100r/
D_attack_noise_ebm_100r/
ABCD_100r_comparison/
```

The per-condition directories are empty until manual execution. Preparation
metadata exists only in the new comparison result directory. No nonIID output
is produced. Each future run saves manifests, round CSVs, actual metrics,
activation/optimizer/channel telemetry, final node weights and failure evidence.

Plots are generated after each completed run. They include full-test, mean-node,
worst-node, full-test CE, ten class trajectories and final per-class accuracy,
sender selection rates and snapshot age. B/C/D show R20 attack markers; A does
not. C/D add channel norms and ratios; D adds EBM penalty, ordinary gradient,
second-order correction and correction-ratio plots.

Metric definitions remain unchanged: full-test accuracy/CE and class trajectories
use the fixed end-of-ring Node 9 honest trained state before transmission;
mean-node and worst-node metrics summarize the ten distinct full-test model
accuracies, without model averaging. Before R20, attacker-designated senders
have not yet corrupted their outbound parameters.

After all four complete, reporting generates the four requested `ABCD_*`
comparison figures, final/best/worst accuracy, pre/post-attack windows, means
over R80–89 and R90–99, and the late-round difference. Signed contrasts are
B−A (attack), C−B (noise), D−C (EBM), D−A (total gap), also for best/late-mean/worst
accuracy. No ordering, benefit, convergence or statistical significance is
assumed; one paired seed does not establish significance. These statistics
never extend the horizon or control training.

To regenerate completed plots/reports later, without training:

```bash
python scripts/report_iid_campaign.py
```

Reports distinguish Not started, Failed, Stopped and Completed. Four-way plots
require all four measured 100-round trajectories; missing rounds are not filled.

## Runtime estimate

Measured 50-round references: clean B ≈98.6 minutes; source-EBM D ≈166.1 minutes.
Linear CPU estimates: A/B/C about 3.3 hours each (C's noise overhead uncertain),
D about 5.5 hours; **fresh queue ≈15.4 hours**, with machine-load uncertainty.
If exact B/D resume were available, the hypothetical total would be ≈11.0 hours.
That resume mode is **not available** and is not queued.

## Verification executed without research training

```bash
environment/basil-noise-env/bin/python scripts/prepare_iid_campaign.py
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests
DISPLAY= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python -m pytest -q tests/test_iid_campaign.py tests/test_gui_architecture.py tests/test_config_library.py tests/test_execution_policy.py tests/test_iid_plot_pipeline.py tests/test_iid_study.py -k 'not test_no_lambda_loss_gradients_updates_and_strict_load'
DISPLAY=:98 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python -m pytest -q tests/test_iid_campaign.py::test_real_gui_loads_four_idle_configs_and_presets_without_launch
environment/basil-noise-env/bin/python scripts/run_iid_condition.py --config gui/configs/IID/sequential_basil_100r/D_iid_basil_attack_noise_ebm_100r.json
environment/basil-noise-env/bin/python scripts/report_iid_campaign.py --help
```

Preparation reproduced data/model hashes without training. Compile succeeded.
Focused tests: **67 passed, 644 subtests passed, 1 display test skipped,
1 training test deselected**. The separate Xvfb GUI test passed (1 test) with
research process launch blocked; it loaded the preset and applied every Builder
preset without starting execution. Two existing protobuf deprecation warnings
remain. The condition CLI returned `executionStarted: false`.
Unrelated ROS pytest plugin auto-loading was disabled after it required an
unavailable `lark` package. The legacy convergence module was not collected;
its known import-time training/collection problem is not claimed fixed.

Objective checks compute losses/gradients and algebraic prospective updates
only, including lambda 1 versus 999999, zero EBM under noisy C, and independent
finite-difference/Hessian-gradient verification. They do not perform local SGD.
Synthetic plot fixtures are temporary files, not research evidence.

SHA-256 manifests before/after contain **5,485 files each, all matched, zero
changed**: 5,363 protected historical files plus 122 completed 50-round evidence
files. No new run manifest, research worker, or active-queue sentinel was created.

CONFIGURATION PREPARATION COMPLETE — RESEARCH RUNS NOT STARTED
