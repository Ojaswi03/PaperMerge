# One-class BASIL: four-condition 50-round validation

The completed 500-round IID evidence is preserved separately. This study uses
**one class per node**, not IID. All four conditions must complete the 50-round
technical and learning-trend review before a 500-round GUI queue is prepared.
No 500-round non-IID run is authorized to start automatically.

## Completed results and continuation decision

All four completed **50/50 full-data GPU rounds**, with 500 activations and
25,000 optimizer updates each. Execution contracts passed, but the learning
trend did not. **No 500-round non-IID queue has been prepared or started.**

| Condition | Final Node 9 | Final mean node | Final worst node | Own-class mean | Other-class mean | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| A | 10.190% | 10.065% | 10.000% | 99.960% | 0.077% | 31.2 min |
| B | 10.000% | 10.000% | 10.000% | 100.000% | 0.000% | 31.2 min |
| C | 10.000% | 10.000% | 10.000% | 100.000% | 0.000% | 31.2 min |
| D | 10.000% | 10.000% | 10.000% | 100.000% | 0.000% | 35.3 min |

The anchor did not prevent specialization in this setting. Confusion matrices
before training and after each epoch show forgetting plus constant-class
prediction collapse. A’s first-ten to last-ten mean improvement was only
0.0351 percentage points. D’s mean second-order correction/CE-gradient ratio
was 0.02747%; it was nonzero but small, with no C–D global accuracy gain.
These observations do not establish universal EBM failure or isolate the
anchor’s causal effect without an unanchored control.

B/C/D each selected 299 actually corrupted snapshots out of 300 post-attack
decisions. The selector still obeyed minimum receiver-local CE: a corrupted
snapshot can fit the receiver’s sole class better than honest snapshots
specialized to other classes. This is not evidence of successful Byzantine
filtering under the extreme partition.

All four share the partition hash
`fe885d5e2e9680b32e05705485d67aed3dc7ef6a77964617abb7da2b0d9abcd2`
and initial-model hash
`7c717dbbd9bd1f2f230b1ff3d288bd8fec33dcd3ec730eb43e8a14e496cdc2df`.
All 2,500 corresponding C/D link-noise hashes matched; different links within
an activation had independent draws. Test images remained evaluation-only.

No numerical failure or OOM was recorded. TensorFlow allocator peak usage
was about 273.3 MiB for A/B/C and 327.6 MiB for D, below the 6 GiB cap;
this is allocator telemetry, not total system GPU usage. Post-run integrity
verification matched **20,247/20,247 retained IID result/plot and paper files**,
with zero changed files.

The detailed local report, contract table and comparison plots are at
`newResults/nonIID/ABCD_50r_comparison/NONIID_50_ROUND_REVIEW.md`.
Research outputs remain local/gitignored; this guide records their summary.

## Predeclared conditions

| Condition | Actual attackers | Channel | Local objective |
|---|---|---|---|
| A | None | Clean | CE + μ/2 · ‖θ − θ_selected‖² |
| B | [0,1,5,7], current Hidden from R20 | Clean | Same as A |
| C | Same as B | Absolute Gaussian, σₑ = 0.010 | Same as A |
| D | Same as B | Same as C | CE + 0.0001 · ‖∇CE‖² + μ/2 · ‖θ − θ_selected‖² |

**μ = 0.01** is fixed before reading these results. It is not weight decay:
the reference is the complete selected *received* model, frozen throughout
the activation, rather than zero parameters. The anchor is our
snapshot-preserving adaptation, not a mechanism claimed by the BASIL paper.

For D, with g = ∇CE and H = Hessian(CE), the update direction is

```text
g_applied = g + 2 σₑ² H g + μ (θ − θ_selected)
θ_next   = θ − η_round g_applied
```

The CE gradient norm excludes the anchor. The existing nested TensorFlow
GradientTapes produce the Hessian-gradient contribution without materializing
the Hessian; the anchor gradient is added analytically. No legacy lambda is
read. A/B/C have no EBM contribution.

## Next diagnostic: isolate anchoring before changing the method

ACDS remains excluded and the current Hidden attack is retained unchanged.
No new class-preservation algorithm has been silently enabled. Two clean
controls are prepared separately in `gui/configs/nonIID/retention_diagnostic/`.
The initial proposal was five rounds; the operator subsequently authorized
**30 rounds per control**, with all other settings unchanged:

| Control | Rounds | Local epochs | Local objective | Attack/noise |
|---|---:|---:|---|---|
| `clean_ce_control_30r_gpu` | 30 | 5 | CE exactly (same worker, μ=0) | Off |
| `clean_fixed_anchor_control_30r_gpu` | 30 | 5 | CE + 0.01/2 · ‖θ−θ_selected‖² | Off |

Both use all 5,000 local examples, full-test evaluation, the original seeded
initialization, S=5, Snapshot Selection, strict handoff, SGD and unchanged
round decay. Only anchor strength changes scientifically. They are short
full-data mechanism diagnostics, marked `researchValid=false`, not 50-/500-round
research evidence. Preparation alone never starts them. The subsequent
30-round execution was separately authorized; W&B is disabled.
Their output paths are under `newResults/nonIID/retention_diagnostic/` and
`newPlots/nonIID/retention_diagnostic/`; existing four-condition evidence is
not overwritten. Preparation never launches a worker or changes the queue:

```bash
python scripts/prepare_retention_diagnostic.py --prepare --rounds 30
# Explicit training commands; do not rerun just to inspect saved evidence:
python scripts/run_noniid_basil_study.py --execute --config gui/configs/nonIID/retention_diagnostic/clean_ce_control_30r_gpu.json
python scripts/run_noniid_basil_study.py --execute --config gui/configs/nonIID/retention_diagnostic/clean_fixed_anchor_control_30r_gpu.json
# After both complete, regenerate the comparison without training:
python scripts/report_retention_diagnostic.py --rounds 30
```

The zero-μ control is represented by the existing `snapshot_anchor_only`
worker with an exactly zero anchor penalty/gradient, plus explicit
`effectiveLocalObjective=cross_entropy` metadata. This avoids introducing a
different training/data-augmentation code path. Contract tests verify equality
of CE loss, gradients and SGD updates. The runner now respects the configured
`researchValid` flag instead of always writing `true` for diagnostics.

For a single-class batch, CE's logit derivative is `p_c − 1[c=local_class]`.
At uniform predictions that is −0.9 for the local class and +0.1 for each
other class: the ordinary local objective rewards specialization. The anchor
gradient starts at zero because the received weights equal the frozen
reference, and subsequently opposes parameter drift; it is not an explicit
constraint on previous-class predictions. These analytical facts do not
predict the exact full-network learning trajectory.

Existing legacy CART is not a drop-in fix: its engine uses consensus
averaging, whereas this protocol requires strict handoff, and the registry
requires local class support for verified per-class gaps. A receiver with
only one class cannot legitimately verify the other nine using test images
or invented metrics. A class-preservation extension therefore requires a
separately agreed mechanism and identifier, not simply switching `approach`
to `cart`. Any new training-side coefficient selection must use training or
explicit validation evidence, never final test accuracy.

The fixed 30-round horizon is declared before reading the final curves. No
automatic reruns, coefficient searches or extensions are triggered by accuracy.
The paired comparison is stored in
`newResults/nonIID/retention_diagnostic/comparison_30r/RETENTION_DIAGNOSTIC_REVIEW.md`
with plots under the corresponding `newPlots/nonIID/` path. It distinguishes
execution contracts from observed class retention and does not promise a
successful result or authorize an unchanged 500-round queue.

### Completed 30-round clean controls

Both controls finished 30/30 full-data GPU rounds: 300 activations and 15,000
optimizer updates each. Every saved-result contract passed, including full
epochs, partial batches, frozen reference hashes, rolling memory and source
CSV/NPZ agreement. No non-finite failure, OOM or memory-pressure event was
recorded. Both shared the same partition and canonical initialization hashes
as the preserved 50-round study.

| Control | Node 9 final | Mean final | Worst final | Own-class mean | Other-class mean | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| CE, μ=0 | 10.030% | 10.005% | 10.000% | 99.980% | 0.0078% | 18.6 min |
| Fixed anchor, μ=0.01 | 10.000% | 10.000% | 10.000% | 100.000% | 0.0000% | 18.3 min |

Anchor minus CE final mean accuracy was −0.005 percentage points; neither
retained meaningful ten-class knowledge. The anchor gradient was nonzero,
averaging 4.4588% of the CE gradient norm. Mean squared parameter distance
from the selected snapshot was 1.134112 for CE versus 1.071725 for anchoring,
but this small observed reduction did not preserve previous-class predictions.
These quantities compare evolving trajectories, not identical-input gradients.

All 300 independently recomputed anchor activation input/output hashes and
selected source/round identities matched the first 30 rounds of the preserved
A50 run. This provides same-device repeatability evidence, not a claim of
CPU/GPU bit-identical execution. The earlier A/B/C/D outputs were not reused
as continuation checkpoints or overwritten.

Post-run focused tests: **176 passed, 1 display-dependent skip, 2 dependency
deprecation warnings in 19.29 seconds**. Compileall succeeded. The synthetic
GPU CE/EBM and anchored objective checks also passed; no new research method
was trained during those contract checks. Hash verification matched all
20,247 protected IID/paper files and all 2,281 preserved 50-round non-IID
result/plot files, with zero changed files. The completed clean controls have
45 plots each plus three paired comparison plots.

No coefficient was selected from these test accuracies and no 500-round queue
was prepared. Repeating the unchanged objective until a desired score appears
is not the continuation plan. A separate class-preservation mechanism must be
agreed before further method changes; no ACDS or attack change is implied.

The operator subsequently approved a **separate class-aware retention
mechanism**. Its predeclared 30-round diagnostic, gradient-importance estimator,
weighted anchor, additional communicated state and checkpoint contracts are
documented in [CLASS_AWARE_RETENTION_PROTOCOL.md](CLASS_AWARE_RETENTION_PROTOCOL.md).
The original baselines and their negative findings are retained unchanged;
ACDS and threat modifications are not enabled by that approval.

## Fixed training and data rules

| Setting | Value |
|---|---|
| Protocol | `sequential_basil_one_class_v1` |
| Study revision | `source_grounded_no_lambda_v1` |
| Dataset | Full CIFAR-10: 50,000 training, 10,000 evaluation-only test images |
| Node i | All 5,000 training examples of class i; no overlap |
| Node order | 0 → 1 → … → 9 → 0 |
| CNN | Existing 117,706-parameter BASIL CNN |
| Local training | Five complete epochs, batch 512, final batch 392 |
| Steps | 50 per activation; 25,000 per 50-round condition |
| SGD | Reset each activation; momentum 0 |
| Learning rate | 0.05 / (1 + 0.05 · zero-based round) |
| Clipping / weight decay / CART | Disabled |
| Snapshot memory | S=5, including A with actual attackers disabled |
| Snapshot Selection | Lowest receiver-local CE on the same unaugmented local batch |
| Tie break | Nearest counterclockwise predecessor |
| Channel start | R0 for C/D; independent keyed draw per directed link |
| Cloud tracking | Disabled; no W&B initialization or logging |
| Device | Verified GPU, float32, TF32 off, deterministic single process |

Selection uses neither sender telemetry nor global test performance. The
selected parameters completely replace the receiver's training start state.
No averaging occurs. After training, the attack is applied before channel noise.

```text
selected received snapshot → frozen anchor → five full epochs → attack → noise
                                                                    ↓
sender 0 → receiver 1 (next trainer)
         → receivers 2,3,4,5 (cache only)

Before Node 0, R11: {5:R10, 6:R10, 7:R10, 8:R10, 9:R10}
Before Node 1, R11: {6:R10, 7:R10, 8:R10, 9:R10, 0:R11}
```

## Hidden attack provenance

The user explicitly chose to retain `delayed_hidden_parameter_attack_v1` for
this exploratory study while seeking advisor approval for a source-faithful
adaptation. Its existing per-tensor transformation is unchanged:

```text
ζ ~ N(0, [0.05 · max(std(w), 0.01)]² I)
w_out = (1 − 0.55)w + 0.55(−1.4w + ζ) = −0.32w + 0.55ζ
```

This is not the BASIL paper's separate Gaussian attack (model replacement
with N(0,I)); the project's own Gaussian-attack implementation is also a
separate code path.
It is also **not an exact reproduction** of the original Hidden attack.
[The Hidden Vulnerability of Distributed Learning in Byzantium](https://proceedings.mlr.press/v80/mhamdi18a.html)
constructs B(γ)=mean(honest gradients)+γE against gradient aggregators;
the BASIL description does not specify all parameters of a model-space port
against receiver-local-loss selection. That change is deferred, not guessed.

For the noise term, we adapt the source gradient-norm regularization to
multiclass logits-based CE. The source squared-loss approximation does not
make scalar gradient scaling a general CNN identity.

## Commands and recovery

```bash
source environment/basil-noise-env/bin/activate
python scripts/run_noniid_basil_study.py --prepare --mu 0.01 --device GPU
# Only the following explicitly starts the authorized four 50-round runs:
python scripts/run_noniid_basil_study.py --execute --mu 0.01 --device GPU
```

The conditions execute serially A/B/C/D in separate processes. Live graphs and
GPU allocators are released between conditions. VRAM is capped at 6 GiB;
full data and memory snapshots stay on CPU, and GPU batches are reused.
No absolute guarantee against OS/device failure or OOM is implied.

Ctrl+C requests a graceful stop. After investigating the interruption, use
`--execute --recover` with the same μ and device. Recovery verifies source
hashes, data, initialization, device policy, checkpoint contents and the
activation cursor. It restores all model states and rolling memories, not
only final weights. Completed conditions are skipped. A non-finite failure
must be investigated rather than automatically resumed.

## Evidence and continuation review

- Results: `newResults/nonIID/<condition>_50r_gpu/`.
- Plots: `newPlots/nonIID/<condition>_50r_gpu/`.
- Combined report: `newResults/nonIID/ABCD_50r_comparison/NONIID_50_ROUND_REVIEW.md`.
- Machine-readable review: `continuation_review.json` beside the report.

After all four finish, `python scripts/report_noniid_study.py` regenerates
the saved-evidence plots and review without any training.

### Pre-run and post-run validation evidence

`python -m compileall -q gui basil_core reporting scripts tests` succeeded.
The focused protocol/recovery/memory/IID/non-IID suite passed **163 tests**
with **one optional GUI/display test skipped**; the display was unavailable
in the sandbox. The snapshot-anchor objective suite separately passed
**13 tests**. Both `scripts/verify_iid_gpu.py` and
`scripts/verify_snapshot_anchor_gpu.py` passed synthetic GPU checks, including
the 512- and 392-example batches, unchanged reference parameters and exact
same-device no-lambda comparisons. CPU/GPU floating-point execution is not
claimed bit-identical. The new anchor GPU check peaked at approximately
304 MiB under the 6 GiB cap.

After all four completed, the same focused suite was rerun: **163 passed,
1 skipped, 2 dependency deprecation warnings in 18.14 seconds**. The skipped
test requires an available GUI display. This is not a claim that every legacy
repository test passed. The 13 objective tests are included in the 163;
do not add these overlapping counts together.

Exact final commands (from the repository root):

```bash
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests

DISPLAY= PAPERMERGE_LIVE_TRACKING=disabled \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 \
TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
environment/basil-noise-env/bin/python -m pytest -q \
  tests/test_noniid_study.py tests/test_iid_study.py \
  tests/test_research_protocol.py tests/test_protocol_recovery.py \
  tests/test_iid_500r.py tests/test_memory_safety.py \
  tests/test_research_audit.py tests/test_iid_gpu.py tests/test_iid_campaign.py

environment/basil-noise-env/bin/python scripts/verify_saved_evidence.py \
  --manifest gui/worker_state/noniid_50r_iid_evidence_before.json
```

Compile completed successfully. The integrity command returned `unchanged`,
with 20,247 files before/after, 20,247 matched and an empty changed-file list.
CSV round-level Node-9 and mean-node accuracy were independently compared
against NPZ arrays and matched for all four runs. Activation per-class tensors
have shape `[50,10,10]`; after-epoch class tensors have `[50,10,5,10]`.
There are 45 PNGs per condition and three combined comparison PNGs, including
representative six-stage confusion and class-retention plots.

Each condition stores all 50 measured round points, 500 activation records,
25,000 optimizer records, selected-source/noise metadata, reference hashes,
and before/after-each-epoch class accuracies and confusion matrices.
Technical checks include finite values, exact objective sums, unchanged
anchors, correct batches, and paired partition/initialization hashes.

Learning is assessed separately using first-ten versus last-ten mean node
accuracy and class retention. Near-100% own-class accuracy with near-zero
other-class accuracy indicates prediction specialization, not successful
ten-class transfer. Correct mathematics alone does not guarantee learning.
Poor or flat trends do not justify silently changing epochs, μ or the loss.
If any condition fails the review, the 500-round queue is not prepared.
Even promising 50-round trends do not assure 500-round convergence.

There is no CE-only/unanchored control among these four runs, so any benefit
of anchoring cannot be causally isolated here. C–D is the matched EBM contrast.
One seed does not establish statistical significance. No hyperparameter is
selected or retuned using the test set.
