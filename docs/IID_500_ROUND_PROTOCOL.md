# Paired 500-round IID BASIL protocol

## Prepared, not running

Four fresh GPU runs are queued A → B → C → D, targeting R0–499. No research
worker is launched by preparation.

| Condition | Actual attackers | Channel | Loss |
|---|---|---|---|
| A | None | None | CE |
| B | [0,1,5,7], Hidden R20 | None | CE |
| C | Same four | Absolute σₑ=0.010 | CE |
| D | Same four | Same absolute channel | CE + σₑ² ‖∇CE‖² |

Protocol: `sequential_basil_iid_v1`. The backward-compatible metadata key
`iidCampaignId` identifies `iid_abcd_500r_v1`; it does not change the method.
Configurations: `gui/configs/IID/sequential_basil_500r/gpu/fresh/`.
Queue: `gui/queues/iid_abcd_500r_gpu_fresh.json`.

All four share seed 2025, partition, canonical initialization, ring order,
data-order/augmentation keys, SS batch generation, memory, optimizer and LR.
B/C/D share attack configuration; C/D share underlying keyed channel draws.
Different model trajectories between C/D are expected because their losses differ.

## Fixed data and training

| Quantity | Value |
|---|---|
| Training set | CIFAR-10: 50,000 examples, 5,000/class |
| IID split | Existing seeded random disjoint equal-size split |
| Nodes | 10; 5,000 examples/node; all ten classes/node |
| Node class counts | Approximately 500/class; audited range 450–545 |
| Test set | Full separate 10,000 images; 1,000/class; evaluation-only |
| CNN | Approved BASIL CNN; 117,706 trainable parameters |
| Local training | 5 complete epochs; batch 512 |
| Each epoch | 9×512 plus final 392; no dropped remainder |
| Steps/activation | 50 |
| Per run | 500 rounds; 5,000 activations; 250,000 SGD steps |
| Optimizer | Reset SGD; momentum=0 |
| Learning rate | ηᵣ=0.05/(1+0.05r), unchanged |
| Clipping, weight decay, anchor, CART | Disabled |

Only the horizon changes. The LR schedule is not stretched: η₄₉₉≈0.001927.
Longer training does not guarantee higher accuracy. Test metrics never affect
selection, optimization, stopping, sigma or hyperparameter choices.

Verified partition hash:
`db79edd6b35d709b0a40fcc9ca5b5253e1da52a414cfba30a87de6aff4339b29`.

Verified initialization hash:
`7c717dbbd9bd1f2f230b1ff3d288bd8fec33dcd3ec730eb43e8a14e496cdc2df`.

The partition histogram and scientific config diffs are saved in
`gui/queues/iid_abcd_500r_preparation.json`. C/D differ scientifically only in
`localObjective` and its corresponding `ebmMode`.

## Snapshot Selection and strict handoff

For receiver i, Mᵢ contains the latest received snapshots from five **distinct**
counterclockwise senders. Evaluate each candidate on the same local training
selection batch Bᵢ:

```text
CE_i(θ; B_i) = mean sparse ten-class cross-entropy from logits
θ_selected = argmin_{θ in M_i} CE_i(θ; B_i)
θ_start = θ_selected                    # full replacement, no averaging
```

The deterministic unaugmented batch contains the first up-to-512 indices of
the receiver's seeded local partition. Sender-reported metrics and global
test accuracy are never selection inputs. Loss ties within 1e-8 choose the
nearest counterclockwise predecessor. A retains S=5 even with zero attackers.

```text
Before Node 0, R11:  5:R10  6:R10  7:R10  8:R10  9:R10
Node 0 trains → refreshes memories of nodes 1–5; only Node 1 activates next
Before Node 1, R11:  6:R10  7:R10  8:R10  9:R10  0:R11
```

Implementation: `select_snapshot`, `RollingMemory`, `broadcast` in
`basil_core/research_protocol.py`. No FedAvg, pairwise consensus or stale-model
averaging occurs.

## Confirmed attack: project Hidden parameter attack

B/C/D call `applyAttack(trained, "hidden", rng=keyed_rng(...))` only for
configured nodes at source round ≥20. R0–19 is honest warm-up. A never attacks.
All alternate attack flags are false; validation rejects them if enabled.

For each parameter tensor w, the unchanged implementation is:

```text
ζ ~ Normal(0, [0.05 max(std(w),0.01)]² I)
h = -1.4w + ζ
w_attack = (1-0.55)w + 0.55h = -0.32w + 0.55ζ
```

This is `delayed_hidden_parameter_attack_v1`, not the separate `gaussianAttack`
weight/noise blend. Hidden contains a small Gaussian concealment perturbation;
that does not make it the separate Gaussian attack mode. It is the project's
existing transformation, not an independently verified attack-paper reproduction.
The honest trained state is evaluated before outbound corruption.
500 rounds include 20 warm-up and 480 attack-active rounds. Actual telemetry
must confirm executed attacks once training starts.

## Absolute Gaussian channel: C/D only

```text
θ_received(i→j) = θ_outbound(i) + ξ(i→j)
ξ(i→j) ~ Normal(0, σ_e² I)
σ_e = 0.010                 # fixed coordinate standard deviation
σ_e² = 0.0001               # variance
```

Noise begins R0. Draws are independent across directed links and keyed by
seed/round/sender/receiver. Model norm does not determine sigma; noise/model
ratio is telemetry only. Relative-L2 noise remains a separate unused branch.

Order: receive → SS → load → reset SGD → train → honest evaluation → active
Hidden corruption → independent per-link noise → receiver memories.

## Full source-grounded EBM: D only

For F=CE, g=∇F and H=∇²F:

```text
L_D = F + σ_e² ||g||² = CE + 0.0001 ||∇CE||²
∇L_D = g + 2σ_e² H g = g + 0.0002 H g
θ_next = θ - η_r ∇L_D
```

Nested TensorFlow tapes differentiate the real objective; no full Hessian is
materialized. There is **no lambda** or predetermined gradient multiplier.
`IidWorker` and its GPU adapter ignore legacy lambda, verified by comparing
loss/gradient/update with lambda 1 and 999999. A/B/C use ordinary CE only.

We adapt Ang et al.'s gradient-norm regularization to CIFAR-10 cross-entropy
and a decentralized BASIL ring. The source's Eq.14 approximates a **squared
loss quantity**, not exact expected noisy CNN CE. Eq.15b is a local update;
Eq.23 scalar-gradient scaling is not a general CNN identity. Our derivative
contains 2σₑ²Hg. No guaranteed noise-resilience or accuracy benefit is assumed.
Original PDFs remain unchanged in `papers/`; no non-IID proximal loss is used.

## GPU and resume safety

Refreshed RTX 4070 Ti synthetic verification passed with TF2.18, float32,
TF32 off and serial deterministic execution. Analytic EBM gradient errors were
approximately 2.77e-8 CPU and 4.67e-8 GPU. CPU/GPU gradient relative difference
was ~0.2994%, below the 1% tolerance; they are not bit-identical backends.
Synthetic peak memory was **390,430,976 bytes (~372.3 MiB)** under the new
**6,144 MiB cap**. The GPU check requires another **1,024 MiB free headroom**
(at least 7,168 MiB free before launch), allows one worker and refuses silent
CPU fallback. This is a representative synthetic test, not an overnight OOM
guarantee. External GPU applications can still cause OOM.

Verification is local: `gui/worker_state/GPU_EXECUTION_VERIFICATION.json`.
Checkpoints preserve node states, rolling memories, snapshot identities/rounds,
measured telemetry and the activation cursor. Keyed RNG avoids mutable stream
reconstruction. Retry the same stopped entry; fingerprints, config, data and
device must match. Never fake continuation using only final model weights.

### Memory reuse and cleanup

| Resource | Policy | What is deliberately retained |
|---|---|---|
| GPU | Bounded TensorFlow allocator, one experiment at a time | Live model, gradients and CE/EBM graphs; cached blocks for reuse |
| CIFAR data | Read-only CPU memory-mapped cache, loaded once per condition | Original training/test arrays; only minibatches move to GPU |
| Host RAM | Refuse startup below 4 GiB available; inspect after each committed checkpoint | Required BASIL snapshots, node states, complete numeric telemetry |
| Dead Python objects | Generation-0 collection normally; full collection under memory pressure | Reachable training state is never cleared |
| Checkpoints | Atomic activation-boundary state plus hash-verified immutable telemetry chunks | Current and previous checkpoint, all completed activation records |

`RuntimeMemoryGuard` records host availability, GPU current/peak usage, cleanup
counts and pressure events in `run.json`. A pressure event requests garbage
collection; it does not retune the model, batch size, EBM or LR. Collection only
releases unreachable objects. It does **not** forcibly flush CUDA or clear the
Keras session mid-training: those actions can invalidate live state or make
subsequent allocations slower. TensorFlow keeps reusable blocks within its cap.
Between conditions the worker process exits, releasing its CUDA context.

Format-2 checkpoints write each completed telemetry record once as a compressed,
SHA-256-addressed chunk. Subsequent checkpoints contain references, not another
copy of the entire growing history. All model states, five-sender memories,
metric arrays and the cursor still commit at **every activation**, with fsync
and previous-checkpoint fallback. Missing or corrupt chunks are rejected, not
silently replaced. Legacy embedded-telemetry checkpoints remain readable.

Gradient auditing also materializes each gradient once for both statistics and
its hash, instead of two GPU-to-CPU copies. Scientific outputs remain identical.
The full five epochs, final partial batch, all existing evaluation points,
selection, attacks, noise, finite checks and mathematical objectives are unchanged.

### I/O-only performance evidence

The following no-training benchmark uses one backed-up real record as a shape
template, synthetic parameters and repeated writes of the same completed history:

| Completed activation records | Old checkpoint size | New checkpoint size | Old median write | New median write |
|---:|---:|---:|---:|---:|
| 100 | 17,530,790 bytes | 4,920,406 bytes | 0.0615 s | 0.0213 s |
| 1,000 | 131,272,798 bytes | 5,165,214 bytes | 0.5077 s | 0.0233 s |

Indexing all 1,000 records initially took 9.27 s, paid incrementally when records
first complete (or once when converting a legacy history). The steady-write
comparison excludes creation of a new chunk; this is **not a full-run speedup
measurement**. Chunk files remain on disk and are not included in the new NPZ
size. Reproduce with `scripts/benchmark_checkpoint_io.py --sample-jsonl <saved
activation log>`. Fresh run manifests record actual `checkpointIo` timing.

Each condition still performs **250,000 optimizer steps**, plus all mandated
test evaluations and BASIL probes. More available VRAM does not automatically
accelerate those computations. Representative synthetic audited GPU step medians
were 0.0517 s CE and 0.0777 s EBM; these do not predict complete campaign runtime.

## W&B Models: live A/B/C/D overlay plots

W&B is optional; Weave is unnecessary for this CNN. `wandb==0.28.0` is isolated
in `environment/requirements-tracking.txt`. It provides the hosted live/shared
dashboard that local Matplotlib and the standard library do not provide.
The SDK is already installed in the current environment.
The tracking requirements retain protobuf 4.25.8. An unrestricted W&B upgrade
can otherwise install protobuf incompatible with TensorFlow 2.18. Only the W&B
SDK was upgraded from 0.24.0; TensorFlow, CUDA and scientific implementation
fingerprints remain unchanged for checkpoint recovery.

```bash
source environment/basil-noise-env/bin/activate
pip install -r environment/requirements-tracking.txt
wandb login
python scripts/configure_live_tracking.py --entity ojaswisinha15-ohio-university --project "Research Paper"
python scripts/verify_live_tracking.py
```

Never share/commit an API key. Settings live in ignored local worker state.
Your supplied project link is
`https://forge.coreweave.com/wandb/ojaswisinha15-ohio-university/Research%20Paper`.
Set project visibility to **Public** in W&B project settings before sharing.
Local setup does not train a model or override account privacy.

**Project access verified on 2026-10-07:** the read-only verification command
returned `accessible` for `ojaswisinha15-ohio-university/Research Paper`. The
subsequent failure was a remotely deleted A run ID, not a login failure. Fresh
tracking IDs avoid reusing that deleted ID. Actual training metrics remain to
be observed after manual queue launch; public visibility must still be confirmed
in project settings. Recheck access with `python scripts/verify_live_tracking.py` after
changing credentials. The official Forge quickstart uses the standard W&B
SDK/login flow; the browser's Forge URL does not require changing the API host.

### Fresh tracking attempt

Four empty records were created through the actual SDK and checked through the
API: no history points, `research_training_started=false`, status `prepared`.
W&B displays these closed setup records as finished until the worker resumes
their tracking IDs. This is **not** research completion. All four local models
still start fresh at R0; cloud-ID reuse only preserves the prepared record.

Filter Workspace to group **`iid_abcd_500r_v1_fresh_20261008T034125755900Z`**
and select these four runs to overlay their round-based curves:

| Condition | Prepared tracking ID |
|---|---|
| A | `ptmgbkim` |
| B | `qlgn5xhn` |
| C | `05yyrdm1` |
| D | `rv7mbu95` |

No fictitious accuracy/loss points were logged to make graphs appear. An empty
curve is expected until its first measured round. Earlier cloud attempts are
not deleted automatically and should be excluded using the group filter.

In Workspace select A/B/C/D from this study's run group; shared metric names
and custom round axes allow four lines on the same accuracy/loss panels:

| Metric | Definition |
|---|---|
| `IID RUN/Accuracy` | Primary live accuracy panel; choose title `IID RUN — Accuracy` |
| `IID RUN/Loss` | Primary live CE panel; choose title `IID RUN — Loss` |
| `round/full_test_accuracy` | Fixed Node 9 honest trained model, full test set |
| `round/mean_node_accuracy` | Mean across ten trained node states |
| `round/worst_node_accuracy` | Minimum of the ten node accuracies |
| `round/full_test_loss` | Node 9 full-test CE |
| `round/mean_local_ce` | Mean local training CE |
| `round/class_0_accuracy` … `class_9_accuracy` | Ten Node 9 class trajectories |
| `round/honest_selection_rate` | Non-attacker-designated sender fraction |
| `round/attacked_source_selection_rate` | Actually corrupted source fraction |
| `round/channel_noise_norm` | Sampled channel perturbation L2 |
| `round/noise_to_model_norm_ratio` | Diagnostic only |
| `round/ebm_penalty` | D's σₑ²‖g‖²; zero for CE |
| `round/ebm_correction_ratio` | D's ‖2σₑ²Hg‖/(‖g‖+ε) |

Each measured round logs once. The currently running line grows; completed
curves remain fixed. A condition has no curve before its first point. Accuracy
uses fractions 0–1. Real activation metrics use a separate `activation` axis.
Browser updates may lag slightly because SDK uploads are asynchronous.

Logging consumes existing scalars only—no new evaluation, training or RNG
choices. It uploads no code, console logs, papers, handoff documents, datasets,
weights or checkpoints. Network errors disable tracking; local evidence remains
authoritative. The manifest saves URL/run ID; retries reuse the ID and skip
already logged points. Offline mode stores logs for later `wandb sync`:

```bash
python scripts/configure_live_tracking.py --entity ojaswisinha15-ohio-university --project "Research Paper" --mode offline
```

SDK references: [initialization/resume](https://docs.wandb.ai/models/ref/python/functions/init),
[custom axes](https://docs.wandb.ai/models/track/log/customize-logging-axes),
[Forge authentication quickstart](https://docs.coreweave.com/products/wandb/quickstart).

## Execution and output paths

Launch `python run_gui.py`; reopen an already-open GUI to reload the new queue.
In Queue confirm A/B/C/D, pending, GPU and 0/500. Load
`gui/queues/iid_abcd_500r_gpu_fresh.json` if needed. Click **Run Queue** manually.

### Remote terminal: no GUI needed

The headless command uses the same serial GPU worker, output paths, W&B
tracking and checkpoint recovery as the GUI. It does not need a display.
Workers start in their own process session; terminal interrupts are forwarded
to the scientific worker as graceful stop requests, rather than broadcast to
its W&B service. This reduces a shutdown hazard but does not guarantee network
availability or cloud upload completion.
Validate without training with `python scripts/run_iid_queue.py`.
To start or resume manually, paste:

```bash
tmux new-session -A -s papermerge-iid 'cd /home/ojaswi/PaperMerge && environment/basil-noise-env/bin/python scripts/run_iid_queue.py --execute'
```

If the session exists, this command reattaches rather than launching a second
queue. `Ctrl+B`, then `D` detaches without stopping training; an SSH disconnect
also leaves the tmux session running. Press `Ctrl+C` inside the session to
request a graceful stop of the active experiment **and** the remaining queue.
Wait for checkpointing and worker exit. Repeat the command to resume: completed
conditions are skipped and the incomplete condition is restored before later
conditions run. GUI queue state is updated too. Do not concurrently start the
GUI queue or another worker.

Checkpoints preserve completed **activations**, not individual minibatches. An
interrupted node's partial five-epoch activation is discarded and repeats from
its selected input; earlier completed activations and rounds are preserved.
Abrupt termination can require falling back to the previous checkpoint and
verifying/replaying saved activations. Corrupt evidence, changed scientific code,
changed config/data/device or genuine numerical failures block automatic resume
rather than silently restarting or overwriting evidence. Avoid `kill -9`.

To reset stopped entries to an idle A-first queue without altering research
configs or results, run `python scripts/prepare_iid_queue_resume.py` while no
worker is active. In an already-open GUI, click **Load Queue** and select
`gui/queues/iid_abcd_500r_gpu_resume.json` so stale in-memory stopped statuses
are replaced. The recovery column shows the actual checkpoint cursor;
"fresh" in the original immutable configuration names is not a restart request.

The latest explicitly requested **from-scratch reset** moved 32 partial output
files to `gui/worker_state/iid-reset-n_l7jwqq/`, verified their SHA-256 hashes,
and left A/B/C/D pending at 0/500. Those files are recoverable locally, ignored
by Git, and not mixed into the fresh results. Dataset caches and papers were
not moved. Reopen the GUI or load the fresh preset before starting; no research
process was launched by the reset or cloud preparation.

```text
newResults/IID/                       newPlots/IID/
  A_clean_no_attack_ce_500r_gpu/         A_clean_no_attack_ce_500r_gpu/
  B_attack_clean_ce_500r_gpu/            B_attack_clean_ce_500r_gpu/
  C_attack_noise_ce_500r_gpu/            C_attack_noise_ce_500r_gpu/
  D_attack_noise_ebm_500r_gpu/           D_attack_noise_ebm_500r_gpu/
  ABCD_500r_comparison_gpu/             ABCD_500r_comparison_gpu/
```

Plots use actual measured rounds, including per-class and EBM/noise curves.
`python scripts/report_iid_campaign.py --device GPU --rounds 500` regenerates
completed evidence without training. Late windows are R480–489 and R490–499;
no automatic extension occurs. Non-IID directories remain empty.
Prior observed per-condition runtime was roughly 8–9 hours. The new checkpoint
format removes growing history-rewrite overhead, but a complete post-change
research runtime has not been measured. Budget for a long multi-condition run;
do not interpret the I/O benchmark or increased VRAM cap as a guaranteed ETA.

Current focused CPU validation: **202 passed, 2 skipped** (17.67 s), covering
resource guards, checkpoint equivalence/corruption, protocol math, IID/cache,
tracking, queue, plotting and GUI state contracts. Both skips were display-only
integration tests; running those separately under Xvfb gave **2 passed** (2.31 s),
including loading the fresh four-entry 500-round queue with process creation
explicitly forbidden. The synthetic RTX 4070 Ti verification passed separately;
these are tests, not research run results. Commands executed:

```bash
python -m compileall -q gui basil_core reporting scripts tests
DISPLAY= PAPERMERGE_LIVE_TRACKING=disabled PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
environment/basil-noise-env/bin/python -m pytest -q \
  tests/test_memory_safety.py tests/test_iid_terminal_queue.py \
  tests/test_protocol_recovery.py tests/test_iid_500r.py \
  tests/test_live_tracking.py tests/test_iid_gpu.py tests/test_gui_live_updates.py \
  tests/test_iid_study.py tests/test_research_protocol.py tests/test_research_audit.py \
  tests/test_cifar_cache.py tests/test_iid_campaign.py tests/test_iid_plot_pipeline.py \
  tests/test_gui_architecture.py tests/test_performance_profiles.py
environment/basil-noise-env/bin/python scripts/verify_iid_gpu.py
environment/basil-noise-env/bin/python scripts/run_iid_queue.py
PAPERMERGE_LIVE_TRACKING=disabled PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
xvfb-run -a environment/basil-noise-env/bin/python -m pytest -q \
  tests/test_gui_live_updates.py::test_real_widgets_stay_stable_and_network_and_chart_render \
  tests/test_iid_campaign.py::test_real_gui_loads_four_idle_configs_and_presets_without_launch
```

The queue command is a read-only dry-run and confirmed A/B/C/D fresh at 0/500.
The suite is focused, not a claim that every historical/legacy test was run.

## Authorized cleanup and publication

5,822 old output files (~6.05 GB) were removed after SHA-256 inventory, including
formerly protected results4/plots4 and completed IID evidence. 51 obsolete
documents were deleted; README files and this guide remain. All 25 papers
retain their original hashes. Tracked deletions are recoverable from Git;
deleted untracked files are not. The deletion manifest is ignored local state.

Handoff documents are excluded by ignore rules, commit/push hooks and CI.
Enable hooks in new clones with `git config core.hooksPath .githooks`.
These are safeguards, not an unbypassable security boundary. Existing Git
history was not rewritten and can contain previously committed documents.
The push hook also checks newly pushed commits; it can therefore block a
push containing an older unpushed handoff commit even if the current files
were deleted. Resolving that needs a separately authorized history cleanup.
No commit/push or research execution occurs during preparation.
