# IID GPU execution and memory safety

For the current fresh GPU-only A rerun, follow
[Fresh GPU-only A](IID_FRESH_A_GPU_QUEUE.md); completed B/C/D are not rerun.
The prior [Fresh B/C/D GPU queue](IID_FRESH_GPU_QUEUE.md) is documented separately.
The remaining-work instructions below describe the earlier CPU-prefix continuation
preset, not either fresh queue.

The manual A → B → C → D queue now requests the NVIDIA GeForce RTX 4070 Ti
(12,282 MiB), TensorFlow 2.18.0, CUDA build 12.5.1 and cuDNN 9.
No research experiment was launched during implementation or verification.

## Start the prepared remaining-work queue

```bash
source environment/basil-noise-env/bin/activate
python run_gui.py
```

Open Queue. Load `gui/queues/iid_abcd_100r_gpu.json` if necessary. Confirm four
pending GPU entries, A → B → C → D, 100 rounds, and Idle / Workers: 0. Then
manually press **Run Queue**. Opening the application or loading a preset never
starts a worker. The prior CPU queue is backed up at
`gui/queues/iid_abcd_cpu_before_gpu.json`.

The prepared queue's **Done / total** values come from complete round evaluation:

| Condition | Complete rounds | Completed activations | First GPU launch |
|---|---:|---:|---|
| A | 3/100 | 38 | Restore complete CPU checkpoint |
| B | 97/100 | 976 | Reconstruct/verify missing state on CPU |
| C | 3/100 | 39 | Reconstruct/verify missing state on CPU |
| D | 0/100 | 0 | Fresh canonical initialization on GPU |

Incomplete rounds do not increase the complete-round counter. B's reconstruction
may take roughly three CPU hours, and C's several minutes. Both are one-time
costs, checkpointed after each verified activation so reconstruction can itself
resume after a graceful stop. Dashboard shows CPU during reconstruction and the
actual GPU name when GPU execution begins. GPU ETA is measured rather than
copied from historical CPU estimates.

## Memory safeguards

- TensorFlow's logical GPU is capped at **4,096 MiB**, before initialization.
- `nvidia-smi` must report at least **5,120 MiB free** (cap plus 1,024 MiB headroom).
  If memory cannot be checked or is insufficient, the run does not start.
- A process-held OS file lock permits only **one IID GPU process** at a time.
  Process exit/crash releases the lock; stale files do not imply active locks.
- Float32/TF32-off execution preserves the scientific precision policy; batch 512,
  final batch 392, five complete epochs, SGD and the loss remain unchanged.
- Images/augmentation and snapshot/checkpoint arrays remain on CPU. Evaluation is
  batched, not an allocation of the complete test set on the GPU.
- GPU OOM aborts rather than reducing batches/epochs or changing the objective.
  The audit saves pre-batch CPU evidence and skips allocation-heavy GPU graph
  replay. Recovery uses the last committed complete activation, never a partially
  applied SGD update.

Measured TensorFlow peak was **377.81 MiB** during synthetic CE/second-order EBM
verification with batch 512. This excludes driver/context memory and does not
prove the peak of an entire 100-round run. The cap leaves substantial headroom;
other applications can still consume VRAM, so an unconditional OOM guarantee is
not possible. Close unnecessary GPU applications before starting.

## Verification evidence and limits

Machine-readable results: [GPU_EXECUTION_VERIFICATION.json](GPU_EXECUTION_VERIFICATION.json).
The command performs bounded synthetic batches only, never CIFAR research runs:

```bash
python scripts/verify_iid_gpu.py
```

Checks covered the approved 117,706 parameters, canonical initialization,
512/392 training batches, 512/272 evaluation batches, CE, true second-order EBM,
finite tensors, SGD updates, same-device repeatability and no-lambda invariance.
The analytic ten-class toy EBM gradient errors were 2.77e-8 CPU / 4.67e-8 GPU.
The GPU adapter matched the original GPU objective's gradients/values exactly;
keyed CPU augmentation produced identical image hashes for CPU/GPU comparisons.

| Audited synthetic optimizer step | CPU median | GPU median | Speedup |
|---|---:|---:|---:|
| CE | 0.1610 s | 0.0695 s | 2.32× |
| Full EBM | 0.3109 s | 0.0545 s | 5.71× |

These are medians of five measured steps after two warm-up steps, not a whole-run
forecast. Evaluation, CPU augmentation, hashing, logging and compressed
checkpoints also cost time. Diagnostic evaluation visits about 800,000 test
images per full ring round across all repeated evaluations, not just the final
10,000-image round curve.

CPU/GPU gradients are not bit-identical: relative L2 difference was approximately
0.2994%; maximum SGD update difference was 8.24e-7. Initial strict per-coordinate
comparison failed; the report explicitly uses a 1% relative-L2 gradient bound and
2e-6 absolute update tolerance, not a claim of exact CPU equivalence. Predictions
matched on the tested synthetic input. Full-run numerical agreement/convergence
and multiprocessing correctness remain unverified.

## Scientific provenance and unchanged protocol

All ten node weights, fifty rolling-memory entries, cursor and recorded metrics
are transferred—not just the last model. CPU reconstruction uses original CPU
kernels/settings and requires exact existing activation telemetry/hash matches.
The original CPU result files and completed 50-round reference evidence are
read-only. GPU outputs add `_gpu` to the existing A/B/C/D directories; comparison
uses `ABCD_100r_comparison_gpu`. Reports do not silently mix old CPU and GPU paths.

The manifest records `deviceTransition`, original evidence hashes and
`executionDeviceSegments`. Transferred state is exact, but future GPU arithmetic
is not bit-identical CPU continuation. A/B/C therefore have CPU prefixes and GPU
suffixes at different boundaries; D is fresh GPU. This is an explicitly mixed-device
study, not a uniformly all-GPU paired study. Device-transition differences must
be considered when interpreting contrasts; new all-GPU reruns would require
separate authorization and are not silently substituted for remaining work.

Partition/initialization hashes, keyed seeds, BASIL S=5, strict selection/handoff,
five epochs, SGD round decay, attackers and attack schedule remain unchanged.
C/D use fixed coordinate sigma 0.010, and D uses exactly
`CE + 0.0001 ||grad CE||²`, no lambda, anchoring, clipping, momentum or weight decay.

Regenerate GPU plots without training:

```bash
python scripts/report_iid_campaign.py --device GPU
```

The runtime rejects missing/stale hardware verification and never silently falls
back to CPU. Changes to verified mathematical sources/GPU worker require another
synthetic verification before launching GPU research.

## Executed checks

```bash
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests
DISPLAY= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python -m pytest -q tests/test_iid_gpu.py tests/test_protocol_recovery.py tests/test_gui_live_updates.py tests/test_iid_campaign.py tests/test_gui_architecture.py tests/test_config_library.py tests/test_execution_policy.py tests/test_research_protocol.py tests/test_iid_study.py tests/test_iid_plot_pipeline.py -k 'not test_no_lambda_loss_gradients_updates_and_strict_load' --tb=short
DISPLAY=:98 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python -m pytest -q tests/test_gui_live_updates.py::test_real_widgets_stay_stable_and_network_and_chart_render tests/test_iid_campaign.py::test_real_gui_loads_four_idle_configs_and_presets_without_launch --tb=short
env -u CUDA_VISIBLE_DEVICES environment/basil-noise-env/bin/python scripts/verify_iid_gpu.py
```

Compile succeeded. Focused headless suites: **146 passed, 644 subtests passed,
2 display tests skipped, 1 CIFAR-training test deselected**, two protobuf warnings.
The two real-widget tests passed separately on temporary Xvfb with subprocess
launch forbidden. The first sandbox attempt could not connect to the display;
the same tests passed with local-display access. The existing legacy convergence
collection issue is not claimed fixed or included in these focused results.

Four zero-training CLI validations used `scripts/run_iid_condition.py --config`
with each JSON in `gui/configs/IID/sequential_basil_100r/gpu/`; all returned valid,
100 rounds and `executionStarted=false`. Config contrasts remained A/B attack
fields only, B/C channel enable/sigma only, C/D objective/EBM mode only. A's actual
checkpoint was read/verified without execution: 38 completed activations, three
evaluated rounds, ten node states and fifty memory entries.

Path-and-SHA-256 rechecks matched **5,363 historical files**, **121 completed
50-round reference/comparison files** and **39 interrupted CPU output files**
(A: 19, B: 9, C: 11, D: 0), with no additions, deletions or changed hashes in those
trees. Specifically `experiments/results4/` stayed at 2,153 files and `plots4/`
at 737. The GPU research result directories were not populated; queue remains idle.
