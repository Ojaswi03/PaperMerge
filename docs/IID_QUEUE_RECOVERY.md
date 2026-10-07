# Interrupted IID queue recovery

The queue remains manual and ordered **A → B → C → D**. Stop gracefully and
wait for Workers: 0 before reopening the application. No research run is started
by loading a configuration, opening the GUI, preparing a queue, or running tests.

```bash
source environment/basil-noise-env/bin/activate
python run_gui.py
```

Open Queue and inspect **Start / recovery**. Click **Run Queue** only when ready.
The saved queue has all four conditions pending; completed runs discovered later
are skipped. Retry makes a stopped entry pending without erasing its evidence.

## Two recovery modes

**Checkpoint resume:** `checkpoint.npz` commits after every completed activation
and round evaluation. It contains all ten node models, all fifty per-receiver
snapshot entries (sender, round, received weights and channel metadata), complete
activation telemetry, measured metric arrays and the next activation/round cursor.
Parameters are deduplicated by SHA-256. Writes use a temporary file, fsync and
atomic replacement, retaining `checkpoint.previous.npz` as fallback. Loading
checks configuration, partition, initialization, scientific source hashes,
parameter hashes, finite values and cursor consistency.

An interrupted activation is restarted from its selected input; it is never
counted as complete. SGD has no momentum and resets every activation, so no
optimizer slots need transmission/restoration. Randomness is already keyed by
seed/round/node/epoch/batch or sender/receiver, rather than a mutable shared RNG.
An interruption after Node 9 but before round evaluation is also recoverable.

**Legacy reconstruction:** interrupted outputs without weights or rolling-memory
checkpoints retain their completed activation records. A now has a checkpoint;
B/C still require reconstruction before their first GPU continuation.
The application rebuilds state from canonical initialization, requiring exact
agreement with every existing record: selected/trained/attacked and per-link
weight hashes, selection, losses, gradient telemetry and predictions. Completed
round CSV values are also checked. On disagreement the run aborts rather than
claiming continuation. Reconstruction is computationally expensive—B must replay
976 recorded activations before adding new measurements. This is explicit in
the Queue and Dashboard, not an instant resume from round 98.

Reconstruction can itself be interrupted: each verified activation commits a
checkpoint. The next launch restores that prefix instead of rebuilding from zero.
Previously recorded activation lines and round CSV rows are not duplicated or
replaced. Each recovery keeps the previous manifest and a separate audit directory
under `recovery_attempts/`; a torn final activation-log write is backed up before
repair. Dataset byte hashes and TensorFlow version must match the original run.
Original scientific settings, initialization and paired keyed RNG are unchanged.
Full legacy CIFAR reconstruction has **not** been executed during this change;
its equivalence is checked at runtime, not assumed from simulated tests.

## Dashboard

- Running experiment: the actual experiment identifier, independent of Builder edits.
- Experiment elapsed: all recorded attempts plus the current session, including reconstruction.
- Experiment ETA remaining: measured current-session work rate, accounting for
  checkpoint progress already restored. Until updates arrive it says Estimating.
- Queue elapsed: wall time for this queue launch, across A/B/C/D, excluding prior sessions.
- Queue ETA remaining: current experiment ETA plus usable pending estimates.
  GPU runs do not reuse CPU runtime estimates; unavailable estimates are explicit.
  Neither ETA controls training.
- Latest-node accuracy: blue dotted; round mean: green solid; round worst: orange dashed.
  Colors and line patterns distinguish curves without relying on color alone.

## Verification

Recovery tests simulate scalar node updates, actual memory delivery, keyed noise,
outbound attacks, strict handoff and stop boundaries. A runner integration test
uses the simulated worker to verify unchanged evidence prefixes, no duplicate
round rows, complete final telemetry and manifest provenance. Tiny TensorFlow
CE/source-EBM gradient evaluations check exact loss/gradient/next-update equality
after serialization; no CIFAR training or research queue is launched. Display
tests block subprocess launch and verify persistent widgets, experiment identity,
scoped timing labels and three distinct chart colors/patterns.

Full-size interrupted-run reconstruction has not been executed during these
changes. Hash mismatch or numerical failure aborts the affected condition.
Synthetic GPU CE/EBM checks now passed; see [GPU execution](IID_GPU_EXECUTION.md).
The remaining-work preset `gui/queues/iid_abcd_100r_gpu.json` transfers all CPU
weights/memories exactly into separate `_gpu` outputs after any necessary verified
CPU reconstruction. Subsequent GPU arithmetic is **not** bit-identical CPU
continuation. Manifests record the transition boundary and both device segments;
original CPU evidence remains untouched. Loading the preset leaves it idle.

Commands executed (no research training):

```bash
environment/basil-noise-env/bin/python -m compileall -q gui basil_core reporting scripts tests
DISPLAY= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python -m pytest -q tests/test_protocol_recovery.py tests/test_gui_live_updates.py tests/test_iid_campaign.py tests/test_gui_architecture.py tests/test_config_library.py tests/test_execution_policy.py tests/test_research_protocol.py tests/test_iid_study.py tests/test_iid_plot_pipeline.py -k 'not test_no_lambda_loss_gradients_updates_and_strict_load'
DISPLAY=:98 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MPLCONFIGDIR=/tmp/papermerge-mpl environment/basil-noise-env/bin/python -m pytest -q tests/test_gui_live_updates.py::test_real_widgets_stay_stable_and_network_and_chart_render tests/test_iid_campaign.py::test_real_gui_loads_four_idle_configs_and_presets_without_launch
```

Compile succeeded. Headless: **129 passed, 2 display tests skipped, 1 training
test deselected, 644 subtests passed**. Both display tests passed separately
under a temporary Xvfb display with research launch forbidden: **2 passed**.
The legacy convergence collection problem is outside this focused suite and is
not claimed fixed. The only warnings were two protobuf deprecations.

Before/after path-and-SHA-256 manifests matched all **5,363 historical files**,
**121 old paired-reference/comparison files**, and **31 interrupted A/B/C files**
(A: 11, B: 9, C: 11). Counts and hashes matched; no research result or plot was
rewritten during implementation. Queue-state status changes are intentional:
A/B/C/D are Pending and idle, not automatically started.
