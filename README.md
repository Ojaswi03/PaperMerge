# PaperMerge

## Latest completed results — 100-round IID BASIL

| Start here | Contents |
|---|---|
| [Latest A/B/C/D results and scientific analysis](newResults/IID/ABCD_100r_comparison_gpu/FINAL_ABCD_100R_ANALYSIS.md) | Ten evidence tables, comparison plots, convergence, attack filtering, noise/EBM diagnostics, and the accuracy-limit discussion |
| [Code changes and training mathematics](docs/IID_CODE_CHANGES_AND_TRAINING_MATHEMATICS.md) | Actual Python excerpts, current source line ranges, diagrams, and mathematical/training explanations |
| [Repository and study handoff](docs/RESEARCH_HANDOFF_PROMPT.md) | Complete context to accompany a repository ZIP |

All four fresh GPU conditions completed **100 measured rounds**. Final fixed
Node-9 full-test accuracies: **A 58.83% · B 58.32% · C 50.09% · D 50.26%**.
These are one-seed results, not universal accuracy or robustness guarantees.

The committed sharing bundle includes the reports, verification summary,
per-run configs/manifests, CSV/NPZ metrics, partition audits, compressed
activation telemetry, final node weights, execution logs, and current PNGs.
Large activation-boundary checkpoints, raw batch traces, datasets, caches,
machine-local worker state, and earlier ignored archives remain local. A
GitHub ZIP is therefore a review bundle, not a complete checkpoint-resume
backup or a self-contained rerun of every deep raw-trace audit.

## Project overview

PaperMerge is a Python and TensorFlow research framework for studying
Byzantine-resilient decentralized and federated learning over noisy
communication channels. It combines experiment configuration, isolated worker
execution, queue management, runtime estimation, telemetry, result inspection,
and research plotting in a Tkinter desktop application.

The framework supports four approaches:

| Approach | Topology | Purpose |
|---|---|---|
| BASIL | Decentralized ring | Byzantine resilience through Snapshot Selection |
| Noisy Channel | Server/client | Communication-noise experiments and EBM/WCM mitigation |
| Merged | Decentralized ring | BASIL-style resilience combined with noise mitigation |
| CART | Decentralized ring | Class-aware training for non-IID data, with optional SS and EBM |

## Quick start

Requirements:

- Python 3.12
- Tkinter
- Dependencies from `environment/requirements.txt`
- NVIDIA GPU with compatible CUDA/cuDNN recommended for full experiments

Create the environment:

```bash
python3 -m venv environment/basil-noise-env
source environment/basil-noise-env/bin/activate
pip install -r environment/requirements.txt
python scripts/check_setup.py
```

Start the desktop application:

```bash
python run_gui.py
```

CPU execution is supported but full experiments can be substantially slower.

## Four-condition IID BASIL configurations and manual execution

Four fixed 100-round configurations are available in the Builder presets and
the saved GUI queue: A clean/no attack/CE, B attack/clean/CE,
C attack/absolute noise/CE, D attack/absolute noise/source EBM.
All use the same IID data, initialization, five full epochs, batch 512,
approved SGD round decay, receiver-local Snapshot Selection and BASIL memory S=5.
C/D use coordinate Gaussian standard deviation 0.010 (variance 0.0001).
D alone differentiates `CE + 0.0001 ||grad CE||²`, without lambda or anchoring.

The queue starts only when you press **Run Queue**. Interrupted runs now restore
activation-boundary checkpoints containing all node weights and BASIL memories.
Older interrupted runs without checkpoints require deterministic reconstruction:
saved activations are replayed and checked against their original hashes and
telemetry before continuation. This costs training time; it is not an instant
weights-only resume. Completed 50-round reference results remain untouched.

```bash
source environment/basil-noise-env/bin/activate
python run_gui.py
```

The latest **fresh GPU A rerun and fresh GPU B/C/D are complete**. The saved
queue presets remain available for explicitly authorized future runs; loading
a preset never starts training. Do not start them merely to inspect the
completed evidence. `gui/queues/iid_abcd_100r_gpu_fresh.json` describes the
fresh four-condition setup; `gui/queues/iid_a_100r_gpu_fresh.json` describes
the now-completed A-only rerun. Open **Queue → Load Queue** only when you
intend to prepare a new execution, inspect its entries and paths, and use
**Run Queue** only after separately authorizing that execution.
No CPU-trained prefix was restored in the current completed comparison.
The previous mixed-device A results/plots and comparisons are preserved in
`reference_archive/`; see [fresh A rerun instructions](docs/IID_FRESH_A_GPU_QUEUE.md).
The earlier `iid_abcd_100r_gpu.json` remains a CPU-continuation preset and should
not be loaded for this fresh run. The prior B/C/D preparation is documented in
[fresh GPU queue instructions](docs/IID_FRESH_GPU_QUEUE.md).
Dashboard names the running experiment and separates experiment elapsed/ETA
from queue-session elapsed/ETA. Accuracy uses blue dotted (latest node), green
solid (round mean), and orange dashed (round worst) lines.

GPU outputs use the existing A/B/C/D 100-round directory names with an **`_gpu`**
suffix under `newResults/IID/` and `newPlots/IID/`; originals remain unchanged.
Four-way GPU plots/reports use `ABCD_100r_comparison_gpu/`. Regenerate these
without training with `python scripts/report_iid_campaign.py --device GPU`.
TensorFlow is capped at 4 GiB, requires 5 GiB free before launch, and allows only
one IID GPU process. Synthetic 512-image CE/EBM checks peaked at about 378 MiB.
Other applications can still consume VRAM; no absolute OOM guarantee is possible.
See [GPU execution and recovery](docs/IID_GPU_EXECUTION.md) for measured speed,
verification scope, memory safeguards and CPU→GPU provenance limitations.
A/B measures attacks under BASIL, B/C measures channel noise, and C/D is the
matched EBM comparison. No outcome is assumed in advance.
See [preparation, pairing and manual execution](docs/IID_ABCD_100R_PREPARATION.md).
See [interrupted-run recovery](docs/IID_QUEUE_RECOVERY.md) before restarting a stopped queue.

## Completed paired 50-round IID BASIL reference

The separate `sequential_basil_iid_v1` protocol pairs clean-channel BASIL with
ordinary cross-entropy against absolute-Gaussian-channel BASIL with
`CE + sigma_e² ||grad CE||²`. Both use ten IID nodes, the same four seeded
Hidden attackers, five-predecessor memory, receiver-local Snapshot Selection,
strict handoff, and five complete local epochs. No anchoring or legacy lambda
is used. The existing one-class protocol is unchanged.

The standard deviation is 0.010; variance/EBM coefficient is 0.0001.
This moderate calibration-based value was declared before the research results,
not selected by test accuracy. Do not overwrite this completed evidence to
prepare the new four-condition queue; it uses separate 100-round paths.
Execution is deterministic single-process CPU; GPU equivalence is not asserted.

Results and plots are routed exclusively to `newResults/IID/` and
`newPlots/IID/`; the corresponding `nonIID/` directories remain empty.
Primary figures represent full research runs, not three-round preflights.
Full-test accuracy/loss use the fixed end-of-ring Node 9 model. Mean-node
accuracy averages the ten model accuracies, without parameter averaging.
Both authorized 50-round full-data runs completed: final full-test accuracy
was 54.35% for clean CE and 50.57% for noisy EBM (fixed Node 9 reference).
This two-condition comparison does not isolate EBM's causal benefit.
See [the study summary](newResults/IID/IID_TWO_TEST_SUMMARY.md) for measured
evidence and [the accuracy comparison](newPlots/IID/comparison/full_test_accuracy_comparison.png).

Verify completed artifacts without training:

```bash
python scripts/verify_iid_research_outputs.py --write-evidence
```
These historical two tests do not isolate EBM's causal benefit. The completed
four-condition study now supplies that matched noisy-channel/CE-only control
as C; C versus D is the primary EBM comparison. No further run starts
automatically.

## Desktop application

PaperMerge uses a persistent five-workspace shell with a shared command bar.

- **Dashboard** shows active-run progress, rounds, elapsed time, estimated
  completion, accuracy, worker count, queue length, recent runs, and a live
  accuracy chart.
- **Experiment Builder** provides grouped experiment settings, presets,
  validation, conditional controls, defaults, and a human-readable summary.
- **Queue** provides filtering, multi-selection, ordering, retry, removal,
  runtime estimates, configuration-library loading, and persistent execution.
- **Results** discovers retained runs lazily and provides filters, metadata,
  convergence charts, comparison, export, plot generation, and output access.
- **Network** visualizes the directed ring, Byzantine nodes, Snapshot Selection
  candidates, selected sources, channel state, per-node accuracy, and telemetry
  replay for compatible runs.

Live IID accuracy updates after each completed node activation; round mean and
worst-node accuracy update after a complete ring traversal. They are distinct
metrics. The first sample is visible immediately. Dashboard widgets and queue
rows update in place, preserving focus/selection, rather than rebuilding on
every event. Network telemetry uses the same read-only adapter as replay.

The interface uses a light slate theme. Entries, dropdown fields, popup lists,
selected rows, disabled controls, and keyboard focus states have explicit
foreground and background colors to remain readable across operating systems.

## Experiment configuration

The GUI uses a typed internal experiment model while preserving the existing
JSON contract:

- Persisted keys remain camelCase.
- Schema versions 1–5 are supported; version 5 adds the explicit research protocol.
- Unknown compatible fields survive load/save round trips.
- Invalid numeric or conditional values remain visible and block Run/Queue.
- Tkinter variables are not used as the business model.

Configuration libraries are stored under:

```text
gui/configs/current/
gui/configs/adaptive_study/
gui/configs/IID/
gui/configs/nonIID/
gui/configs/sequential_basil/
```

Historical campaign or study identifiers inside persisted metadata are
experimental provenance and are intentionally not renamed.

## Execution and queue safety

Versioned studies run in isolated child processes. The application preserves:

- atomic queue persistence,
- duplicate-run prevention,
- structured worker events,
- main-thread-only Tkinter updates,
- graceful stop requests,
- stopped and failed queue recovery,
- orphan-worker detection,
- worker crash logs,
- conservative single- or multi-lane execution policy,
- runtime estimates based on compatible historical runs.

Closing the application during active work requests a graceful stop and waits
for managed workers instead of immediately destroying the process tree.

## Results and plots

Result discovery reads lightweight `run.json` metadata first. Large metric and
telemetry arrays are loaded only when a run is selected or replayed.

The retained research artifacts are:

```text
experiments/results4/
plots4/
```

These directories are protected historical evidence. Application code must
not rename, normalize, or overwrite their contents. New plot generation must
use a separate output directory selected by the user.

Plotting implementations remain in:

```text
reporting/adaptive_study_plots.py
reporting/baseline_study_plots.py
```

## Architecture

```text
run_gui.py
gui/
├── app.py                    # application lifecycle and service composition
├── experiment_app.py         # compatibility import only
├── theme.py                  # shared visual design system
├── state/                    # typed experiment, queue, navigation, execution state
├── services/                 # config, queue, execution, results, and plot operations
├── views/                    # Dashboard, Builder, Queue, Results, Network shell
├── components/               # command bar, navigation, scrolling, empty states
├── worker_pool.py            # isolated worker lifecycle
├── network_view.py           # live ring and telemetry replay
├── runtime_estimator.py      # per-run and queue estimates
├── baseline_study.py         # baseline-study protocol and identifiers
└── adaptive_study.py         # retained adaptive-study protocol and identifiers

basil_core/
├── experiment_engine.py
├── adaptive_experiment_engine.py
├── basil.py
├── cart.py
├── attacks.py
├── trainer.py
├── models.py
└── data/

reporting/
├── adaptive_study_plots.py
├── baseline_study_plots.py
├── experiment_plots.py
└── comparison_plots.py
```

Research algorithms remain in `basil_core/`; GUI services call those modules
without duplicating experiment mathematics in visual components.

## Command-line execution

Run a compatible JSON configuration directly:

```bash
python scripts/run_single_config.py path/to/config.json
```

Use a temporary round override for a lightweight check:

```bash
python scripts/run_single_config.py path/to/config.json --rounds 2
```

Do not use a shortened run as scientific evidence.

## Separate one-class sequential BASIL protocol — paused

The earlier one-class/high-relative-noise verdict is **NOT READY FOR
PRODUCTION**. That numerical/reproducibility
audit records high-relative-noise failures and residual cross-process gradient
differences. See [production readiness](docs/RESEARCH_PRODUCTION_READINESS.md)
before launching that separate protocol. This does not replace the completed
IID evidence linked above. The IID GPU path has representative synthetic
[hardware verification](docs/IID_GPU_EXECUTION.md) and completed 100-round
results; neither establishes bit-identical CPU/GPU execution or resolves the
separate multiprocessing discrepancy.

The new protocol has its own engine and outputs, leaving historical studies
unchanged. It uses 10 CIFAR-10 nodes, one complete class per node, strict weight
handoff, five full local epochs, batch 512, reset SGD, and rolling BASIL memory
from five distinct predecessors. Each full-data activation performs 50 updates,
including the final 392-image batch of each epoch.

Channel noise and training defenses are separate settings. Choose **Paper
absolute Gaussian** (coordinate standard deviation) or **Relative-L2 Gaussian**
(model-relative perturbation). Full gradient-norm EBM uses nested autodiff;
legacy gradient scaling is a separate comparison, not the same loss.

Generate configs and run the bounded diagnostic gate:

```bash
python scripts/generate_research_protocol_configs.py
python scripts/calibrate_channel_noise.py --semantics relative_l2_gaussian --evaluate
python scripts/calibrate_channel_noise.py --semantics paper_absolute_gaussian --evaluate
python scripts/run_research_diagnostics.py --family smoke --jobs 2
python scripts/run_research_diagnostics.py --family preflight
python scripts/report_research_diagnostics.py --plots
```

The 47 smoke configurations use three rounds and reduced data, **not reduced
epochs**. Three clean full-data preflights use five rounds. Production configs
are prepared but require explicit CLI approval via `--allow-production` after
reviewing the baseline; the GUI cannot silently launch the 100-round matrix.

New outputs live in `experiments/research_protocol_results/` and
`plots/research_protocol/`, separated into smoke, preflight, and production.
Results include before-training and five-epoch class metrics, outgoing-link
noise statistics, candidate losses, optimizer-step objectives, and all ten final
node checkpoints. Results browsing, plotting, and network replay support them.

See the [protocol guide](docs/SEQUENTIAL_BASIL_PROTOCOL.md),
[source-equation note](docs/SOURCE_EQUATIONS.md), and
[measured diagnostic report](docs/RESEARCH_DIAGNOSTIC_REPORT.md).
[Verification commands and artifact hashes](docs/RESEARCH_PROTOCOL_VERIFICATION.md)
record the actual checks and environment boundaries. The >50%
worst-node target is a research objective, never a software correctness test.

The sequential BASIL research protocol uses five complete local epochs, strict
snapshot handoff, bounded five-snapshot memory, receiver-local Snapshot Selection,
and optional Gaussian channel noise. Old saved protocol identifiers are accepted
as deprecated compatibility aliases; new configurations use
`sequential_basil_one_class_v1`. Historical results remain read-only and retain
their original provenance. See [the naming migration](docs/RESEARCH_NAMING_MIGRATION.md).

## Testing

Latest publication checks: compilation passed; the focused IID/protocol,
recovery, GPU-policy, GUI-state, configuration-library, runtime-estimator and
worker suites completed with **178 passed, 2 display-dependent skips and
644 subtests passed**. This is not a claim that the entire legacy suite passed.
No new scientific condition was executed for publication.

Compile the active source tree:

```bash
python -m compileall -q gui basil_core reporting scripts tests
```

Run the display-independent suite without unrelated globally installed pytest
plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  --ignore=tests/test_convergence.py
```

Run the fast GUI architecture contracts:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_gui_architecture.py \
  tests/test_config_library.py \
  tests/test_runtime_estimator.py \
  tests/test_worker_pool.py
```

`tests/test_convergence.py` loads MNIST during collection and may require
network access when the dataset is not already cached. Tests should not launch
full research campaigns merely to verify the GUI.

## Documentation

- [Documentation index](docs/README.md)
- [Repository overview](docs/GetToKnow.md)
- [GUI architecture](docs/GUI_REDESIGN_PLAN.md)
- [Repository audit](docs/REPOSITORY_AUDIT.md)
- [Baseline study protocol](docs/BASELINE_STUDY_GUIDE.md)
- [Adaptive study plan](docs/ADAPTIVE_STUDY_PLAN.md)
- [Adaptive runtime research](docs/ADAPTIVE_RUNTIME_RESEARCH.md)
- [CART gamma explanation](docs/gammaExplained.md)
- [WCM pilot](docs/WCM_PILOT.md)

Detailed experimental rationale, equations, protocol gates, and historical
findings belong in these documents rather than in this README.
