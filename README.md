# PaperMerge

PaperMerge is a TensorFlow research framework for Byzantine-resilient federated learning with noisy communication channels. It includes a Tkinter GUI, JSON experiment configs, CLI config execution, automatic result saving, and plot generation.

The project compares four approaches:

| Approach | Topology | Main purpose |
|---|---|---|
| BASIL | decentralized ring | Snapshot Selection for Byzantine resilience |
| Noisy Channel | FedAvg/server-client | EBM/WCM for channel-noise mitigation |
| Merged | decentralized ring | combines BASIL-style Byzantine defense with EBM noise mitigation |
| CART | decentralized ring | class-aware ring training for non-IID data, with optional SS and EBM |

Merged and CART are the project additions built on the two main papers.

## Documentation

- [Get To Know PaperMerge](docs/GetToKnow.md) is the advisor-facing repository
  map. It starts with what has been tested so far, explains every execution
  stage with diagrams, documents result meaning and runtime, and includes the
  file/function reference.
- [Campaign 3 R2 Guide](docs/Campaign3Guide.md) contains the exact current
  protocol, equations, calibration rules, and paper-facing interpretation.
- [Campaign 4 Engineering And Evaluation Plan](docs/Campaign4Plan.md) defines
  the isolated `results4`/`plots4` campaign, sigma `0.4-0.6` diagnosis,
  adaptive EBM extension, live node-ring GUI, and GPU runtime targets.
- [Gamma Explained](docs/gammaExplained.md) explains CART `gamma` and the
  active proximal coefficient `mu`.
- [WCM Pilot](docs/WCM_PILOT.md) documents the isolated WCM implementation,
  safety boundary, and evaluation plan.
- [Documentation index](docs/README.md) links the guides and generated result
  report/tables.

## Setup

Requirements:

- Python 3.12
- NVIDIA GPU with CUDA/cuDNN recommended
- CPU fallback works but is slow

```bash
python3 -m venv environment/basil-noise-env
source environment/basil-noise-env/bin/activate
pip install -r environment/requirements.txt
python scripts/testSetup.py
```

## Run The GUI

```bash
source environment/basil-noise-env/bin/activate
python runGui.py
```

The GUI supports:

- IID and non-IID config loading
- BASIL, Noisy, Merged, and CART approaches
- a dark interface across the main window, queue, pickers, log, and live chart
- manual runs and queue runs
- automatic result saving
- Campaign 3 plot generation at manual-run and queue-batch boundaries
- persistent queue state in `gui/queue_state.json`
- lane-aware live average/worst accuracy charts for isolated Campaign 3 workers
- a complete per-run configuration block in the Output experiment log
- empirical per-config durations plus total queue ETA, uncertainty range, and
  projected completion time

The queue can be edited while it is running. New configs are appended to the bottom and picked up after earlier items finish.
`Paper Core 101 · Add / Restore Missing` reconstructs only unfinished paper
runs without replacing custom queue items. Custom selections can be moved to
the top and run next while the paper campaign remains persisted underneath.

Use **Queue > Add from File** to choose individual configurations. The picker
defaults to the versioned **Current** library, supports multi-selection, and
shows separate IID/non-IID counts for BASIL, Noisy, Merged, and CART. When the
queue is stopped, adding selected files automatically sorts the resulting queue
by ascending estimated runtime. During a run, the selected block is sorted and
appended at the bottom without disturbing active work. **Shortest First** can
repeat the full sort, and stopped rows can be dragged into any manual order.
**Legacy / Custom** remains available as an explicit archive choice; it is
never mixed into the Current list.

The generated Current library contains:

| Split | BASIL | Noisy | Merged | CART |
|---|---:|---:|---:|---:|
| non-IID | 54 | 66 | 99 | 99 |
| IID | 54 | 66 | 99 | 99 |

Merged and CART files are exact Campaign 3 R2 confirmation configs and write
to `experiments/results3/r2` and `plots3/r2`. BASIL and Noisy files are
deterministic standalone paper-baseline configs and retain their existing
standalone result routes. Regenerate the checked-in library after changing its
source contract with:

```bash
python scripts/sync_gui_config_library.py
```

This config-picker and theme work does not change the SS selection rule, the
EBM objective, their coefficients, CART, attack timing, or training batch size.

## Campaign 3 R2

Campaign 3 R2 is the isolated hidden-attack study used for new conference-paper
experiments. Its Queue buttons load CART Low-Noise Refinement (6), Low-Noise
Repair (6), Calibration (49), Merged Core (99), CART Add-on (99), IID Controls
(48), CART Non-IID Controls (9), the Full Campaign (246), or the paper-focused
Core Confirmation matrix (105). The Paper Core button appends only missing runs
without replacing custom items; an explicit confirmed replacement action is
also available. Four current confirmation results are reused, leaving 101
pending. It uses deterministic seeds, one shared GPU model per experiment,
atomic run metadata, and a blocking CART calibration gate.

Campaign queue items run in isolated child processes. TensorFlow state is
cleared explicitly and then the child exits, guaranteeing CUDA allocator
release before the next item. Normalized CIFAR arrays and deterministic
partition indices are cached on CPU/disk; model, optimizer, iterator, registry,
and snapshot state are never reused between experiments.

The Queue window can benchmark one versus two GPU workers. Two lanes are
enabled only when both workers fit the configured memory cap, produce identical
result fingerprints, and deliver at least 1.4x measured throughput. On the
current RTX 4070 Ti setup the measured speedup was 1.01x, so one lane is used.
Each queue row shows its estimated duration. The queue summary reports remaining
wall-clock time and, when multiple lanes are active, summed config work
separately. Completion logs preserve the measured wall time for each run.

```text
experiments/results3/r2/   # corrected R2 results only
plots3/r2/                 # corrected R2 plots, diagnostics, and tables only
```

The original `results3`/`plots3` campaign is preserved outside `r2`; R2 never
mixes those records with corrected runs. After completing the first repair,
run **CART Low-Noise Refinement R2 - 6**. Fresh campaigns can load
**Calibration R2 - 49**, which skips completed records and adds only missing
runs. Confirmation buttons remain locked unless `campaign_state.json` is
`frozen`.

See the [Campaign 3 R2 Guide](docs/Campaign3Guide.md) for the complete
protocol, equations, diagrams, result layout, pilot findings, and paper-draft
synchronization notes.

## Run One Config From CLI

```bash
source environment/basil-noise-env/bin/activate
python scripts/run_single_config.py \
  "gui/configs/current/nonIID/cart/2026 - clean_no_mitigation.json" \
  --rounds 30
```

Campaign 3 pilots can be run without creating official result files:

```bash
python scripts/run_single_config.py campaign3:cart:hidden_noise:ss_ebm:0.4 --rounds 10
```

## Directory Layout

```text
PaperMerge/
├── runGui.py
├── docs/
│   ├── GetToKnow.md
│   ├── Campaign3Guide.md
│   ├── Campaign4Plan.md
│   ├── gammaExplained.md
│   └── WCM_PILOT.md
├── basil_core/
│   ├── basil.py
│   ├── cart.py
│   ├── campaign_engine.py
│   ├── trainer.py
│   ├── attacks.py
│   ├── models.py
│   └── data/
├── gui/
│   ├── experimentGui.py
│   ├── campaign3.py
│   ├── campaign_workers.py
│   ├── config_library.py
│   ├── runtime_estimator.py
│   ├── queue_state.json
│   └── configs/
│       ├── current/{IID,nonIID}/{basil,noisy,merged,cart}/
│       ├── IID/{basil,noisy,merged,cart}/
│       └── nonIID/{basil,noisy,merged,cart}/
├── plots/
│   ├── plotGui.py
│   └── plotCampaign3.py
├── scripts/
│   ├── benchmark_campaign_workers.py
│   ├── run_campaign_worker.py
│   ├── run_single_config.py
│   ├── sync_gui_config_library.py
│   └── testSetup.py
├── experiments/results/gui/
└── experiments/results3/r2/gui/
```

## Outputs

No-channel-noise runs save here:

```text
experiments/results/gui/{IID|nonIID}/{dataset}/{attackKey}/{approach}/
plots/images/gui/{IID|nonIID}/{dataset}/{attackKey}/{approach}/
```

Channel-noise runs are bucketed by sigma:

```text
experiments/results/gui/{IID|nonIID}/{dataset}/{attackKey}/{approach}/sigma_0_4/
plots/images/gui/{IID|nonIID}/{dataset}/{attackKey}/{approach}/sigma_0_4/
```

This keeps IID/non-IID, approach, attack type, and noise level separated.

## Experiment Semantics

Clean environment means:

- 0 Byzantine nodes
- no channel noise
- no mitigation

Mitigations are independent:

- SS / `useBasil=true` is only for Byzantine attacks.
- EBM/WCM is only for channel noise.
- CART class-aware proximal regularization is for non-IID forgetting.
- Clean runs should not use SS or EBM just to improve accuracy.

Campaign 3 R2 clean runs use exact all-node consensus and no mitigation. Other
Campaign 3 R2 arms use pairwise consensus before local training. Consensus is
normal decentralized aggregation, not SS mitigation.

R2 SS first rejects received snapshots outside a declared relative-L2
plausibility budget and then applies BASIL's lowest-local-loss rule to the
remaining received neighbors. The plausibility guard and pairwise consensus are
project integration mechanisms; they are not claimed as unchanged BASIL.

## Aggregation Mode

Configs may include:

```json
"aggregationMode": "consensus"
```

Defaults:

- `useBasil=true` -> `handoff`
- `approach=merged|cart` and `useBasil=false` -> `consensus`
- other cases -> `handoff`

`handoff` matches the BASIL ring handoff behavior. `consensus` averages current and received models before training.

## Channel Noise Sweeps

Legacy JSON sweeps may use the older fixed-scale shortcut:

```text
scale = 1 + lambda * sigma^2
lambda = 1 / sigma^2
```

Current values:

| sigma | lambda |
|---:|---:|
| 0.2 | 25.0 |
| 0.3 | 11.111111 |
| 0.4 | 6.25 |
| 0.5 | 4.0 |
| 0.6 | 2.777778 |

The sigma-specific legacy configs are generated for IID/non-IID, CART/Merged,
0-Byzantine channel-noise cases, and 4-hidden-Byzantine channel-noise cases.

Campaign 3 R2 does **not** use that shortcut. It differentiates the
noisy-communication objective
`F + lambda * sigma^2 * ||grad F||^2` with bounded second-order microbatches.
The configured effective batch remains 512. Its predeclared objective
coefficient is `0.00100` for sigma 0.2, `0.00025` for sigma 0.3-0.4, and
`0.00010` for sigma 0.5-0.6; `lambda` is derived by dividing that coefficient
by `sigma^2`. The bounded CART low-noise refinement separately tests
coefficient `0.00025` at sigma 0.2 because CART's proximal term changes the
combined optimization problem. The selected CART confirmation configs preserve
that coefficient (`lambda=0.00625`) at sigma 0.2. The objective and SS rule are
unchanged.

## Plot Outputs

The plot button and automatic post-run plotting generate:

- `experiments_avg.png`
- `experiments_avg_zoom.png`
- `grid_avg.png`
- `final_accuracy_avg.png`
- `improvement_over_no_mitigation_avg.png`
- `ablation_groups_avg.png`

Plots are saved into the matching split/dataset/attack/approach folder, and channel-noise plots are additionally separated by `sigma_*` folders.

## Key Source Files

| File | Purpose |
|---|---|
| `gui/experimentGui.py` | GUI, configs, queue, experiment execution |
| `gui/campaign3.py` | Campaign matrix, run IDs, calibration, result paths |
| `gui/campaign_workers.py` | isolated one/two-lane Campaign 3 process pool |
| `gui/config_library.py` | versioned individual BASIL/Noisy/Merged/CART config matrices |
| `gui/runtime_estimator.py` | matched-history per-config and whole-queue ETA |
| `scripts/run_campaign_worker.py` | one official Campaign 3 run per process |
| `scripts/benchmark_campaign_workers.py` | deterministic two-lane safety/throughput gate |
| `scripts/run_single_config.py` | CLI config runner |
| `basil_core/basil.py` | BASIL ring and FedAvg loops |
| `basil_core/cart.py` | CART ring loop |
| `basil_core/campaign_engine.py` | deterministic shared-worker Campaign 3 engine |
| `basil_core/trainer.py` | parameter helpers and training/evaluation |
| `plots/plotGui.py` | result discovery and plot generation |
| `plots/plotCampaign3.py` | Campaign 3 paper and diagnostic plots |

## Validation

```bash
environment/basil-noise-env/bin/python -m unittest tests.test_config_library tests.test_campaign3_contracts tests.test_campaign3_engine tests.test_runtime_estimator tests.test_campaign_workers tests.test_cifar_cache -v
environment/basil-noise-env/bin/python -m py_compile gui/experimentGui.py gui/config_library.py gui/campaign3.py gui/campaign_workers.py gui/runtime_estimator.py plots/plotCampaign3.py scripts/sync_gui_config_library.py scripts/run_single_config.py scripts/run_campaign_worker.py scripts/benchmark_campaign_workers.py basil_core/campaign_engine.py
git diff --check
```
