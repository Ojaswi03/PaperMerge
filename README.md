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
- manual runs and queue runs
- automatic result saving
- automatic plot generation after each completed experiment
- persistent queue state in `gui/queue_state.json`

The queue can be edited while it is running. New configs are appended to the bottom and picked up after earlier items finish.

## Run One Config From CLI

```bash
source environment/basil-noise-env/bin/activate
python scripts/run_single_config.py "gui/configs/nonIID/cart/0 - Byzantine Nodes + No Channel Noise + No Mitigation.json" --rounds 30
```

## Directory Layout

```text
PaperMerge/
├── runGui.py
├── basil_core/
│   ├── basil.py
│   ├── cart.py
│   ├── trainer.py
│   ├── attacks.py
│   ├── models.py
│   └── data/
├── gui/
│   ├── experimentGui.py
│   ├── queue_state.json
│   └── configs/
│       ├── IID/{basil,noisy,merged,cart}/
│       └── nonIID/{basil,noisy,merged,cart}/
├── plots/
│   └── plotGui.py
├── scripts/
│   ├── run_single_config.py
│   └── testSetup.py
└── experiments/results/gui/
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
- CART proximal distillation is for non-IID forgetting.
- Clean runs should not use SS or EBM just to improve accuracy.

For Merged and CART with no SS, the clean decentralized baseline uses consensus averaging: current model plus received neighbor models, then local training. This is normal decentralized aggregation and is not SS mitigation.

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

For EBM configs that keep scale fixed at `2.0`:

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

The sigma-specific configs are generated for IID/non-IID, CART/Merged, 0-Byzantine channel-noise cases, and 4-hidden-Byzantine channel-noise cases.

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
| `scripts/run_single_config.py` | CLI config runner |
| `basil_core/basil.py` | BASIL ring and FedAvg loops |
| `basil_core/cart.py` | CART ring loop |
| `basil_core/trainer.py` | parameter helpers and training/evaluation |
| `plots/plotGui.py` | result discovery and plot generation |

## Validation

```bash
environment/basil-noise-env/bin/python -m py_compile gui/experimentGui.py plots/plotGui.py scripts/run_single_config.py basil_core/basil.py basil_core/cart.py
git diff --check
```
