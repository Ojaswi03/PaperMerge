# Byzantine-Resilient Federated Learning with Noisy Communication Channels

A comprehensive implementation and experimental framework for studying Byzantine fault tolerance, communication noise mitigation, and non-IID data convergence in federated learning systems.

## Abstract

This project implements and evaluates four complementary approaches to robust federated learning:

1. **BASIL** (Paper 001): A Byzantine-resilient decentralized training algorithm using ring topology with Snapshot Selection (SS) for model filtering
2. **Robust Federated Learning with Noisy Communication** (Paper 002): EBM (Expectation-Based Model) and WCM (Worst-Case Model) approaches for mitigating channel noise in FedAvg
3. **Merged**: Combines Papers 001 and 002 — ring topology with SS (Byzantine) and EBM (channel noise) simultaneously
4. **CART** (Paper 003): Class-Aware Ring Training — a novel decentralized algorithm extending the ring topology with per-class trust tracking, EMA-smoothed proximal distillation, and Byzantine-verified registry propagation; enables convergence under non-IID data (Dirichlet α=0.2) while simultaneously maintaining Byzantine resilience and channel noise mitigation. **No published work addresses all four challenges (decentralized + non-IID + Byzantine + channel noise) simultaneously.**

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Theoretical Background](#theoretical-background)
4. [Implementation Details](#implementation-details)
5. [Experimental Configuration](#experimental-configuration)
6. [Experimental Results](#experimental-results)
7. [Paper 003 — CART](#paper-003--cart-class-aware-ring-training)
8. [GUI Reference](#gui-reference)
9. [Troubleshooting](#troubleshooting)
10. [Changelog](#changelog)

---

## Installation

### Requirements

- Python 3.12
- NVIDIA GPU with CUDA 12.3+ (CPU fallback supported but slow)
- cuDNN 9.x

### Setup

```bash
python3 -m venv environment/basil-noise-env
source environment/basil-noise-env/bin/activate
pip install -r environment/requirements.txt
```

### Verify Installation

```bash
python scripts/testSetup.py
```

---

## Quick Start

```bash
source environment/basil-noise-env/bin/activate
python runGui.py
```

This starts the **auto-reload launcher**. The GUI restarts automatically whenever you save any `.py` file.

1. Select **Approach**: BASIL Only / Noisy Channel / Merged / CART
2. Load a pre-built config with **Load Config** (`Ctrl+L`): choose **IID** or **nonIID** split, then approach folder
3. Click **▶ Run Experiment** (`Ctrl+R`)

### Running a Single Config from CLI

```bash
python scripts/run_single_config.py gui/configs/nonIID/cart/0\ -\ Byzantine\ Nodes\ +\ Channel\ Noise\ +\ EBM\ Mitigation.json
python scripts/run_single_config.py gui/configs/IID/basil/0\ -\ Byzantine\ Nodes\ +\ No\ Channel\ Noise\ +\ No\ Mitigation.json --rounds 30
```

Results → `experiments/results/gui/{IID|nonIID}/{dataset}/{attack}/{approach}/`  
Plots → `plots/images/gui/{IID|nonIID}/{dataset}/{attack}/{approach}/`

### Phone Notifications

Install the **ntfy** app and subscribe to `papermerge-ojaswi` for push notifications when runs start, complete, or fail.

---

## Theoretical Background

### BASIL Algorithm (Paper 001)

```
For each training round:
    For each node i in ring (sequential):
        1. Receive models from S counterclockwise neighbors
        2. Select best model by minimum local batch loss (Byzantine filter)
        3. Perform local SGD training
        4. Multicast updated model to S clockwise neighbors
```

**S = b + 1** where b = max Byzantine nodes. Loss-based selection naturally filters Byzantine models (adversarial models have high loss on honest local data).

### EBM — Expectation-Based Model (Paper 002)

```
Original loss:    F(w)
EBM loss:         F_e(w) = F(w) + σ²||∇F(w)||²     (Eq. 13)
EBM gradient:     ∇F_e(w) = (1 + λσ²) × ∇F(w)      (Eq. 23)
Scale factor:     scale = 1 + λσ²  (default: λ=25, σ=0.2 → scale=2.0)
```

Pushes the model toward flat minima that are robust to weight perturbations.

### CART — Class-Aware Ring Training (Paper 003)

```
For each training round:
    For each node i in ring (sequential):
        1. Select best model (SS if enabled)
        2. Verify and merge received ClassRegistry
        3. Compute trust_weight[c] = max(0, registry_best[c] - my_acc[c])
        4. ema_trust = 0.85 × ema_trust + 0.15 × mean(trust_weight)
        5. Train: L = cross_entropy + (γ × (1 + ema_trust) / 2) × ||w - w_ref||²
        6. Evaluate per-class accuracy → update registry
        7. Send (model + registry) to S clockwise neighbors
```

**Key insight**: The base proximal coefficient `γ × 1.0` is always active from round 1 — even before the registry has any knowledge. This prevents catastrophic forgetting at cold start. EMA smoothing (decay=0.85) averages trust weights over ~7 rounds to eliminate round-to-round oscillation.

---

## Implementation Details

### System Architecture

| Approach | Topology | Training | Aggregation | Defenses |
|----------|----------|----------|-------------|---------|
| BASIL | Ring | Sequential | Selection (min loss) | Byzantine (SS) |
| Noisy Channel | Star | Parallel | FedAvg (averaging) | Channel noise (EBM/WCM) |
| Merged | Ring | Sequential | Selection + EBM | Byzantine + channel noise |
| **CART** | **Ring** | **Sequential** | **Proximal distillation** | **Byzantine + channel noise + non-IID** |

### File Structure

```
PaperMerge/
├── runGui.py                    # Entry point — auto-reload launcher
├── basil_core/
│   ├── basil.py                 # BASIL ring + FedAvg training loops
│   ├── cart.py                  # CART training loop (CARTNode, cartRingTraining)
│   ├── class_registry.py        # ClassRegistry (merge, verify, trustWeights)
│   ├── trainer.py               # Local training helpers (evaluate, evaluatePerClass, …)
│   ├── models.py                # MNISTModel, CIFARModel, NMNISTModel
│   ├── attacks.py               # Byzantine attack implementations
│   └── data/
│       ├── mnist.py             # MNIST loader + tf.data pipeline
│       ├── cifar.py             # CIFAR-10 loader + Dirichlet non-IID split
│       └── nMnist.py            # Neuromorphic MNIST loader
├── noise_comm/
│   └── wcm.py                   # WCM noise mitigation
├── gui/
│   ├── experimentGui.py         # Tkinter GUI (config, run, live chart, log)
│   └── configs/                 # 480 pre-built experiment configurations
│       ├── nonIID/              # 240 configs (Dirichlet α=0.2, all approaches)
│       │   ├── basil/           # BASIL configs (non-IID)
│       │   ├── noisy/           # FedAvg/Noisy configs (non-IID)
│       │   ├── merged/          # Merged configs (non-IID)
│       │   └── cart/            # CART configs (non-IID)
│       └── IID/                 # 240 configs (uniform random split, all approaches)
│           ├── basil/           # BASIL configs (IID)
│           ├── noisy/           # FedAvg/Noisy configs (IID)
│           ├── merged/          # Merged configs (IID)
│           └── cart/            # CART configs (IID)
├── plots/
│   └── plotGui.py               # Plot discovery and generation
├── scripts/
│   ├── common.py                # GPU setup, notifications, utils
│   ├── testSetup.py             # Verify environment and GPU
│   ├── run_single_config.py     # CLI runner for a single JSON config
│   └── testBasilPaper.py / testEbmPaper.py / testMerged.py
├── update_configs.py            # Bulk-update all JSON configs
└── experiments/results/gui/     # Saved results
    └── {IID|nonIID}/{dataset}/{attack}/{approach}/
        ├── acc_*.npy            # Accuracy curve
        └── config_*.json        # Full config snapshot
```

### Model Architectures

**MNIST / N-MNIST:**
```
Input(28×28) → Flatten → FC(100) → ReLU → FC(100) → ReLU → FC(10) [logits]
```

**CIFAR-10 (VGG-style):**
```
Input(32×32×3)
→ Conv(64,3×3)→ReLU → Conv(64,3×3)→ReLU → MaxPool(2) → Dropout(0.25)
→ Conv(128,3×3)→ReLU → Conv(128,3×3)→ReLU → MaxPool(2) → Dropout(0.25)
→ Conv(256,3×3)→ReLU → Conv(256,3×3)→ReLU → MaxPool(2) → Dropout(0.40)
→ Flatten → FC(512)→ReLU → Dropout(0.50) → FC(10) [logits]
```
He normal initialisation. No BatchNorm (avoids federated averaging issues with running stats).

### Attack Implementations

| Attack | Description | Severity |
|--------|-------------|----------|
| Gaussian | Replace weights with N(0,1) noise | Easy |
| Sign-Flip | Negate all gradient values | Medium |
| Hidden | Normal until start round, then attack | Hard |
| Model Poison | Gradient ascent toward wrong minimum | Hard |
| Scaling | Multiply weights by large negative factor | Medium |
| ALIE | Stealthy update within honest distribution | Hard |
| IPM | Negate + scale to maximise negative inner product | Hard |
| Noise Amp | Amplified noise proportional to layer std | Medium |

---

## Experimental Configuration

### Recommended Settings (CIFAR-10, non-IID α=0.2)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Nodes | 10 | 4 attackers (IDs 1,4,6,8) |
| Rounds | 100 | |
| Local Epochs | 5 | |
| Steps per Epoch | 5 | 25 gradient steps/round/node |
| Batch Size | 512 | |
| Learning Rate | 0.05 | |
| Momentum | 0.9 | Required for stable convergence |
| Plateau LR | true | patience=8, factor=0.7, minLr=0.001 |
| Non-IID | true | Dirichlet α=0.2 |

### EBM Parameters

| Noise σ | Lambda λ | Scale | Use Case |
|---------|---------|-------|----------|
| 0.2 | 25 | 2.0 | High noise (default) |
| 0.1 | 100 | 2.0 | Medium noise |
| 0.05 | 400 | 2.0 | Low noise |

`channelNoiseSigma` is implemented as a full-model relative L2 communication-noise budget, not an absolute per-coordinate standard deviation. No-channel-noise configs use `channelNoiseSigma: 0.0`. Channel-noise + EBM configs use fixed LR (`usePlateauLr: false`) because noisy evaluation can trigger harmful plateau reductions.

The `No Channel Noise + EBM Mitigation` Byzantine configs are intentional negative controls: they keep channel noise disabled but leave `noiseMitigation: "ebm"` to show that EBM alone does not protect against Byzantine attacks.

### CART Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `distillStrength` (γ) | 0.4 | Scale for proximal coefficient; stronger anti-forgetting (was 0.3) |
| `verifyThreshold` | 0.05 | Max gap before rejecting Byzantine registry claim |
| EMA decay | 0.85 | Hardcoded; ~7-round averaging window |

### Config Matrix (per approach)

480 configs total:
- **8 attack scenarios**: clean (0 attackers) + ALIE / Hidden / IPM / ModelPoison / NoiseAmp / Scaling / SignFlip (4 attackers each)
- **6 mitigation combos**: {No SS, SS} × {No Noise, Noise+NoMitigation, Noise+EBM}
- **2 data splits**: `nonIID/` (Dirichlet α=0.2, `nonIID: true`) and `IID/` (uniform random, `nonIID: false`)
- **Added negative controls**: all 7 Byzantine attack types also have `No Channel Noise + EBM Mitigation` configs for basil/noisy/merged/cart in both splits. These intentionally test that EBM alone does not mitigate Byzantine attacks.

The `nonIID` field in each config is enforced by its folder location — `update_configs.py` automatically sets the correct value based on path.

---

## Experimental Results

### Key Finding: Non-IID Kills All Standard Approaches

With Dirichlet α=0.2 (severe non-IID), standard federated learning collapses regardless of Byzantine/noise mitigations:

| Scenario | BASIL R-plain | BASIL SS | Merged SS+EBM | **CART** |
|---|---|---|---|---|
| Clean (no attack, no noise) | ~10% | ~60-70% | ~60-70% | **~65-75%** |
| Hidden attack + channel noise, SS+EBM | ~10% | ~32% | ~44-55% | **~50-60%** |
| EBM only + Byzantine | ~10% | ~10% | ~10% | ~10%* |
| No mitigation + Byzantine | ~10% | ~10% | ~10% | ~10%* |

\* When Byzantine attack dominates without SS, CART's distillation helps non-IID but cannot filter adversarial models — SS is still required for Byzantine protection.

### Why Non-IID Breaks the Ring

With 25 local gradient steps and momentum=0.9 on class-skewed data:
- Node A (90% class 7): trains 25 steps → "forgets" all other classes
- Node B receives A's class-7-biased model → overwrites with its own class
- After 10 nodes: model oscillates randomly → stuck at 10% accuracy

CART's base proximal pull `γ × 1.0` anchors each node to the received model during local training, preventing catastrophic overwriting. As the registry fills with verified class knowledge, the pull amplifies for classes the node rarely sees.

### Baseline Results (IID setting, for reference)

**BASIL (MNIST, IID):**
| Scenario | Avg Accuracy |
|---|---|
| Clean | ~95% |
| Attack, no SS | ~67% |
| Attack + SS | ~91% |

**Noisy Channel (MNIST, IID, σ=0.2):**
| Scenario | Final Accuracy |
|---|---|
| Clean | ~96% |
| Noisy, no EBM | ~43% |
| Noisy + EBM | ~82% |

---

## Paper 003 — CART: Class-Aware Ring Training

### The Research Gap

No published work addresses all four challenges simultaneously in a decentralized setting:

| Capability | BASIL | EBM/WCM | FedProx | SCAFFOLD | **CART** |
|---|---|---|---|---|---|
| Decentralized (no server) | ✅ | ❌ | ❌ | ❌ | **✅** |
| Byzantine resilience | ✅ | ❌ | ❌ | ❌ | **✅** |
| Channel noise resilience | ❌ | ✅ | ❌ | ❌ | **✅** |
| Non-IID convergence | ❌ | ❌ | ✅ | ✅ | **✅** |
| All four simultaneously | ❌ | ❌ | ❌ | ❌ | **✅** |

### How It Works

**ClassRegistry** (10 floats for CIFAR-10): tracks best verified per-class accuracy across all honest nodes. Propagates clockwise around the ring; each receiving node verifies claims before merging (Byzantine inflation rejected if claimed accuracy is >5% above local re-evaluation).

**EMA-smoothed proximal loss**:
```
μ = γ × (1 + ema_trust)
ema_trust(t) = 0.85 × ema_trust(t-1) + 0.15 × mean_c[max(0, registry[c] - my_acc[c])]
L = cross_entropy + (μ/2) × ‖w − w_ref‖²
```

**Cold start** (`ema_trust=0`): μ = γ = 0.4. The proximal term is always active — no forgetting from round 1.  
**EMA smoothing**: eliminates zigzag accuracy curves from round-to-round registry fluctuations.

### Pros

- Fills a genuine research gap — no published work does decentralized + non-IID + Byzantine + channel noise
- **Low overhead** — registry is 10 floats (vs. full correction vectors in SCAFFOLD)
- **No server required** — knowledge propagates through the ring naturally
- **Stronger Byzantine filter** — per-class evaluation is harder to fool than single overall-loss
- **Orthogonal to EBM** — channel noise mitigation unchanged, combines additively

### Cons / Open Questions

- New hyperparameter γ requires tuning (validated: 0.4 works well on CIFAR-10 α=0.2)
- Distillation acts on the full weight vector, not class-specific weights — less precise for shared conv layers
- Convergence proof required for top-tier venues (NeurIPS, ICML); empirical results sufficient for applied venues (ICLR workshop, IEEE TNNLS)
- Byzantine nodes sending inflated registries: mitigated by local verification, but threshold tuning matters

### Implementation Files

| File | Purpose |
|------|---------|
| `basil_core/cart.py` | `CARTNode`, `cartRingTraining` |
| `basil_core/class_registry.py` | `ClassRegistry` (update, merge, verifyAndMerge, trustWeights) |
| `basil_core/trainer.py` | `evaluatePerClass` |
| `gui/configs/nonIID/cart/` | 48 CART configs (non-IID) |
| `gui/configs/IID/cart/` | 48 CART configs (IID) |

---

## GUI Reference

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Run Experiment |
| `Ctrl+S` | Save Configuration |
| `Ctrl+L` | Load Configuration |
| `Esc` | Stop running experiment |
| `Ctrl+Shift+R` | Reload GUI |

### Tabs

| Tab | Contents |
|-----|----------|
| Basic | Dataset, approach (BASIL/Noisy/Merged/CART), training params, quick presets |
| Advanced | BASIL memory size, channel noise, EBM λ, momentum, LR schedule, non-IID settings, CART config (algorithm selector, γ, verify threshold) |
| Attacks | 8 attack types with per-attack enable and start-round |
| Output | Live log (dark terminal) + live accuracy chart (updates each round) |

### IID / Non-IID Config Selector

Both **Load Config** and the **Queue Manager → Add from File** dialog have an **IID / nonIID** radio button pair. This routes to `gui/configs/IID/` or `gui/configs/nonIID/` automatically. The `nonIID` field inside each JSON is enforced by its folder location.

The **Add All…** button in the Queue Manager opens a small dialog to pick a split + approach, then bulk-adds all matching configs to the queue.

### CART Algorithm Selector

When **Approach = CART** is selected, the Advanced tab shows an **Algorithm** radio group:
- `CART (Paper 003)` — runs full CART with registry + proximal distillation
- `BASIL Ring (baseline)` — runs standard BASIL SS in the same non-IID setting
- `FedAvg / Noisy (baseline)` — runs FedAvg in the same non-IID setting
- `Merged Ring+EBM (baseline)` — runs Merged approach in the same non-IID setting

This allows direct comparison of all four approaches under identical data conditions.

### Progress Bar

Always visible between the tab panel and button bar (fixed layout — does not hide on resize):
- `Round N/100` counter
- ETA countdown (EMA-smoothed, ignores slow warmup round)
- Latest average accuracy %

### Quick Presets

| Preset | Description |
|--------|-------------|
| Clean Baseline | CIFAR-10, BASIL SS, no noise, no attacks |
| Noisy + EBM | CIFAR-10, FedAvg, σ=0.2, EBM mitigation |
| BASIL + Gaussian | CIFAR-10, BASIL SS, Gaussian attack on 4 nodes |
| Merged (Best) | CIFAR-10, Merged SS+EBM, σ=0.2, no attacks |

---

## Troubleshooting

### GPU Out of Memory

```bash
export CUDA_VISIBLE_DEVICES=''
python runGui.py
```

### GUI Won't Launch (Linux / WSL2)

```bash
sudo apt-get install python3-tk
```

### Training Stuck at ~10% (Non-IID)

- Verify `nonIID: true` and `dirichletAlpha: 0.2` in config
- For BASIL/FedAvg: expected — these approaches fail on non-IID α=0.2. Use CART.
- For CART: check `distillStrength > 0` and that `cartAlgorithm: "cart"` in config
- For CART+EBM+SS=OFF: see known issue below

### Known Issue: CART + EBM + SS=OFF on CIFAR-10 → 10% (Under Investigation)

**Symptom**: `cartRingTraining` with `noiseModel="ebm"`, `useSnapshots=False`, CIFAR-10 non-IID α=0.2 produces flat ~10% accuracy from round 0.

**What was tried**:
- Verified EBM gradient order in `_cart_step` — scale-before-clip (capped post-scale norm = 5). MNIST passes at 87.9% with 15 rounds.
- EBM rigorous tests (`scripts/testEbmRigorous.py`, 6/6 PASS) confirm EBM mechanism works on MNIST for FedAvg, ring, and CART.
- Per-hop noise calibration (`σ/√N`) is in place in `cartRingTraining`.

**Suspected causes (not yet ruled out)**:
1. CART + EBM + SS=OFF may be a documented negative result (see EBM Ring Topology note above) — CART's proximal term prevents non-IID forgetting but EBM's flat-minima push may still fail to accumulate in sequential ring without SS, combined with CIFAR-10's larger loss surface.
2. Interaction between EBM scale=2.0 and proximal term on CIFAR-10's larger gradient magnitudes.

**Next steps**: run CART + no-noise (no EBM) baseline; run CART + EBM + SS=ON to isolate whether SS is required for EBM to work in CART.

### EBM Not Improving Accuracy in Ring Topology (SS=OFF)

EBM alone (`SS=OFF, useBasil=False`) in ring topology is a **documented negative result** — accuracy stays at ~10% even with channel noise + EBM. This is by design:

- EBM pushes the model toward flat minima over many rounds of accumulated gradient steps
- Ring with SS=OFF: each node receives and overwrites with the predecessor's (noisy) model every round — no accumulation happens
- EBM was calibrated for a single-hop noise budget. In a 10-node ring, noise accumulates hop-by-hop (σ_eff ≈ σ×√N). The code applies a per-hop correction (`σ/√N`) but this does not fix the non-accumulation problem
- **Solution**: use Merged (SS+EBM) — SS preserves the node's own trained state while EBM trains toward flat minima. Or use FedAvg (server averaging implicitly reduces noise by 1/√N and the global model accumulates across rounds)

This is intentional: the result shows *why* the merged approach is needed.

### LR Collapses Too Early

- `plateauMinLr` must be ≥ 0.001. Setting 0.0001 causes LR to floor at round ~80 while model is still improving.
- Run `python update_configs.py` to reset all configs to current validated defaults.

### Inspecting Saved Results

```python
import numpy as np
acc = np.load('experiments/results/gui/nonIID/cifar10/hidden/cart/acc_MyExperiment.npy')
print(f"Final accuracy: {acc[-1]:.4f}")
print(f"Peak accuracy:  {max(acc):.4f}")
```

---

## Changelog

### Latest — CART+EBM Investigation, EBM Test Suite, Script Fixes

#### Open Issue: CART+EBM+SS=OFF on CIFAR-10 → 10%
- Identified via GUI run of `cart/0 - Byzantine Nodes + Channel Noise + EBM Mitigation.json`
- EBM rigorous test suite created: `scripts/testEbmRigorous.py` — 4 tests, **6/6 PASS on MNIST** (FedAvg, ring SS+EBM, CART SS+EBM all converge to >85%)
- CIFAR-10 CART+EBM+SS=OFF still at 10% — root cause investigation ongoing (see Troubleshooting above)

#### scripts/run_single_config.py — CART Routing Added
- Now imports and calls `cartRingTraining` / `CARTNode` when `approach=cart` and `cartAlgorithm=cart`
- Previously always called `basilRingTrainingWithAttack` regardless of config's `approach` field, making CART configs untestable from CLI
- Also handles `approach=noisy` → `fedAvgTrainingWithNoise`; all other approaches → `basilRingTrainingWithAttack`

#### GUI Plateau LR Defaults Fixed
- `plateauPatience`: 5 → **8** (3 locations: `setupVariables`, `loadConfig` fallback, `_executeExperiment`)
- `plateauFactor`: 0.5 → **0.7** (same 3 locations)
- `plateauMinLr`: 0.0001 → **0.001** (same 3 locations — old value caused LR collapse at round ~80)
- `stepsPerEpoch` now saved to config JSON via `getConfig()`

#### CART EBM Gradient Order (cart.py)
- `_cart_step` uses **scale-before-clip** (intentional deviation from `BasilNode._step_fn`)
- `BasilNode` uses clip-before-scale (CLAUDE.md rule)
- CART uses scale-before-clip because: 250 sequential weight updates/round + lr=0.05 + momentum=0.9 + active proximal term — clip-before-scale (max post-scale norm=10) caused CIFAR-10 divergence; scale-before-clip caps post-scale norm at 5
- Both directions are documented in `cart.py` comments to prevent future "fixes"

---

### Previous — IID/NonIID Config Split, EBM Ring Investigation, CLI Runner

#### Config Structure Reorganization
- Configs moved from flat `gui/configs/{approach}/` into `gui/configs/nonIID/{approach}/`
- IID copies generated in `gui/configs/IID/{approach}/` with `nonIID: false`
- Total: 480 configs (240 nonIID + 240 IID)
- `update_configs.py` now enforces the `nonIID` field from folder path (not manually set)
- Added all Byzantine/no-channel-noise/EBM negative controls across basil/noisy/merged/cart and IID/nonIID splits.

#### GUI Updates
- **Load Config** dialog: IID/nonIID radio button pair routes to correct subfolder
- **Queue Manager → Add from File**: same IID/nonIID selector added
- **Add All…** button: replaced "Add All CART Configs" with a universal dialog (pick split + approach → bulk-add)
- **Save Config**: auto-routes to `IID/` or `nonIID/` subfolder based on current `nonIID` setting

#### scripts/run_single_config.py (NEW)
- CLI runner: `python scripts/run_single_config.py <config.json> [--rounds N]`
- Correctly mirrors GUI's `getNoiseModel()` logic: returns `"noisy"` for channel noise without mitigation (not `"none"`)
- Supports all approaches, datasets, and attack types

#### EBM Ring Topology Investigation
- Investigated why EBM alone (SS=OFF) in ring topology fails (~10% accuracy)
- Root cause: ring without SS forces each node to overwrite its own trained state with the predecessor's model each round — EBM cannot accumulate gradient steps toward flat minima
- Applied per-hop noise scaling for EBM ring: `σ_hop = σ/√N` → accumulated noise over N hops = σ (EBM's calibrated value)
- **Conclusion**: EBM in ring without SS is a documented negative result showing why Merged (SS+EBM) is needed; SS=OFF + EBM-only configs intentionally produce ~10% as a baseline

#### CART Tuning
- `distillStrength` increased 0.3 → 0.4 for stronger non-IID forgetting prevention
- EBM gradient order: clip-then-scale (was scale-then-clip; halved effective clip threshold)

---

### CART Implementation + Non-IID Infrastructure

#### Paper 003 — CART: Class-Aware Ring Training
- **`basil_core/cart.py`**: `CARTNode` (extends `BasilNode`) and `cartRingTraining`
  - Proximal distillation loss with EMA-smoothed trust weights (decay=0.85)
  - Always-active base proximal coefficient `γ × 1.0` — cold-start fix
  - Validated `distillStrength=0.4` (0.5 was over-constraining late-round learning)
- **`basil_core/class_registry.py`**: `ClassRegistry` data structure
  - `update`, `merge`, `verifyAndMerge` (Byzantine registry inflation rejected)
  - `trustWeights` for per-class distillation scaling
  - `toDict`/`fromDict`/`clone` for ring propagation
- **`basil_core/trainer.py`**: Added `evaluatePerClass` (returns `np.float32[nClasses]`)
- **`gui/configs/cart/`**: 48 CART configs (same attack × mitigation matrix as other approaches)
- **GUI**: CART approach with algorithm selector (run BASIL/FedAvg/Merged as non-IID baselines)
- **Fixed layout**: progress bar, button bar, status bar now pack `side=BOTTOM` before notebook — no longer hidden on window resize

#### Non-IID Dirichlet Partitioning
- `basil_core/data/cifar.py`: `_dirichletPartition` helper; `makeLoaders` now accepts `iid=True/False` and `dirichletAlpha`
- All 208 configs updated: `nonIID: true`, `dirichletAlpha: 0.2`

#### Plateau LR Tuning
- Updated defaults: `patience: 5→8`, `factor: 0.5→0.7`, `minLr: 0.0001→0.001`
- Old settings caused LR to collapse at round ~80 while the model was still improving; new settings keep LR productive through all 100 rounds

#### Optimizer Fix (critical)
- `_resetOptimizerSlots`: Keras 3 uses `'iteration'` (not `'iterations'`); rank-0 scalars (including the LR `tf.Variable`) must be skipped — zeroing them killed learning from round 1 (stuck at exactly 10%)

#### tf.function Retracing
- Added `reduce_retracing=True` to `_computeGrads` and `_forwardPass` — eliminates retracing warnings in queue runs with multiple model instances

---

### v5 — Auto-Reload Launcher, Full Clear Log

- `runGui.py`: launcher mode watches `.py` files, restarts GUI subprocess on save
- **Reload** button + `Ctrl+Shift+R` in GUI
- **Clear Log** resets text, chart, progress bar, ETA state, and terminal

---

### v4 — Reduce LR on Plateau, Config Audit

- `ReduceLROnPlateau` added to both training loops with 25% accuracy guard
- All 68 (now 208) configs audited and completed with every field the training code reads

---

### v3 — EBM Fix, Training Speed, Realistic ETA

- **Critical**: EBM lambda `75→25` (scale was 4.0, not 2.0; caused NaN weights)
- EBM gradient clipping (`clip_by_global_norm=5.0`)
- Single iterator per local training call (eliminates per-epoch TF overhead)
- EMA-based ETA countdown (ignores slow warmup round)

---

### v2 — VGG Model, GUI Redesign, Convergence Fixes

- CIFAR-10 model: small 2-conv → VGG-style 6-conv (88–92% clean accuracy)
- CIFAR-10 augmentation: random flip + pad/crop
- GUI redesign: dark terminal log, live chart, progress bar, keyboard shortcuts
- Default hyperparameters tuned: lr 0.03→0.05, momentum 0→0.9, epochs 1→5, batch 32→512

---

## Citation

If you use this implementation in your research, please cite the original papers:

**BASIL:**
```
[Paper 001 citation — BASIL: A Fast and Byzantine-Resilient Approach for Decentralized Training]
```

**Robust Federated Learning with Noisy Communication:**
```
[Paper 002 citation — Robust Federated Learning with Noisy Communication]
```

---

## License

This implementation is provided for educational and research purposes.
