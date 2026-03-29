# Byzantine-Resilient Federated Learning with Noisy Communication Channels

A comprehensive implementation and experimental framework for studying Byzantine fault tolerance and communication noise mitigation in federated learning systems.

## Abstract

This project implements and evaluates two complementary approaches to robust federated learning:

1. **BASIL** (Paper 001): A Byzantine-resilient decentralized training algorithm using ring topology with performance-based model selection
2. **Robust Federated Learning with Noisy Communication** (Paper 002): EBM (Expectation-Based Model) and WCM (Worst-Case Model) approaches for mitigating channel noise

The implementation also explores a **merged approach** that combines both defense mechanisms for real-world scenarios where systems face both malicious nodes and unreliable communication channels.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Theoretical Background](#theoretical-background)
4. [Implementation Details](#implementation-details)
5. [Experimental Configuration](#experimental-configuration)
6. [Experimental Results](#experimental-results)
7. [GUI Reference](#gui-reference)
8. [Troubleshooting](#troubleshooting)
9. [Changelog](#changelog)

---

## Installation

### Requirements

- Python 3.12
- NVIDIA GPU with CUDA 12.3+ (CPU fallback supported but slow)
- cuDNN 9.x

### Setup

```bash
# Create and activate the virtual environment
python3 -m venv environment/basil-noise-env
source environment/basil-noise-env/bin/activate

# Install all dependencies
pip install -r environment/requirements.txt
```

### Verify Installation

```bash
python scripts/testSetup.py
```

The GUI will display whether it is running on CUDA (GPU) or CPU at the start of every run.

---

## Quick Start

### Step 1 - Activate the environment

```bash
source environment/basil-noise-env/bin/activate
```

### Step 2 - Launch the GUI

```bash
python runGui.py
```

This starts the **auto-reload launcher**. The GUI opens as a subprocess and will restart automatically whenever you save any `.py` file. Use the **Reload** button (or `Ctrl+Shift+R`) inside the GUI for a manual restart.

### Step 3 - Run an experiment

1. Select a **Dataset** (MNIST is fastest for testing; CIFAR-10 for publication-quality results)
2. Select an **Approach** (BASIL Only, Noisy Channel, or Merged)
3. Optionally load a pre-built config from `gui/configs/` with **Load Config**
4. Give your experiment a name or click **Auto-generate Name**
5. Click **▶ Run Experiment** (or press `Ctrl+R`)

Results are saved automatically to:
```
experiments/results/gui/{dataset}/{attack_type}/{approach}/
```

Plots are saved to:
```
plots/images/gui/{dataset}/{attack_type}/{approach}/
```

Errors are automatically logged to:
```
error.txt
```

### Phone Notifications

Install the **ntfy** app on your phone and subscribe to your topic (default: `papermerge-ojaswi`) to receive push notifications when a run starts, completes, or fails.

To change the topic name edit `scripts/common.py`:
```python
NTFY_TOPIC = "papermerge-ojaswi"
```

### Command-Line Scripts (no GUI)

```bash
python scripts/testBasilPaper.py    # BASIL paper reproduction
python scripts/testEbmPaper.py      # Noisy channel paper reproduction
python scripts/testMerged.py        # Merged approach
```

---

## Theoretical Background

### BASIL Algorithm (Paper 001)

BASIL employs a **ring topology** with **sequential training** to achieve Byzantine resilience:

```
For each training round:
    For each node i in ring (sequential):
        1. Receive models from S counterclockwise neighbors
        2. Select best model using local batch loss (Eq. 3)
        3. Perform local SGD training
        4. Multicast updated model to S clockwise neighbors
```

**Key Parameters:**
- **S (Memory Size)**: Number of neighbor models stored; paper recommends S = b + 1 where b = max Byzantine nodes
- **Selection Criterion**: Model with minimum local batch loss (filters Byzantine models)

**Why BASIL Works:**
- Byzantine models produce **high loss** on honest local data
- Honest models produce **low loss** on honest local data
- Loss-based selection naturally filters out Byzantine contributions

### EBM - Expectation-Based Model (Paper 002)

EBM mitigates channel noise by pushing the model toward **flat minima** where weight perturbations cause minimal accuracy degradation.

```
Original loss:    F(w)
EBM loss:         F_e(w) = F(w) + σ²||∇F(w)||²     (Eq. 13)
EBM gradient:     ∇F_e(w) = (1 + λσ²) × ∇F(w)      (Eq. 23)
Scale factor:     scale = 1 + λσ²
```

### Merged Approach

Combines both defense mechanisms:
- **Ring topology** with **BASIL selection** (filters Byzantine nodes)
- **EBM gradient scaling** during local training (handles channel noise)

---

## Implementation Details

### System Architecture

| Approach | Topology | Training | Aggregation | Defense |
|----------|----------|----------|-------------|---------|
| BASIL | Ring | Sequential | Selection (min loss) | Byzantine filtering |
| Noisy Channel | Star | Parallel | FedAvg (averaging) | EBM/WCM gradient scaling |
| Merged | Ring | Sequential | Selection + EBM | Both |

### File Structure

```
PaperMerge/
├── runGui.py                    # Entry point - launches the GUI
├── error.txt                    # Auto-generated error log from GUI
├── basil_core/
│   ├── basil.py                 # BASIL ring topology + FedAvg training loops
│   ├── trainer.py               # Local training with EBM/WCM
│   ├── models.py                # Neural network architectures
│   ├── attacks.py               # Byzantine attack implementations
│   └── data/
│       ├── mnist.py             # MNIST loader + tf.data pipeline
│       ├── cifar.py             # CIFAR-10 loader + augmentation pipeline
│       └── nMnist.py            # Neuromorphic MNIST loader
├── noise_comm/
│   └── wcm.py                   # WCM noise mitigation
├── gui/
│   ├── experimentGui.py         # Tkinter GUI (config, run, live chart, log)
│   └── configs/                 # 68 pre-built experiment configurations
├── scripts/
│   ├── common.py                # GPU setup, phone notifications, shared utils
│   ├── testSetup.py             # Verify environment and GPU
│   ├── testBasilPaper.py        # BASIL paper reproduction
│   ├── testEbmPaper.py          # EBM paper reproduction
│   └── testMerged.py            # Merged approach experiments
├── plots/
│   └── plotGui.py               # Plot discovery and generation
├── environment/
│   ├── requirements.txt         # Pinned dependencies
│   └── basil-noise-env/         # Python virtual environment
└── experiments/results/gui/     # Saved results
    └── {dataset}/{attack}/{approach}/
        ├── acc_*.npy            # Accuracy curve (numpy array)
        └── config_*.json        # Full experiment configuration snapshot
```

### Model Architectures

**MNIST / N-MNIST:**
```
Input(28×28) → Flatten → FC(100) → ReLU → FC(100) → ReLU → FC(10) [logits]
```

**CIFAR-10 (VGG-style, updated):**
```
Input(32×32×3)
→ Conv(64,3×3) → ReLU → Conv(64,3×3) → ReLU → MaxPool(2) → Dropout(0.25)
→ Conv(128,3×3) → ReLU → Conv(128,3×3) → ReLU → MaxPool(2) → Dropout(0.25)
→ Conv(256,3×3) → ReLU → Conv(256,3×3) → ReLU → MaxPool(2) → Dropout(0.40)
→ Flatten → FC(512) → ReLU → Dropout(0.50) → FC(10) [logits]
He normal initialisation throughout. Targets 88–92% on clean CIFAR-10.
```

**Training augmentation (CIFAR-10 only):**
- Random horizontal flip
- Pad 4 px → random crop back to 32×32

### Attack Implementations

| Attack | Description | Difficulty |
|--------|-------------|------------|
| Gaussian | Replace weights with N(0,1) random noise | Easy |
| Sign-Flip | Negate all gradient values | Medium |
| Hidden | Behave normally until a specified round, then attack | Hard |
| Model Poison | Gradient ascent to push model toward wrong minimum | Hard |
| Scaling | Multiply weights by large negative factor | Medium |
| ALIE | Stealthy update within honest distribution bounds | Hard |
| IPM | Negate and scale to maximise negative inner product | Hard |
| Noise Amp | Amplified noise proportional to layer std | Medium |

---

## Experimental Configuration

### Recommended Settings (CIFAR-10)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | cifar10 | VGG model, augmentation enabled |
| Nodes | 10 | |
| Rounds | 100 | |
| Local Epochs | 5 | |
| Batch Size | 512 | |
| Learning Rate | 0.05 | |
| Momentum | 0.9 | Required for stable convergence |
| LR Decay | false (clean) / false (EBM) | See below |
| Steps per Epoch | 5 | Controls per-round granularity |

### LR Schedule Policy

| Config type | Recommended schedule | Reason |
|-------------|---------------------|--------|
| Clean (no noise) | `usePlateauLr=true` or `useLrDecay=false` | Plateau LR auto-reduces near convergence; fixed LR also works |
| Noisy + EBM | `useLrDecay=false` | LR decay degrades EBM effectiveness in later rounds |
| Noisy (no mitigation) | `useLrDecay=true` | Decay needed to prevent oscillation |

**`usePlateauLr` overrides `useLrDecay`** - if plateau is enabled, the decay schedule is ignored.

### EBM Parameters

| Noise σ | Lambda λ | Scale | Use Case |
|---------|---------|-------|----------|
| 0.05 | 400 | 2.0 | Low noise |
| 0.1 | 100 | 2.0 | Medium noise |
| 0.2 | 25 | 2.0 | High noise - default |

**Critical:** EBM with σ=0.2 requires `momentum=0.9` and `useLrDecay=false` for stable high accuracy.

### Pre-built Configs (`gui/configs/`)

68 JSON configurations covering all combinations of:
- 0 or 4 Byzantine attackers
- 8 attack types (Gaussian, SignFlip, Hidden, Model Poison, Scaling, ALIE, IPM, Noise Amp)
- With/without channel noise (σ=0.2)
- With/without EBM mitigation
- BASIL standalone and Noisy Channel standalone variants

---

## Experimental Results

### Expected Clean Baseline (CIFAR-10, 100 rounds)

| Round | Expected Accuracy |
|-------|------------------|
| 0 | ~18% |
| 10 | ~75% |
| 30 | ~84% |
| 100 | **88–92%** |

### BASIL Standalone (MNIST)

| Scenario | Avg Accuracy | Worst Accuracy |
|----------|-------------|----------------|
| Clean (no attack) | ~95% | ~95% |
| Attack, no BASIL | ~67% | ~61% |
| **Attack + BASIL** | **~91%** | **~90%** |

### Noisy Channel (MNIST, σ=0.2)

| Scenario | Final Accuracy |
|----------|---------------|
| Clean | ~96% |
| Noisy, no EBM | ~43% |
| **Noisy + EBM** | **~82%** |

---

## GUI Reference

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Run Experiment |
| `Ctrl+S` | Save Configuration |
| `Ctrl+L` | Load Configuration |
| `Esc` | Stop running experiment |
| `Ctrl+A` | Select all log text |

### Tabs

| Tab | Contents |
|-----|----------|
| Basic | Dataset, approach, training parameters, quick presets |
| Advanced | BASIL memory size, channel noise, EBM lambda, momentum, LR schedule |
| Attacks | 8 attack types with per-attack enable and start-round |
| Output | Live log (dark terminal) + live accuracy chart (updates each round) |

### Quick Presets

| Preset | Description |
|--------|-------------|
| Clean Baseline | CIFAR-10, BASIL, no noise, no attacks |
| Noisy + EBM | CIFAR-10, FedAvg, σ=0.2, EBM mitigation |
| BASIL + Gaussian | CIFAR-10, BASIL, Gaussian attack on 4 nodes |
| Merged (Best) | CIFAR-10, BASIL + EBM, σ=0.2, no attacks |

### Output Log

- **Right-click** the log for Copy / Select All / Copy All / Save Errors / Clear
- **Copy All** button copies the entire log to clipboard
- **Save Errors** button extracts all red error lines and appends them to `error.txt`
- Errors are also **automatically** appended to `error.txt` as they occur (with timestamp)

### Run All Configs

Click **▶▶ Run All Configs** to batch-run all 68 configs in `gui/configs/`. A dialog shows how many configs were found, how many are pending, and how many are already done. Already-completed experiments (matched by config hash) are skipped automatically.

### Progress Bar

Displays between the tab panel and button bar:
- Current round number (e.g. `Round 12/100`)
- Estimated time remaining
- Latest average accuracy

---

## Troubleshooting

### GPU Out of Memory

```bash
export CUDA_VISIBLE_DEVICES=''
python runGui.py
```

### GUI Won't Launch

```bash
sudo apt-get install python3-tk   # Linux / WSL2
```

### Slow Training

- Use MNIST instead of CIFAR-10 for quick tests
- Reduce `nRounds` to 10–20 for sanity checks
- Ensure GPU is detected at startup (`[GPU] Successfully configured …`)

### Inspecting Saved Results

```python
import numpy as np

acc = np.load('experiments/results/gui/cifar10/none/basil/acc_MyExperiment.npy')
print(f"Rounds recorded : {len(acc)}")
print(f"Final accuracy  : {acc[-1]:.4f}")
print(f"Peak accuracy   : {max(acc):.4f}")
```

### Reading the Error Log

```bash
cat error.txt          # all timestamped errors
tail -50 error.txt     # last 50 error lines
```

---

## Changelog

### v5 - Auto-Reload GUI, Hotfix

#### Auto-Reload Launcher (`runGui.py`)
- `runGui.py` now runs in two modes: **launcher** (default) and **GUI subprocess** (`--gui` flag)
- The launcher polls every 0.8 seconds for changes to any `.py` file under `basil_core/`, `gui/`, `scripts/`, `noise_comm/`, and `runGui.py` itself
- When a change is detected the old GUI subprocess is terminated and a fresh one is started automatically - no manual restart needed
- The launcher exits cleanly on Ctrl+C, and shuts down when the user closes the GUI window normally

#### Reload Button and Shortcut (`gui/experimentGui.py`)
- Added **Reload** button to the button bar (right side, next to Exit)
- Added `Ctrl+Shift+R` keyboard shortcut for reload
- Clicking Reload (or pressing the shortcut) exits the GUI with code 42, which the launcher treats as an immediate restart signal
- If an experiment is running, a confirmation dialog is shown before reloading
- Status bar updated to show the new shortcut hint

#### Hotfix - Tkinter Key Binding
- Fixed `_tkinter.TclError: bad event type or keysym "shift"` - Tkinter requires `Shift` (capital S) and uppercase letter: `<Control-Shift-R>` not `<Control-shift-R>`

---

### v4 - Adaptive LR (Reduce on Plateau), Config Audit

#### Reduce LR on Plateau (`basil_core/basil.py`, `gui/experimentGui.py`)
- Added **ReduceLROnPlateau** to both `basilRingTrainingWithAttack` and `fedAvgTrainingWithNoise`
- When enabled, LR starts at `lr0` and is multiplied by `plateauFactor` whenever accuracy fails to improve by more than `plateauThreshold` for `plateauPatience` consecutive rounds
- A **cooldown** period (`plateauPatience // 2` rounds) prevents back-to-back reductions after each adjustment
- Overrides the existing `useLrDecay` schedule - the two are mutually exclusive (plateau gives more precise control)
- Prints `ReduceLROnPlateau: 0.050000 → 0.025000` in the log each time it fires
- Off by default (`usePlateauLr: false`) - existing experiments are unaffected

**New parameters (all configurable in GUI Advanced tab and per-config JSON):**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `usePlateauLr` | `false` | Enable/disable feature |
| `plateauPatience` | `10` | Rounds of no improvement before reducing |
| `plateauFactor` | `0.5` | LR multiplier on plateau (e.g. 0.5 = halve) |
| `plateauMinLr` | `0.0001` | Floor - LR never drops below this |
| `plateauThreshold` | `0.002` | Minimum accuracy gain to count as improvement |

**Recommended use:** Enable on clean CIFAR-10 baselines - LR stays at 0.05 during fast convergence, then auto-halves when approaching the accuracy ceiling (~round 70–80), potentially pushing final accuracy above 88%.

#### Config Audit - All 68 Configs Brought Up to Date
- Audited every field read via `config.get(...)` in the GUI against all 68 JSON files
- The following fields were present in code but missing from every config (causing silent fallback to defaults with no way to override per-config):

| Field added | Default value | Why it matters |
|-------------|--------------|----------------|
| `stepsPerEpoch` | `5` | Controls gradient steps per node per round; was GUI-hardcoded, now per-config |
| `usePlateauLr` | `false` | New plateau LR feature |
| `plateauPatience` | `10` | New plateau LR feature |
| `plateauFactor` | `0.5` | New plateau LR feature |
| `plateauMinLr` | `0.0001` | New plateau LR feature |
| `plateauThreshold` | `0.002` | New plateau LR feature (was also missing from GUI entirely) |

- All 68 configs now contain every parameter the training code reads, making each config a complete, self-contained experiment specification

---

### v3 - EBM Fix, Training Speed, Realistic ETA

#### EBM Lambda Correction (critical bug fix)
- **Root cause**: `lambda=75, sigma=0.2` gives `scale = 1 + 75×0.04 = 4.0`, not 2.0 as the GUI hint claimed. With `momentum=0.9` the effective gradient amplification caused NaN weights from round 1 onward, keeping accuracy stuck at ~10% (random chance) for the entire run.
- **Fix**: Changed `lambda` from `75 → 25` everywhere. With `sigma=0.2`: `scale = 1 + 25×0.04 = 2.0` (matches paper's intended scale).
- Updated in: `basil_core/basil.py`, `basil_core/trainer.py`, `gui/experimentGui.py` (default, presets, load fallback, hint text), all 68 JSON configs, all test scripts, `tests/test_convergence.py`

**Correct lambda reference table:**

| Noise σ | Lambda λ | Scale | Use Case |
|---------|---------|-------|----------|
| 1.0 | 1 | 2.0 | Paper's original setting |
| 0.2 | 25 | 2.0 | High noise - default |
| 0.1 | 100 | 2.0 | Medium noise |
| 0.05 | 400 | 2.0 | Low noise |

#### EBM Gradient Clipping
- Added `tf.clip_by_global_norm(scaled, 5.0)` to the EBM compiled training step in `basil_core/basil.py`
- Prevents NaN weights when training resumes from channel-noise-corrupted starting weights (round after noise activates)
- Only active for EBM path; standard and WCM paths are unaffected

#### Training Pipeline Speed-up (WSL2 / WDDM overhead reduction)
- **Single iterator per local training call**: the nested epoch loop (`for epoch: for batch`) was creating one TF dataset iterator per epoch per node per round (5 iterators/node/round = 4,000+ across a 100-round run). Each iterator startup on WSL2 carries significant WDDM overhead. Replaced with a single `for batch in _iterLimited(dataLoader, localEpochs * stepsPerEpoch)` loop. Change applied in both `BasilNode.localTrain` and `trainer.localUpdate`.
- **`drop_remainder=True`** on all training loaders (`mnist.py`, `cifar.py`, `nMnist.py`): guarantees every batch is the same shape, preventing `@tf.function` retracing for partial end-of-epoch batches.
- **`.repeat()` on training datasets**: makes each client's dataset infinite so it never exhausts mid-training. The `_iterLimited` step-count cap is the only stop condition.

#### `stepsPerEpoch` Now Configurable per Config
- GUI previously hardcoded `stepsPerEpoch=5` in both the BASIL ring and FedAvg training calls, ignoring any value in the JSON config.
- Changed to `config.get('stepsPerEpoch', 5)` - existing configs default to 5 (no behaviour change); individual configs can now override with `"stepsPerEpoch": N`.

#### Realistic ETA Countdown (gui/experimentGui.py)
- **Before**: ETA used a simple all-time average (`elapsed / roundsDone`). The first round is always slow (TF graph compilation warmup), permanently inflating the estimate.
- **After**:
  - Per-round durations are recorded in `_roundTimes`
  - **Exponential moving average** (α = 0.25) maintained in `_emaRoundTime` - recent rounds receive ~4× more weight than older rounds; the slow warmup round is nearly forgotten after 4–5 rounds
  - **Smooth countdown**: the displayed ETA decreases by 1 second per tick normally. If the EMA drops significantly (model sped up), the display catches up at 30% of the gap per tick rather than jumping instantly
  - State (`_roundTimes`, `_emaRoundTime`, `_smoothedEta`) is reset at the start of each experiment, including between runs in a "Run All" queue, so timing from one experiment never bleeds into the next

---

### v2 - VGG Model, GUI Redesign, Convergence Fixes

#### GPU & Performance
- `runGui.py`: `setupGpu()` is now called **before** any TensorFlow imports, eliminating the "GPU forced to CPU on second call" bug
- All data loaders (`mnist.py`, `cifar.py`, `nMnist.py`) converted from plain Python lists to `tf.data.Dataset` with `.prefetch(AUTOTUNE)` - GPU now prefetches the next batch during compute
- `TF_CPP_MIN_LOG_LEVEL=2` suppresses TensorFlow INFO messages

#### CIFAR-10 Model (models.py)
- Replaced the original paper's small 2-conv architecture (Conv16 → Conv64 → FC384 → FC192) with a **VGG-style 6-conv model** (64→128→256 filters, He normal init, Dropout)
- Expected clean accuracy improved from ~70% to **88–92%**
- No BatchNorm (avoids federated parameter-averaging issues with running statistics)

#### CIFAR-10 Data Pipeline (data/cifar.py)
- Added **random horizontal flip** and **pad-4 / random-crop** augmentation to training loader
- Augmentation applied **before** batching (fixes `random_crop` dimension mismatch error)
- Training loader now shuffles each epoch

#### Training Loops (basil_core/basil.py)
- Added `roundCallback(roundNum, avgAcc, worstAcc, totalRounds)` parameter to both `basilRingTrainingWithAttack` and `fedAvgTrainingWithNoise` - called after every round evaluation
- Removed all pre-training evaluation (no model eval before the first training round)
- Round labels in logs are now 0-indexed and consistent across both training functions

#### Convergence Fixes
- Default `learningRate`: `0.03 → 0.05` (matches original BASIL paper)
- Default `momentum`: `0.0 → 0.9` (required for stable CIFAR-10 convergence with SGD)
- Default `nRounds`: `30 → 100`
- Default `localEpochs`: `1 → 5`
- Default `batchSize`: `32 → 512`
- Fixed `fedAvgTrainingWithNoise` ignoring the config's `localEpochs` (was hardcoded to 1)
- Fixed both training functions ignoring the config's `useLrDecay` (was hardcoded `True`)
- `stepsPerEpoch`: `100 → 5` - reduces per-round gradient steps so the accuracy curve starts near ~10–20% and rises gradually rather than jumping to ~50% in round 0
- All 68 configs updated: `useLrDecay=false` for no-noise configs; `useLrDecay=false` for EBM configs
- All 68 configs updated to `attackHiddenStart=0`, `dataset=cifar10`, `momentum=0.9`, `learningRate=0.05`, `nRounds=100`

#### GUI Redesign (gui/experimentGui.py)
- Window enlarged to 1200×900 with `clam` ttk theme and custom colour palette
- **Quick Presets**: one-click buttons for Clean Baseline, Noisy+EBM, BASIL+Gaussian, Merged
- **Auto-generate Name**: derives experiment name from current settings
- **Live accuracy chart**: embedded matplotlib figure (right panel of Output tab), updates after every round via `roundCallback`
- **Progress bar**: shows `Round N/100`, ETA, and latest accuracy % between tabs and buttons
- **Status bar**: persistent one-line status at bottom of window
- **Colour-coded log**: dark terminal style - blue for round info, green for success, red for errors, yellow for warnings
- **Keyboard shortcuts**: `Ctrl+R` run, `Ctrl+S` save, `Ctrl+L` load, `Esc` stop
- **Copy from log**: right-click context menu (Copy / Select All / Copy All / Save Errors), `Ctrl+A`, and "Copy All" header button
- **Error logging**: errors auto-appended to `error.txt` with timestamps; "Save Errors" button/menu item for manual export
- **Run All Configs**: dialog now shows counts only (found / to run / already done) without listing every file
- **Skip completed experiments**: `_isAlreadyRun()` compares saved config JSON against source config - if any field changed the experiment re-runs automatically

---

## Citation

If you use this implementation in your research, please cite the original papers:

**BASIL:**
```
[Paper 001 citation]
```

**Robust Federated Learning with Noisy Communication:**
```
[Paper 002 citation]
```

---

## License

This implementation is provided for educational and research purposes.
