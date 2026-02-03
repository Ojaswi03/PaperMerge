# BASIL + Noisy Channel Implementation

Implementation and testing framework for Byzantine-resilient federated learning with noisy communication channels.

## Papers Implemented

1. **BASIL**: Ring topology with performance-based snapshot selection for Byzantine resilience
2. **Robust Federated Learning with Noisy Communication**: EBM (Expectation-Based Model) and WCM (Worst-Case Model) approaches

---

## Quick Start

### Option 1: Graphical Interface (Recommended) ⭐

```bash
python runGui.py
```

**Features**:
- Visual configuration of all parameters
- Select datasets, approaches, and attacks
- **Specify when attacks start** (e.g., "start Gaussian attack at round 5 out of 30")
- **Specify when noise starts** (e.g., "start channel noise at round 10")
- Real-time progress monitoring
- **Stop experiments mid-run** (partial results are saved)
- **Safe window closing** (confirms before stopping running experiments)
- Save/load configurations

### Option 2: Script-Based Testing

```bash
# Test BASIL approach
python scripts/runBasilOnly.py
python plots/plotBasil.py

# Test Noisy Channel approaches (EBM, WCM)
python scripts/runNoisyChannel.py
python plots/plotNoisyChannel.py
```

**Note**: Edit the CONFIGURATION section at the top of each script (no argparse used).

---

## Three Testing Approaches

### 1. BASIL Only
- **Purpose**: Test Byzantine resilience using ring topology with snapshot selection
- **Tests**: Clean (no BASIL) vs BASIL with snapshot selection
- **Key Parameter**: S (memory size, default: 10 past models)
- **Results**: `experiments/results/basil/[dataset]/`
- **Plots**: `plots/images/basil/`

### 2. Noisy Channel Only
- **Purpose**: Test channel noise mitigation using EBM and WCM
- **Training**: FedAvg (parallel training + model averaging) - matches paper's system model
- **EBM**: Adds regularization `F_e(w) = F(w) + λσ²||∇F(w)||²`
- **WCM**: Worst-case optimization with boundary sampling
- **Tests**: Clean (no mitigation), EBM, WCM
- **Results**: `experiments/results/noisyChannel/[dataset]/`
- **Plots**: `plots/images/noisyChannel/`

### 3. Merged (BASIL + Noisy Channel)
- **Purpose**: Handle both Byzantine attacks AND channel noise
- **Training**: Ring topology (BASIL) with EBM/WCM regularization
- **Combines**: BASIL snapshot selection + EBM/WCM gradient scaling
- **Key insight**: BASIL's loss-based selection filters noisy models (replaces averaging)
- **Best for**: Real-world scenarios with multiple failure modes

---

## Configuration (No argparse)

All scripts use configuration sections at the top. Example from `scripts/runBasilOnly.py`:

```python
# ============================================================================
# CONFIGURATION - EDIT THESE TO CUSTOMIZE YOUR TEST
# ============================================================================

# Which datasets to test? Options: "mnist", "cifar10", "nmnist"
DATASETS_TO_TEST = ["mnist"]

# Which attacks to test? Options: "clean", "gaussian", "signFlip", "hidden"
ATTACKS_TO_TEST = ["clean", "gaussian"]

# Which modes to test? Options: "clean", "basil", "both"
MODE_TO_TEST = "both"

# Quick test mode (reduces rounds for faster testing)
QUICK_TEST = True  # Set to False for full training

# ============================================================================
```

Simply edit these values and run: `python scripts/runBasilOnly.py`

---

## Attack and Noise Timing

### In GUI
1. Set total training rounds (e.g., 30)
2. Attack tab: Select attacks and specify start rounds
   - Gaussian attack: Start at round 5
   - Sign-Flip attack: Start at round 15
3. Advanced tab: Channel noise start round (e.g., round 10)

### In Scripts
Configure in the appropriate section:
```python
# Example: runNoisyChannel.py
CHANNEL_NOISE_START = 5  # Noise starts at round 5
```

**Result**: You can observe how defenses handle attacks/noise that appear during training.

---

## Datasets

- **MNIST**: 28×28 grayscale handwritten digits
- **CIFAR-10**: 32×32 color images (10 classes)
- **N-MNIST**: Neuromorphic MNIST (event-based, 34×34)

---

## Attack Types

### Byzantine Attacks (Malicious Nodes)
1. **Gaussian**: Attackers send random Gaussian noise
2. **Sign-Flip**: Attackers flip signs of their gradients
3. **Hidden/Backdoor**: Attackers behave normally initially, then attack

### Channel Noise (Communication)
- **Gaussian noise**: Added to all communications
- **Configurable**: Noise strength (σ), start round, mitigation method

---

## File Structure

```
/home/ojaswi/PaperMerge/
├── runGui.py                         # Launch GUI
├── scripts/
│   ├── runBasilOnly.py              # BASIL tests (edit config at top)
│   ├── runNoisyChannel.py           # Noisy channel tests (edit config at top)
│   ├── runComprehensiveTest.py      # All approaches
│   └── testSetup.py                 # Verify installation
├── plots/
│   ├── plotBasil.py                 # BASIL plots (edit config at top)
│   ├── plotNoisyChannel.py          # Noisy channel plots (edit config at top)
│   └── plotComprehensiveComparison.py
├── gui/
│   └── experimentGui.py             # GUI implementation
├── basil_core/                       # Core implementation (camelCase)
│   ├── basil.py                      # Ring topology (BASIL) + FedAvg (Noisy Channel)
│   ├── trainer.py                    # Training with EBM/WCM + model averaging
│   ├── models.py                     # Model architectures
│   ├── attacks.py                    # Byzantine attacks
│   └── data/                         # Dataset loaders
├── noise_comm/
│   └── wcm.py                        # WCM implementation
├── experiments/results/
│   ├── basil/[dataset]/              # BASIL results
│   ├── noisyChannel/[dataset]/       # Noisy channel results
│   ├── gui/[dataset]/                # GUI results
│   └── [dataset]/                    # Comprehensive results
└── plots/images/
    ├── basil/                        # BASIL plots
    ├── noisyChannel/                 # Noisy channel plots
    └── *.png                         # Comprehensive plots
```

---

## Results Format

### Numpy Files
```python
import numpy as np

# Load accuracy curve
acc = np.load('experiments/results/basil/mnist/acc_basil_gaussian_avg.npy')

# Print final accuracy
print(f"Final accuracy: {acc[-1]:.4f}")
```

### File Naming Convention
- **BASIL**: `acc_{mode}_{attack}_{metric}.npy`
  - mode: "clean" or "basil"
  - attack: "clean", "gaussian", "signFlip", "hidden"
  - metric: "avg" or "worst"

- **Noisy Channel**: `acc_{mode}_{attack}_{metric}.npy`
  - mode: "clean", "ebm", "wcm"

---

## GUI Example Scenarios

### Scenario 1: BASIL with Delayed Attack
1. Dataset: MNIST, Approach: BASIL Only
2. Training Rounds: 30
3. Attack tab: Enable Gaussian, **Start at round: 10**
4. Result: Clean training rounds 0-9, attack starts round 10

### Scenario 2: EBM with Delayed Noise
1. Dataset: MNIST, Approach: Noisy Channel Only
2. Training Rounds: 30
3. Advanced tab: Enable Channel Noise, **Start at round: 5**, Mitigation: EBM
4. Result: Clean rounds 0-4, noise+EBM rounds 5-29

### Scenario 3: Merged with Multiple Attacks
1. Dataset: MNIST, Approach: Merged
2. Training Rounds: 30
3. Advanced tab: Channel Noise from round 0, Mitigation: EBM
4. Attack tab:
   - Gaussian: Start at round 5
   - Sign-Flip: Start at round 15
   - Hidden: Start at round 20

---

## Common Workflows

### Workflow 1: Quick Exploration
```bash
# Verify setup
python scripts/testSetup.py

# Launch GUI and experiment
python runGui.py
```

### Workflow 2: BASIL Standalone
```bash
# Edit scripts/runBasilOnly.py configuration
# Set: QUICK_TEST=True, DATASETS_TO_TEST=["mnist"]
python scripts/runBasilOnly.py

# Generate plots
python plots/plotBasil.py
```

### Workflow 3: Noisy Channel Standalone
```bash
# Edit scripts/runNoisyChannel.py configuration
# Set: QUICK_TEST=True, MODES_TO_TEST=["clean", "ebm", "wcm"]
python scripts/runNoisyChannel.py

# Generate plots
python plots/plotNoisyChannel.py
```

### Workflow 4: Full Paper Validation
```bash
# BASIL on all datasets (set QUICK_TEST=False)
python scripts/runBasilOnly.py

# Noisy Channel on all datasets
python scripts/runNoisyChannel.py

# Comprehensive test (merged approaches)
python scripts/runComprehensiveTest.py

# Generate all plots
python plots/plotBasil.py
python plots/plotNoisyChannel.py
python plots/plotComprehensiveComparison.py
```

---

## Key Parameters

### Training Topologies

| Approach | Topology | Training | Aggregation | Noise Behavior |
|----------|----------|----------|-------------|----------------|
| BASIL | Ring | Sequential | Selection (best model) | Filtered by selection |
| Noisy Channel | Star | Parallel | Averaging (FedAvg) | Reduced by √N |
| Merged | Ring | Sequential | Selection + EBM/WCM | Both mechanisms |

### BASIL (Paper 001: Algorithm 1)
- **S (Memory Size)**: Number of models to store from S counterclockwise neighbors (default: 10)
- Paper: S = b+1 where b = max Byzantine nodes. Larger S = better resilience
- **Sequential Training**: Nodes process one at a time around the ring (paper's algorithm)
- **S-Neighbor Multicast**: Each node sends to next S clockwise neighbors

### EBM (Paper 002: Eq. 13 & 23)
- **Training**: Uses FedAvg (parallel + averaging) to match paper's system model
- **Formula**: grad_Fe(w) = (1 + λσ²) × grad_F(w)
- **λ (Lambda)**: Amplification factor - must be tuned based on σ
- **σ (Sigma)**: Channel noise standard deviation
- **Scale Factor**: scale = 1 + λσ² (paper targets scale=2.0 for flat minima)

**Lambda (λ) values for scale=2.0:**
| σ (Sigma) | λ (Lambda) | Scale | Notes |
|-----------|------------|-------|-------|
| 0.05 | 400 | 2.0 | Default, low noise |
| 0.1 | 100 | 2.0 | Medium noise |
| 0.2 | 25 | 2.0 | High noise |

**For high noise (σ=0.2), higher scale may improve results:**
| σ (Sigma) | λ (Lambda) | Scale | Expected Improvement |
|-----------|------------|-------|---------------------|
| 0.2 | 25 | 2.0 | ~10-15% over noisy baseline |
| 0.2 | 50 | 3.0 | ~15-18% over noisy baseline |
| 0.2 | 75 | 4.0 | ~18-22% over noisy baseline |

### WCM (Worst-Case Model)
- **λ (Lambda)**: Regularization strength (default: 0.1)
- **Samples**: Number of boundary samples (default: 5)
- **ρ (Rho)**: SCA convex combination parameter (default: 0.5)

### Training (Paper Values)
- **Number of Nodes**: Default 10
- **Training Rounds**: 15-30 for MNIST/N-MNIST, 25-50 for CIFAR-10
- **Learning Rate**: 0.03 for MNIST/N-MNIST (paper), 0.01 for CIFAR-10
- **LR Schedule**: lr(t) = lr0 / (1 + lr0 × t) as per BASIL paper
- **Batch Size**: Default 32

---

## Expected Results

### BASIL Performance
- **No attack**: BASIL similar to clean baseline
- **With attacks**: BASIL significantly outperforms clean baseline
- **Recovery**: BASIL quickly recovers after attack begins

### Noisy Channel Performance (MNIST)

**Understanding EBM behavior:**
- EBM pushes model toward flat minima where noise causes less accuracy degradation
- Training (SGD) and noise reach an equilibrium - accuracy doesn't drop to 0%
- EBM provides ~5-20% improvement depending on noise level

**Expected accuracy with noise from round 0:**
| Configuration | σ=0.05 | σ=0.1 | σ=0.2 |
|--------------|--------|-------|-------|
| Noisy (no mitigation) | ~85% | ~78% | ~15-25% |
| Noisy + EBM (scale=2.0) | ~88% | ~84% | ~30-40% |
| Noisy + EBM (scale=4.0) | - | - | ~35-45% |

**Key findings:**
- Low noise (σ≤0.1): Both noisy baseline and EBM perform well due to training recovery
- High noise (σ=0.2): EBM provides significant improvement (+15-20% accuracy)
- Scale factor matters: Higher noise may benefit from scale > 2.0

### Merged Approach
- **Best overall**: Handles both Byzantine attacks AND channel noise
- **BASIL+EBM**: Good balance of performance and efficiency
- **BASIL+WCM**: Most robust to worst-case scenarios

---

## Troubleshooting

### GPU Out of Memory
```bash
export CUDA_VISIBLE_DEVICES=''
python scripts/runBasilOnly.py
```

### Can't Find Results
```bash
# Check BASIL results
ls experiments/results/basil/mnist/

# Check Noisy Channel results
ls experiments/results/noisyChannel/mnist/

# Check GUI results
ls experiments/results/gui/mnist/
```

### GUI Won't Launch
```bash
# Install tkinter
sudo apt-get install python3-tk  # Linux
```

### Slow Training
- Set `QUICK_TEST=True` in scripts
- Reduce number of rounds
- Reduce number of nodes
- Use GPU if available

### GUI: Stopping Experiments
- Click **Stop** button to terminate training at the next round boundary
- Partial results are automatically saved
- Closing the window (X button) while running prompts for confirmation
- Choose "Yes" to stop and exit, or "No" to continue running

---

## Dependencies

```bash
pip install tensorflow numpy matplotlib
pip install tonic  # Optional, for real N-MNIST data
```

**Note**: The code automatically falls back to CPU if GPU is not available or fails.

---

## Implementation Details

### Paper-Compliant Implementation

**BASIL Algorithm (Paper 001: Algorithm 1) - Ring Topology**
- Sequential training: nodes process one at a time around the ring
- S-neighbor multicast: each node sends to next S clockwise neighbors
- Neighbor memory: each node stores S models from S counterclockwise neighbors
- Loss-based selection: pick model with lowest local batch loss (Eq. 3)

**Noisy Channel (Paper 002: Figure 1, Eq. 3a/3b) - Star Topology with FedAvg**
- Parallel training: all nodes train simultaneously from same global model
- Model averaging: FedAvg aggregation after each round (reduces noise by √N)
- Single noise point: channel noise added once after averaging
- Broadcast: noisy averaged model sent to all nodes

**EBM Implementation (Paper 002: Eq. 13, 23)**
- Gradient scaling: `grad_Fe(w) = (1 + λσ²) × grad_F(w)`
- No Hessian computation, no noise sampling - simple scalar multiplication
- Pushes model into flat minima robust to channel noise
- Default: σ=0.1, λ=100 → scale=2.0 (matches paper)

**Model Architectures (Paper Tables I & II)**
- MNIST: 784→100→100→10 fully connected (Table I)
- CIFAR-10: conv1(16,3×3)→pool→conv2(64,4×4)→pool→fc(384)→fc(192)→fc(10) (Table II)
- N-MNIST: 34×34 input, same FC as MNIST

**Attacks (Paper Section V-A)**
- Gaussian: Replace weights with N(0,1)
- Sign-flip: Layer-wise random sign flip (50% probability per layer)
- Hidden: Subtle perturbation that's hard to detect (omniscient-style)

### Naming Convention
- **camelCase** used throughout: `localUpdate`, `dataLoader`, `noiseModel`
- Consistent across all files

### GPU/CPU Handling
- Automatic GPU detection and configuration
- Graceful fallback to CPU on any GPU error
- Memory growth enabled to prevent OOM errors

### Error Handling
- WCM failures automatically fall back to standard training
- N-MNIST loads simulated data if tonic not available
- Comprehensive error messages with suggestions

---

## Quick Reference

| Task | Command |
|------|---------|
| Launch GUI | `python runGui.py` |
| Test BASIL | `python scripts/runBasilOnly.py` |
| Test Noisy Channel | `python scripts/runNoisyChannel.py` |
| Generate BASIL plots | `python plots/plotBasil.py` |
| Generate Noisy plots | `python plots/plotNoisyChannel.py` |
| Verify setup | `python scripts/testSetup.py` |

---

## Features

✅ **No argparse** - Edit configuration sections directly
✅ **GUI** - Complete visual interface for all parameters
✅ **Stop button** - Terminate experiments mid-run, partial results saved
✅ **Safe closing** - Window close confirms before stopping experiments
✅ **Attack timing** - Specify exactly when attacks start
✅ **Noise timing** - Specify when channel noise begins
✅ **Multiple attacks** - Different attacks at different times
✅ **Three approaches** - BASIL, Noisy Channel, Merged
✅ **Three datasets** - MNIST, CIFAR-10, N-MNIST
✅ **Auto GPU/CPU** - Automatic fallback
✅ **Full camelCase** - Consistent naming

---

## Getting Started

```bash
# 1. Verify everything works
python scripts/testSetup.py

# 2. Try the GUI
python runGui.py

# 3. Or run a quick script test
python scripts/runBasilOnly.py
```

**Start with the GUI for the easiest experience!**
