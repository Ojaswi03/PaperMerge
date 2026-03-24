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
7. [Test Scripts](#test-scripts)
8. [GUI Reference](#gui-reference)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Python 3.12
- NVIDIA GPU with CUDA 12.3+ (CPU fallback is supported but slow)
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

### Step 3 - Run an experiment

1. Select a **Dataset** (MNIST is fastest for testing)
2. Select an **Approach** (BASIL Only, Noisy Channel, or Merged)
3. Configure attacks and noise in the tabs
4. Give your experiment a name in the **Experiment Name** field
5. Click **Run Experiment**

Results are saved automatically to:
```
experiments/results/gui/{dataset}/{attack_type}/
```

Plots are saved to:
```
plots/images/gui/{dataset}/{attack_type}/
```

### Phone Notifications

Install the **ntfy** app on your phone and subscribe to your topic (default: `papermerge-ojaswi`) to receive push notifications when a run starts, completes, or fails.

To change the topic name edit this line in `scripts/common.py`:
```python
NTFY_TOPIC = "papermerge-ojaswi"
```

### Command-Line Scripts (no GUI)

```bash
# BASIL paper reproduction
python scripts/testBasilPaper.py

# Noisy channel paper reproduction
python scripts/testEbmPaper.py

# Merged approach
python scripts/testMerged.py
```

---

## Theoretical Background

### BASIL Algorithm (Paper 001)

BASIL employs a **ring topology** with **sequential training** to achieve Byzantine resilience:

**Algorithm Overview:**
```
For each training round:
    For each node i in ring (sequential):
        1. Receive models from S counterclockwise neighbors
        2. Select best model using local batch loss (Eq. 3)
        3. Perform local SGD training
        4. Multicast updated model to S clockwise neighbors
```

**Key Parameters:**
- **S (Memory Size)**: Number of neighbor models stored; paper recommends S = b + 1 where b = maximum Byzantine nodes
- **Selection Criterion**: Model with minimum local batch loss (filters Byzantine models)

**Why BASIL Works:**
- Byzantine models (random noise, sign-flipped gradients) produce **high loss** on honest local data
- Honest models produce **low loss** on honest local data
- Loss-based selection naturally filters out Byzantine contributions

### EBM - Expectation-Based Model (Paper 002)

EBM mitigates channel noise by pushing the model toward **flat minima** where weight perturbations cause minimal accuracy degradation.

**Mathematical Formulation:**
```
Original loss:     F(w)
EBM loss:          F_e(w) = F(w) + σ²||∇F(w)||²     (Eq. 13)
EBM gradient:      ∇F_e(w) = (1 + λσ²) × ∇F(w)      (Eq. 23)
Scale factor:      scale = 1 + λσ²
```

**Intuition:**
- Flat minima have small gradient norms
- Adding σ²||∇F(w)||² penalizes sharp minima
- Model converges to regions robust to noise perturbations

### Merged Approach

Combines both defense mechanisms:
- **Ring topology** with **BASIL selection** (filters Byzantine nodes)
- **EBM gradient scaling** during local training (handles channel noise)
- Provides defense against both threat models simultaneously

---

## Implementation Details

### System Architecture

| Approach | Topology | Training | Aggregation | Defense Mechanism |
|----------|----------|----------|-------------|-------------------|
| BASIL | Ring | Sequential | Selection (min loss) | Byzantine filtering |
| Noisy Channel | Star | Parallel | FedAvg (averaging) | EBM/WCM gradient scaling |
| Merged | Ring | Sequential | Selection + EBM | Both mechanisms |

### File Structure

```
PaperMerge/
├── runGui.py                    # Entry point - launches the GUI
├── basil_core/
│   ├── basil.py                 # BASIL ring topology + FedAvg
│   ├── trainer.py               # Local training with EBM/WCM
│   ├── models.py                # Neural network architectures (MNIST, CIFAR, N-MNIST)
│   ├── attacks.py               # Byzantine attack implementations
│   └── data/                    # Dataset loaders (MNIST, CIFAR-10, N-MNIST)
├── noise_comm/
│   └── wcm.py                   # WCM noise mitigation
├── gui/
│   └── experimentGui.py         # Tkinter GUI (config, run, plot)
├── scripts/
│   ├── common.py                # GPU setup, phone notifications, shared utils
│   ├── testSetup.py             # Verify environment and GPU
│   ├── testBasilPaper.py        # BASIL paper reproduction
│   ├── testEbmPaper.py          # EBM paper reproduction
│   ├── testMerged.py            # Merged approach experiments
│   ├── runBasilOnly.py          # Full BASIL test suite
│   ├── runNoisyChannel.py       # Full noisy channel test suite
│   └── runComprehensiveTest.py  # All combinations
├── plots/
│   └── plotGui.py               # Plot discovery and generation
├── environment/
│   ├── requirements.txt         # Pinned dependencies
│   └── basil-noise-env/         # Python virtual environment
└── experiments/results/gui/     # Saved results organized by dataset and attack type
    └── {dataset}/
        └── {attack_type}/
            ├── acc_*.npy        # Accuracy curve
            └── config_*.json    # Experiment configuration
```

### Model Architectures

**MNIST (Paper Table I):**
```
Input(784) → FC(100) → ReLU → FC(100) → ReLU → FC(10) → Softmax
```

**CIFAR-10 (Paper Table II):**
```
Input(32×32×3) → Conv(16,3×3) → Pool → Conv(64,4×4) → Pool → FC(384) → FC(192) → FC(10)
```

### Attack Implementations

| Attack | Description | Detection Difficulty |
|--------|-------------|---------------------|
| **Gaussian** | Replace weights with N(0,1) random noise | Easy (high loss) |
| **Sign-Flip** | Negate gradient signs | Medium |
| **Hidden** | Behave normally, then attack at specified round | Hard |
| **Model Poison** | Gradient ascent to push model in wrong direction | Hard |

---

## Experimental Configuration

### BASIL Parameters

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| S (Memory Size) | b + 1 | b = max Byzantine nodes |
| Learning Rate | 0.03 | With decay: lr(t) = lr₀/(1 + lr₀×t) |
| Training Rounds | 30 | For MNIST |
| Nodes | 10 | 2 attackers = 20% Byzantine |

**Example Configuration (2 attackers out of 10 nodes):**
```python
S_MEMORY = 3          # S = b + 1 = 2 + 1 = 3
ATTACKER_IDS = [0, 5] # Nodes 0 and 5 are Byzantine
ATTACK_TYPE = "gaussian"
```

### EBM Parameters

The scale factor `scale = 1 + λσ²` determines gradient amplification:

| Noise (σ) | Lambda (λ) | Scale | Use Case |
|-----------|------------|-------|----------|
| 0.05 | 400 | 2.0 | Low noise |
| 0.1 | 100 | 2.0 | Medium noise |
| 0.2 | 25 | 2.0 | High noise (conservative) |
| **0.2** | **75** | **4.0** | **High noise (optimal)** |

**Critical Finding: Momentum Significantly Improves EBM Performance**

For high noise scenarios (σ = 0.2), adding SGD momentum dramatically improves results:

| Configuration | Best Accuracy | Notes |
|---------------|---------------|-------|
| EBM (λ=75, no momentum) | ~52% | Unstable |
| **EBM (λ=75, momentum=0.9)** | **~83%** | Stable, recommended |

**Optimal EBM Configuration for σ=0.2:**
```python
SIGMA = 0.2
EBM_LAMBDA = 75        # scale = 4.0
MOMENTUM = 0.9         # Critical for stability
LR = 0.03              # With decay
```

---

## Experimental Results

### BASIL Standalone (Gaussian Attack)

**Setup:** 10 nodes, 2 attackers (nodes 0, 5), S = 3, 30 rounds

| Test | Avg Accuracy | Worst Accuracy | Stability |
|------|--------------|----------------|-----------|
| Clean (no attack) | 95.97% | 95.87% | Stable |
| Attack (no BASIL) | 66.99% | 61.00% | Unstable |
| **Attack + BASIL** | **91.03%** | **89.69%** | **Stable** |

**Key Finding:** BASIL recovers **+24% average accuracy** under Byzantine attack.

### Noisy Channel Standalone (σ = 0.2)

**Setup:** 10 nodes, FedAvg topology, 30 rounds, momentum = 0.9

| Test | Best Accuracy | Final Accuracy |
|------|---------------|----------------|
| Clean (no noise) | 96.16% | 96.16% |
| Noisy (no EBM) | 68.09% | 43.22% |
| **Noisy + EBM** | **82.34%** | **81.87%** |

**Key Finding:** EBM recovers **+38% final accuracy** with high channel noise.

### Merged Approach (Attack + Noise)

**Setup:** 10 nodes, 2 attackers, σ = 0.2, S = 3, EBM λ = 75

| Test | Expected Accuracy |
|------|-------------------|
| Clean | ~96% |
| Attack + Noise (no defense) | ~30-50% |
| Attack + Noise + BASIL | ~60-70% |
| **Attack + Noise + BASIL + EBM** | **~80-85%** |

---

## Test Scripts

### BASIL Paper Reproduction

```bash
python scripts/testBasilPaper.py
```

Runs 3 tests:
1. Clean (no attacks)
2. Attacks without BASIL
3. Attacks with BASIL

### EBM Paper Reproduction

```bash
python scripts/testEbmPaper.py
```

Runs 3 tests:
1. Clean (no noise)
2. Noisy (no mitigation)
3. Noisy + EBM

### Merged Approach

```bash
python scripts/testMerged.py
```

Runs 4 tests:
1. Clean baseline
2. Attacks + Noise (no defense)
3. Attacks + Noise + BASIL
4. Attacks + Noise + BASIL + EBM

### Debug Script (Verify Attacks)

```bash
python scripts/testBasilDebug.py
```

Shows loss values for each candidate model to verify Byzantine models are being filtered.

---

## GUI Reference

### Basic Configuration Tab

| Setting | Description | Recommended |
|---------|-------------|-------------|
| Dataset | MNIST, CIFAR-10, N-MNIST | MNIST for testing |
| Approach | BASIL Only, Noisy Channel Only, Merged | Based on experiment |
| Nodes | Number of training nodes | 10 |
| Rounds | Training iterations | 30 |
| Learning Rate | Initial LR | 0.03 |
| Momentum | SGD momentum | 0.9 (for EBM) |

### Advanced Configuration Tab

| Setting | Description | Recommended |
|---------|-------------|-------------|
| BASIL Memory (S) | Neighbor models stored | b + 1 |
| Channel Noise | Enable/disable noise | Based on experiment |
| Sigma (σ) | Noise standard deviation | 0.2 for high noise |
| Mitigation | None, EBM, WCM | EBM |
| EBM Lambda | Gradient scaling factor | 75 for σ=0.2 |

### Attack Configuration Tab

| Setting | Description |
|---------|-------------|
| Attacker IDs | Comma-separated node IDs (e.g., "0,5") |
| Gaussian Attack | Random noise attack |
| Sign-Flip Attack | Gradient negation attack |
| Hidden Attack | Delayed attack activation |
| Model Poison Attack | Gradient ascent attack |

---

## Troubleshooting

### Common Issues

**GPU Out of Memory:**
```bash
export CUDA_VISIBLE_DEVICES=''
python scripts/testBasilPaper.py
```

**GUI Won't Launch:**
```bash
sudo apt-get install python3-tk  # Linux
```

**Slow Training:**
- Reduce `N_ROUNDS` for quick tests
- Use MNIST (fastest dataset)
- Enable GPU if available

### Verifying Results

```python
import numpy as np

# Load and inspect results - path follows {dataset}/{attack_type}/acc_{name}.npy
acc = np.load('experiments/results/gui/mnist/gaussian/acc_my_experiment.npy')
print(f"Final accuracy: {acc[-1]:.4f}")
print(f"Best accuracy: {max(acc):.4f}")
```

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

---

## Acknowledgments

This project implements algorithms from academic papers for experimental validation and comparison. The implementation follows the original papers' specifications while providing a unified framework for testing different defense mechanisms.
