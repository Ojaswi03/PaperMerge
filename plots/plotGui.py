"""
Plotting for GUI experiment results.
Auto-discovers all experiments saved in experiments/results/gui/
and generates comparison plots per dataset.

Plots are saved to plots/images/gui/

CONFIGURATION: Edit the variables below to customize your plots
"""
import os
import sys
import json
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ============================================================================
# CONFIGURATION - EDIT THESE TO CUSTOMIZE YOUR PLOTS
# ============================================================================

# Which datasets to plot? Options: "mnist", "cifar10", "nmnist"
# Set to None to auto-detect all available datasets
DATASETS_TO_PLOT = None

# Which metric? Options: "avg" only (worst is no longer saved)
METRIC = "avg"

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Robust import
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')

# All distinct matplotlib markers
_ALL_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p', 'H', '8', '+', 'x', '1', '2', '3', '4']


def getColors(n):
    """Generate n visually distinct colors dynamically using matplotlib colormaps."""
    if n <= 0:
        return []
    if n <= 10:
        cmap = cm.get_cmap('tab10', n)
    elif n <= 20:
        cmap = cm.get_cmap('tab20', n)
    else:
        cmap = cm.get_cmap('hsv', n)
    return [cmap(i) for i in range(n)]


def getMarkers(n):
    """Return n markers, cycling through all available distinct markers."""
    return [m for _, m in zip(range(n), itertools.cycle(_ALL_MARKERS))]


def discoverDatasets():
    """Find all datasets that have GUI results."""
    guiDir = "experiments/results/gui"
    if not os.path.isdir(guiDir):
        print(f"No GUI results directory found at {guiDir}")
        return []

    datasets = []
    for entry in sorted(os.listdir(guiDir)):
        entryPath = os.path.join(guiDir, entry)
        if os.path.isdir(entryPath):
            # Check if it has any .npy files
            npyFiles = glob.glob(os.path.join(entryPath, "*.npy"))
            if npyFiles:
                datasets.append(entry)

    return datasets


def discoverExperiments(dataset):
    """
    Find all experiments for a given dataset.
    Returns a list of dicts with keys: name, config, avgPath, label
    """
    resultDir = f"experiments/results/gui/{dataset}"
    if not os.path.isdir(resultDir):
        return []

    # Find all config files
    configFiles = sorted(glob.glob(os.path.join(resultDir, "config_*.json")))

    experiments = []
    for configPath in configFiles:
        configName = os.path.basename(configPath)
        # Pattern: config_{name}.json  (name is the sanitized experiment title)
        name = configName.replace("config_", "").replace(".json", "")

        avgPath = os.path.join(resultDir, f"acc_{name}.npy")
        if not os.path.exists(avgPath):
            continue

        # Load config
        with open(configPath, 'r') as f:
            config = json.load(f)

        experiments.append({
            'name': name,
            'config': config,
            'avgPath': avgPath,
            'label': buildLabel(config),
        })

    return experiments


def buildLabel(config):
    """
    Build a human-readable label from a config dict.
    Uses experimentName if set by user, otherwise auto-generates a descriptive label.
    Example auto labels:
      - Ring Topology (Clean)
      - Ring Topology (Noisy + No EBM + No Byzantine Nodes)
      - Ring Topology (Noisy + With EBM + No Byzantine Nodes)
      - Ring Topology (Noisy + With WCM + Gaussian@5)
    """
    # If user provided a custom name, use it
    customName = config.get('experimentName', '').strip()
    if customName:
        return customName

    # Auto-generate descriptive label
    useNoise = config.get('useChannelNoise', False)
    mitigation = config.get('noiseMitigation', 'none')
    useBasil = config.get('useBasil', False)

    # Noise description
    if not useNoise:
        noisePart = "Clean"
    else:
        noiseStart = config.get('channelNoiseStart', 0)
        noisePart = f"Noisy@Round {noiseStart}" if noiseStart > 0 else "Noisy"

        # Mitigation
        if mitigation == 'ebm':
            noisePart += " + With EBM"
        elif mitigation == 'wcm':
            noisePart += " + With WCM"
        else:
            noisePart += " + No EBM"

    # Attack description
    attacks = []
    if config.get('attackGaussian'):
        attacks.append(f"Gaussian@{config.get('attackGaussianStart', 0)}")
    if config.get('attackSignFlip'):
        attacks.append(f"SignFlip@{config.get('attackSignFlipStart', 0)}")
    if config.get('attackHidden'):
        attacks.append(f"Hidden@{config.get('attackHiddenStart', 0)}")
    if config.get('attackModelPoison'):
        attacks.append(f"ModelPoison@{config.get('attackModelPoisonStart', 0)}")

    if attacks:
        attackPart = " + ".join(attacks)
    else:
        attackPart = "No Byzantine Nodes"

    # Topology
    if useBasil:
        topo = "BASIL Ring"
    else:
        topo = "Ring Topology"

    return f"{topo} ({noisePart} + {attackPart})"


def plotDatasetExperiments(dataset, experiments):
    """
    Plot all experiments for a single dataset on one figure.
    """
    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = getColors(len(experiments))
    markers = getMarkers(len(experiments))

    for idx, exp in enumerate(experiments):
        if not os.path.exists(exp['avgPath']):
            continue

        acc = np.load(exp['avgPath'])
        rounds = np.arange(len(acc))

        ax.plot(rounds, acc,
                label=exp['label'],
                color=colors[idx], linewidth=2.5,
                marker=markers[idx], markersize=5,
                markevery=max(1, len(rounds) // 10))

    title = datasetTitles.get(dataset, dataset.upper())
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title(f'GUI Experiments on {title} - Average Accuracy',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    savePath = f"plots/images/gui/{dataset}_experiments_avg.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotDatasetGrid(dataset, experiments):
    """
    Plot each experiment in its own subplot for a dataset.
    Useful when there are many experiments.
    """
    if len(experiments) <= 1:
        return

    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    nExps = len(experiments)
    nCols = min(nExps, 3)
    nRows = (nExps + nCols - 1) // nCols

    fig, axes = plt.subplots(nRows, nCols, figsize=(6 * nCols, 5 * nRows))
    if nRows == 1 and nCols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    title = datasetTitles.get(dataset, dataset.upper())
    fig.suptitle(f'GUI Experiments on {title} - Average Accuracy',
                 fontsize=16, fontweight='bold')

    colors = getColors(len(experiments))
    markers = getMarkers(len(experiments))

    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        if not os.path.exists(exp['avgPath']):
            continue

        acc = np.load(exp['avgPath'])
        rounds = np.arange(len(acc))

        ax.plot(rounds, acc, color=colors[idx], linewidth=2.5,
                marker=markers[idx], markersize=4, markevery=1)
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Average Accuracy', fontsize=10)
        ax.set_title(exp['label'], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Annotate final accuracy
        if len(acc) > 0:
            ax.annotate(f'{acc[-1]:.3f}', xy=(len(acc) - 1, acc[-1]),
                       fontsize=9, fontweight='bold',
                       xytext=(-30, 10), textcoords='offset points')

    # Hide unused subplots
    for idx in range(nExps, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    savePath = f"plots/images/gui/{dataset}_grid_avg.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotFinalAccuracyBar(dataset, experiments):
    """
    Bar chart comparing final accuracy across all experiments for a dataset.
    """
    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    labels = []
    finalAccs = []

    for exp in experiments:
        if not os.path.exists(exp['avgPath']):
            continue
        acc = np.load(exp['avgPath'])
        labels.append(exp['label'])
        finalAccs.append(acc[-1] if len(acc) > 0 else 0)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))

    colors = getColors(len(labels))
    bars = ax.bar(range(len(labels)), finalAccs, color=colors, width=0.6)

    for bar, acc in zip(bars, finalAccs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    title = datasetTitles.get(dataset, dataset.upper())
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title(f'Final Average Accuracy - {title}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    savePath = f"plots/images/gui/{dataset}_final_accuracy_avg.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def generateGuiPlots():
    """Generate all GUI experiment plots."""
    print("\n" + "=" * 80)
    print("GENERATING GUI EXPERIMENT PLOTS")
    print("=" * 80)

    datasets = DATASETS_TO_PLOT if DATASETS_TO_PLOT else discoverDatasets()

    if not datasets:
        print("No GUI experiment results found in experiments/results/gui/")
        print("Run experiments via the GUI first (python runGui.py)")
        return

    print(f"Datasets found: {datasets}")
    print("=" * 80)

    for dataset in datasets:
        experiments = discoverExperiments(dataset)
        if not experiments:
            print(f"\nNo experiments found for {dataset}, skipping...")
            continue

        print(f"\nDataset: {dataset.upper()} ({len(experiments)} experiment(s))")
        for exp in experiments:
            print(f"  - {exp['label']}")

        # Overlay plot (all experiments on one chart)
        print(f"  1. Overlay comparison...")
        plotDatasetExperiments(dataset, experiments)

        # Grid plot (one subplot per experiment)
        if len(experiments) > 1:
            print(f"  2. Grid view...")
            plotDatasetGrid(dataset, experiments)

        # Final accuracy bar chart
        print(f"  3. Final accuracy bar chart...")
        plotFinalAccuracyBar(dataset, experiments)

    print("\nAll GUI plots generated!")
    print("Plots saved to: plots/images/gui/")


if __name__ == "__main__":
    print("""
=========================================================================
                      GUI EXPERIMENT PLOTTING SCRIPT                     
                                                                        
   Auto-discovers and plots all experiments from experiments/results/gui 
                                                                        
   To customize: Edit the CONFIGURATION section at the top of this file  
=========================================================================
    """)

    generateGuiPlots()
    print("\nDone!")
