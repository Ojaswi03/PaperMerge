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
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - EDIT THESE TO CUSTOMIZE YOUR PLOTS
# ============================================================================

# Which datasets to plot? Options: "mnist", "cifar10", "nmnist"
# Set to None to auto-detect all available datasets
DATASETS_TO_PLOT = None

# Which metric? Options: "avg", "worst", "both"
METRIC = "both"

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Robust import
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
COLORS_LIST = [
    '#2E86AB', '#A23B72', '#E63946', '#457B9D',
    '#6A994E', '#F18F01', '#C73E1D', '#A8DADC',
]

MARKERS_LIST = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


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
    Returns a list of dicts with keys: timestamp, approach, config, avgPath, worstPath
    """
    resultDir = f"experiments/results/gui/{dataset}"
    if not os.path.isdir(resultDir):
        return []

    # Find all config files
    configFiles = sorted(glob.glob(os.path.join(resultDir, "config_*.json")))

    experiments = []
    for configPath in configFiles:
        configName = os.path.basename(configPath)
        # Pattern: config_{approach}_{timestamp}.json
        parts = configName.replace("config_", "").replace(".json", "")
        # Split on last underscore-separated timestamp (YYYYMMDD_HHMMSS)
        # approach might contain underscores, so split from the right
        tokens = parts.rsplit("_", 2)
        if len(tokens) >= 3:
            approach = tokens[0]
            timestamp = f"{tokens[1]}_{tokens[2]}"
        else:
            approach = parts
            timestamp = ""

        avgPath = os.path.join(resultDir, f"acc_{approach}_{timestamp}_avg.npy")
        worstPath = os.path.join(resultDir, f"acc_{approach}_{timestamp}_worst.npy")

        if not os.path.exists(avgPath):
            continue

        # Load config
        with open(configPath, 'r') as f:
            config = json.load(f)

        experiments.append({
            'timestamp': timestamp,
            'approach': approach,
            'config': config,
            'avgPath': avgPath,
            'worstPath': worstPath,
            'label': buildLabel(config),
        })

    return experiments


def buildLabel(config):
    """Build a human-readable label from a config dict."""
    parts = []

    # Approach
    approachLabels = {'basil': 'BASIL', 'noisy': 'Noisy Channel', 'merged': 'Merged'}
    parts.append(approachLabels.get(config.get('approach', ''), config.get('approach', '')))

    # Attacks
    attacks = []
    if config.get('attackGaussian'):
        attacks.append(f"Gauss@{config.get('attackGaussianStart', 0)}")
    if config.get('attackSignFlip'):
        attacks.append(f"SignFlip@{config.get('attackSignFlipStart', 0)}")
    if config.get('attackHidden'):
        attacks.append(f"Hidden@{config.get('attackHiddenStart', 0)}")

    if attacks:
        parts.append("+".join(attacks))
    else:
        parts.append("No Attack")

    # Mitigation
    mitigation = config.get('noiseMitigation', 'none')
    if mitigation != 'none':
        parts.append(mitigation.upper())

    return " | ".join(parts)


def plotDatasetExperiments(dataset, experiments, metric='avg'):
    """
    Plot all experiments for a single dataset on one figure.
    """
    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"
    metricSuffix = metric

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, exp in enumerate(experiments):
        accPath = exp['avgPath'] if metric == 'avg' else exp['worstPath']
        if not os.path.exists(accPath):
            continue

        acc = np.load(accPath)
        rounds = np.arange(len(acc))

        color = COLORS_LIST[idx % len(COLORS_LIST)]
        marker = MARKERS_LIST[idx % len(MARKERS_LIST)]

        ax.plot(rounds, acc,
                label=exp['label'],
                color=color, linewidth=2.5,
                marker=marker, markersize=5,
                markevery=max(1, len(rounds) // 10))

    title = datasetTitles.get(dataset, dataset.upper())
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel(metricLabel, fontsize=12)
    ax.set_title(f'GUI Experiments on {title} - {metricLabel}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    savePath = f"plots/images/gui/{dataset}_experiments_{metricSuffix}.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotDatasetGrid(dataset, experiments, metric='avg'):
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

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    nExps = len(experiments)
    nCols = min(nExps, 3)
    nRows = (nExps + nCols - 1) // nCols

    fig, axes = plt.subplots(nRows, nCols, figsize=(6 * nCols, 5 * nRows))
    if nRows == 1 and nCols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    title = datasetTitles.get(dataset, dataset.upper())
    fig.suptitle(f'GUI Experiments on {title} - {metricLabel}',
                 fontsize=16, fontweight='bold')

    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        accPath = exp['avgPath'] if metric == 'avg' else exp['worstPath']
        if not os.path.exists(accPath):
            continue

        acc = np.load(accPath)
        rounds = np.arange(len(acc))

        color = COLORS_LIST[idx % len(COLORS_LIST)]

        ax.plot(rounds, acc, color=color, linewidth=2.5)
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel(metricLabel, fontsize=10)
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
    savePath = f"plots/images/gui/{dataset}_grid_{metric}.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotFinalAccuracyBar(dataset, experiments, metric='avg'):
    """
    Bar chart comparing final accuracy across all experiments for a dataset.
    """
    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    labels = []
    finalAccs = []

    for idx, exp in enumerate(experiments):
        accPath = exp['avgPath'] if metric == 'avg' else exp['worstPath']
        if not os.path.exists(accPath):
            continue

        acc = np.load(accPath)
        labels.append(exp['label'])
        finalAccs.append(acc[-1] if len(acc) > 0 else 0)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))

    colors = [COLORS_LIST[i % len(COLORS_LIST)] for i in range(len(labels))]
    bars = ax.bar(range(len(labels)), finalAccs, color=colors, width=0.6)

    # Add value labels on bars
    for bar, acc in zip(bars, finalAccs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    title = datasetTitles.get(dataset, dataset.upper())
    ax.set_ylabel(metricLabel, fontsize=12)
    ax.set_title(f'Final {metricLabel} - {title}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    savePath = f"plots/images/gui/{dataset}_final_accuracy_{metric}.png"
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
    print(f"Metric: {METRIC}")
    print("=" * 80)

    metrics = ['avg', 'worst'] if METRIC == 'both' else [METRIC]

    for dataset in datasets:
        experiments = discoverExperiments(dataset)
        if not experiments:
            print(f"\nNo experiments found for {dataset}, skipping...")
            continue

        print(f"\nDataset: {dataset.upper()} ({len(experiments)} experiment(s))")
        for exp in experiments:
            print(f"  - {exp['label']} ({exp['timestamp']})")

        for metric in metrics:
            print(f"\n  Generating {metric} plots...")

            # Overlay plot (all experiments on one chart)
            print(f"    1. Overlay comparison...")
            plotDatasetExperiments(dataset, experiments, metric=metric)

            # Grid plot (one subplot per experiment)
            if len(experiments) > 1:
                print(f"    2. Grid view...")
                plotDatasetGrid(dataset, experiments, metric=metric)

            # Final accuracy bar chart
            print(f"    3. Final accuracy bar chart...")
            plotFinalAccuracyBar(dataset, experiments, metric=metric)

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
