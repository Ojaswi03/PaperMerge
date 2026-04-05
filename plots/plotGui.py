"""
Plotting for GUI experiment results.
Auto-discovers all experiments saved in experiments/results/gui/
and generates comparison plots per dataset / attack type / approach.

Folder structure (results):
  experiments/results/gui/{dataset}/{attackKey}/{approach}/
      acc_*.npy
      config_*.json

Folder structure (plots):
  plots/images/gui/{dataset}/{attackKey}/{approach}/
      experiments_avg.png
      grid_avg.png
      final_accuracy_avg.png

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
    # pick colormap based on how many colors are needed
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
    # cycle through marker list to get n markers
    return [m for _, m in zip(range(n), itertools.cycle(_ALL_MARKERS))]


def discoverDatasets():
    # scan gui results directory for dataset subdirs
    # supports both old flat structure (dataset/attack/*.npy)
    # and new structure (dataset/attack/approach/*.npy)
    guiDir = "experiments/results/gui"
    if not os.path.isdir(guiDir):
        print(f"No GUI results directory found at {guiDir}")
        return []

    datasets = []
    for entry in sorted(os.listdir(guiDir)):
        entryPath = os.path.join(guiDir, entry)
        if not os.path.isdir(entryPath):
            continue
        # new structure: 3 levels deep
        newNpy = glob.glob(os.path.join(entryPath, "*", "*", "*.npy"))
        # old flat structure: 2 levels deep
        oldNpy = glob.glob(os.path.join(entryPath, "*", "*.npy"))
        if newNpy or oldNpy:
            datasets.append(entry)

    return datasets


def discoverAttackTypes(dataset):
    # return attack type subfolders that have results (new or old structure)
    datasetDir = f"experiments/results/gui/{dataset}"
    if not os.path.isdir(datasetDir):
        return []

    attackTypes = []
    for entry in sorted(os.listdir(datasetDir)):
        entryPath = os.path.join(datasetDir, entry)
        if not os.path.isdir(entryPath):
            continue
        # new structure: approach subdir contains npy
        newNpy = glob.glob(os.path.join(entryPath, "*", "*.npy"))
        # old flat structure: npy directly in attack folder
        oldNpy = glob.glob(os.path.join(entryPath, "*.npy"))
        if newNpy or oldNpy:
            attackTypes.append(entry)

    return attackTypes


def discoverApproaches(dataset, attackKey):
    # return approach subfolders (basil / noisy / merged) that have npy files
    attackDir = f"experiments/results/gui/{dataset}/{attackKey}"
    if not os.path.isdir(attackDir):
        return []

    approaches = []
    for entry in sorted(os.listdir(attackDir)):
        entryPath = os.path.join(attackDir, entry)
        if os.path.isdir(entryPath) and glob.glob(os.path.join(entryPath, "*.npy")):
            approaches.append(entry)

    # fall back to old flat structure: npy files directly in attack folder
    if not approaches and glob.glob(os.path.join(attackDir, "*.npy")):
        approaches.append("_legacy")

    return approaches


def discoverExperiments(dataset, attackKey=None, approach=None):
    # find config JSON files and pair each with its .npy accuracy file
    if attackKey and approach and approach != "_legacy":
        resultDir = f"experiments/results/gui/{dataset}/{attackKey}/{approach}"
    elif attackKey:
        resultDir = f"experiments/results/gui/{dataset}/{attackKey}"
    else:
        resultDir = f"experiments/results/gui/{dataset}"

    if not os.path.isdir(resultDir):
        return []

    configFiles = sorted(glob.glob(os.path.join(resultDir, "config_*.json")))

    experiments = []
    for configPath in configFiles:
        configName = os.path.basename(configPath)
        name = configName.replace("config_", "").replace(".json", "")

        avgPath = os.path.join(resultDir, f"acc_{name}.npy")
        if not os.path.exists(avgPath):
            continue

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
    # use custom experiment name if provided, otherwise auto-generate from config fields
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
    if config.get('attackScaling'):
        attacks.append(f"Scaling@{config.get('attackScalingStart', 0)}")
    if config.get('attackAlie'):
        attacks.append(f"ALIE@{config.get('attackAlieStart', 0)}")
    if config.get('attackIpm'):
        attacks.append(f"IPM@{config.get('attackIpmStart', 0)}")
    if config.get('attackNoiseAmp'):
        attacks.append(f"NoiseAmp@{config.get('attackNoiseAmpStart', 0)}")

    if attacks:
        attackPart = " + ".join(attacks)
    else:
        attackPart = "No Byzantine Nodes"

    # Topology
    topo = "BASIL Ring" if useBasil else "Ring Topology"

    return f"{topo} ({noisePart} + {attackPart})"


def _approachTitle(approach):
    # human-readable approach label for plot titles
    titles = {
        'basil':   'BASIL Ring',
        'noisy':   'Noisy Channel (FedAvg)',
        'merged':  'Merged (BASIL + EBM)',
        '_legacy': 'Legacy',
    }
    return titles.get(approach, approach.title())


CLEAN_COLOR = '#16a34a'   # green
CLEAN_STYLE = '--'
CLEAN_LW    = 2.0
CLEAN_ALPHA = 0.75


def loadCleanBaselines(dataset, approach):
    """Return experiments from the 'none' attack folder — the clean reference runs."""
    return discoverExperiments(dataset, 'none', approach)


def bestCleanAcc(dataset, approach):
    """Final accuracy of the highest-performing clean experiment, or None."""
    best = None
    for exp in loadCleanBaselines(dataset, approach):
        if os.path.exists(exp['avgPath']):
            acc = np.load(exp['avgPath'])
            if len(acc) > 0 and (best is None or acc[-1] > best):
                best = float(acc[-1])
    return best


def plotDatasetExperiments(dataset, attackKey, approach, experiments):
    # overlay all experiment curves on one axes and save the figure
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

    # Overlay clean baselines as dashed green reference lines
    if attackKey != 'none':
        for exp in loadCleanBaselines(dataset, approach):
            if not os.path.exists(exp['avgPath']):
                continue
            acc    = np.load(exp['avgPath'])
            rounds = np.arange(len(acc))
            ax.plot(rounds, acc, label=f"[Clean] {exp['label']}",
                    color=CLEAN_COLOR, linewidth=CLEAN_LW,
                    linestyle=CLEAN_STYLE, alpha=CLEAN_ALPHA)

    title = datasetTitles.get(dataset, dataset.upper())
    attackTitle = attackKey.replace("_", " + ").title()
    approachTitle = _approachTitle(approach)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title(f'{title} | {attackTitle} | {approachTitle} — Average Accuracy',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    savePath = f"plots/images/gui/{dataset}/{attackKey}/{approach}/experiments_avg.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotDatasetGrid(dataset, attackKey, approach, experiments):
    # one subplot per experiment arranged in a grid layout
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
    attackTitle = attackKey.replace("_", " + ").title()
    approachTitle = _approachTitle(approach)
    fig.suptitle(f'{title} | {attackTitle} | {approachTitle} — Average Accuracy',
                 fontsize=16, fontweight='bold')

    colors  = getColors(len(experiments))
    markers = getMarkers(len(experiments))

    # Pre-load best clean accuracy for this dataset/approach
    _cleanAcc = bestCleanAcc(dataset, approach) if attackKey != 'none' else None

    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        if not os.path.exists(exp['avgPath']):
            continue

        acc    = np.load(exp['avgPath'])
        rounds = np.arange(len(acc))

        ax.plot(rounds, acc, color=colors[idx], linewidth=2.5,
                marker=markers[idx], markersize=4, markevery=1)
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Average Accuracy', fontsize=10)
        ax.set_title(exp['label'], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Dashed green clean-baseline reference line
        if _cleanAcc is not None:
            ax.axhline(y=_cleanAcc, color=CLEAN_COLOR, linestyle=CLEAN_STYLE,
                       linewidth=CLEAN_LW, alpha=CLEAN_ALPHA,
                       label=f'Clean: {_cleanAcc:.3f}')
            ax.legend(fontsize=8, loc='lower right')

        # annotate the final accuracy value on the last point
        if len(acc) > 0:
            ax.annotate(f'{acc[-1]:.3f}', xy=(len(acc) - 1, acc[-1]),
                       fontsize=9, fontweight='bold',
                       xytext=(-30, 10), textcoords='offset points')

    # hide unused subplots
    for idx in range(nExps, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    savePath = f"plots/images/gui/{dataset}/{attackKey}/{approach}/grid_avg.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotFinalAccuracyBar(dataset, attackKey, approach, experiments):
    # bar chart of each experiment's final accuracy value
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
    bars   = ax.bar(range(len(labels)), finalAccs, color=colors, width=0.6)

    for bar, acc in zip(bars, finalAccs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Dashed green clean-baseline reference line
    if attackKey != 'none':
        _cleanAcc = bestCleanAcc(dataset, approach)
        if _cleanAcc is not None:
            ax.axhline(y=_cleanAcc, color=CLEAN_COLOR, linestyle=CLEAN_STYLE,
                       linewidth=CLEAN_LW, alpha=CLEAN_ALPHA,
                       label=f'Clean baseline: {_cleanAcc:.3f}', zorder=5)
            ax.legend(fontsize=10, loc='upper right')

    title = datasetTitles.get(dataset, dataset.upper())
    attackTitle = attackKey.replace("_", " + ").title()
    approachTitle = _approachTitle(approach)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title(f'Final Accuracy | {title} | {attackTitle} | {approachTitle}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    savePath = f"plots/images/gui/{dataset}/{attackKey}/{approach}/final_accuracy_avg.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def generateGuiPlots():
    # discover datasets → attack types → approaches, then generate plots per combination
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
        attackTypes = discoverAttackTypes(dataset)
        if not attackTypes:
            print(f"\nNo attack type subfolders found for {dataset}, skipping...")
            continue

        print(f"\nDataset: {dataset.upper()} - Attack types: {attackTypes}")

        for attackKey in attackTypes:
            approaches = discoverApproaches(dataset, attackKey)
            if not approaches:
                continue

            print(f"\n  Attack: {attackKey} - Approaches: {approaches}")

            for approach in approaches:
                experiments = discoverExperiments(dataset, attackKey, approach)
                if not experiments:
                    continue

                print(f"\n    Approach: {approach} ({len(experiments)} experiment(s))")
                for exp in experiments:
                    print(f"      - {exp['label']}")

                plotDatasetExperiments(dataset, attackKey, approach, experiments)

                if len(experiments) > 1:
                    plotDatasetGrid(dataset, attackKey, approach, experiments)

                plotFinalAccuracyBar(dataset, attackKey, approach, experiments)

    print("\nAll GUI plots generated!")
    print("Plots saved to: plots/images/gui/")


if __name__ == "__main__":
    print("""
=========================================================================
                      GUI EXPERIMENT PLOTTING SCRIPT

   Auto-discovers and plots all experiments from experiments/results/gui

   Organized by: dataset / attack type / approach
=========================================================================
    """)

    generateGuiPlots()
    print("\nDone!")
