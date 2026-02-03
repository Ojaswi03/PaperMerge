"""
Plotting for Noisy Channel experiments.
Compares clean (no mitigation) vs EBM vs WCM under different attacks.
Plots are saved to plots/images/noisyChannel/

CONFIGURATION: Edit the variables below to customize your plots
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - EDIT THESE TO CUSTOMIZE YOUR PLOTS
# ============================================================================

# Which datasets to plot? Options: "mnist", "cifar10", "nmnist"
DATASETS_TO_PLOT = ["mnist"]

# Which attacks to plot? Options: "clean", "gaussian", "signFlip", "hidden"
ATTACKS_TO_PLOT = ["clean", "gaussian"]

# Which metric to plot? Options: "avg", "worst", "both"
METRIC_TO_PLOT = "both"  # Use "both" to generate plots for both metrics

# Generate which plots? Set to True/False
GENERATE_BY_DATASET = True      # Clean vs EBM vs WCM for each dataset
GENERATE_ACROSS_DATASETS = True  # Compare across datasets
GENERATE_IMPACT = True           # Impact bar chart
GENERATE_SINGLE = True           # Single comprehensive plot per dataset

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Robust import
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'clean': '#F18F01',  # Orange - baseline noisy
    'ebm': '#C73E1D',    # Red - EBM
    'wcm': '#6A994E',    # Green - WCM
}

MARKERS = {
    'clean': 'o',
    'ebm': 's',
    'wcm': '^',
}


def loadNoisyChannelResults(dataset, mode, attack, metric='avg'):
    """
    Load noisy channel experiment results.

    Args:
        dataset: 'mnist', 'cifar10', or 'nmnist'
        mode: 'clean' (no mitigation), 'ebm', or 'wcm'
        attack: 'clean', 'gaussian', 'signFlip', or 'hidden'
        metric: 'avg' or 'worst'

    Returns:
        numpy array of accuracy history, or None if file doesn't exist
    """
    resultDir = f"experiments/results/noisyChannel/{dataset}"
    filename = f"acc_{mode}_{attack}_{metric}.npy"
    filepath = os.path.join(resultDir, filename)

    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        print(f"Warning: {filepath} not found")
        return None


def plotByDataset(datasets, attacks, metric='avg'):
    """
    Plot clean vs EBM vs WCM for each dataset.
    One plot per dataset showing all attacks.
    """
    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    attackLabels = {
        'clean': 'No Attack',
        'gaussian': 'Gaussian Attack',
        'signFlip': 'Sign-Flip Attack',
        'hidden': 'Hidden Attack'
    }

    modeLabels = {
        'clean': 'Noisy (no mitigation)',
        'ebm': 'EBM',
        'wcm': 'WCM'
    }

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    for dataset in datasets:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Noisy Channel Approaches on {datasetTitles[dataset]} - {metricLabel}',
                     fontsize=16, fontweight='bold')

        for idx, attack in enumerate(attacks):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # Plot all modes
            for mode in ['clean', 'ebm', 'wcm']:
                acc = loadNoisyChannelResults(dataset, mode, attack, metric=metric)
                if acc is not None:
                    rounds = np.arange(len(acc))
                    ax.plot(rounds, acc, label=modeLabels[mode],
                           color=COLORS[mode], linewidth=2.5,
                           marker=MARKERS[mode], markersize=5, markevery=max(1, len(rounds)//10))

            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel(metricLabel, fontsize=11)
            ax.set_title(attackLabels[attack], fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()
        savePath = f"plots/images/noisyChannel/{dataset}_comparison_{metric}.png"
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        print(f"Saved: {savePath}")
        plt.close()


def plotAcrossDatasets(attacks, metric='avg'):
    """
    Plot comparison across all datasets for each approach.
    """
    datasets = ['mnist', 'cifar10', 'nmnist']
    datasetLabels = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'N-MNIST'
    }

    attackLabels = {
        'clean': 'No Attack',
        'gaussian': 'Gaussian Attack',
        'signFlip': 'Sign-Flip Attack',
        'hidden': 'Hidden Attack'
    }

    modes = ['ebm', 'wcm']
    modeLabels = {
        'ebm': 'EBM',
        'wcm': 'WCM'
    }

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    for mode in modes:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{modeLabels[mode]} Across Datasets - {metricLabel}',
                     fontsize=16, fontweight='bold')

        for idx, attack in enumerate(attacks):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            for dataset in datasets:
                acc = loadNoisyChannelResults(dataset, mode, attack, metric=metric)
                if acc is not None:
                    rounds = np.arange(len(acc))
                    ax.plot(rounds, acc, label=datasetLabels[dataset],
                           linewidth=2.5, marker='o', markersize=4, markevery=max(1, len(rounds)//10))

            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel(metricLabel, fontsize=11)
            ax.set_title(attackLabels[attack], fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()
        savePath = f"plots/images/noisyChannel/{mode}_across_datasets_{metric}.png"
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        print(f"Saved: {savePath}")
        plt.close()


def plotImpact(datasets, attacks, metric='avg'):
    """
    Plot the impact of EBM and WCM (improvement over clean baseline).
    """
    datasetLabels = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'N-MNIST'
    }

    attackLabels = {
        'gaussian': 'Gaussian',
        'signFlip': 'Sign-Flip',
        'hidden': 'Hidden'
    }

    # Filter out 'clean' attack for impact (only show under attacks)
    impactAttacks = [a for a in attacks if a != 'clean']
    if not impactAttacks:
        print("Skipping impact plot (no attacks to compare)")
        return

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Noisy Channel Mitigation Impact - {metricLabel}', fontsize=16, fontweight='bold')

    x = np.arange(len(impactAttacks))
    width = 0.25

    # Plot 1: EBM impact
    for didx, dataset in enumerate(datasets):
        improvements = []
        for attack in impactAttacks:
            cleanAcc = loadNoisyChannelResults(dataset, 'clean', attack, metric=metric)
            ebmAcc = loadNoisyChannelResults(dataset, 'ebm', attack, metric=metric)

            if cleanAcc is not None and ebmAcc is not None:
                cleanFinal = cleanAcc[-1]
                ebmFinal = ebmAcc[-1]
                improvement = (ebmFinal - cleanFinal) * 100
                improvements.append(improvement)
            else:
                improvements.append(0)

        offset = (didx - 1) * width
        ax1.bar(x + offset, improvements, width, label=datasetLabels[dataset])

    ax1.set_xlabel('Attack Type', fontsize=12)
    ax1.set_ylabel('Improvement (percentage points)', fontsize=12)
    ax1.set_title('EBM Impact', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([attackLabels[a] for a in impactAttacks])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 2: WCM impact
    for didx, dataset in enumerate(datasets):
        improvements = []
        for attack in impactAttacks:
            cleanAcc = loadNoisyChannelResults(dataset, 'clean', attack, metric=metric)
            wcmAcc = loadNoisyChannelResults(dataset, 'wcm', attack, metric=metric)

            if cleanAcc is not None and wcmAcc is not None:
                cleanFinal = cleanAcc[-1]
                wcmFinal = wcmAcc[-1]
                improvement = (wcmFinal - cleanFinal) * 100
                improvements.append(improvement)
            else:
                improvements.append(0)

        offset = (didx - 1) * width
        ax2.bar(x + offset, improvements, width, label=datasetLabels[dataset])

    ax2.set_xlabel('Attack Type', fontsize=12)
    ax2.set_ylabel('Improvement (percentage points)', fontsize=12)
    ax2.set_title('WCM Impact', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([attackLabels[a] for a in impactAttacks])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    savePath = f"plots/images/noisyChannel/mitigation_impact_{metric}.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotSingleDataset(dataset, attacks, metric='avg'):
    """
    Create a single comprehensive plot for one dataset.
    Shows clean vs EBM vs WCM for all attacks.
    """
    attackLabels = {
        'clean': 'No Attack',
        'gaussian': 'Gaussian Attack',
        'signFlip': 'Sign-Flip Attack',
        'hidden': 'Hidden Attack'
    }

    datasetTitles = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'nmnist': 'Neuromorphic MNIST'
    }

    modeLabels = {
        'clean': 'Noisy',
        'ebm': 'EBM',
        'wcm': 'WCM'
    }

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    fig, ax = plt.subplots(figsize=(12, 6))

    for attack in attacks:
        for mode in ['clean', 'ebm', 'wcm']:
            acc = loadNoisyChannelResults(dataset, mode, attack, metric=metric)
            if acc is not None:
                rounds = np.arange(len(acc))
                linestyle = '--' if mode == 'clean' else '-'
                linewidth = 2 if mode == 'clean' else 2.5
                ax.plot(rounds, acc,
                       label=f'{attackLabels[attack]} - {modeLabels[mode]}',
                       color=COLORS[mode], linestyle=linestyle, linewidth=linewidth,
                       marker=MARKERS[mode], markersize=4, markevery=max(1, len(rounds)//10), alpha=0.8)

    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel(metricLabel, fontsize=12)
    ax.set_title(f'Noisy Channel Approaches on {datasetTitles[dataset]} - {metricLabel}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    savePath = f"plots/images/noisyChannel/{dataset}_all_attacks_{metric}.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def generateAllPlots():
    """Generate all plots based on configuration."""
    print("\n" + "="*80)
    print("GENERATING NOISY CHANNEL PLOTS")
    print("="*80)
    print(f"Datasets: {DATASETS_TO_PLOT}")
    print(f"Attacks: {ATTACKS_TO_PLOT}")
    print(f"Metric: {METRIC_TO_PLOT}")
    print("="*80)

    metrics = ['avg', 'worst'] if METRIC_TO_PLOT == 'both' else [METRIC_TO_PLOT]

    for metric in metrics:
        print(f"\nGenerating plots for {metric} metric...")

        if GENERATE_BY_DATASET:
            print(f"  1. Comparison by dataset ({metric})...")
            plotByDataset(DATASETS_TO_PLOT, ATTACKS_TO_PLOT, metric=metric)

        if GENERATE_ACROSS_DATASETS:
            print(f"  2. Comparison across datasets ({metric})...")
            plotAcrossDatasets(ATTACKS_TO_PLOT, metric=metric)

        if GENERATE_IMPACT:
            print(f"  3. Impact comparison ({metric})...")
            plotImpact(DATASETS_TO_PLOT, ATTACKS_TO_PLOT, metric=metric)

        if GENERATE_SINGLE:
            print(f"  4. Single dataset comprehensive plots ({metric})...")
            for dataset in DATASETS_TO_PLOT:
                print(f"     - {dataset.upper()}...")
                plotSingleDataset(dataset, ATTACKS_TO_PLOT, metric=metric)

    print("\nAll noisy channel plots generated successfully!")
    print("Plots saved to: plots/images/noisyChannel/")


if __name__ == "__main__":
    print("""
===================================================================================
                  NOISY CHANNEL PLOTTING SCRIPT                             
                                                                            
  Generates plots comparing Clean (no mitigation), EBM, and WCM            
                                                                            
  To customize: Edit the CONFIGURATION section at the top of this file    
===================================================================================
    """)

    generateAllPlots()
    print("\nDone!")
