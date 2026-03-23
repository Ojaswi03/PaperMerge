"""
Comprehensive plotting for BASIL + Noisy Channel experiments.
Generates 3 types of comparison plots:
1. Standalone BASIL performance across datasets with different attacks
2. Noise mitigation comparison (noisy baseline vs EBM vs WCM)
3. Merged approach performance (BASIL-only vs BASIL+EBM vs BASIL+WCM)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Robust import
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'clean': '#2E86AB',
    'basilOnly': '#A23B72',
    'noisy': '#F18F01',
    'ebm': '#C73E1D',
    'wcm': '#6A994E',
    'basilEbm': '#BC4B51',
    'basilWcm': '#5E503F',
    'gaussian': '#E63946',
    'signFlip': '#457B9D',
    'hidden': '#A8DADC',
}

LINESTYLES = {
    'clean': '-',
    'gaussian': '--',
    'signFlip': '-.',
    'hidden': ':',
}


def loadResults(dataset, mode, attack, metric='avg'):
    # build file path from dataset/mode/attack/metric and load, returning None if absent
    resultDir = f"experiments/results/{dataset}"
    filename = f"acc_{mode}_{attack}_{metric}.npy"
    filepath = os.path.join(resultDir, filename)

    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        print(f"Warning: {filepath} not found")
        return None


def plotBasilPerformanceAcrossDatasets(savePath="plots/images/basil_performance_comparison.png"):
    # 3 subplots (one per dataset) showing BASIL accuracy under all attack types
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = ['mnist', 'cifar10', 'nmnist']
    datasetTitles = ['MNIST', 'CIFAR-10', 'Neuromorphic MNIST']
    attacks = ['clean', 'gaussian', 'signFlip', 'hidden']
    attackLabels = {
        'clean': 'No Attack',
        'gaussian': 'Gaussian Attack',
        'signFlip': 'Sign-Flip Attack',
        'hidden': 'Hidden Attack'
    }

    for idx, (dataset, title) in enumerate(zip(datasets, datasetTitles)):
        ax = axes[idx]

        for attack in attacks:
            # Load BASIL-only results under this attack
            acc = loadResults(dataset, 'basilOnly', attack, metric='avg')
            if acc is not None:
                rounds = np.arange(len(acc))
                ax.plot(rounds, acc, label=attackLabels[attack],
                       color=COLORS.get(attack, '#000000'),
                       linestyle=LINESTYLES.get(attack, '-'),
                       linewidth=2, marker='o', markersize=4, markevery=5)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'BASIL Performance on {title}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotNoiseMitigationComparison(savePath="plots/images/noise_mitigation_comparison.png"):
    # 3 subplots comparing noisy/EBM/WCM per dataset under Gaussian attack
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = ['mnist', 'cifar10', 'nmnist']
    datasetTitles = ['MNIST', 'CIFAR-10', 'Neuromorphic MNIST']
    modes = ['noisy', 'ebm', 'wcm']
    modeLabels = {
        'noisy': 'Noisy (no mitigation)',
        'ebm': 'EBM Regularization',
        'wcm': 'WCM Regularization'
    }

    for idx, (dataset, title) in enumerate(zip(datasets, datasetTitles)):
        ax = axes[idx]

        for mode in modes:
            # Load results under Gaussian attack
            acc = loadResults(dataset, mode, 'gaussian', metric='avg')
            if acc is not None:
                rounds = np.arange(len(acc))
                ax.plot(rounds, acc, label=modeLabels[mode],
                       color=COLORS.get(mode, '#000000'),
                       linewidth=2, marker='s', markersize=4, markevery=5)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'Noise Mitigation on {title}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotMergedApproachComparison(savePath="plots/images/merged_approach_comparison.png"):
    # 3 subplots comparing noisy/BASIL-only/BASIL+EBM/BASIL+WCM per dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = ['mnist', 'cifar10', 'nmnist']
    datasetTitles = ['MNIST', 'CIFAR-10', 'Neuromorphic MNIST']
    modes = ['noisy', 'basilOnly', 'basilEbm', 'basilWcm']
    modeLabels = {
        'noisy': 'Noisy Baseline',
        'basilOnly': 'BASIL Only',
        'basilEbm': 'BASIL + EBM',
        'basilWcm': 'BASIL + WCM'
    }

    for idx, (dataset, title) in enumerate(zip(datasets, datasetTitles)):
        ax = axes[idx]

        for mode in modes:
            # Load results under Gaussian attack
            acc = loadResults(dataset, mode, 'gaussian', metric='avg')
            if acc is not None:
                rounds = np.arange(len(acc))
                ax.plot(rounds, acc, label=modeLabels[mode],
                       color=COLORS.get(mode, '#000000'),
                       linewidth=2, marker='D', markersize=4, markevery=5)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'Merged Approach on {title}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotDetailedComparison(dataset, savePath=None):
    # 2x2 grid showing all modes under each attack type for one dataset
    if savePath is None:
        savePath = f"plots/images/{dataset}_detailed_comparison.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    attacks = ['clean', 'gaussian', 'signFlip', 'hidden']
    attackTitles = ['No Attack', 'Gaussian Attack', 'Sign-Flip Attack', 'Hidden Attack']
    modes = ['clean', 'basilOnly', 'noisy', 'ebm', 'wcm', 'basilEbm', 'basilWcm']
    modeLabels = {
        'clean': 'Clean (baseline)',
        'basilOnly': 'BASIL Only',
        'noisy': 'Noisy',
        'ebm': 'EBM',
        'wcm': 'WCM',
        'basilEbm': 'BASIL+EBM',
        'basilWcm': 'BASIL+WCM'
    }

    for idx, (attack, attackTitle) in enumerate(zip(attacks, attackTitles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        for mode in modes:
            acc = loadResults(dataset, mode, attack, metric='avg')
            if acc is not None:
                rounds = np.arange(len(acc))
                ax.plot(rounds, acc, label=modeLabels[mode],
                       color=COLORS.get(mode, '#000000'),
                       linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Round', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title(f'{dataset.upper()}: {attackTitle}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotWorstCaseComparison(savePath="plots/images/worst_case_comparison.png"):
    # 3 subplots showing worst-node accuracy for BASIL variants under Gaussian attack
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = ['mnist', 'cifar10', 'nmnist']
    datasetTitles = ['MNIST', 'CIFAR-10', 'Neuromorphic MNIST']
    modes = ['basilOnly', 'basilEbm', 'basilWcm']
    modeLabels = {
        'basilOnly': 'BASIL Only',
        'basilEbm': 'BASIL + EBM',
        'basilWcm': 'BASIL + WCM'
    }

    for idx, (dataset, title) in enumerate(zip(datasets, datasetTitles)):
        ax = axes[idx]

        for mode in modes:
            # Load worst-case results under Gaussian attack
            acc = loadResults(dataset, mode, 'gaussian', metric='worst')
            if acc is not None:
                rounds = np.arange(len(acc))
                ax.plot(rounds, acc, label=modeLabels[mode],
                       color=COLORS.get(mode, '#000000'),
                       linewidth=2, marker='v', markersize=4, markevery=5)

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Worst-Node Accuracy', fontsize=12)
        ax.set_title(f'Worst-Case Performance on {title}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def generateAllPlots():
    # run all five plot functions in sequence
    print("Generating comprehensive comparison plots...")

    print("\n1. BASIL performance across datasets...")
    plotBasilPerformanceAcrossDatasets()

    print("\n2. Noise mitigation comparison...")
    plotNoiseMitigationComparison()

    print("\n3. Merged approach comparison...")
    plotMergedApproachComparison()

    print("\n4. Worst-case comparison...")
    plotWorstCaseComparison()

    print("\n5. Detailed comparisons for each dataset...")
    for dataset in ['mnist', 'cifar10', 'nmnist']:
        print(f"   - {dataset.upper()}...")
        plotDetailedComparison(dataset)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate comprehensive comparison plots")
    parser.add_argument("--plot", choices=[
        'all', 'basil', 'noise', 'merged', 'worst', 'detailed'
    ], default='all', help="Which plot(s) to generate")
    parser.add_argument("--dataset", choices=['mnist', 'cifar10', 'nmnist'],
                        help="For detailed plot, which dataset to use")

    args = parser.parse_args()

    if args.plot == 'all':
        generateAllPlots()
    elif args.plot == 'basil':
        plotBasilPerformanceAcrossDatasets()
    elif args.plot == 'noise':
        plotNoiseMitigationComparison()
    elif args.plot == 'merged':
        plotMergedApproachComparison()
    elif args.plot == 'worst':
        plotWorstCaseComparison()
    elif args.plot == 'detailed':
        if args.dataset:
            plotDetailedComparison(args.dataset)
        else:
            for dataset in ['mnist', 'cifar10', 'nmnist']:
                plotDetailedComparison(dataset)

    print("\nDone!")
