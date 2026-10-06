"""
Plotting for BASIL-only experiments.
Compares clean (no BASIL) vs BASIL approach under different attacks.
Plots are saved to plots/images/basil/

CONFIGURATION: Edit the variables below to customize your plots
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - EDIT THESE TO CUSTOMIZE YOUR PLOTS
# ============================================================================

# Which plot type to generate? Options: "all", "bydataset", "acrossdatasets", "impact", "single"
PLOT_TYPE = "all"

# For single plot, which dataset? Options: "mnist", "cifar10", "nmnist"
SINGLE_DATASET = "mnist"

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
COLORS = {
    'clean': '#2E86AB',
    'basil': '#A23B72',
    'gaussian': '#E63946',
    'signFlip': '#457B9D',
    'hidden': '#A8DADC',
}

MARKERS = {
    'clean': 'o',
    'basil': 's',
}


def loadBasilResults(dataset, mode, attack, metric='avg'):
    # build file path and load .npy array, returning None if missing
    resultDir = f"experiments/results/basil/{dataset}"
    filename = f"acc_{mode}_{attack}_{metric}.npy"
    filepath = os.path.join(resultDir, filename)

    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        print(f"Warning: {filepath} not found")
        return None


def plotCleanVsBasilByDataset(datasets=['mnist', 'cifar10', 'nmnist'],
                                attacks=['clean', 'gaussian', 'signFlip', 'hidden'],
                                metric='avg'):
    # one 2x2 figure per dataset comparing clean and BASIL under all attacks
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

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    for dataset in datasets:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Clean vs BASIL on {datasetTitles[dataset]} - {metricLabel}',
                     fontsize=16, fontweight='bold')

        for idx, attack in enumerate(attacks):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # Plot clean (no BASIL)
            cleanAcc = loadBasilResults(dataset, 'clean', attack, metric=metric)
            if cleanAcc is not None:
                rounds = np.arange(len(cleanAcc))
                ax.plot(rounds, cleanAcc, label='Clean (no BASIL)',
                       color=COLORS['clean'], linewidth=2.5,
                       marker=MARKERS['clean'], markersize=5, markevery=5)

            # Plot BASIL
            basilAcc = loadBasilResults(dataset, 'basil', attack, metric=metric)
            if basilAcc is not None:
                rounds = np.arange(len(basilAcc))
                ax.plot(rounds, basilAcc, label='BASIL',
                       color=COLORS['basil'], linewidth=2.5,
                       marker=MARKERS['basil'], markersize=5, markevery=5)

            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel(metricLabel, fontsize=11)
            ax.set_title(attackLabels[attack], fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()
        savePath = f"plots/images/basil/{dataset}_clean_vs_basil_{metric}.png"
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        print(f"Saved: {savePath}")
        plt.close()


def plotBasilComparisonAcrossDatasets(attacks=['clean', 'gaussian', 'signFlip', 'hidden'],
                                       metric='avg'):
    # one 2x2 figure across datasets showing BASIL accuracy per attack
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

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'BASIL Performance Across Datasets - {metricLabel}',
                 fontsize=16, fontweight='bold')

    for idx, attack in enumerate(attacks):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        for dataset in datasets:
            basilAcc = loadBasilResults(dataset, 'basil', attack, metric=metric)
            if basilAcc is not None:
                rounds = np.arange(len(basilAcc))
                ax.plot(rounds, basilAcc, label=datasetLabels[dataset],
                       linewidth=2.5, marker='o', markersize=4, markevery=5)

        ax.set_xlabel('Training Round', fontsize=11)
        ax.set_ylabel(metricLabel, fontsize=11)
        ax.set_title(attackLabels[attack], fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    savePath = f"plots/images/basil/basil_across_datasets_{metric}.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotBasilImpact(datasets=['mnist', 'cifar10', 'nmnist'],
                     attacks=['gaussian', 'signFlip', 'hidden'],
                     metric='avg'):
    # grouped bar chart of BASIL accuracy gain over clean baseline per attack
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

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(attacks))
    width = 0.25

    for didx, dataset in enumerate(datasets):
        improvements = []

        for attack in attacks:
            cleanAcc = loadBasilResults(dataset, 'clean', attack, metric=metric)
            basilAcc = loadBasilResults(dataset, 'basil', attack, metric=metric)

            if cleanAcc is not None and basilAcc is not None:
                # Calculate final accuracy improvement
                cleanFinal = cleanAcc[-1]
                basilFinal = basilAcc[-1]
                improvement = (basilFinal - cleanFinal) * 100  # Percentage points
                improvements.append(improvement)
            else:
                improvements.append(0)

        offset = (didx - 1) * width
        ax.bar(x + offset, improvements, width, label=datasetLabels[dataset])

    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Improvement (percentage points)', fontsize=12)
    ax.set_title(f'BASIL Impact: Improvement Over Clean Baseline - {metricLabel}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([attackLabels[a] for a in attacks])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    savePath = f"plots/images/basil/basil_impact_{metric}.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def plotSingleDatasetComparison(dataset='mnist', metric='avg'):
    # overlay all attacks (clean and BASIL lines) on one figure for one dataset
    attacks = ['clean', 'gaussian', 'signFlip', 'hidden']
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

    metricLabel = "Average Accuracy" if metric == 'avg' else "Worst-Node Accuracy"

    fig, ax = plt.subplots(figsize=(12, 6))

    for attack in attacks:
        # Plot clean
        cleanAcc = loadBasilResults(dataset, 'clean', attack, metric=metric)
        if cleanAcc is not None:
            rounds = np.arange(len(cleanAcc))
            ax.plot(rounds, cleanAcc,
                   label=f'{attackLabels[attack]} - Clean',
                   linestyle='--', linewidth=2, alpha=0.7)

        # Plot BASIL
        basilAcc = loadBasilResults(dataset, 'basil', attack, metric=metric)
        if basilAcc is not None:
            rounds = np.arange(len(basilAcc))
            ax.plot(rounds, basilAcc,
                   label=f'{attackLabels[attack]} - BASIL',
                   linestyle='-', linewidth=2.5, marker='o', markersize=4, markevery=5)

    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel(metricLabel, fontsize=12)
    ax.set_title(f'Clean vs BASIL on {datasetTitles[dataset]} - {metricLabel}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    savePath = f"plots/images/basil/{dataset}_all_attacks_{metric}.png"
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close()


def generateAllBasilPlots():
    # run all plot functions for both avg and worst metrics
    print("Generating BASIL comparison plots...")

    print("\n1. Clean vs BASIL by dataset (average accuracy)...")
    plotCleanVsBasilByDataset(metric='avg')

    print("\n2. Clean vs BASIL by dataset (worst-node accuracy)...")
    plotCleanVsBasilByDataset(metric='worst')

    print("\n3. BASIL across datasets (average accuracy)...")
    plotBasilComparisonAcrossDatasets(metric='avg')

    print("\n4. BASIL across datasets (worst-node accuracy)...")
    plotBasilComparisonAcrossDatasets(metric='worst')

    print("\n5. BASIL impact comparison (average accuracy)...")
    plotBasilImpact(metric='avg')

    print("\n6. BASIL impact comparison (worst-node accuracy)...")
    plotBasilImpact(metric='worst')

    print("\n7. Single dataset comprehensive plots...")
    for dataset in ['mnist', 'cifar10', 'nmnist']:
        print(f"   - {dataset.upper()} (average)...")
        plotSingleDatasetComparison(dataset, metric='avg')
        print(f"   - {dataset.upper()} (worst)...")
        plotSingleDatasetComparison(dataset, metric='worst')

    print("\nAll BASIL plots generated successfully!")
    print("Plots saved to: plots/images/basil/")


if __name__ == "__main__":
    print("""
==================================================================================
                     BASIL PLOTTING SCRIPT                                  
                                                                            
  Generates plots comparing Clean vs BASIL approach                        
                                                                            
  To customize: Edit the CONFIGURATION section at the top of this file    
==================================================================================
    """)

    metrics = ['avg', 'worst'] if METRIC == 'both' else [METRIC]

    if PLOT_TYPE == 'all':
        generateAllBasilPlots()

    elif PLOT_TYPE == 'bydataset':
        for metric in metrics:
            print(f"\nGenerating clean vs BASIL by dataset ({metric})...")
            plotCleanVsBasilByDataset(metric=metric)

    elif PLOT_TYPE == 'acrossdatasets':
        for metric in metrics:
            print(f"\nGenerating BASIL across datasets ({metric})...")
            plotBasilComparisonAcrossDatasets(metric=metric)

    elif PLOT_TYPE == 'impact':
        for metric in metrics:
            print(f"\nGenerating BASIL impact plot ({metric})...")
            plotBasilImpact(metric=metric)

    elif PLOT_TYPE == 'single':
        for metric in metrics:
            print(f"\nGenerating single dataset plot for {SINGLE_DATASET} ({metric})...")
            plotSingleDatasetComparison(SINGLE_DATASET, metric=metric)

    print("\nDone!")
