"""
Test BASIL approach by itself.
Tests BASIL ring topology with snapshot selection under different attack scenarios.
Compares clean (no BASIL) vs BASIL approach.

CONFIGURATION: Edit the variables below to customize your test
"""
import os
import sys
import numpy as np

# ============================================================================
# CONFIGURATION - EDIT THESE TO CUSTOMIZE YOUR TEST
# ============================================================================

# Which datasets to test? Options: "mnist", "cifar10", "nmnist"
DATASETS_TO_TEST = ["mnist"]  # Change to ["mnist", "cifar10", "nmnist"] for all

# Which attacks to test? Options: "clean", "gaussian", "signFlip", "hidden"
ATTACKS_TO_TEST = ["clean", "gaussian"]  # Add more attacks as needed

# Which modes to test? Options: "clean", "basil", "both"
# - clean: baseline without BASIL snapshot selection
# - basil: with BASIL snapshot selection
# - both: test both approaches
MODE_TO_TEST = "both"  # Change to "clean" or "basil" to test only one

# Quick test mode (reduces rounds for faster testing)
QUICK_TEST = True  # Set to False for full training

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Robust import
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.common import ensureDirs, setupGpu, handleGpuMemoryError
else:
    from .common import ensureDirs, setupGpu, handleGpuMemoryError

from basil_core.data.mnist import loadMnist, makeLoaders as makeMnistLoaders
from basil_core.data.cifar import loadCifar10, makeLoaders as makeCifarLoaders
from basil_core.data.nMnist import loadNMnist, makeLoaders as makeNMnistLoaders
from basil_core.models import MNISTModel, CIFARModel, NMNISTModel
from basil_core.basil import BasilNode, basilRingTrainingWithAttack
from basil_core.trainer import evaluateAll


# Dataset configurations
DATASETS = {
    "mnist": {
        "loadFn": loadMnist,
        "loaderFn": makeMnistLoaders,
        "modelClass": MNISTModel,
        "nNodes": 10,
        "rounds": 15 if QUICK_TEST else 30,
        "localEpochs": 1,
        "lr": 0.05,
        "batchSize": 32,
    },
    "cifar10": {
        "loadFn": loadCifar10,
        "loaderFn": makeCifarLoaders,
        "modelClass": CIFARModel,
        "nNodes": 10,
        "rounds": 25 if QUICK_TEST else 50,
        "localEpochs": 1,
        "lr": 0.01,
        "batchSize": 32,
    },
    "nmnist": {
        "loadFn": loadNMnist,
        "loaderFn": makeNMnistLoaders,
        "modelClass": NMNISTModel,
        "nNodes": 10,
        "rounds": 15 if QUICK_TEST else 30,
        "localEpochs": 1,
        "lr": 0.05,
        "batchSize": 32,
    },
}


# Attack configurations
ATTACKS = {
    "clean": {
        "attackerIds": [],
        "attackTypes": ["none"],
        "description": "No attackers"
    },
    "gaussian": {
        "attackerIds": [0, 5],
        "attackTypes": ["gaussian"],
        "description": "2 Gaussian noise attackers (nodes 0, 5)"
    },
    "signFlip": {
        "attackerIds": [0, 5],
        "attackTypes": ["signFlip"],
        "description": "2 sign-flip attackers (nodes 0, 5)"
    },
    "hidden": {
        "attackerIds": [0, 5],
        "attackTypes": ["hidden"],
        "hiddenStartRound": 10,
        "description": "2 hidden/backdoor attackers (nodes 0, 5, start round 10)"
    },
}


def runBasilExperiment(dataset, attack, useBasil=True, verbose=True):
    """
    Run BASIL experiment on specified dataset and attack.

    Args:
        dataset: 'mnist', 'cifar10', or 'nmnist'
        attack: 'clean', 'gaussian', 'signFlip', or 'hidden'
        useBasil: True to use BASIL snapshot selection, False for baseline
        verbose: print progress

    Returns:
        (avgAccHist, worstAccHist, finalAvg, finalWorst)
    """
    if verbose:
        mode = "BASIL" if useBasil else "Clean (no BASIL)"
        print(f"\n{'='*80}")
        print(f"Running: {dataset.upper()} - {mode} - {ATTACKS[attack]['description']}")
        print(f"{'='*80}")

    # Get configurations
    dsConfig = DATASETS[dataset]
    attackConfig = ATTACKS[attack]

    # Load data
    if verbose:
        print("Loading data...")
    train, test = dsConfig["loadFn"]()
    trainLoaders, testLoader = dsConfig["loaderFn"](
        train, test,
        batchSize=dsConfig["batchSize"],
        iid=True,
        nClients=dsConfig["nNodes"]
    )

    # Create nodes
    if verbose:
        print(f"Creating {dsConfig['nNodes']} nodes...")
    nodes = []
    for i in range(dsConfig["nNodes"]):
        model = dsConfig["modelClass"]()
        nodes.append(BasilNode(
            nodeId=i,
            model=model,
            dataLoader=trainLoaders[i],
            S=10,  # BASIL memory size (only used if useBasil=True)
            noiseModel="none",
            sigma=0.0,
            lr0=dsConfig["lr"],
            localEpochs=dsConfig["localEpochs"],
        ))

    # Run training
    if verbose:
        print(f"Training for {dsConfig['rounds']} rounds...")
    avgAccHist, worstAccHist = basilRingTrainingWithAttack(
        nodes=nodes,
        rounds=dsConfig["rounds"],
        testLoader=testLoader,
        attackTypes=attackConfig["attackTypes"],
        attackerIds=attackConfig["attackerIds"],
        hiddenStartRound=attackConfig.get("hiddenStartRound", 999),
        sigma=0.0,
        noiseModel="none",
        lr0=dsConfig["lr"],
        lrAlpha=0.6,
        stepsPerEpoch=100,
        useSnapshots=useBasil,  # Key parameter: enable/disable BASIL
    )

    # Final evaluation
    finalAvg, finalWorst, _ = evaluateAll(nodes, testLoader)

    if verbose:
        print(f"\nFinal Results:")
        print(f"  Average Accuracy: {finalAvg:.4f}")
        print(f"  Worst Accuracy: {finalWorst:.4f}")

    return avgAccHist, worstAccHist, finalAvg, finalWorst


def runBasilTests(datasets=None, attacks=None, testClean=True, testBasil=True):
    """
    Run BASIL tests.

    Args:
        datasets: list of datasets to test (None = all)
        attacks: list of attacks to test (None = all)
        testClean: whether to test clean (no BASIL) baseline
        testBasil: whether to test BASIL approach
    """
    setupGpu()
    ensureDirs()

    # Default to all if not specified
    if datasets is None:
        datasets = list(DATASETS.keys())
    if attacks is None:
        attacks = list(ATTACKS.keys())

    if not testClean and not testBasil:
        print("Error: Must test at least one of clean or basil")
        return

    results = {}

    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}

        for attack in attacks:
            if attack not in results[dataset]:
                results[dataset][attack] = {}

            # Test clean baseline (no BASIL)
            if testClean:
                print(f"\n\n{'#'*80}")
                print(f"# Testing CLEAN (no BASIL): {dataset.upper()} / {attack}")
                print(f"{'#'*80}")

                try:
                    avgHist, worstHist, finalAvg, finalWorst = runBasilExperiment(
                        dataset, attack, useBasil=False, verbose=True
                    )

                    results[dataset][attack]['clean'] = {
                        'avgAccHist': avgHist,
                        'worstAccHist': worstHist,
                        'finalAvg': finalAvg,
                        'finalWorst': finalWorst,
                    }

                    # Save results
                    resultDir = f"experiments/results/basil/{dataset}"
                    os.makedirs(resultDir, exist_ok=True)

                    avgPath = f"{resultDir}/acc_clean_{attack}_avg.npy"
                    worstPath = f"{resultDir}/acc_clean_{attack}_worst.npy"

                    np.save(avgPath, np.array(avgHist))
                    np.save(worstPath, np.array(worstHist))

                    print(f"Saved: {avgPath}")
                    print(f"Saved: {worstPath}")

                except Exception as e:
                    print(f"ERROR in clean/{dataset}/{attack}: {e}")
                    if not handleGpuMemoryError(e):
                        import traceback
                        traceback.print_exc()
                    results[dataset][attack]['clean'] = {"error": str(e)}

            # Test BASIL
            if testBasil:
                print(f"\n\n{'#'*80}")
                print(f"# Testing BASIL: {dataset.upper()} / {attack}")
                print(f"{'#'*80}")

                try:
                    avgHist, worstHist, finalAvg, finalWorst = runBasilExperiment(
                        dataset, attack, useBasil=True, verbose=True
                    )

                    results[dataset][attack]['basil'] = {
                        'avgAccHist': avgHist,
                        'worstAccHist': worstHist,
                        'finalAvg': finalAvg,
                        'finalWorst': finalWorst,
                    }

                    # Save results
                    resultDir = f"experiments/results/basil/{dataset}"
                    os.makedirs(resultDir, exist_ok=True)

                    avgPath = f"{resultDir}/acc_basil_{attack}_avg.npy"
                    worstPath = f"{resultDir}/acc_basil_{attack}_worst.npy"

                    np.save(avgPath, np.array(avgHist))
                    np.save(worstPath, np.array(worstHist))

                    print(f"Saved: {avgPath}")
                    print(f"Saved: {worstPath}")

                except Exception as e:
                    print(f"ERROR in basil/{dataset}/{attack}: {e}")
                    if not handleGpuMemoryError(e):
                        import traceback
                        traceback.print_exc()
                    results[dataset][attack]['basil'] = {"error": str(e)}

    # Print summary table
    print("\n\n" + "="*80)
    print("SUMMARY OF BASIL TESTS")
    print("="*80)

    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        print("-" * 80)
        print(f"{'Mode':<15} {'Attack':<15} {'Avg Acc':<12} {'Worst Acc':<12} {'Status'}")
        print("-" * 80)

        for attack in attacks:
            if testClean and 'clean' in results[dataset][attack]:
                res = results[dataset][attack]['clean']
                if "error" in res:
                    print(f"{'Clean':<15} {attack:<15} {'N/A':<12} {'N/A':<12} ERROR")
                else:
                    print(f"{'Clean':<15} {attack:<15} {res['finalAvg']:<12.4f} {res['finalWorst']:<12.4f} OK")

            if testBasil and 'basil' in results[dataset][attack]:
                res = results[dataset][attack]['basil']
                if "error" in res:
                    print(f"{'BASIL':<15} {attack:<15} {'N/A':<12} {'N/A':<12} ERROR")
                else:
                    print(f"{'BASIL':<15} {attack:<15} {res['finalAvg']:<12.4f} {res['finalWorst']:<12.4f} OK")

    print(f"\n\nResults saved to: experiments/results/basil/")
    print(f"Generate plots with: python plots/plotBasil.py")

    return results


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                       BASIL TESTING SCRIPT                                 ║
║                                                                            ║
║  Tests: Clean (no BASIL) vs BASIL approach                               ║
║                                                                            ║
║  To customize: Edit the CONFIGURATION section at the top of this file    ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Determine what to test based on configuration
    testClean = MODE_TO_TEST in ["clean", "both"]
    testBasil = MODE_TO_TEST in ["basil", "both"]

    print(f"Configuration:")
    print(f"  Datasets: {DATASETS_TO_TEST}")
    print(f"  Attacks: {ATTACKS_TO_TEST}")
    print(f"  Mode: {MODE_TO_TEST}")
    print(f"  Quick Test: {QUICK_TEST}")
    print()

    results = runBasilTests(
        datasets=DATASETS_TO_TEST,
        attacks=ATTACKS_TO_TEST,
        testClean=testClean,
        testBasil=testBasil
    )

    print("\n\nAll BASIL tests complete!")
    print("Next step: Generate plots with 'python plots/plotBasil.py'")
