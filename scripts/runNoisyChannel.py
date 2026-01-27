"""
Test Noisy Channel approaches by themselves.
Tests clean baseline, EBM (Expectation-Based Model), and WCM (Worst-Case Model).

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

# Which modes to test? Options: "clean", "ebm", "wcm"
# - clean: baseline with noisy channel (no mitigation)
# - ebm: Expectation-Based Model (regularization approach)
# - wcm: Worst-Case Model (robust to worst-case noise)
MODES_TO_TEST = ["clean", "ebm", "wcm"]  # Test all three

# Noise parameters
CHANNEL_NOISE_SIGMA = 0.1  # Standard deviation of channel noise
EBM_LAMBDA = 0.01          # EBM regularization strength
WCM_LAMBDA = 0.1           # WCM regularization strength
WCM_SAMPLES = 5            # Number of boundary samples for WCM
WCM_RHO = 0.5              # WCM SCA parameter

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
        "description": "No attackers (only channel noise)"
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


def runNoisyChannelExperiment(dataset, attack, mode, verbose=True):
    """
    Run noisy channel experiment.

    Args:
        dataset: 'mnist', 'cifar10', or 'nmnist'
        attack: 'clean', 'gaussian', 'signFlip', or 'hidden'
        mode: 'clean' (no mitigation), 'ebm', or 'wcm'
        verbose: print progress

    Returns:
        (avgAccHist, worstAccHist, finalAvg, finalWorst)
    """
    modeLabels = {
        'clean': 'Clean (no mitigation)',
        'ebm': 'EBM (Expectation-Based Model)',
        'wcm': 'WCM (Worst-Case Model)'
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: {dataset.upper()} - {modeLabels[mode]} - {ATTACKS[attack]['description']}")
        print(f"Channel Noise Sigma: {CHANNEL_NOISE_SIGMA}")
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

    # Determine noise model
    if mode == 'clean':
        noiseModel = 'noisy'  # Channel noise without mitigation
    elif mode == 'ebm':
        noiseModel = 'ebm'
    elif mode == 'wcm':
        noiseModel = 'wcm'

    # Create nodes
    if verbose:
        print(f"Creating {dsConfig['nNodes']} nodes...")
    nodes = []
    for i in range(dsConfig["nNodes"]):
        model = dsConfig["modelClass"]()
        nodeConfig = {
            "nodeId": i,
            "model": model,
            "dataLoader": trainLoaders[i],
            "S": 2,  # Small S since we're not using BASIL snapshot selection
            "noiseModel": noiseModel,
            "sigma": CHANNEL_NOISE_SIGMA,
            "lr0": dsConfig["lr"],
            "localEpochs": dsConfig["localEpochs"],
        }

        # Add mode-specific parameters
        if mode == 'ebm':
            nodeConfig["ebmLambda"] = EBM_LAMBDA
        elif mode == 'wcm':
            nodeConfig["wcmLambda"] = WCM_LAMBDA
            nodeConfig["wcmSamples"] = WCM_SAMPLES
            nodeConfig["wcmRho"] = WCM_RHO

        nodes.append(BasilNode(**nodeConfig))

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
        sigma=CHANNEL_NOISE_SIGMA,
        noiseModel=noiseModel,
        lr0=dsConfig["lr"],
        lrAlpha=0.6,
        stepsPerEpoch=100,
        useSnapshots=False,  # Don't use BASIL snapshots for noisy channel tests
    )

    # Final evaluation
    finalAvg, finalWorst, _ = evaluateAll(nodes, testLoader)

    if verbose:
        print(f"\nFinal Results:")
        print(f"  Average Accuracy: {finalAvg:.4f}")
        print(f"  Worst Accuracy: {finalWorst:.4f}")

    return avgAccHist, worstAccHist, finalAvg, finalWorst


def runNoisyChannelTests():
    """Run noisy channel tests based on configuration."""
    setupGpu()
    ensureDirs()

    print("\n" + "="*80)
    print("NOISY CHANNEL TESTING")
    print("="*80)
    print(f"Datasets: {DATASETS_TO_TEST}")
    print(f"Attacks: {ATTACKS_TO_TEST}")
    print(f"Modes: {MODES_TO_TEST}")
    print(f"Channel Noise Sigma: {CHANNEL_NOISE_SIGMA}")
    print(f"Quick Test: {QUICK_TEST}")
    print("="*80)

    results = {}

    for dataset in DATASETS_TO_TEST:
        if dataset not in results:
            results[dataset] = {}

        for attack in ATTACKS_TO_TEST:
            if attack not in results[dataset]:
                results[dataset][attack] = {}

            for mode in MODES_TO_TEST:
                print(f"\n\n{'#'*80}")
                print(f"# Testing: {dataset.upper()} / {mode.upper()} / {attack}")
                print(f"{'#'*80}")

                try:
                    avgHist, worstHist, finalAvg, finalWorst = runNoisyChannelExperiment(
                        dataset, attack, mode, verbose=True
                    )

                    results[dataset][attack][mode] = {
                        'avgAccHist': avgHist,
                        'worstAccHist': worstHist,
                        'finalAvg': finalAvg,
                        'finalWorst': finalWorst,
                    }

                    # Save results
                    resultDir = f"experiments/results/noisyChannel/{dataset}"
                    os.makedirs(resultDir, exist_ok=True)

                    avgPath = f"{resultDir}/acc_{mode}_{attack}_avg.npy"
                    worstPath = f"{resultDir}/acc_{mode}_{attack}_worst.npy"

                    np.save(avgPath, np.array(avgHist))
                    np.save(worstPath, np.array(worstHist))

                    print(f"Saved: {avgPath}")
                    print(f"Saved: {worstPath}")

                except Exception as e:
                    print(f"ERROR in {mode}/{dataset}/{attack}: {e}")
                    if not handleGpuMemoryError(e):
                        import traceback
                        traceback.print_exc()
                    results[dataset][attack][mode] = {"error": str(e)}

    # Print summary table
    print("\n\n" + "="*80)
    print("SUMMARY OF NOISY CHANNEL TESTS")
    print("="*80)

    for dataset in DATASETS_TO_TEST:
        print(f"\n{dataset.upper()}:")
        print("-" * 80)
        print(f"{'Mode':<15} {'Attack':<15} {'Avg Acc':<12} {'Worst Acc':<12} {'Status'}")
        print("-" * 80)

        for attack in ATTACKS_TO_TEST:
            for mode in MODES_TO_TEST:
                if mode in results[dataset][attack]:
                    res = results[dataset][attack][mode]
                    if "error" in res:
                        print(f"{mode:<15} {attack:<15} {'N/A':<12} {'N/A':<12} ERROR")
                    else:
                        print(f"{mode:<15} {attack:<15} {res['finalAvg']:<12.4f} {res['finalWorst']:<12.4f} OK")

    print(f"\n\nResults saved to: experiments/results/noisyChannel/")
    print(f"Generate plots with: python plots/plotNoisyChannel.py")

    return results


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    NOISY CHANNEL TESTING SCRIPT                            ║
║                                                                            ║
║  Tests: Clean (no mitigation), EBM, WCM approaches                       ║
║                                                                            ║
║  To customize: Edit the CONFIGURATION section at the top of this file    ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    results = runNoisyChannelTests()

    print("\n\nAll noisy channel tests complete!")
    print("Next step: Generate plots with 'python plots/plotNoisyChannel.py'")
