"""
Comprehensive testing framework for BASIL + Noisy Channel approaches.
Tests all combinations of:
- Datasets: MNIST, CIFAR-10, N-MNIST
- Modes: clean, basilOnly, noisy, ebm, wcm, basilEbm, basilWcm
- Attacks: clean, gaussian, signFlip, hidden
"""
import os
import sys
import numpy as np
import pickle

# Robust import
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.common import ensureDirs, saveCurve, setupGpu, handleGpuMemoryError
else:
    from .common import ensureDirs, saveCurve, setupGpu, handleGpuMemoryError

from basil_core.data.mnist import loadMnist, makeLoaders as makeMnistLoaders
from basil_core.data.cifar import loadCifar10, makeLoaders as makeCifarLoaders
from basil_core.data.n_mnist import loadNMnist, makeLoaders as makeNMnistLoaders
from basil_core.models import MNISTModel, CIFARModel, NMNISTModel
from basil_core.basil import BasilNode, basilRingTrainingWithAttack
from basil_core.trainer import evaluateAll, getParams

# Mode configurations
MODES = {
    "clean": {
        "noiseModel": "none",
        "sigma": 0.0,
        "useSnapshots": False,
        "description": "Baseline without BASIL or noise"
    },
    "basilOnly": {
        "noiseModel": "none",
        "sigma": 0.0,
        "useSnapshots": True,
        "description": "BASIL snapshot selection only"
    },
    "noisy": {
        "noiseModel": "noisy",
        "sigma": 0.1,
        "useSnapshots": False,
        "description": "Channel noise without regularization"
    },
    "ebm": {
        "noiseModel": "ebm",
        "sigma": 0.1,
        "ebmLambda": 100.0,
        "useSnapshots": False,
        "description": "EBM regularization only"
    },
    "wcm": {
        "noiseModel": "wcm",
        "sigma": 0.1,
        "wcmLambda": 0.1,
        "wcmSamples": 5,
        "wcmRho": 0.5,
        "useSnapshots": False,
        "description": "WCM regularization only"
    },
    "basilEbm": {
        "noiseModel": "ebm",
        "sigma": 0.1,
        "ebmLambda": 100.0,
        "useSnapshots": True,
        "description": "BASIL + EBM combined"
    },
    "basilWcm": {
        "noiseModel": "wcm",
        "sigma": 0.1,
        "wcmLambda": 0.1,
        "wcmSamples": 5,
        "wcmRho": 0.5,
        "useSnapshots": True,
        "description": "BASIL + WCM combined"
    },
}

# Attack configurations
ATTACK_CONFIGS = {
    "clean": {
        "attackerIds": [],
        "attackTypes": ["none"],
        "description": "No attackers"
    },
    "gaussian": {
        "attackerIds": [0, 5],
        "attackTypes": ["gaussian"],
        "description": "2 Gaussian noise attackers"
    },
    "signFlip": {
        "attackerIds": [0, 5],
        "attackTypes": ["signFlip"],
        "description": "2 sign-flip attackers"
    },
    "hidden": {
        "attackerIds": [0, 5],
        "attackTypes": ["hidden"],
        "hiddenStartRound": 10,
        "description": "2 hidden/backdoor attackers (start round 10)"
    },
}

# Dataset configurations
DATASETS = {
    "mnist": {
        "loadFn": loadMnist,
        "loaderFn": makeMnistLoaders,
        "modelClass": MNISTModel,
        "nNodes": 10,
        "rounds": 30,
        "localEpochs": 1,
        "lr": 0.05,
        "batchSize": 32,
    },
    "cifar10": {
        "loadFn": loadCifar10,
        "loaderFn": makeCifarLoaders,
        "modelClass": CIFARModel,
        "nNodes": 10,
        "rounds": 50,
        "localEpochs": 1,
        "lr": 0.01,
        "batchSize": 32,
    },
    "nmnist": {
        "loadFn": loadNMnist,
        "loaderFn": makeNMnistLoaders,
        "modelClass": NMNISTModel,
        "nNodes": 10,
        "rounds": 30,
        "localEpochs": 1,
        "lr": 0.05,
        "batchSize": 32,
    },
}


def runExperiment(datasetName, modeName, attackName, verbose=True):
    """
    Run a single experiment with specified configuration.

    Args:
        datasetName: one of DATASETS keys
        modeName: one of MODES keys
        attackName: one of ATTACK_CONFIGS keys
        verbose: print progress

    Returns:
        (avgAccHist, worstAccHist, finalAvg, finalWorst)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: {datasetName} / {modeName} / {attackName}")
        print(f"{'='*80}")

    try:
        import tensorflow as tf
        # Verify TensorFlow can run basic operations
        _ = tf.constant([1.0])
        if verbose:
            print(f"[TensorFlow] Version {tf.__version__} initialized successfully")
    except Exception as e:
        print(f"[ERROR] TensorFlow initialization failed: {e}")
        raise

    # Get configurations
    dsConfig = DATASETS[datasetName]
    modeConfig = MODES[modeName]
    attackConfig = ATTACK_CONFIGS[attackName]

    # Load data
    train, test = dsConfig["loadFn"]()
    trainLoaders, testLoader = dsConfig["loaderFn"](
        train, test,
        batchSize=dsConfig["batchSize"],
        iid=True,
        nClients=dsConfig["nNodes"]
    )

    # Create nodes
    nodes = []
    for i in range(dsConfig["nNodes"]):
        model = dsConfig["modelClass"]()
        nodeConfig = {
            "nodeId": i,
            "model": model,
            "dataLoader": trainLoaders[i],
            "S": 10,  # BASIL memory size
            "noiseModel": modeConfig.get("noiseModel", "none"),
            "sigma": modeConfig.get("sigma", 0.0),
            "lr0": dsConfig["lr"],
            "localEpochs": dsConfig["localEpochs"],
        }

        # Add mode-specific parameters
        if "ebmLambda" in modeConfig:
            nodeConfig["ebmLambda"] = modeConfig["ebmLambda"]
        if "wcmLambda" in modeConfig:
            nodeConfig["wcmLambda"] = modeConfig["wcmLambda"]
        if "wcmSamples" in modeConfig:
            nodeConfig["wcmSamples"] = modeConfig["wcmSamples"]
        if "wcmRho" in modeConfig:
            nodeConfig["wcmRho"] = modeConfig["wcmRho"]

        nodes.append(BasilNode(**nodeConfig))

    # Run training
    trainConfig = {
        "nodes": nodes,
        "rounds": dsConfig["rounds"],
        "testLoader": testLoader,
        "attackTypes": attackConfig["attackTypes"],
        "attackerIds": attackConfig["attackerIds"],
        "hiddenStartRound": attackConfig.get("hiddenStartRound", 999),
        "sigma": modeConfig.get("sigma", 0.0),
        "noiseModel": modeConfig.get("noiseModel", "none"),
        "lr0": dsConfig["lr"],
        "lrAlpha": 0.6,
        "stepsPerEpoch": 100,
        "useSnapshots": modeConfig.get("useSnapshots", False),
    }

    avgAccHist, worstAccHist = basilRingTrainingWithAttack(**trainConfig)

    # Final evaluation
    finalAvg, finalWorst, _ = evaluateAll(nodes, testLoader)

    if verbose:
        print(f"\nFinal Results:")
        print(f"  Average Accuracy: {finalAvg:.4f}")
        print(f"  Worst Accuracy: {finalWorst:.4f}")

    return avgAccHist, worstAccHist, finalAvg, finalWorst


def runAllExperiments(datasetsToRun=None, modesToRun=None, attacksToRun=None):
    """
    Run comprehensive test suite.

    Args:
        datasetsToRun: list of dataset names (None = all)
        modesToRun: list of mode names (None = all)
        attacksToRun: list of attack names (None = all)
    """
    setupGpu()
    ensureDirs()

    # Default to all if not specified
    if datasetsToRun is None:
        datasetsToRun = list(DATASETS.keys())
    if modesToRun is None:
        modesToRun = list(MODES.keys())
    if attacksToRun is None:
        attacksToRun = list(ATTACK_CONFIGS.keys())

    results = {}
    totalExperiments = len(datasetsToRun) * len(modesToRun) * len(attacksToRun)
    currentExperiment = 0

    for dataset in datasetsToRun:
        if dataset not in results:
            results[dataset] = {}

        for mode in modesToRun:
            if mode not in results[dataset]:
                results[dataset][mode] = {}

            for attack in attacksToRun:
                currentExperiment += 1
                print(f"\n\nProgress: {currentExperiment}/{totalExperiments}")

                try:
                    avgHist, worstHist, finalAvg, finalWorst = runExperiment(
                        dataset, mode, attack, verbose=True
                    )

                    # Save results
                    resultKey = f"{mode}_{attack}"
                    results[dataset][mode][attack] = {
                        "avgAccHist": avgHist,
                        "worstAccHist": worstHist,
                        "finalAvg": finalAvg,
                        "finalWorst": finalWorst,
                    }

                    # Save accuracy curves
                    resultDir = f"experiments/results/{dataset}"
                    os.makedirs(resultDir, exist_ok=True)

                    avgPath = f"{resultDir}/acc_{mode}_{attack}_avg.npy"
                    worstPath = f"{resultDir}/acc_{mode}_{attack}_worst.npy"

                    np.save(avgPath, np.array(avgHist))
                    np.save(worstPath, np.array(worstHist))

                    print(f"Saved: {avgPath}")
                    print(f"Saved: {worstPath}")

                except Exception as e:
                    print(f"ERROR in {dataset}/{mode}/{attack}: {e}")

                    # Check if this is a GPU memory error and provide helpful guidance
                    if not handleGpuMemoryError(e):
                        # If not a GPU error, print full traceback
                        import traceback
                        traceback.print_exc()

                    results[dataset][mode][attack] = {"error": str(e)}

    # Save summary
    summaryPath = "experiments/results/comprehensive_test_summary.pkl"
    with open(summaryPath, "wb") as f:
        pickle.dump(results, f)
    print(f"\n\nSaved summary: {summaryPath}")

    # Print summary table
    print("\n\n" + "="*80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*80)
    for dataset in datasetsToRun:
        print(f"\n{dataset.upper()}:")
        print("-" * 80)
        print(f"{'Mode':<15} {'Attack':<15} {'Avg Acc':<12} {'Worst Acc':<12} {'Status'}")
        print("-" * 80)
        for mode in modesToRun:
            for attack in attacksToRun:
                if attack in results[dataset][mode]:
                    res = results[dataset][mode][attack]
                    if "error" in res:
                        print(f"{mode:<15} {attack:<15} {'N/A':<12} {'N/A':<12} ERROR")
                    else:
                        print(f"{mode:<15} {attack:<15} {res['finalAvg']:<12.4f} {res['finalWorst']:<12.4f} OK")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive BASIL + Noisy Channel tests")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                        help="Datasets to test (default: all)")
    parser.add_argument("--modes", nargs="+", choices=list(MODES.keys()),
                        help="Modes to test (default: all)")
    parser.add_argument("--attacks", nargs="+", choices=list(ATTACK_CONFIGS.keys()),
                        help="Attacks to test (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: only MNIST with subset of configs")

    args = parser.parse_args()

    if args.quick:
        # Quick test: only MNIST with basic configs
        print("Running QUICK TEST mode (MNIST only, subset of configs)")
        results = runAllExperiments(
            datasetsToRun=["mnist"],
            modesToRun=["clean", "basilOnly", "ebm", "basilEbm"],
            attacksToRun=["clean", "gaussian"]
        )
    else:
        results = runAllExperiments(
            datasetsToRun=args.datasets,
            modesToRun=args.modes,
            attacksToRun=args.attacks
        )

    print("\n\nAll experiments complete!")
