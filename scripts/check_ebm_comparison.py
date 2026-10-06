#!/usr/bin/env python3
"""
Test script to compare EBM vs Noisy Baseline
Tests with sigma=0.2 and different lambda values
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.basil import BasilNode, fedAvgTrainingWithNoise
from scripts.common import setupGpu

# Configuration
SIGMA = 0.2
N_NODES = 10
N_ROUNDS = 30
LR = 0.03
BATCH_SIZE = 32

def runTest(noiseModel, ebmLambda, label):
    """Run a single test configuration."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  noiseModel={noiseModel}, sigma={SIGMA}, ebmLambda={ebmLambda}")
    if noiseModel == "ebm":
        scale = 1.0 + ebmLambda * SIGMA * SIGMA
        print(f"  EBM scale factor = {scale:.2f}")
    print(f"{'='*60}\n")

    # Load data
    train, test = loadMnist()
    trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

    # Create nodes
    nodes = []
    for i in range(N_NODES):
        model = MNISTModel()
        node = BasilNode(
            nodeId=i,
            model=model,
            dataLoader=trainLoaders[i],
            S=10,
            noiseModel=noiseModel,
            sigma=SIGMA,
            lr0=LR,
            localEpochs=1,
            ebmLambda=ebmLambda,
        )
        nodes.append(node)

    # Run training
    avgAccHist, worstAccHist = fedAvgTrainingWithNoise(
        nodes=nodes,
        rounds=N_ROUNDS,
        testLoader=testLoader,
        attackTypes=["none"],
        attackerIds=None,
        sigma=SIGMA,
        noiseModel=noiseModel,
        channelNoiseStart=0,
        lr0=LR,
        stepsPerEpoch=100,
    )

    finalAcc = avgAccHist[-1] if avgAccHist else 0.0
    print(f"\n>>> FINAL ACCURACY: {finalAcc:.4f} ({finalAcc*100:.2f}%)")

    return avgAccHist, worstAccHist, finalAcc


def main():
    setupGpu()

    results = {}

    # Test 1: Noisy baseline (no mitigation)
    acc1, _, final1 = runTest(
        noiseModel="noisy",
        ebmLambda=0,  # Not used for noisy
        label="NOISY BASELINE (no mitigation)"
    )
    results["noisy"] = final1

    # Test 2: EBM with lambda=25 (scale=2.0)
    acc2, _, final2 = runTest(
        noiseModel="ebm",
        ebmLambda=25,
        label="EBM (lambda=25, scale=2.0)"
    )
    results["ebm_25"] = final2

    # Test 3: EBM with lambda=50 (scale=3.0) — higher than recommended for comparison
    acc3, _, final3 = runTest(
        noiseModel="ebm",
        ebmLambda=50,
        label="EBM (lambda=50, scale=3.0)"
    )
    results["ebm_50"] = final3

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (sigma=0.2)")
    print("="*60)
    print(f"Noisy baseline:     {results['noisy']*100:.2f}%")
    print(f"EBM (λ=25, s=2.0):  {results['ebm_25']*100:.2f}%  (Δ = {(results['ebm_25']-results['noisy'])*100:+.2f}%)")
    print(f"EBM (λ=50, s=3.0):  {results['ebm_50']*100:.2f}%  (Δ = {(results['ebm_50']-results['noisy'])*100:+.2f}%)")
    print("="*60)

    # Save results
    os.makedirs("experiments/results/ebm_test", exist_ok=True)
    np.save("experiments/results/ebm_test/noisy_baseline.npy", acc1)
    np.save("experiments/results/ebm_test/ebm_lambda25.npy", acc2)
    np.save("experiments/results/ebm_test/ebm_lambda50.npy", acc3)
    print("\nResults saved to experiments/results/ebm_test/")


if __name__ == "__main__":
    main()
