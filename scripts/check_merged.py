#!/usr/bin/env python3
"""
Merged approach: BASIL + Noisy Channel (EBM)
Tests defense against BOTH Byzantine attacks AND channel noise.

4 Tests:
1. Clean (no attacks, no noise)
2. Attacks only (Byzantine, no channel noise)
3. Noise only (channel noise, no attacks)
4. Attacks + Noise + BASIL + EBM (full merged defense)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.basil import BasilNode, basilRingTrainingWithAttack
from scripts.common import setupGpu

# Configuration
N_NODES = 10
N_ROUNDS = 30
LR = 0.03
BATCH_SIZE = 32
MOMENTUM = 0.9

# BASIL config
S_MEMORY = 3  # S = b + 1

# Noise config
SIGMA = 0.2
EBM_LAMBDA = 25

# Attack config
ATTACK_TYPE = "gaussian"
ATTACKER_IDS = [0, 5]


def runTest(testName, useAttacks, useNoise, useBasil, useEbm, trainLoaders, testLoader):
    # configure nodes with the given flags, run training, return histories and stats
    print(f"\n{'='*70}")
    print(f"TEST: {testName}")
    print(f"  Attacks: {ATTACK_TYPE if useAttacks else 'None'}, Attackers: {ATTACKER_IDS if useAttacks else 'None'}")
    print(f"  Channel Noise: {'σ=' + str(SIGMA) if useNoise else 'None'}")
    print(f"  BASIL Snapshot: {'Yes (S=' + str(S_MEMORY) + ')' if useBasil else 'No'}")
    print(f"  EBM: {'Yes (λ=' + str(EBM_LAMBDA) + ')' if useEbm else 'No'}")
    print(f"  Momentum: {MOMENTUM}, LR: {LR} (decay)")
    print(f"{'='*70}\n")

    # Determine noise model
    if useEbm and useNoise:
        noiseModel = "ebm"
    elif useNoise:
        noiseModel = "noisy"
    else:
        noiseModel = "none"

    # Create nodes
    nodes = []
    for i in range(N_NODES):
        model = MNISTModel()
        node = BasilNode(
            nodeId=i,
            model=model,
            dataLoader=trainLoaders[i],
            S=S_MEMORY,
            noiseModel=noiseModel,
            sigma=SIGMA if useNoise else 0.0,
            lr0=LR,
            localEpochs=1,
            ebmLambda=EBM_LAMBDA,
            momentum=MOMENTUM,
        )
        nodes.append(node)

    # Run training
    attackTypes = [ATTACK_TYPE] if useAttacks else ["none"]
    attackerIds = ATTACKER_IDS if useAttacks else []

    avgAccHist, worstAccHist = basilRingTrainingWithAttack(
        nodes=nodes,
        rounds=N_ROUNDS,
        testLoader=testLoader,
        attackTypes=attackTypes,
        attackerIds=attackerIds,
        sigma=SIGMA if useNoise else 0.0,
        noiseModel=noiseModel,
        channelNoiseStart=0,
        lr0=LR,
        stepsPerEpoch=100,
        useSnapshots=useBasil,
        useSequential=True,
    )

    bestAvg = max(avgAccHist) if avgAccHist else 0
    finalAvg = avgAccHist[-1] if avgAccHist else 0
    bestWorst = max(worstAccHist) if worstAccHist else 0
    finalWorst = worstAccHist[-1] if worstAccHist else 0

    print(f"\n>>> AVG  - Final: {finalAvg:.4f}, Best: {bestAvg:.4f}")
    print(f">>> WORST - Final: {finalWorst:.4f}, Best: {bestWorst:.4f}")

    return avgAccHist, worstAccHist, bestAvg, finalAvg


def main():
    setupGpu()

    print("Loading MNIST...")
    train, test = loadMnist()
    trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

    results = {}

    # Test 1: Clean baseline
    _, _, best1, final1 = runTest(
        "1. CLEAN (no attacks, no noise)",
        useAttacks=False, useNoise=False, useBasil=False, useEbm=False,
        trainLoaders=trainLoaders, testLoader=testLoader
    )
    results["Clean"] = (best1, final1)

    # Test 2: Attacks + Noise (NO defense)
    _, _, best2, final2 = runTest(
        "2. ATTACKS + NOISE (no defense)",
        useAttacks=True, useNoise=True, useBasil=False, useEbm=False,
        trainLoaders=trainLoaders, testLoader=testLoader
    )
    results["Attacks+Noise (no defense)"] = (best2, final2)

    # Test 3: Attacks + Noise + BASIL only
    _, _, best3, final3 = runTest(
        "3. ATTACKS + NOISE + BASIL (no EBM)",
        useAttacks=True, useNoise=True, useBasil=True, useEbm=False,
        trainLoaders=trainLoaders, testLoader=testLoader
    )
    results["Attacks+Noise+BASIL"] = (best3, final3)

    # Test 4: Attacks + Noise + BASIL + EBM (full merged)
    _, _, best4, final4 = runTest(
        "4. ATTACKS + NOISE + BASIL + EBM (full merged)",
        useAttacks=True, useNoise=True, useBasil=True, useEbm=True,
        trainLoaders=trainLoaders, testLoader=testLoader
    )
    results["MERGED (BASIL+EBM)"] = (best4, final4)

    # Summary
    print("\n" + "="*70)
    print("MERGED APPROACH RESULTS SUMMARY")
    print(f"Attack: {ATTACK_TYPE}, Attackers: {ATTACKER_IDS}")
    print(f"Noise: σ={SIGMA}, EBM λ={EBM_LAMBDA}")
    print(f"BASIL: S={S_MEMORY}, Momentum={MOMENTUM}")
    print("="*70)
    print(f"{'Test':<35} {'Best Avg':>12} {'Final Avg':>12}")
    print("-"*70)

    for name, (best, final) in results.items():
        print(f"{name:<35} {best*100:>11.2f}% {final*100:>11.2f}%")

    print("-"*70)

    # Improvements
    baseline = results["Attacks+Noise (no defense)"][1]
    basil_only = results["Attacks+Noise+BASIL"][1]
    merged = results["MERGED (BASIL+EBM)"][1]

    print(f"\nImprovements over no-defense baseline ({baseline*100:.1f}%):")
    print(f"  BASIL only:      {(basil_only - baseline)*100:+.2f}%")
    print(f"  MERGED (BASIL+EBM): {(merged - baseline)*100:+.2f}%")
    print("="*70)

    # Save results
    os.makedirs("experiments/results/merged_test", exist_ok=True)
    for name, (best, final) in results.items():
        safeName = name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
        np.save(f"experiments/results/merged_test/{safeName}.npy", [best, final])
    print("\nResults saved to experiments/results/merged_test/")


if __name__ == "__main__":
    main()
