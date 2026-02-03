#!/usr/bin/env python3
"""
BASIL Paper comparison:
1. Clean (no attacks, no BASIL)
2. Attacks without BASIL (no snapshot selection)
3. Attacks with BASIL (snapshot selection)

Uses ring topology as per BASIL paper (001).
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
S_MEMORY = 10  # BASIL memory size (paper: S = b+1 where b = max Byzantine nodes)

# Attack configuration
ATTACK_TYPE = "gaussian"  # Options: "gaussian", "signFlip", "hidden"
ATTACKER_IDS = [0, 5]  # 2 out of 10 nodes are attackers (20%)


def runTest(testName, useAttacks, useBasil, trainLoaders, testLoader):
    """Run a single BASIL test configuration."""
    print(f"\n{'='*60}")
    print(f"TEST: {testName}")
    print(f"  Attacks: {ATTACK_TYPE if useAttacks else 'None'}")
    print(f"  Attackers: {ATTACKER_IDS if useAttacks else 'None'}")
    print(f"  BASIL Snapshot Selection: {'Yes' if useBasil else 'No'}")
    print(f"  Ring Topology, S={S_MEMORY}, lr={LR} (decay)")
    print(f"{'='*60}\n")

    # Create nodes
    nodes = []
    for i in range(N_NODES):
        model = MNISTModel()
        node = BasilNode(
            nodeId=i,
            model=model,
            dataLoader=trainLoaders[i],
            S=S_MEMORY,
            noiseModel="none",
            sigma=0.0,
            lr0=LR,
            localEpochs=1,
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
        sigma=0.0,
        noiseModel="none",
        lr0=LR,
        stepsPerEpoch=100,
        useSnapshots=useBasil,  # Key parameter!
        useSequential=True,  # Paper's Algorithm 1
    )

    bestAvg = max(avgAccHist) if avgAccHist else 0
    finalAvg = avgAccHist[-1] if avgAccHist else 0
    bestWorst = max(worstAccHist) if worstAccHist else 0
    finalWorst = worstAccHist[-1] if worstAccHist else 0

    print(f"\n>>> AVG  - Final: {finalAvg:.4f}, Best: {bestAvg:.4f}")
    print(f">>> WORST - Final: {finalWorst:.4f}, Best: {bestWorst:.4f}")

    return avgAccHist, worstAccHist, bestAvg, finalAvg, bestWorst, finalWorst


def main():
    setupGpu()

    # Load data
    print("Loading MNIST...")
    train, test = loadMnist()
    trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

    results = {}

    # Test 1: Clean (no attacks, no BASIL)
    avg1, worst1, bestAvg1, finalAvg1, bestWorst1, finalWorst1 = runTest(
        "1. CLEAN (no attacks)",
        useAttacks=False, useBasil=False,
        trainLoaders=trainLoaders, testLoader=testLoader
    )
    results["Clean"] = {
        "avgHist": avg1, "worstHist": worst1,
        "bestAvg": bestAvg1, "finalAvg": finalAvg1,
        "bestWorst": bestWorst1, "finalWorst": finalWorst1
    }

    # Test 2: Attacks WITHOUT BASIL
    avg2, worst2, bestAvg2, finalAvg2, bestWorst2, finalWorst2 = runTest(
        f"2. ATTACKS ({ATTACK_TYPE}) - NO BASIL",
        useAttacks=True, useBasil=False,
        trainLoaders=trainLoaders, testLoader=testLoader
    )
    results["Attacks (no BASIL)"] = {
        "avgHist": avg2, "worstHist": worst2,
        "bestAvg": bestAvg2, "finalAvg": finalAvg2,
        "bestWorst": bestWorst2, "finalWorst": finalWorst2
    }

    # Test 3: Attacks WITH BASIL
    avg3, worst3, bestAvg3, finalAvg3, bestWorst3, finalWorst3 = runTest(
        f"3. ATTACKS ({ATTACK_TYPE}) + BASIL",
        useAttacks=True, useBasil=True,
        trainLoaders=trainLoaders, testLoader=testLoader
    )
    results["Attacks + BASIL"] = {
        "avgHist": avg3, "worstHist": worst3,
        "bestAvg": bestAvg3, "finalAvg": finalAvg3,
        "bestWorst": bestWorst3, "finalWorst": finalWorst3
    }

    # Summary
    print("\n" + "="*70)
    print("BASIL PAPER RESULTS SUMMARY")
    print(f"Attack: {ATTACK_TYPE}, Attackers: {ATTACKER_IDS} ({len(ATTACKER_IDS)}/{N_NODES} nodes)")
    print(f"Settings: lr={LR} (decay), S={S_MEMORY}, {N_ROUNDS} rounds")
    print("="*70)
    print(f"{'Test':<25} {'Best Avg':>12} {'Final Avg':>12} {'Best Worst':>12} {'Final Worst':>12}")
    print("-"*70)

    for name, r in results.items():
        print(f"{name:<25} {r['bestAvg']*100:>11.2f}% {r['finalAvg']*100:>11.2f}% {r['bestWorst']*100:>11.2f}% {r['finalWorst']*100:>11.2f}%")

    print("-"*70)

    # BASIL improvement
    noBasil_avg = results["Attacks (no BASIL)"]["finalAvg"]
    basil_avg = results["Attacks + BASIL"]["finalAvg"]
    improvement = basil_avg - noBasil_avg
    print(f"\nBASIL improvement (Avg Acc): {improvement*100:+.2f}%")

    noBasil_worst = results["Attacks (no BASIL)"]["finalWorst"]
    basil_worst = results["Attacks + BASIL"]["finalWorst"]
    improvement_worst = basil_worst - noBasil_worst
    print(f"BASIL improvement (Worst Acc): {improvement_worst*100:+.2f}%")
    print("="*70)

    # Save results
    os.makedirs("experiments/results/basil_paper", exist_ok=True)
    np.save("experiments/results/basil_paper/clean_avg.npy", results["Clean"]["avgHist"])
    np.save("experiments/results/basil_paper/clean_worst.npy", results["Clean"]["worstHist"])
    np.save("experiments/results/basil_paper/attacks_nobasil_avg.npy", results["Attacks (no BASIL)"]["avgHist"])
    np.save("experiments/results/basil_paper/attacks_nobasil_worst.npy", results["Attacks (no BASIL)"]["worstHist"])
    np.save("experiments/results/basil_paper/attacks_basil_avg.npy", results["Attacks + BASIL"]["avgHist"])
    np.save("experiments/results/basil_paper/attacks_basil_worst.npy", results["Attacks + BASIL"]["worstHist"])
    print("\nResults saved to experiments/results/basil_paper/")


if __name__ == "__main__":
    main()
