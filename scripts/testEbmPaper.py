#!/usr/bin/env python3
"""
Paper comparison: Clean vs Noisy vs Noisy+EBM
All tests use momentum=0.9 for fair comparison.
Noise σ=0.2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.trainer import getParams, setParams, evaluate, averageParams, addChannelNoiseToParams, lossFn, makeLrScheduler
from scripts.common import setupGpu

# Configuration
SIGMA = 0.2
EBM_LAMBDA = 75
LR = 0.03
MOMENTUM = 0.9  # Same for all tests (fair comparison)
N_NODES = 10
N_ROUNDS = 30
BATCH_SIZE = 32


def runTest(testName, useNoise, useEbm, trainLoaders, testLoader):
    """Run a single test configuration."""
    print(f"\n{'='*60}")
    print(f"TEST: {testName}")
    print(f"  Noise: {'σ='+str(SIGMA) if useNoise else 'None'}")
    print(f"  EBM: {'λ='+str(EBM_LAMBDA)+' (scale='+str(1+EBM_LAMBDA*SIGMA*SIGMA)+')' if useEbm else 'No'}")
    print(f"  Momentum: {MOMENTUM}, LR: {LR} (with decay)")
    print(f"{'='*60}\n")

    # Create fresh models
    models = [MNISTModel() for _ in range(N_NODES)]

    # Initialize all to same weights
    globalParams = getParams(models[0])
    for m in models:
        setParams(m, globalParams)

    # LR scheduler with decay
    lrSched = makeLrScheduler(LR, useLrDecay=True)

    accHistory = []
    bestAcc = 0.0
    bestRound = 0

    # Pre-eval
    acc = evaluate(models[0], testLoader)
    print(f"[round -1] pre-train acc={acc:.4f}")
    accHistory.append(acc)

    for r in range(N_ROUNDS):
        currentLr = lrSched(r)

        # Train all nodes in parallel
        localUpdates = []
        for i, model in enumerate(models):
            setParams(model, globalParams)

            # SGD with momentum (same for all tests)
            optimizer = tf.keras.optimizers.SGD(learning_rate=currentLr, momentum=MOMENTUM)

            # One epoch of training
            batchCount = 0
            for batch in trainLoaders[i]:
                if batchCount >= 100:
                    break
                xBatch, yBatch = batch
                xb = tf.convert_to_tensor(xBatch, dtype=tf.float32)
                yb = tf.convert_to_tensor(yBatch, dtype=tf.int32)

                with tf.GradientTape() as tape:
                    logits = model(xb, training=True)
                    loss = lossFn(yb, logits)
                grads = tape.gradient(loss, model.trainable_weights)

                if useEbm:
                    # EBM: scale gradients
                    scale = 1.0 + EBM_LAMBDA * SIGMA * SIGMA
                    grads = [g * scale for g in grads]

                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                batchCount += 1

            localUpdates.append(getParams(model))

        # Average models (FedAvg)
        avgParams = averageParams(localUpdates)

        # Add channel noise only if useNoise=True
        if useNoise:
            noisyParams = addChannelNoiseToParams(avgParams, sigma=SIGMA)
        else:
            noisyParams = avgParams
        globalParams = noisyParams

        # Update all models
        for m in models:
            setParams(m, globalParams)

        # Evaluate
        acc = evaluate(models[0], testLoader)
        accHistory.append(acc)

        if acc > bestAcc:
            bestAcc = acc
            bestRound = r

        print(f"[round {r}] lr={currentLr:.6f} acc={acc:.4f} (best={bestAcc:.4f} @ r{bestRound})")

    print(f"\n>>> FINAL: {acc:.4f} ({acc*100:.2f}%), BEST: {bestAcc:.4f} ({bestAcc*100:.2f}%) @ round {bestRound}")
    return accHistory, bestAcc, acc


def main():
    setupGpu()

    # Load data once
    print("Loading MNIST...")
    train, test = loadMnist()
    trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

    results = {}

    # Test 1: Clean (no noise, no EBM)
    acc1, best1, final1 = runTest("1. CLEAN (no noise)",
                                   useNoise=False, useEbm=False,
                                   trainLoaders=trainLoaders, testLoader=testLoader)
    results["Clean"] = (best1, final1, acc1)

    # Test 2: Noisy (noise, no EBM)
    acc2, best2, final2 = runTest("2. NOISY (σ=0.2, no EBM)",
                                   useNoise=True, useEbm=False,
                                   trainLoaders=trainLoaders, testLoader=testLoader)
    results["Noisy"] = (best2, final2, acc2)

    # Test 3: Noisy + EBM
    acc3, best3, final3 = runTest("3. NOISY + EBM (σ=0.2, λ=75)",
                                   useNoise=True, useEbm=True,
                                   trainLoaders=trainLoaders, testLoader=testLoader)
    results["Noisy+EBM"] = (best3, final3, acc3)

    # Summary
    print("\n" + "="*70)
    print("PAPER RESULTS SUMMARY")
    print(f"Settings: σ={SIGMA}, λ={EBM_LAMBDA}, lr={LR} (decay), momentum={MOMENTUM}")
    print("="*70)
    print(f"{'Test':<25} {'Best Acc':>12} {'Final Acc':>12}")
    print("-"*70)

    for name, (best, final, _) in results.items():
        print(f"{name:<25} {best*100:>11.2f}% {final*100:>11.2f}%")

    print("-"*70)

    # EBM improvement
    noisy_best = results["Noisy"][0]
    ebm_best = results["Noisy+EBM"][0]
    improvement = ebm_best - noisy_best
    print(f"\nEBM improvement over Noisy: {improvement*100:+.2f}%")
    print("="*70)

    # Save results
    os.makedirs("experiments/results/paper_comparison", exist_ok=True)
    np.save("experiments/results/paper_comparison/clean.npy", results["Clean"][2])
    np.save("experiments/results/paper_comparison/noisy.npy", results["Noisy"][2])
    np.save("experiments/results/paper_comparison/noisy_ebm.npy", results["Noisy+EBM"][2])
    print("\nResults saved to experiments/results/paper_comparison/")


if __name__ == "__main__":
    main()
