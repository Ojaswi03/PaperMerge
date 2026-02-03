#!/usr/bin/env python3
"""
Fair comparison: Noisy baseline vs EBM
Both with and without momentum to isolate EBM's contribution.
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
N_NODES = 10
N_ROUNDS = 30
BATCH_SIZE = 32


def runTest(useEbm, useMomentum, trainLoaders, testLoader):
    """Run a single test configuration."""
    momentum = 0.9 if useMomentum else 0.0
    ebmLabel = "EBM" if useEbm else "Noisy"
    momLabel = "+momentum" if useMomentum else ""

    print(f"\n{'='*60}")
    print(f"TEST: {ebmLabel}{momLabel}")
    print(f"  σ={SIGMA}, momentum={momentum}, lr={LR} (decay)")
    if useEbm:
        scale = 1.0 + EBM_LAMBDA * SIGMA * SIGMA
        print(f"  EBM λ={EBM_LAMBDA}, scale={scale:.1f}")
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

            # SGD with or without momentum
            optimizer = tf.keras.optimizers.SGD(learning_rate=currentLr, momentum=momentum)

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

        # Add channel noise
        noisyParams = addChannelNoiseToParams(avgParams, sigma=SIGMA)
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

    print(f"\n>>> FINAL: {acc:.4f}, BEST: {bestAcc:.4f} @ round {bestRound}")
    return accHistory, bestAcc, acc


def main():
    setupGpu()

    # Load data once
    print("Loading MNIST...")
    train, test = loadMnist()
    trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

    results = {}

    # Test 1: Noisy baseline (no momentum)
    _, best1, final1 = runTest(useEbm=False, useMomentum=False,
                                trainLoaders=trainLoaders, testLoader=testLoader)
    results["Noisy (no momentum)"] = (best1, final1)

    # Test 2: Noisy + momentum
    _, best2, final2 = runTest(useEbm=False, useMomentum=True,
                                trainLoaders=trainLoaders, testLoader=testLoader)
    results["Noisy + momentum"] = (best2, final2)

    # Test 3: EBM (no momentum)
    _, best3, final3 = runTest(useEbm=True, useMomentum=False,
                                trainLoaders=trainLoaders, testLoader=testLoader)
    results["EBM (no momentum)"] = (best3, final3)

    # Test 4: EBM + momentum
    _, best4, final4 = runTest(useEbm=True, useMomentum=True,
                                trainLoaders=trainLoaders, testLoader=testLoader)
    results["EBM + momentum"] = (best4, final4)

    # Summary
    print("\n" + "="*70)
    print("FAIR COMPARISON SUMMARY (σ=0.2, lr=0.03 with decay)")
    print("="*70)
    print(f"{'Configuration':<25} {'Best':>10} {'Final':>10} {'EBM Δ':>10}")
    print("-"*70)

    # No momentum comparison
    noisy_no_mom = results["Noisy (no momentum)"][0]
    ebm_no_mom = results["EBM (no momentum)"][0]
    delta_no_mom = ebm_no_mom - noisy_no_mom
    print(f"{'Noisy (no momentum)':<25} {noisy_no_mom*100:>9.2f}% {results['Noisy (no momentum)'][1]*100:>9.2f}%")
    print(f"{'EBM (no momentum)':<25} {ebm_no_mom*100:>9.2f}% {results['EBM (no momentum)'][1]*100:>9.2f}% {delta_no_mom*100:>+9.2f}%")
    print("-"*70)

    # With momentum comparison
    noisy_mom = results["Noisy + momentum"][0]
    ebm_mom = results["EBM + momentum"][0]
    delta_mom = ebm_mom - noisy_mom
    print(f"{'Noisy + momentum':<25} {noisy_mom*100:>9.2f}% {results['Noisy + momentum'][1]*100:>9.2f}%")
    print(f"{'EBM + momentum':<25} {ebm_mom*100:>9.2f}% {results['EBM + momentum'][1]*100:>9.2f}% {delta_mom*100:>+9.2f}%")
    print("="*70)

    print("\nKey findings:")
    print(f"  - Momentum alone improves noisy baseline by: {(noisy_mom - noisy_no_mom)*100:+.2f}%")
    print(f"  - EBM improves over noisy (no momentum) by:  {delta_no_mom*100:+.2f}%")
    print(f"  - EBM improves over noisy (with momentum) by: {delta_mom*100:+.2f}%")


if __name__ == "__main__":
    main()
