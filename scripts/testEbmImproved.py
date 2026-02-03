#!/usr/bin/env python3
"""
Improved EBM test to achieve higher accuracy with sigma=0.2
Tests different configurations to find optimal settings.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.basil import BasilNode, fedAvgTrainingWithNoise
from basil_core.trainer import getParams, setParams, evaluate, averageParams, addChannelNoiseToParams, lossFn
from scripts.common import setupGpu

# Configuration
SIGMA = 0.2
N_NODES = 10
BATCH_SIZE = 32


def runImprovedEbm(trainLoaders, testLoader, ebmLambda, lr, nRounds, useMomentum=False, lrDecay=False):
    """Run EBM with improved settings."""
    scale = 1.0 + ebmLambda * SIGMA * SIGMA
    momentumStr = " +momentum" if useMomentum else ""
    decayStr = " +decay" if lrDecay else " fixed_lr"
    print(f"\n{'='*60}")
    print(f"EBM: λ={ebmLambda}, scale={scale:.1f}, lr={lr}{momentumStr}{decayStr}")
    print(f"Rounds: {nRounds}")
    print(f"{'='*60}\n")

    # Create fresh nodes
    nodes = []
    for i in range(N_NODES):
        model = MNISTModel()
        node = BasilNode(
            nodeId=i,
            model=model,
            dataLoader=trainLoaders[i],
            S=10,
            noiseModel="ebm",
            sigma=SIGMA,
            lr0=lr,
            localEpochs=1,
            ebmLambda=ebmLambda,
        )
        nodes.append(node)

    # Custom training loop with options
    globalParams = getParams(nodes[0].model)
    for nd in nodes:
        setParams(nd.model, globalParams)

    accHistory = []
    bestAcc = 0.0
    bestRound = 0

    # Pre-eval
    acc = evaluate(nodes[0].model, testLoader)
    print(f"[round -1] pre-train acc={acc:.4f}")
    accHistory.append(acc)

    for r in range(nRounds):
        # Learning rate
        if lrDecay:
            currentLr = lr / (1.0 + lr * r)
        else:
            currentLr = lr  # Fixed LR

        # Train all nodes in parallel
        localUpdates = []
        for i, nd in enumerate(nodes):
            setParams(nd.model, globalParams)

            # Custom training with momentum option
            if useMomentum:
                optimizer = tf.keras.optimizers.SGD(learning_rate=currentLr, momentum=0.9)
            else:
                optimizer = tf.keras.optimizers.SGD(learning_rate=currentLr, momentum=0.0)

            # One epoch of training with EBM
            batchCount = 0
            for batch in nd.dataLoader:
                if batchCount >= 100:
                    break
                xBatch, yBatch = batch
                xb = tf.convert_to_tensor(xBatch, dtype=tf.float32)
                yb = tf.convert_to_tensor(yBatch, dtype=tf.int32)

                with tf.GradientTape() as tape:
                    logits = nd.model(xb, training=True)
                    loss = lossFn(yb, logits)
                grads = tape.gradient(loss, nd.model.trainable_weights)

                # EBM scaling
                scale = 1.0 + ebmLambda * SIGMA * SIGMA
                scaledGrads = [g * scale for g in grads]
                optimizer.apply_gradients(zip(scaledGrads, nd.model.trainable_weights))
                batchCount += 1

            localUpdates.append(getParams(nd.model))

        # Average models
        avgParams = averageParams(localUpdates)

        # Add channel noise
        noisyParams = addChannelNoiseToParams(avgParams, sigma=SIGMA)
        globalParams = noisyParams

        # Update all nodes
        for nd in nodes:
            setParams(nd.model, globalParams)

        # Evaluate
        acc = evaluate(nodes[0].model, testLoader)
        accHistory.append(acc)

        if acc > bestAcc:
            bestAcc = acc
            bestRound = r

        print(f"[round {r}] lr={currentLr:.6f} acc={acc:.4f} (best={bestAcc:.4f} @ r{bestRound})")

    print(f"\n>>> FINAL: {acc:.4f} ({acc*100:.2f}%), BEST: {bestAcc:.4f} ({bestAcc*100:.2f}%) @ round {bestRound}")
    return accHistory, bestAcc, bestRound


def main():
    setupGpu()

    # Load data once
    print("Loading MNIST...")
    train, test = loadMnist()
    trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

    results = []

    # Test 1: Original settings (baseline)
    print("\n" + "="*70)
    print("TEST 1: Original EBM (λ=75, lr=0.03, decay)")
    print("="*70)
    acc1, best1, r1 = runImprovedEbm(trainLoaders, testLoader,
                                      ebmLambda=75, lr=0.03, nRounds=30,
                                      useMomentum=False, lrDecay=True)
    results.append(("Original (λ=75, decay)", best1))

    # Test 2: Fixed LR (no decay)
    print("\n" + "="*70)
    print("TEST 2: Fixed LR (λ=75, lr=0.03, no decay)")
    print("="*70)
    acc2, best2, r2 = runImprovedEbm(trainLoaders, testLoader,
                                      ebmLambda=75, lr=0.03, nRounds=30,
                                      useMomentum=False, lrDecay=False)
    results.append(("Fixed LR (λ=75)", best2))

    # Test 3: With momentum
    print("\n" + "="*70)
    print("TEST 3: With Momentum (λ=75, lr=0.01, momentum=0.9)")
    print("="*70)
    acc3, best3, r3 = runImprovedEbm(trainLoaders, testLoader,
                                      ebmLambda=75, lr=0.01, nRounds=30,
                                      useMomentum=True, lrDecay=False)
    results.append(("Momentum (λ=75)", best3))

    # Test 4: Higher lambda with lower LR
    print("\n" + "="*70)
    print("TEST 4: Higher Lambda (λ=100, lr=0.02, no decay)")
    print("="*70)
    acc4, best4, r4 = runImprovedEbm(trainLoaders, testLoader,
                                      ebmLambda=100, lr=0.02, nRounds=30,
                                      useMomentum=False, lrDecay=False)
    results.append(("Higher λ=100", best4))

    # Test 5: More rounds
    print("\n" + "="*70)
    print("TEST 5: More Rounds (λ=75, lr=0.03, 50 rounds)")
    print("="*70)
    acc5, best5, r5 = runImprovedEbm(trainLoaders, testLoader,
                                      ebmLambda=75, lr=0.03, nRounds=50,
                                      useMomentum=False, lrDecay=False)
    results.append(("50 rounds (λ=75)", best5))

    # Test 6: Momentum + More rounds
    print("\n" + "="*70)
    print("TEST 6: Momentum + More Rounds (λ=75, lr=0.01, momentum, 50 rounds)")
    print("="*70)
    acc6, best6, r6 = runImprovedEbm(trainLoaders, testLoader,
                                      ebmLambda=75, lr=0.01, nRounds=50,
                                      useMomentum=True, lrDecay=False)
    results.append(("Momentum + 50 rounds", best6))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Best Accuracy Achieved (σ=0.2)")
    print("="*70)
    results.sort(key=lambda x: x[1], reverse=True)
    for name, acc in results:
        print(f"  {name}: {acc*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
