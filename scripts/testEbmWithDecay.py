#!/usr/bin/env python3
"""
Test EBM with LR decay + momentum
Settings: lr=0.03, decay=ON, momentum=0.9, EBM lambda=25, sigma=0.2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.basil import BasilNode, fedAvgTrainingWithNoise
from basil_core.trainer import getParams, setParams, evaluate, averageParams, addChannelNoiseToParams, lossFn, makeLrScheduler
from scripts.common import setupGpu

# Configuration - User's requested settings
SIGMA = 0.2
EBM_LAMBDA = 25
LR = 0.03
MOMENTUM = 0.9
USE_LR_DECAY = True
N_NODES = 10
N_ROUNDS = 30
BATCH_SIZE = 32


def runTest():
    """Run EBM with lr=0.03, decay, momentum=0.9."""
    scale = 1.0 + EBM_LAMBDA * SIGMA * SIGMA
    print(f"\n{'='*60}")
    print(f"EBM Test: lr={LR}, decay={USE_LR_DECAY}, momentum={MOMENTUM}")
    print(f"  λ={EBM_LAMBDA}, σ={SIGMA}, scale={scale:.1f}")
    print(f"  Rounds: {N_ROUNDS}, Nodes: {N_NODES}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading MNIST...")
    train, test = loadMnist()
    trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

    # Create nodes with momentum
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
            lr0=LR,
            localEpochs=1,
            ebmLambda=EBM_LAMBDA,
            momentum=MOMENTUM,
        )
        nodes.append(node)

    # Custom training with decay + momentum
    globalParams = getParams(nodes[0].model)
    for nd in nodes:
        setParams(nd.model, globalParams)

    # LR scheduler with decay
    lrSched = makeLrScheduler(LR, useLrDecay=USE_LR_DECAY)

    accHistory = []
    bestAcc = 0.0
    bestRound = 0

    # Pre-eval
    acc = evaluate(nodes[0].model, testLoader)
    print(f"[round -1] pre-train acc={acc:.4f}")
    accHistory.append(acc)

    for r in range(N_ROUNDS):
        currentLr = lrSched(r)

        # Train all nodes in parallel
        localUpdates = []
        for i, nd in enumerate(nodes):
            setParams(nd.model, globalParams)

            # SGD with momentum
            optimizer = tf.keras.optimizers.SGD(learning_rate=currentLr, momentum=MOMENTUM)

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
                scale = 1.0 + EBM_LAMBDA * SIGMA * SIGMA
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

    print(f"\n{'='*60}")
    print(f"FINAL: {acc:.4f} ({acc*100:.2f}%)")
    print(f"BEST:  {bestAcc:.4f} ({bestAcc*100:.2f}%) @ round {bestRound}")
    print(f"{'='*60}")

    # Save results
    os.makedirs("experiments/results/ebm_decay_test", exist_ok=True)
    np.save("experiments/results/ebm_decay_test/ebm_decay_momentum.npy", accHistory)
    print(f"\nResults saved to experiments/results/ebm_decay_test/")

    return accHistory, bestAcc


if __name__ == "__main__":
    setupGpu()
    runTest()
