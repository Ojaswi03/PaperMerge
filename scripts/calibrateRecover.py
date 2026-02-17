#!/usr/bin/env python3
"""
Test "noise + partial recovery" attack:
1. Add Gaussian noise (destroys model)
2. Do gradient descent to partially recover loss
3. Model still has embedded noise but loss looks closer to honest
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.trainer import lossFn, getParams, setParams, evaluateBatchLoss, localUpdate
from basil_core.attacks import gaussianAttack
from scripts.common import setupGpu

setupGpu()

print("Loading MNIST...")
train, test = loadMnist()
trainLoaders, testLoader = makeLoaders(train, test, batchSize=32, nClients=10)

# Train a model for 3 rounds
model = MNISTModel()
for r in range(3):
    localUpdate(model, trainLoaders[0], epochs=1, lr=0.03, stepsPerEpoch=100)

honest_params = getParams(model)

# Honest loss across nodes
print("\n=== Honest model losses ===")
honest_losses = []
for i in range(10):
    setParams(model, honest_params)
    loss = evaluateBatchLoss(model, trainLoaders[i])
    honest_losses.append(loss)
avg_honest = np.mean(honest_losses)
std_honest = np.std(honest_losses)
print(f"  Mean={avg_honest:.4f}, Std={std_honest:.4f}, Range=[{min(honest_losses):.4f}, {max(honest_losses):.4f}]")

# Pure Gaussian noise loss
setParams(model, honest_params)
noised = gaussianAttack(getParams(model), std=0.8, blend=0.55)
setParams(model, noised)
pure_noise_loss = evaluateBatchLoss(model, trainLoaders[1])
print(f"\nPure Gaussian noise loss: {pure_noise_loss:.4f} (gap={pure_noise_loss - avg_honest:+.4f})")

# Test: add noise then recover with training
print(f"\n=== Noise + Partial Recovery ===")
print(f"{'recoverSteps':>14} {'recoverLr':>10} {'loss_n0':>10} {'loss_n1':>10} {'loss_n5':>10} {'avg_loss':>10} {'gap':>8} {'detect%':>10}")
print("-" * 85)

settings = [
    (5, 0.03), (5, 0.05),
    (10, 0.03), (10, 0.05),
    (15, 0.03), (15, 0.05),
    (20, 0.03), (20, 0.05),
    (25, 0.03), (25, 0.05),
    (30, 0.03), (30, 0.05),
    (40, 0.03), (40, 0.05),
    (50, 0.03), (50, 0.05),
]

for recoverSteps, recoverLr in settings:
    # Reset, add noise, then recover
    setParams(model, honest_params)
    noised = gaussianAttack(getParams(model), std=0.8, blend=0.55)
    setParams(model, noised)

    # Partial recovery via gradient descent
    for step in range(recoverSteps):
        for xBatch, yBatch in trainLoaders[0]:
            xb = tf.convert_to_tensor(xBatch, dtype=tf.float32)
            yb = tf.convert_to_tensor(yBatch, dtype=tf.int32)
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss = lossFn(yb, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            for var, grad in zip(model.trainable_variables, grads):
                if grad is not None:
                    var.assign_sub(recoverLr * grad)
            break

    # Measure loss on different nodes
    recovered_params = getParams(model)
    losses = []
    for node_id in [0, 1, 5]:
        setParams(model, recovered_params)
        l = evaluateBatchLoss(model, trainLoaders[node_id])
        losses.append(l)

    avg_l = np.mean(losses)
    gap = avg_l - avg_honest
    # Estimate detection probability (how often BASIL picks honest over this)
    # If gap > 2*std, almost always detected. If gap < 0, almost never detected.
    detect_pct = min(100, max(0, 50 + 50 * gap / (2 * std_honest)))
    print(f"{recoverSteps:>14} {recoverLr:>10.3f} {losses[0]:>10.4f} {losses[1]:>10.4f} {losses[2]:>10.4f} {avg_l:>10.4f} {gap:>+8.4f} {detect_pct:>9.0f}%")

print(f"\nTarget: detect% around 60-70% (BASIL catches most but not all)")
print(f"This means gap ~0.05-0.15 (within 1-2x batch std={std_honest:.4f})")
