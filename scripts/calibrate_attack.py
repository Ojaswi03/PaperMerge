#!/usr/bin/env python3
"""
Quick calibration: measure loss of honest vs poisoned models at various settings.
Helps find modelPoisonAttack params where poisoned loss is close to honest loss.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.trainer import lossFn, getParams, setParams, evaluateBatchLoss, localUpdate
from basil_core.attacks import modelPoisonAttack
from scripts.common import setupGpu

setupGpu()

print("Loading MNIST...")
train, test = loadMnist()
trainLoaders, testLoader = makeLoaders(train, test, batchSize=32, nClients=10)

# Train a model for a few rounds to get realistic weights
model = MNISTModel()
print("Pre-training model for 3 rounds...")
for r in range(3):
    localUpdate(model, trainLoaders[0], epochs=1, lr=0.03, stepsPerEpoch=100)

# Measure honest loss on multiple nodes' data
print("\n=== Honest model losses on different nodes' data ===")
honest_params = getParams(model)
losses = []
for i in range(10):
    setParams(model, honest_params)
    loss = evaluateBatchLoss(model, trainLoaders[i])
    losses.append(loss)
    print(f"  Node {i} data: loss={loss:.4f}")
avg_honest = np.mean(losses)
std_honest = np.std(losses)
print(f"  Mean={avg_honest:.4f}, Std={std_honest:.4f}")

# Test various poison parameters
print("\n=== Poisoned model losses at various settings ===")
print(f"{'nSteps':>8} {'poisonLr':>10} {'loss_node0':>12} {'loss_node1':>12} {'loss_node5':>12} {'avg_loss':>10} {'gap':>8}")
print("-" * 75)

settings = [
    (1, 0.005), (1, 0.01), (1, 0.02),
    (2, 0.005), (2, 0.01), (2, 0.02),
    (3, 0.005), (3, 0.008), (3, 0.01), (3, 0.015),
    (4, 0.005), (4, 0.008), (4, 0.01), (4, 0.015),
    (5, 0.005), (5, 0.008), (5, 0.01), (5, 0.015), (5, 0.02),
    (7, 0.008), (7, 0.01), (7, 0.015),
    (10, 0.005), (10, 0.008), (10, 0.01), (10, 0.015), (10, 0.02),
]

for nSteps, poisonLr in settings:
    # Reset to honest params and poison
    setParams(model, honest_params)
    poisoned = modelPoisonAttack(model, trainLoaders[0], nSteps=nSteps, poisonLr=poisonLr, noiseStd=0.01)

    # Measure loss on different nodes' data
    poison_losses = []
    for node_id in [0, 1, 5]:
        setParams(model, poisoned)
        loss = evaluateBatchLoss(model, trainLoaders[node_id])
        poison_losses.append(loss)

    avg_poison = np.mean(poison_losses)
    gap = avg_poison - avg_honest
    print(f"{nSteps:>8} {poisonLr:>10.4f} {poison_losses[0]:>12.4f} {poison_losses[1]:>12.4f} {poison_losses[2]:>12.4f} {avg_poison:>10.4f} {gap:>+8.4f}")

print(f"\nTarget: gap should be ~0.02-0.10 (close enough for batch variance to matter)")
print(f"Honest loss variance across batches: std={std_honest:.4f}")
print(f"If gap < 2*std, BASIL will sometimes pick the poisoned model")
