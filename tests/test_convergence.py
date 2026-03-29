#!/usr/bin/env python3
"""
Convergence test for BASIL and FedAvg training.

Tests:
1. BASIL ring training (basilRingTrainingWithAttack) over 8 rounds on MNIST
2. FedAvg training (fedAvgTrainingWithNoise) over 8 rounds on MNIST

Assertions:
- Pre-training accuracy (index 0) <= 0.20  (random model ~10%)
- Final accuracy (index -1) > 0.50         (should learn significantly)
- Upward trend: avg of last 3 rounds > avg of first 3 rounds (skip index 0)
"""

import sys
sys.path.insert(0, '/home/ojaswi/PaperMerge')

from scripts.common import setupGpu
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.basil import BasilNode, basilRingTrainingWithAttack, fedAvgTrainingWithNoise

# ============================================================
# Setup
# ============================================================
print("=" * 60)
print("CONVERGENCE TEST")
print("=" * 60)

gpuInfo = setupGpu()
print(f"Device: {gpuInfo['device']}")
print()

# ============================================================
# Load MNIST (small subset)
# ============================================================
print("Loading MNIST (3000 train, 1000 test)...")
trainFull, testFull = loadMnist()

trainSmall = trainFull[:3000]
testSmall = testFull[:1000]

N_NODES = 3
BATCH_SIZE = 256

trainLoaders, testLoader = makeLoaders(trainSmall, testSmall, batchSize=BATCH_SIZE, nClients=N_NODES)
print(f"  Train samples: {len(trainSmall)}, Test samples: {len(testSmall)}")
print(f"  Nodes: {N_NODES}, Batch size: {BATCH_SIZE}")
print()

# ============================================================
# TEST 1: BASIL Ring Training
# ============================================================
print("=" * 60)
print("TEST 1: BASIL Ring Training (basilRingTrainingWithAttack)")
print("=" * 60)

# Create nodes
basilNodes = []
for i in range(N_NODES):
    model = MNISTModel()
    node = BasilNode(
        nodeId=i,
        model=model,
        dataLoader=trainLoaders[i],
        S=N_NODES,       # memory size = number of nodes
        noiseModel="none",
        sigma=0.0,
        lr0=0.05,
        localEpochs=5,
        momentum=0.9,
        ebmLambda=25.0,
    )
    basilNodes.append(node)

# Run 8 rounds
ROUNDS = 8
avgAccHist, worstAccHist = basilRingTrainingWithAttack(
    nodes=basilNodes,
    rounds=ROUNDS,
    testLoader=testLoader,
    attackTypes=["none"],
    attackerIds=None,
    sigma=0.0,
    noiseModel="none",
    lr0=0.05,
    useSnapshots=True,
    useSequential=True,
    useLrDecay=True,
)

print()
print("BASIL Accuracy History (avg across nodes):")
for i, acc in enumerate(avgAccHist):
    label = "pre-training" if i == 0 else f"round {i-1}"
    print(f"  [{label:>12}] acc = {acc:.4f}")

print()
print("BASIL Assertions:")
passCount = 0
failCount = 0

# Assertion 1: pre-training accuracy <= 0.20
preAcc = avgAccHist[0]
p1 = preAcc <= 0.20
status1 = "PASS" if p1 else "FAIL"
print(f"  [{status1}] Pre-training accuracy {preAcc:.4f} <= 0.20")
if p1:
    passCount += 1
else:
    failCount += 1

# Assertion 2: final accuracy > 0.50
finalAcc = avgAccHist[-1]
p2 = finalAcc > 0.50
status2 = "PASS" if p2 else "FAIL"
print(f"  [{status2}] Final accuracy {finalAcc:.4f} > 0.50")
if p2:
    passCount += 1
else:
    failCount += 1

# Assertion 3: upward trend (skip index 0 = pre-training)
# first 3 rounds = indices 1,2,3; last 3 rounds = indices -3,-2,-1
roundAccs = avgAccHist[1:]  # skip pre-training
if len(roundAccs) >= 6:
    first3avg = sum(roundAccs[:3]) / 3
    last3avg = sum(roundAccs[-3:]) / 3
    p3 = last3avg > first3avg
    status3 = "PASS" if p3 else "FAIL"
    print(f"  [{status3}] Upward trend: last3 avg {last3avg:.4f} > first3 avg {first3avg:.4f}")
    if p3:
        passCount += 1
    else:
        failCount += 1
else:
    print(f"  [SKIP] Not enough rounds to check trend (need >=6, got {len(roundAccs)})")

print()
print(f"BASIL Test: {passCount} PASS, {failCount} FAIL")

# ============================================================
# TEST 2: FedAvg Training
# ============================================================
print()
print("=" * 60)
print("TEST 2: FedAvg Training (fedAvgTrainingWithNoise)")
print("=" * 60)

# Create fresh nodes for FedAvg
fedAvgNodes = []
for i in range(N_NODES):
    model = MNISTModel()
    node = BasilNode(
        nodeId=i,
        model=model,
        dataLoader=trainLoaders[i],
        S=N_NODES,
        noiseModel="none",
        sigma=0.0,
        lr0=0.05,
        localEpochs=5,
        momentum=0.9,
        ebmLambda=25.0,
    )
    fedAvgNodes.append(node)

# Run 8 rounds
avgAccHistFA, worstAccHistFA = fedAvgTrainingWithNoise(
    nodes=fedAvgNodes,
    rounds=ROUNDS,
    testLoader=testLoader,
    attackTypes=["none"],
    attackerIds=None,
    sigma=0.0,
    noiseModel="none",
    lr0=0.05,
    localEpochs=5,
    useLrDecay=True,
)

print()
print("FedAvg Accuracy History:")
for i, acc in enumerate(avgAccHistFA):
    label = "pre-training" if i == 0 else f"round {i-1}"
    print(f"  [{label:>12}] acc = {acc:.4f}")

print()
print("FedAvg Assertions:")
faPassCount = 0
faFailCount = 0

# Assertion 1: pre-training accuracy <= 0.20
faPreAcc = avgAccHistFA[0]
fa1 = faPreAcc <= 0.20
faStatus1 = "PASS" if fa1 else "FAIL"
print(f"  [{faStatus1}] Pre-training accuracy {faPreAcc:.4f} <= 0.20")
if fa1:
    faPassCount += 1
else:
    faFailCount += 1

# Assertion 2: final accuracy > 0.50
faFinalAcc = avgAccHistFA[-1]
fa2 = faFinalAcc > 0.50
faStatus2 = "PASS" if fa2 else "FAIL"
print(f"  [{faStatus2}] Final accuracy {faFinalAcc:.4f} > 0.50")
if fa2:
    faPassCount += 1
else:
    faFailCount += 1

# Assertion 3: upward trend
faRoundAccs = avgAccHistFA[1:]
if len(faRoundAccs) >= 6:
    faFirst3avg = sum(faRoundAccs[:3]) / 3
    faLast3avg = sum(faRoundAccs[-3:]) / 3
    fa3 = faLast3avg > faFirst3avg
    faStatus3 = "PASS" if fa3 else "FAIL"
    print(f"  [{faStatus3}] Upward trend: last3 avg {faLast3avg:.4f} > first3 avg {faFirst3avg:.4f}")
    if fa3:
        faPassCount += 1
    else:
        faFailCount += 1
else:
    print(f"  [SKIP] Not enough rounds to check trend (need >=6, got {len(faRoundAccs)})")

print()
print(f"FedAvg Test: {faPassCount} PASS, {faFailCount} FAIL")

# ============================================================
# Overall Summary
# ============================================================
print()
print("=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
totalPass = passCount + faPassCount
totalFail = failCount + faFailCount
print(f"  BASIL:  {passCount} PASS, {failCount} FAIL")
print(f"  FedAvg: {faPassCount} PASS, {faFailCount} FAIL")
print(f"  TOTAL:  {totalPass} PASS, {totalFail} FAIL")

if totalFail == 0:
    print()
    print("ALL TESTS PASSED")
else:
    print()
    print(f"SOME TESTS FAILED ({totalFail} failures)")
    sys.exit(1)
