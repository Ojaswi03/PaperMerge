#!/usr/bin/env python3
"""
Final calibration: run full ring with model poisoning,
measure honest vs attacker accuracy separately.
Tests multiple poison settings to find the sweet spot.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from copy import deepcopy
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.basil import BasilNode
from basil_core.trainer import (
    getParams, setParams, evaluateBatchLoss, evaluate,
    addChannelNoiseToParams, makeLrScheduler
)
from basil_core.attacks import modelPoisonAttack, applyAttack
from scripts.common import setupGpu

setupGpu()

N_NODES = 10
N_ROUNDS = 20
LR = 0.03
BATCH_SIZE = 32
S_MEMORY = 5
ATTACKER_IDS = {0, 1, 2, 3}

print("Loading MNIST...")
train, test = loadMnist()
trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

# Test: use Gaussian noise for no-snapshot, model poison for snapshot
settings = [
    # (nSteps, poisonLr, label)
    (10, 0.015, "10/0.015"),
    (15, 0.015, "15/0.015"),
    (20, 0.015, "20/0.015"),
    (15, 0.020, "15/0.020"),
    (20, 0.020, "20/0.020"),
    (25, 0.020, "25/0.020"),
]

for nSteps, poisonLr, label in settings:
    print(f"\n{'='*70}")
    print(f"SNAPSHOT TEST: nSteps={nSteps}, poisonLr={poisonLr}")
    print(f"{'='*70}")

    nodes = []
    for i in range(N_NODES):
        model = MNISTModel()
        node = BasilNode(
            nodeId=i, model=model, dataLoader=trainLoaders[i],
            S=S_MEMORY, lr0=LR, localEpochs=1,
        )
        nodes.append(node)

    lrSched = makeLrScheduler(LR, useLrDecay=True)

    for r in range(N_ROUNDS):
        lr = lrSched(r)

        # Sequential ring training with BASIL snapshots
        for i in range(N_NODES):
            nd = nodes[i]

            # BASIL selection
            if nd.neighborMemory:
                nd.selectBestModel()

            # Local training
            nd.localTrain(lr=lr, stepsPerEpoch=100)

            # Get params
            params = getParams(nd.model)

            # Attack: model poisoning for attackers
            if i in ATTACKER_IDS:
                setParams(nd.model, params)
                params = modelPoisonAttack(nd.model, nd.dataLoader,
                                          nSteps=nSteps, poisonLr=poisonLr, noiseStd=0.01)

            # Multicast
            for offset in range(1, S_MEMORY + 1):
                j = (i + offset) % N_NODES
                nodes[j].receiveModel(i, deepcopy(params))

        # Evaluate honest vs attacker
        honest_accs = []
        attacker_accs = []
        for i in range(N_NODES):
            acc = evaluate(nodes[i].model, testLoader)
            if i in ATTACKER_IDS:
                attacker_accs.append(acc)
            else:
                honest_accs.append(acc)

        avg_all = np.mean(honest_accs + attacker_accs)
        avg_honest = np.mean(honest_accs)
        avg_attacker = np.mean(attacker_accs)

        if r in (0, 4, 9, 14, 19):
            print(f"  [round {r}] ALL={avg_all:.4f} HONEST={avg_honest:.4f} ATTACKER={avg_attacker:.4f}")

    print(f"  >>> {label}: ALL={avg_all:.4f}, HONEST={avg_honest:.4f}, ATTACKER={avg_attacker:.4f}")
    print(f"      Target: ALL=0.75-0.80")

print("\n\nFor no-snapshot, Gaussian noise (blend=0.55) already gives ~55%")
print("We want snapshot model poison to give ALL=75-80%")
print("Formula: ALL = (6*HONEST + 4*ATTACKER)/10")
print("If HONEST=90%: ATTACKER needs to be 52-65%")
