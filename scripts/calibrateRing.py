#!/usr/bin/env python3
"""
Quick ring simulation: test a few poison settings for both no-snapshot and snapshot.
Runs only 15 rounds to save time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.basil import BasilNode, basilRingTrainingWithAttack
from scripts.common import setupGpu

setupGpu()

N_NODES = 10
N_ROUNDS = 15  # Quick test
LR = 0.03
BATCH_SIZE = 32
S_MEMORY = 5
ATTACKER_IDS = [0, 1, 2, 3]

print("Loading MNIST...")
train, test = loadMnist()
trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

# Test settings: (nSteps, poisonLr)
settings = [
    (5, 0.010),
    (7, 0.008),
    (10, 0.005),
]

for nSteps, poisonLr in settings:
    for useSnapshots in [False, True]:
        label = f"nSteps={nSteps}, lr={poisonLr}, snapshot={'YES' if useSnapshots else 'NO'}"
        print(f"\n{'='*60}")
        print(f"TEST: {label}")
        print(f"{'='*60}")

        # Monkey-patch the poison params for this test
        import basil_core.attacks as atk_mod
        orig_fn = atk_mod.modelPoisonAttack.__defaults__
        atk_mod.modelPoisonAttack.__defaults__ = (nSteps, poisonLr, 0.01)

        nodes = []
        for i in range(N_NODES):
            model = MNISTModel()
            node = BasilNode(
                nodeId=i, model=model, dataLoader=trainLoaders[i],
                S=S_MEMORY, lr0=LR, localEpochs=1,
            )
            nodes.append(node)

        avgHist, worstHist = basilRingTrainingWithAttack(
            nodes=nodes, rounds=N_ROUNDS, testLoader=testLoader,
            attackTypes=["gaussian"], attackerIds=ATTACKER_IDS,
            lr0=LR, stepsPerEpoch=100,
            useSnapshots=useSnapshots, useSequential=True, useLrDecay=True,
        )

        # Restore
        atk_mod.modelPoisonAttack.__defaults__ = orig_fn

        final_avg = avgHist[-1] if avgHist else 0
        final_worst = worstHist[-1] if worstHist else 0
        best_avg = max(avgHist) if avgHist else 0
        print(f">>> {label}: final_avg={final_avg:.4f}, best_avg={best_avg:.4f}, final_worst={final_worst:.4f}")

print("\n\nDONE - Compare no-snapshot (~55%) vs snapshot (~75-80%)")
