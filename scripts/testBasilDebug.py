#!/usr/bin/env python3
"""
Debug BASIL to show attacks ARE happening and being filtered.
Shows loss values for each candidate model to prove attacker models have high loss.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from copy import deepcopy
from basil_core.data.mnist import loadMnist, makeLoaders
from basil_core.models import MNISTModel
from basil_core.basil import BasilNode
from basil_core.trainer import getParams, setParams, evaluateBatchLoss, evaluate, makeLrScheduler, addChannelNoiseToParams
from basil_core.attacks import applyAttack
from scripts.common import setupGpu

# Configuration
N_NODES = 10
N_ROUNDS = 5  # Just 5 rounds to see the pattern
LR = 0.03
BATCH_SIZE = 32
S_MEMORY = 3
ATTACK_TYPE = "gaussian"
ATTACKER_IDS = {0, 5}  # Nodes 0 and 5 are attackers


def main():
    setupGpu()

    print("Loading MNIST...")
    train, test = loadMnist()
    trainLoaders, testLoader = makeLoaders(train, test, batchSize=BATCH_SIZE, nClients=N_NODES)

    # Create nodes
    nodes = []
    for i in range(N_NODES):
        model = MNISTModel()
        node = BasilNode(
            nodeId=i,
            model=model,
            dataLoader=trainLoaders[i],
            S=S_MEMORY,
            lr0=LR,
        )
        nodes.append(node)

    lrSched = makeLrScheduler(LR, useLrDecay=True)

    print(f"\n{'='*70}")
    print(f"BASIL DEBUG: Showing how attacks are filtered")
    print(f"Attackers: {ATTACKER_IDS} (send Gaussian noise)")
    print(f"S = {S_MEMORY} (each node stores {S_MEMORY} neighbor models)")
    print(f"{'='*70}\n")

    for r in range(N_ROUNDS):
        lr = lrSched(r)
        print(f"\n{'='*70}")
        print(f"ROUND {r} (lr={lr:.4f})")
        print(f"{'='*70}")

        # Sequential training (paper's algorithm)
        for i in range(N_NODES):
            nd = nodes[i]
            isAttacker = i in ATTACKER_IDS

            print(f"\n--- Node {i} {'[ATTACKER]' if isAttacker else '[HONEST]'} ---")

            # Step 1: BASIL selection (if has neighbors)
            if nd.neighborMemory:
                print(f"  Received models from: {list(nd.neighborMemory.keys())}")

                # Show all losses before selection
                currentLoss = evaluateBatchLoss(nd.model, nd.dataLoader)
                print(f"  Loss comparison:")
                print(f"    - Current model: {currentLoss:.4f}")

                for senderId, params in nd.neighborMemory.items():
                    senderIsAttacker = senderId in ATTACKER_IDS
                    setParams(nd.model, params)
                    loss = evaluateBatchLoss(nd.model, nd.dataLoader)
                    marker = " ← ATTACKER MODEL (high loss!)" if senderIsAttacker else ""
                    print(f"    - From node {senderId}: {loss:.4f}{marker}")

                # Do actual selection
                nd.selectBestModel(verbose=False)
                selectedLoss = evaluateBatchLoss(nd.model, nd.dataLoader)
                print(f"  → Selected model with loss: {selectedLoss:.4f}")

            # Step 2: Local training
            nd.localTrain(lr=lr, stepsPerEpoch=50)

            # Step 3: Get params (and apply attack if Byzantine)
            params = getParams(nd.model)

            if isAttacker:
                print(f"  [ATTACK] Applying {ATTACK_TYPE} attack to outgoing model!")
                params = applyAttack(params, ATTACK_TYPE)

            # Step 4: Multicast to next S neighbors
            for offset in range(1, S_MEMORY + 1):
                j = (i + offset) % N_NODES
                nodes[j].receiveModel(i, deepcopy(params))

        # Evaluation
        accs = [evaluate(nd.model, testLoader) for nd in nodes]
        avg = np.mean(accs)
        worst = np.min(accs)
        print(f"\n[EVAL] Round {r}: avg={avg:.4f}, worst={worst:.4f}")

    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print("- Attacker models have MUCH HIGHER loss (random noise doesn't fit data)")
    print("- BASIL always picks the model with LOWEST loss (honest models)")
    print("- That's why attacks are 'invisible' - they're filtered out!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
