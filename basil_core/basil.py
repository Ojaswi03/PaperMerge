# basil_core/basil.py
"""
BASIL: Byzantine-resilient decentralized training on a ring topology.

Implements Algorithm 1 from the BASIL paper:
- Sequential training over logical ring (one node active at a time)
- Each node stores S latest models from S counterclockwise neighbors
- Aggregation rule: select model with lowest local batch loss
- Multicast updated model to next S clockwise neighbors

Reference: "BASIL: A Fast and Byzantine-Resilient Approach for Decentralized Training"
"""

import sys
from copy import deepcopy

from .attacks import applyAttack, modelPoisonAttack
from .trainer import (
    localUpdate,
    evaluateBatchLoss,
    addChannelNoiseToParams,
    getParams,
    setParams,
    averageParams,
    makeLrScheduler,
    evaluate,
    evaluateAll,
)


class BasilNode:
    """
    Node in BASIL ring topology.

    Memory stores models from S counterclockwise neighbors (not own history).
    This is the key insight from the paper - Byzantine filtering works because
    we select among models from different neighbors.

    Paper reference: Definition 1, Equation 3.
    """
    def __init__(
        self,
        nodeId,
        model,
        dataLoader,
        S,
        noiseModel="none",
        sigma=0.0,
        lr0=0.05,
        localEpochs=1,
        ebmLambda=400.0,  # With sigma=0.05: scale = 1 + 400*0.0025 = 2.0
        wcmLambda=0.1,
        wcmSamples=5,
        wcmRho=0.5,
        momentum=0.0,  # Use 0.9 with EBM for high noise (σ=0.2)
        **kwargs,
    ):
        if "lr" in kwargs and kwargs["lr"] is not None:
            lr0 = kwargs["lr"]

        self.nodeId = nodeId
        self.model = model
        self.dataLoader = dataLoader
        self.S = int(S)
        self.noiseModel = str(noiseModel)
        self.sigma = float(sigma)
        self.lr0 = float(lr0)
        self.localEpochs = int(localEpochs)
        self.ebmLambda = float(ebmLambda)
        self.wcmLambda = float(wcmLambda)
        self.wcmSamples = int(wcmSamples)
        self.wcmRho = float(wcmRho)
        self.momentum = float(momentum)

        # Memory stores models from S counterclockwise neighbors
        # Key: sender node ID, Value: model params (list of numpy arrays)
        # We keep at most S entries (one per counterclockwise neighbor)
        self.neighborMemory = {}
        self.round = 0

    def receiveModel(self, senderId, params):
        # store received params under sender ID
        self.neighborMemory[senderId] = [p.copy() for p in params]
        # If we have more than S entries, remove the oldest
        # (oldest = smallest round distance, i.e., farthest counterclockwise)
        if len(self.neighborMemory) > self.S:
            # Remove the oldest entry (first inserted)
            oldest = next(iter(self.neighborMemory))
            del self.neighborMemory[oldest]

    def selectBestModel(self, verbose=False):
        # evaluate own model as the starting candidate
        # Candidates: current model + all received neighbor models
        currentParams = getParams(self.model)
        currentLoss = evaluateBatchLoss(self.model, self.dataLoader)

        bestParams = currentParams
        bestLoss = currentLoss
        bestSource = "self"

        allLosses = {"self": currentLoss}

        for senderId, params in self.neighborMemory.items():
            # Temporarily set params to evaluate
            setParams(self.model, params)
            loss = evaluateBatchLoss(self.model, self.dataLoader)
            allLosses[f"node_{senderId}"] = loss
            if loss < bestLoss:
                bestLoss = loss
                bestParams = [p.copy() for p in params]
                bestSource = f"node_{senderId}"

        # Restore to best params
        setParams(self.model, bestParams)

        if verbose:
            print(f"    [Node {self.nodeId}] Losses: {', '.join(f'{k}={v:.4f}' for k,v in allLosses.items())} → selected {bestSource}")

    def localTrain(self, lr, stepsPerEpoch=100):
        # delegate to trainer.localUpdate with all noise/mitigation settings
        localUpdate(
            self.model,
            self.dataLoader,
            epochs=self.localEpochs,
            lr=lr,
            noiseModel=self.noiseModel,
            sigma=self.sigma,
            ebmLambda=self.ebmLambda,
            wcmLambda=self.wcmLambda,
            wcmSamples=self.wcmSamples,
            wcmRho=self.wcmRho,
            stepsPerEpoch=stepsPerEpoch,
            momentum=self.momentum,
        )


def basilRingTrainingWithAttack(
    nodes,
    rounds,
    testLoader=None,
    attackTypes=("none",),
    attackerIds=None,
    hiddenStartRound=20,
    sigma=0.0,
    noiseModel="none",
    channelNoiseStart=0,
    lr0=0.05,
    lrAlpha=0.6,
    stepsPerEpoch=100,
    useSnapshots=True,
    useSequential=True,  # True = paper's sequential, False = parallel (faster but less accurate)
    stopCallback=None,
    useLrDecay=True,  # Set False for EBM with high noise + momentum
    roundCallback=None,  # called as roundCallback(roundNum, avgAcc, worstAcc, totalRounds)
    **kwargs,
):

    # Backward compatibility
    if "attack_type" in kwargs and kwargs["attack_type"] is not None:
        attackTypes = [kwargs["attack_type"]]

    if isinstance(attackTypes, str):
        attackTypes = [attackTypes]
    if not attackTypes:
        attackTypes = ["none"]

    n = len(nodes)
    channelNoiseStart = int(channelNoiseStart)
    attackers = set(attackerIds or [])

    # S = memory size (paper: S = b+1 where b = max Byzantine nodes)
    S = nodes[0].S if nodes else 10

    # EBM/WCM mitigation starts from round 0 to pre-condition model
    mitigation = noiseModel if noiseModel in ("ebm", "wcm") else "none"
    for nd in nodes:
        if mitigation != "none" and sigma > 0:
            nd.noiseModel = noiseModel
            nd.sigma = float(sigma)
        else:
            nd.noiseModel = "none"
            nd.sigma = 0.0
        nd.lr0 = float(lr0)

    # Learning rate: paper uses lr0 / (1 + lr0 * t) with lr0=0.03
    # For EBM with high noise + momentum, use useLrDecay=False for best results
    lrSched = makeLrScheduler(lr0, alpha=lrAlpha, useBasilSchedule=True, useLrDecay=useLrDecay)

    avgAccHist, worstAccHist = [], []

    for r in range(rounds):
        # Check stop callback
        if stopCallback is not None and stopCallback():
            print(f"[round {r}] STOP REQUESTED - terminating training early", flush=True)
            break

        # Channel noise only activates at channelNoiseStart
        channelNoiseActive = (r >= channelNoiseStart) and noiseModel not in ("none", "clean") and sigma > 0

        lr = lrSched(r)
        mitigationStr = f" {mitigation.upper()} training" if mitigation != "none" else ""
        if channelNoiseActive:
            noiseStatus = f" channel_noise(sigma={sigma}){' +' + mitigationStr if mitigationStr else ''}"
        else:
            noiseStatus = f"{mitigationStr}" if mitigationStr else " clean"

        # Determine current attack type (cycle through list)
        atk = attackTypes[r % len(attackTypes)]

        attackStr = f" | attack={atk}({len(attackers)} nodes)" if attackers and atk != "none" else ""
        print(f"[round {r}] lr={lr:.6f}{noiseStatus}{attackStr}...", flush=True)

        # Communication sigma
        commSigma = sigma if channelNoiseActive else 0.0

        if useSequential:
            # ===== PAPER'S SEQUENTIAL ALGORITHM (Algorithm 1) =====
            # Process nodes one at a time around the ring
            for i in range(n):
                nd = nodes[i]

                # Step 1: Handle received models
                if nd.neighborMemory:
                    if useSnapshots:
                        # BASIL: Select best model from memory
                        nd.selectBestModel()
                    else:
                        # No BASIL: adopt predecessor's model
                        predecessorId = (i - 1) % n
                        if predecessorId in nd.neighborMemory:
                            setParams(nd.model, nd.neighborMemory[predecessorId])

                # Step 2: Local training
                nd.localTrain(lr=lr, stepsPerEpoch=stepsPerEpoch)

                # Step 3: Get updated params and apply channel noise
                params = getParams(nd.model)
                noisyParams = addChannelNoiseToParams(params, sigma=commSigma)

                # Step 4: Apply attack if this node is Byzantine
                if i in attackers:
                    if atk == "hidden" and r < hiddenStartRound:
                        pass  # Hidden attack not yet active
                    elif atk == "model_poison":
                        # Model poisoning: compute poisoned params to SEND, but restore
                        # nd.model to its trained state so it doesn't corrupt its own future training
                        setParams(nd.model, noisyParams)
                        trainedParams = [p.copy() for p in noisyParams]
                        noisyParams = modelPoisonAttack(nd.model, nd.dataLoader)
                        setParams(nd.model, trainedParams)  # restore own model
                    else:
                        noisyParams = applyAttack(noisyParams, atk)

                # Step 5: Multicast to next S clockwise neighbors (paper's key feature)
                for offset in range(1, S + 1):
                    j = (i + offset) % n
                    nodes[j].receiveModel(i, deepcopy(noisyParams))

        else:
            # ===== PARALLEL TRAINING (faster but less accurate) =====
            # All nodes train simultaneously, then communicate
            for i, nd in enumerate(nodes):
                if nd.neighborMemory:
                    if useSnapshots:
                        # BASIL: Select best model from memory
                        nd.selectBestModel()
                    else:
                        # No BASIL: adopt predecessor's model
                        predecessorId = (i - 1) % n
                        if predecessorId in nd.neighborMemory:
                            setParams(nd.model, nd.neighborMemory[predecessorId])
                nd.localTrain(lr=lr, stepsPerEpoch=stepsPerEpoch)

            # Communication phase
            for i in range(n):
                params = getParams(nodes[i].model)
                noisyParams = addChannelNoiseToParams(params, sigma=commSigma)

                if i in attackers:
                    if atk == "hidden" and r < hiddenStartRound:
                        pass
                    elif atk == "model_poison":
                        setParams(nodes[i].model, noisyParams)
                        trainedParams = [p.copy() for p in noisyParams]
                        noisyParams = modelPoisonAttack(nodes[i].model, nodes[i].dataLoader)
                        setParams(nodes[i].model, trainedParams)  # restore own model
                    else:
                        noisyParams = applyAttack(noisyParams, atk)

                # Multicast to next S clockwise neighbors
                for offset in range(1, S + 1):
                    j = (i + offset) % n
                    nodes[j].receiveModel(i, deepcopy(noisyParams))

        # Evaluation
        if testLoader is not None:
            avg, worst, _ = evaluateAll(nodes, testLoader)
            print(f"[round {r}] eval avg={avg:.4f}", flush=True)
            avgAccHist.append(avg)
            worstAccHist.append(worst)
            if roundCallback is not None:
                roundCallback(r + 1, avg, worst, rounds)

    return avgAccHist, worstAccHist


# ============================================================================
# BACKWARD COMPATIBILITY: Simple ring training without BASIL features
# ============================================================================

def simpleRingTraining(
    nodes,
    rounds,
    testLoader=None,
    sigma=0.0,
    noiseModel="none",
    channelNoiseStart=0,
    lr0=0.05,
    stepsPerEpoch=100,
    stopCallback=None,
):
    # baseline wrapper: no snapshot selection, no S-neighbor multicast
    return basilRingTrainingWithAttack(
        nodes=nodes,
        rounds=rounds,
        testLoader=testLoader,
        attackTypes=["none"],
        attackerIds=None,
        sigma=sigma,
        noiseModel=noiseModel,
        channelNoiseStart=channelNoiseStart,
        lr0=lr0,
        stepsPerEpoch=stepsPerEpoch,
        useSnapshots=False,
        useSequential=False,
        stopCallback=stopCallback,
    )


# ============================================================================
# FEDAVG TRAINING: For Noisy Channel Paper (002)
# Parallel training + Averaging (matches paper's system model)
# ============================================================================

def fedAvgTrainingWithNoise(
    nodes,
    rounds,
    testLoader=None,
    attackTypes=("none",),
    attackerIds=None,
    hiddenStartRound=20,
    sigma=0.0,
    noiseModel="none",
    channelNoiseStart=0,
    lr0=0.03,
    lrAlpha=0.6,
    localEpochs=1,
    stepsPerEpoch=100,
    stopCallback=None,
    useLrDecay=True,  # Set False for EBM with high noise + momentum
    roundCallback=None,  # called as roundCallback(roundNum, avgAcc, worstAcc, totalRounds)
    **kwargs,
):
    # Backward compatibility
    if isinstance(attackTypes, str):
        attackTypes = [attackTypes]
    if not attackTypes:
        attackTypes = ["none"]

    n = len(nodes)
    attackers = set(attackerIds or [])
    channelNoiseStart = int(channelNoiseStart)

    # Configure nodes for EBM/WCM training
    mitigation = noiseModel if noiseModel in ("ebm", "wcm") else "none"
    for nd in nodes:
        if mitigation != "none" and sigma > 0:
            nd.noiseModel = noiseModel
            nd.sigma = float(sigma)
        else:
            nd.noiseModel = "none"
            nd.sigma = 0.0
        nd.lr0 = float(lr0)
        nd.localEpochs = int(localEpochs)

    # Learning rate scheduler (paper uses lr0 / (1 + lr0 * t))
    # For EBM with high noise + momentum, use useLrDecay=False for best results
    lrSched = makeLrScheduler(lr0, alpha=lrAlpha, useBasilSchedule=True, useLrDecay=useLrDecay)

    avgAccHist, worstAccHist = [], []

    # Initialize global model from node 0
    globalParams = getParams(nodes[0].model)

    # Synchronize all nodes to same initial model
    for nd in nodes:
        setParams(nd.model, globalParams)

    for r in range(rounds):
        # Check stop callback
        if stopCallback is not None and stopCallback():
            print(f"[round {r}] STOP REQUESTED - terminating training early", flush=True)
            break

        lr = lrSched(r)
        channelNoiseActive = (r >= channelNoiseStart) and noiseModel not in ("none", "clean") and sigma > 0

        mitigationStr = f" {mitigation.upper()}" if mitigation != "none" else ""
        noiseStatus = f" channel_noise(σ={sigma})" if channelNoiseActive else ""

        # Determine current attack type
        atk = attackTypes[r % len(attackTypes)]

        attackStr = f" | attack={atk}({len(attackers)} nodes)" if attackers and atk != "none" else ""
        print(f"[round {r}] lr={lr:.6f}{noiseStatus}{attackStr} parallel training...", flush=True)

        # ===== STEP 1: ALL NODES TRAIN IN PARALLEL =====
        # Each node starts from the same global model
        localUpdates = []
        dataWeights = []

        for i, nd in enumerate(nodes):
            # Set global model as starting point
            setParams(nd.model, globalParams)

            # Local training (with EBM/WCM if enabled)
            nd.localTrain(lr=lr, stepsPerEpoch=stepsPerEpoch)

            # Get updated local model
            localParams = getParams(nd.model)

            # Apply Byzantine attack if this node is an attacker
            if i in attackers:
                if atk == "hidden" and r < hiddenStartRound:
                    pass  # Hidden attack not yet active
                elif atk == "model_poison":
                    setParams(nd.model, localParams)
                    savedParams = [p.copy() for p in localParams]
                    localParams = modelPoisonAttack(nd.model, nd.dataLoader)
                    setParams(nd.model, savedParams)  # restore own model
                else:
                    localParams = applyAttack(localParams, atk)

            localUpdates.append(localParams)
            # Weight by data size (approximate: use equal weights for now)
            dataWeights.append(1.0)

        # ===== STEP 2: AVERAGE ALL LOCAL MODELS (Eq 3a) =====
        avgParams = averageParams(localUpdates, weights=dataWeights)

        # ===== STEP 3: ADD CHANNEL NOISE (Paper 002) =====
        # Noise added to broadcast when server sends to all nodes
        if channelNoiseActive:
            noisyParams = addChannelNoiseToParams(avgParams, sigma=sigma)
        else:
            noisyParams = avgParams

        # ===== STEP 4: UPDATE GLOBAL MODEL =====
        globalParams = noisyParams

        # Synchronize all nodes to new global model (for evaluation)
        for nd in nodes:
            setParams(nd.model, globalParams)

        # ===== EVALUATION =====
        if testLoader is not None:
            # All nodes have same model, so just evaluate one
            acc = evaluate(nodes[0].model, testLoader)
            print(f"[round {r}] eval acc={acc:.4f}", flush=True)
            avgAccHist.append(acc)
            worstAccHist.append(acc)  # Same as avg since all nodes have same model
            if roundCallback is not None:
                roundCallback(r + 1, acc, acc, rounds)

    return avgAccHist, worstAccHist
