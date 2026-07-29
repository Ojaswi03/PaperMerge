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
import math
import tensorflow as tf
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
    lossFn,
    _iterLimited,
)


MAX_TRAIN_MICRO_BATCH_SIZE = 128


def _batchGradients(model_inner, weights, x, y, microBatchSize=MAX_TRAIN_MICRO_BATCH_SIZE):
    """Compute one effective full-batch gradient using smaller activation slices.

    This keeps the configured batch size unchanged while reducing peak GPU
    activation memory for CIFAR. The gradients are weighted by slice size, so
    for loss functions that average over the batch this is equivalent to one
    full-batch gradient update.
    """
    batchSize = x.shape[0]
    if batchSize is None or batchSize <= microBatchSize:
        with tf.GradientTape() as tape:
            logits = model_inner(x, training=True)
            loss = lossFn(y, logits)
        grads = tape.gradient(loss, weights)
        return [
            tf.zeros_like(w) if g is None else g
            for g, w in zip(grads, weights)
        ]

    gradSums = [tf.zeros_like(w) for w in weights]
    total = float(batchSize)
    for start in range(0, int(batchSize), int(microBatchSize)):
        end = min(start + int(microBatchSize), int(batchSize))
        with tf.GradientTape() as tape:
            logits = model_inner(x[start:end], training=True)
            loss = lossFn(y[start:end], logits)
        grads = tape.gradient(loss, weights)
        factor = float(end - start) / total
        gradSums = [
            acc + (tf.zeros_like(w) if g is None else g) * factor
            for acc, g, w in zip(gradSums, grads, weights)
        ]
    return gradSums


class _MutableLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule backed by a tf.Variable so we can update LR
    between rounds without triggering a @tf.function retrace."""
    def __init__(self, initial_lr):
        super().__init__()
        self._var = tf.Variable(float(initial_lr), trainable=False, dtype=tf.float32)

    def __call__(self, step):
        return self._var   # TF reads the variable value dynamically each step

    def assign(self, value):
        self._var.assign(float(value))

    def get_config(self):
        return {"initial_lr": float(self._var.numpy())}


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
        ebmLambda=25.0,   # With sigma=0.2: scale = 1 + 25*0.04 = 2.0
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
        self.neighborMemory = {}
        self.round = 0

        # Compiled training step - built lazily on first localTrain call so that
        # noiseModel/sigma (which may be overwritten by basilRingTrainingWithAttack
        # before training starts) are finalised before we compile.
        self._lrSchedule = _MutableLR(float(lr0))
        self._opt = None
        self._compiledStep = None

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
        """Select only among predecessor snapshots using one shared mini-batch."""
        if not self.neighborMemory:
            return "self"

        for xBatch, yBatch in _iterLimited(self.dataLoader, maxBatches=1):
            selectionBatch = (
                tf.cast(xBatch, tf.float32),
                tf.cast(yBatch, tf.int32),
            )
            break
        else:
            return "self"

        bestParams = None
        bestLoss = float("inf")
        bestSource = None
        allLosses = {}
        for senderId, params in self.neighborMemory.items():
            setParams(self.model, params)
            logits = self.model(selectionBatch[0], training=False)
            loss = float(lossFn(selectionBatch[1], logits).numpy())
            allLosses[f"node_{senderId}"] = loss
            if loss < bestLoss:
                bestLoss = loss
                bestParams = [p.copy() for p in params]
                bestSource = f"node_{senderId}"

        if bestParams is not None:
            setParams(self.model, bestParams)

        if verbose:
            print(f"    [Node {self.nodeId}] Losses: {', '.join(f'{k}={v:.4f}' for k,v in allLosses.items())} → selected {bestSource}")
        return bestSource


    def _ensureCompiled(self):
        """Build the compiled train step the first time it is needed.
        Called lazily so noiseModel/sigma are set before we compile."""
        if self._compiledStep is not None:
            return
        self._opt = tf.keras.optimizers.SGD(
            learning_rate=self._lrSchedule, momentum=self.momentum
        )
        model_inner = self.model.model   # the underlying Keras Sequential
        opt = self._opt

        if self.noiseModel == "ebm" and self.sigma > 0:
            # Scale = 1 + λσ²  (constant for the lifetime of this node)
            scale = tf.constant(
                1.0 + self.ebmLambda * self.sigma * self.sigma, dtype=tf.float32
            )
            def _step_fn(x, y):
                weights = model_inner.trainable_weights
                grads = _batchGradients(model_inner, weights, x, y)
                # Clip raw gradients first, then apply EBM scale.
                # Clipping AFTER scaling was halving the effective clip threshold
                # (clip=5 on 2x-scaled grads = clip=2.5 on real grads), which
                # severely under-stepped at the start of training.
                clipped, _ = tf.clip_by_global_norm(grads, 5.0)
                scaled = [g * scale for g in clipped]
                opt.apply_gradients(
                    zip(scaled, weights)
                )
        else:
            def _step_fn(x, y):
                weights = model_inner.trainable_weights
                grads = _batchGradients(model_inner, weights, x, y)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                opt.apply_gradients(zip(grads, weights))

        self._compiledStep = tf.function(_step_fn)

    def _resetOptimizerSlots(self):
        """Zero momentum buffers so each round starts fresh (matches original semantics).
        Must skip: 'iteration' step counter (Keras 3 name) and rank-0 scalar variables
        (includes the LR tf.Variable) — zeroing those kills learning from round 1."""
        if self._opt is None:
            return
        for v in self._opt.variables:
            if 'iteration' in v.name:   # Keras 3 uses 'iteration', not 'iterations'
                continue
            if v.shape.rank == 0:       # skip scalars — includes the LR tf.Variable
                continue
            v.assign(tf.zeros_like(v))

    def localTrain(self, lr, stepsPerEpoch=100):
        # WCM requires custom gradient logic - fall back to existing path
        if self.noiseModel == "wcm" and self.sigma > 0:
            localUpdate(
                self.model, self.dataLoader,
                epochs=self.localEpochs, lr=lr,
                noiseModel=self.noiseModel, sigma=self.sigma,
                wcmLambda=self.wcmLambda, wcmSamples=self.wcmSamples,
                wcmRho=self.wcmRho, stepsPerEpoch=stepsPerEpoch,
                momentum=self.momentum,
            )
            return

        # Compiled path: forward + backward + optimizer in a single TF graph.
        # Compiled once per node on first call; reused every round.
        self._ensureCompiled()
        self._lrSchedule.assign(lr)
        self._resetOptimizerSlots()  # fresh momentum each round

        # Single continuous iterator for all epochs combined - the dataset uses
        # .repeat() so it never exhausts, and we do exactly localEpochs*stepsPerEpoch
        # steps with one iterator creation instead of localEpochs separate ones.
        totalSteps = self.localEpochs * stepsPerEpoch
        for xBatch, yBatch in _iterLimited(self.dataLoader, maxBatches=totalSteps):
            self._compiledStep(
                tf.cast(xBatch, tf.float32),
                tf.cast(yBatch, tf.int32),
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
    aggregationMode="handoff",
    stopCallback=None,
    useLrDecay=True,  # Set False for EBM with high noise + momentum
    usePlateauLr=False,   # Reduce LR when accuracy stops improving
    plateauPatience=10,   # Rounds of no improvement before reducing
    plateauFactor=0.5,    # Multiply LR by this on plateau
    plateauMinLr=1e-4,    # Floor for plateau reductions
    plateauThreshold=0.002, # Minimum improvement to count as progress
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
    if aggregationMode not in ("handoff", "consensus"):
        raise ValueError(f"Unknown aggregationMode: {aggregationMode}")

    n = len(nodes)
    channelNoiseStart = int(channelNoiseStart)
    attackers = set(attackerIds or [])

    # S = memory size for BASIL snapshot selection (paper: S = b+1).
    # Consensus mode is a separate decentralized averaging baseline and needs
    # full-ring mixing rather than one-predecessor handoff.
    S = nodes[0].S if nodes else 10
    fanout = n - 1 if aggregationMode == "consensus" else S

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

    # Plateau LR state
    _plateauLr        = float(lr0)
    _plateauBestAcc   = -1.0
    _plateauPatience  = 0
    _plateauCooldown  = 0  # rounds to wait after a reduction before checking again

    avgAccHist, worstAccHist = [], []

    for r in range(rounds):
        # Check stop callback
        if stopCallback is not None and stopCallback():
            print(f"[round {r}] STOP REQUESTED - terminating training early", flush=True)
            break

        # Channel noise only activates at channelNoiseStart
        channelNoiseActive = (r >= channelNoiseStart) and noiseModel not in ("none", "clean") and sigma > 0

        lr = _plateauLr if usePlateauLr else lrSched(r)
        mitigationStr = f" {mitigation.upper()} training" if mitigation != "none" else ""
        if channelNoiseActive:
            noiseStatus = f" channel_noise(sigma={sigma}){' +' + mitigationStr if mitigationStr else ''}"
        else:
            noiseStatus = f"{mitigationStr}" if mitigationStr else " clean"

        # Determine current attack type (cycle through list)
        atk = attackTypes[r % len(attackTypes)]
        cleanConsensusRound = (
            aggregationMode == "consensus"
            and not useSnapshots
            and not attackers
            and not channelNoiseActive
            and mitigation == "none"
            and atk in ("none", "clean")
        )

        attackStr = f" | attack={atk}({len(attackers)} nodes)" if attackers and atk != "none" else ""
        print(f"[round {r}] lr={lr:.6f}{noiseStatus}{attackStr}...", flush=True)

        # Every mitigation arm is exposed to the same configured link noise.
        commSigma = sigma if channelNoiseActive else 0.0

        if cleanConsensusRound:
            # Clean reference: no adversaries, no channel noise, no mitigation.
            # Every communicated update is correct, so the consensus result is
            # exactly the average of all locally trained node models.
            for nd in nodes:
                nd.localTrain(lr=lr, stepsPerEpoch=stepsPerEpoch)
        elif useSequential:
            # ===== PAPER'S SEQUENTIAL ALGORITHM (Algorithm 1) =====
            # Process nodes one at a time around the ring
            for i in range(n):
                nd = nodes[i]

                # Step 1: Handle received models
                if nd.neighborMemory:
                    if useSnapshots:
                        # BASIL: Select best model from memory
                        nd.selectBestModel()
                    elif aggregationMode == "consensus":
                        # Consensus/gossip-inspired baseline: mix the node's
                        # current model with received neighbor models before SGD.
                        paramsList = [getParams(nd.model)]
                        paramsList.extend(
                            [p.copy() for p in params]
                            for params in nd.neighborMemory.values()
                        )
                        setParams(nd.model, averageParams(paramsList))
                    else:
                        # R-plain baseline from the BASIL paper: adopt the
                        # immediate counterclockwise predecessor's model.
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

                # Step 5: Multicast clockwise. BASIL uses S predecessors;
                # consensus mode circulates to the full ring.
                for offset in range(1, fanout + 1):
                    j = (i + offset) % n
                    nodes[j].receiveModel(i, noisyParams)

        else:
            # ===== PARALLEL TRAINING (faster but less accurate) =====
            # All nodes train simultaneously, then communicate
            for i, nd in enumerate(nodes):
                if nd.neighborMemory:
                    if useSnapshots:
                        # BASIL: Select best model from memory
                        nd.selectBestModel()
                    elif aggregationMode == "consensus":
                        paramsList = [getParams(nd.model)]
                        paramsList.extend(
                            [p.copy() for p in params]
                            for params in nd.neighborMemory.values()
                        )
                        setParams(nd.model, averageParams(paramsList))
                    else:
                        # R-plain baseline from the BASIL paper.
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

                # Multicast clockwise. BASIL uses S predecessors;
                # consensus mode circulates to the full ring.
                for offset in range(1, fanout + 1):
                    j = (i + offset) % n
                    nodes[j].receiveModel(i, noisyParams)

        if cleanConsensusRound and nodes:
            consensusParams = averageParams([getParams(nd.model) for nd in nodes])
            for nd in nodes:
                setParams(nd.model, [p.copy() for p in consensusParams])

        # Evaluation - maxBatches=5 (~2500 samples) is fast enough for per-round tracking
        if testLoader is not None:
            avg, worst, _ = evaluateAll(nodes, testLoader, maxBatches=5)
            print(f"[round {r}] eval avg={avg:.4f}", flush=True)
            avgAccHist.append(avg)
            worstAccHist.append(worst)
            if roundCallback is not None:
                roundCallback(r + 1, avg, worst, rounds)

            # Plateau LR check (only after model clears 25% baseline)
            if usePlateauLr:
                if avg <= 0.25:
                    _plateauBestAcc  = -1.0
                    _plateauPatience = 0
                elif _plateauCooldown > 0:
                    _plateauCooldown -= 1
                elif avg > _plateauBestAcc + plateauThreshold:
                    _plateauBestAcc  = avg
                    _plateauPatience = 0
                else:
                    _plateauPatience += 1
                    if _plateauPatience >= plateauPatience:
                        new_lr = max(plateauMinLr, _plateauLr * plateauFactor)
                        if new_lr < _plateauLr:
                            print(f"[round {r}] ReduceLROnPlateau: {_plateauLr:.6f} → {new_lr:.6f}", flush=True)
                            _plateauLr = new_lr
                        _plateauPatience = 0
                        _plateauCooldown = plateauPatience // 2

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
    usePlateauLr=False,
    plateauPatience=10,
    plateauFactor=0.5,
    plateauMinLr=1e-4,
    plateauThreshold=0.002,
    useSnapshots=False,  # Server-side SS: discard worst S models before averaging
    S=1,                 # Number of models to discard (= basilMemorySize)
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

    # Plateau LR state
    _plateauLr        = float(lr0)
    _plateauBestAcc   = -1.0
    _plateauPatience  = 0
    _plateauCooldown  = 0

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

        lr = _plateauLr if usePlateauLr else lrSched(r)
        channelNoiseActive = (r >= channelNoiseStart) and noiseModel not in ("none", "clean") and sigma > 0

        mitigationStr = f" {mitigation.upper()}" if mitigation != "none" else ""
        noiseStatus = f" channel_noise(σ={sigma})" if channelNoiseActive else ""
        ssStr = f" SS(discard={S})" if useSnapshots else ""

        # Determine current attack type
        atk = attackTypes[r % len(attackTypes)]

        attackStr = f" | attack={atk}({len(attackers)} nodes)" if attackers and atk != "none" else ""
        print(f"[round {r}] lr={lr:.6f}{noiseStatus}{ssStr}{attackStr} parallel training...", flush=True)

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

        # ===== STEP 2: AVERAGE LOCAL MODELS (with optional server-side SS) =====
        if useSnapshots and testLoader is not None and len(localUpdates) > S:
            # Evaluate each submitted model on one test batch (balanced, server-held)
            # Use node 0's model as scratch space — it gets reset to globalParams next round
            evalNode = nodes[0]
            savedParams = getParams(evalNode.model)
            losses = []
            for params in localUpdates:
                setParams(evalNode.model, params)
                losses.append(evaluateBatchLoss(evalNode.model, testLoader))
            setParams(evalNode.model, savedParams)
            # Keep the n-S lowest-loss models (discard S worst)
            order = sorted(range(len(losses)), key=lambda i: losses[i])
            kept = order[:len(localUpdates) - S]
            print(f"[round {r}] SS discarded nodes {[order[i] for i in range(len(localUpdates)-S, len(localUpdates))]} (highest loss)", flush=True)
            avgParams = averageParams([localUpdates[i] for i in kept],
                                      weights=[dataWeights[i] for i in kept])
        else:
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

            # Plateau LR check (only after model clears 25% baseline)
            if usePlateauLr:
                if acc <= 0.25:
                    _plateauBestAcc  = -1.0
                    _plateauPatience = 0
                elif _plateauCooldown > 0:
                    _plateauCooldown -= 1
                elif acc > _plateauBestAcc + plateauThreshold:
                    _plateauBestAcc  = acc
                    _plateauPatience = 0
                else:
                    _plateauPatience += 1
                    if _plateauPatience >= plateauPatience:
                        new_lr = max(plateauMinLr, _plateauLr * plateauFactor)
                        if new_lr < _plateauLr:
                            print(f"[round {r}] ReduceLROnPlateau: {_plateauLr:.6f} → {new_lr:.6f}", flush=True)
                            _plateauLr = new_lr
                        _plateauPatience = 0
                        _plateauCooldown = plateauPatience // 2

    return avgAccHist, worstAccHist
