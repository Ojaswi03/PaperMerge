# basil_core/cart.py
"""
CART: Class-Aware Ring Training (Paper 003)

Extends the BASIL ring topology with:
1. ClassRegistry — per-class best accuracy tracked and propagated around the ring
2. Class-aware proximal regularisation — prevents catastrophic forgetting of
   classes the node doesn't see much (non-IID fix)
3. Class-aware Snapshot Selection — harder to fool than overall-loss SS

Algorithm per node per round
-----------------------------
1. Receive (model_params, registry) from ring.
2. (Optional) Snapshot Selection: evaluate each candidate per-class; select best.
3. Compute per-class accuracy of selected model on local data → ref_class_acc.
4. Compute trust weights: trust[c] = max(0, ref_class_acc[c] - my_class_acc[c])
5. Train with proximal term:
       L = cross_entropy(x, y) + (proximal_mu / 2) * ||w - w_ref||²
   where  proximal_mu = distillStrength × max(trust_weights)
6. After training, evaluate per-class accuracy → update own registry.
7. Merge received registry (verified), update with own per-class accuracy.
8. Send (new_params, updated_registry) to next S clockwise nodes.
"""

import math
import numpy as np
import tensorflow as tf

from .attacks import applyAttack, modelPoisonAttack
from .class_registry import ClassRegistry
from .trainer import (
    addChannelNoiseToParams,
    evaluateBatchLoss,
    evaluatePerClass,
    getParams,
    setParams,
    averageParams,
    makeLrScheduler,
    evaluate,
    evaluateAll,
    lossFn,
    _iterLimited,
)
from .basil import BasilNode, _MutableLR


# ---------------------------------------------------------------------------
# CARTNode
# ---------------------------------------------------------------------------

class CARTNode(BasilNode):
    """BasilNode extended with ClassRegistry and class-aware proximal training."""

    def __init__(
        self,
        nodeId,
        model,
        dataLoader,
        S,
        nClasses=10,
        distillStrength=0.5,    # γ: scales proximal coefficient
        verifyThreshold=0.05,   # tolerance for registry Byzantine verification
        **kwargs,
    ):
        super().__init__(nodeId=nodeId, model=model, dataLoader=dataLoader, S=S, **kwargs)
        self.nClasses = int(nClasses)
        self.distillStrength = float(distillStrength)
        self.verifyThreshold = float(verifyThreshold)

        # Per-class accuracy from last evaluation (starts at 0)
        self.classAcc = np.zeros(nClasses, dtype=np.float32)

        # Ring registry — travels with the model
        self.classRegistry = ClassRegistry(nClasses=nClasses)

        # EMA-smoothed mean trust weight — prevents round-to-round mu oscillation
        # that causes the zigzag pattern in accuracy curves.
        # Decay=0.85: trust signal responds in ~7 rounds, not 1.
        self._emaTrust = 0.0
        self._emaDecay = 0.85

        # Compiled proximal step — rebuilt when mu changes significantly
        self._proxMu = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._refParams = None          # reference weights as tf.Variables (built lazily)
        self._cartStep = None           # compiled tf.function for proximal update

    # ------------------------------------------------------------------
    # Class-aware snapshot selection
    # ------------------------------------------------------------------

    def selectBestModelClassAware(self, verbose=False, maxBatches=1):
        """Select from memory using overall loss (same as standard SS).

        Enhancement over standard SS: after selection, return the per-class
        accuracy of the winner so the caller can verify registry claims.
        The class-aware aspect comes from how trust weights are computed
        downstream — the selection itself still uses overall loss to keep
        this O(S) rather than O(S × nClasses).
        """
        self.selectBestModel(verbose=verbose)
        # Evaluate per-class accuracy of the selected model
        return evaluatePerClass(
            self.model,
            self.dataLoader,
            nClasses=self.nClasses,
            maxBatches=maxBatches,
        )

    # ------------------------------------------------------------------
    # Proximal training (class-aware FedProx)
    # ------------------------------------------------------------------

    def _buildRefParams(self):
        """Build a list of tf.Variables that mirror the model's trainable weights.
        Used as the proximal reference point (frozen during local training)."""
        return [
            tf.Variable(w.numpy(), trainable=False, dtype=tf.float32)
            for w in self.model.trainable_weights
        ]

    def _ensureCartCompiled(self):
        """Build the compiled proximal train step."""
        if self._cartStep is not None:
            return
        if self._refParams is None:
            self._refParams = self._buildRefParams()

        # Ensure the base optimizer is ready
        self._ensureCompiled()

        model_inner = self.model.model
        opt = self._opt
        refParams = self._refParams
        proxMu = self._proxMu

        ebmEnabled = (self.noiseModel == "ebm" and self.sigma > 0)
        if ebmEnabled:
            scale = tf.constant(
                1.0 + self.ebmLambda * self.sigma * self.sigma, dtype=tf.float32
            )

        def _cart_step(x, y):
            with tf.GradientTape() as tape:
                logits = model_inner(x, training=True)
                ce_loss = lossFn(y, logits)
                # Proximal term: (mu/2) * ||w - w_ref||²
                prox = tf.add_n([
                    tf.reduce_sum(tf.square(w - r))
                    for w, r in zip(model_inner.trainable_weights, refParams)
                ])
                loss = ce_loss + (proxMu / 2.0) * prox
            grads = tape.gradient(loss, model_inner.trainable_weights)
            if ebmEnabled:
                grads = [g * scale for g in grads]
            clipped, _ = tf.clip_by_global_norm(grads, 5.0)
            opt.apply_gradients(zip(clipped, model_inner.trainable_weights))

        self._cartStep = tf.function(_cart_step)

    def cartLocalTrain(self, lr, trustWeights, stepsPerEpoch=100):
        """Local training with class-aware proximal regularisation.

        Parameters
        ----------
        lr : float
        trustWeights : np.ndarray[nClasses] — output of ClassRegistry.trustWeights()
        stepsPerEpoch : int
        """
        # WCM path: fall back to base training (no proximal for WCM)
        if self.noiseModel == "wcm" and self.sigma > 0:
            self.localTrain(lr=lr, stepsPerEpoch=stepsPerEpoch)
            return

        # Proximal coefficient: base + EMA-smoothed class-aware amplification.
        #
        # Raw mean_trust jumps sharply each round as the registry fills,
        # causing the proximal mu to oscillate → zigzag accuracy curves.
        #
        # Fix: EMA-smooth mean_trust before scaling.
        #   ema_trust(t) = 0.85 × ema_trust(t-1) + 0.15 × mean_trust(t)
        #
        # This averages over the last ~7 rounds, damping per-round registry
        # fluctuations while still responding to genuine knowledge gaps.
        #
        #   mu = γ × (1 + ema_trust)
        #   cold start  → mu = γ × 1.0  (always active — prevents forgetting)
        #   registry ↑  → mu grows smoothly as class gaps accumulate
        raw_mean_trust = float(np.mean(trustWeights))
        self._emaTrust = self._emaDecay * self._emaTrust + (1.0 - self._emaDecay) * raw_mean_trust
        mu = float(self.distillStrength * (1.0 + self._emaTrust))

        # Freeze reference point = current model params (before local training)
        self._ensureCartCompiled()
        self._lrSchedule.assign(lr)
        self._resetOptimizerSlots()

        # Update ref params to current model state
        if self._refParams is None:
            self._refParams = self._buildRefParams()
            self._cartStep = None  # force rebuild with correct refParams
            self._ensureCartCompiled()
        for ref, w in zip(self._refParams, self.model.trainable_weights):
            ref.assign(w)

        # Update proximal coefficient (tf.Variable so no retrace)
        self._proxMu.assign(mu)

        totalSteps = self.localEpochs * stepsPerEpoch
        for xBatch, yBatch in _iterLimited(self.dataLoader, maxBatches=totalSteps):
            self._cartStep(
                tf.cast(xBatch, tf.float32),
                tf.cast(yBatch, tf.int32),
            )



# ---------------------------------------------------------------------------
# CART ring training loop
# ---------------------------------------------------------------------------

def cartRingTraining(
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
    aggregationMode="consensus",
    stopCallback=None,
    useLrDecay=True,
    usePlateauLr=False,
    plateauPatience=10,
    plateauFactor=0.5,
    plateauMinLr=1e-4,
    plateauThreshold=0.002,
    roundCallback=None,
    nClasses=10,
    verifyThreshold=0.05,
    evalMaxBatches=5,
    classEvalMaxBatches=2,
    **kwargs,
):
    """CART ring training: ring topology + class-aware registry + proximal distillation.

    Same interface as basilRingTrainingWithAttack but uses CARTNode features.
    Plain BasilNode instances are silently wrapped into CARTNode if needed.
    """
    if isinstance(attackTypes, str):
        attackTypes = [attackTypes]
    if not attackTypes:
        attackTypes = ["none"]
    if aggregationMode not in ("handoff", "consensus"):
        raise ValueError(f"Unknown aggregationMode: {aggregationMode}")

    n = len(nodes)
    channelNoiseStart = int(channelNoiseStart)
    attackers = set(attackerIds or [])
    S = nodes[0].S if nodes else 10
    fanout = n - 1 if aggregationMode == "consensus" else S

    # Promote plain BasilNodes to CARTNodes if necessary
    cartNodes = []
    for nd in nodes:
        if not isinstance(nd, CARTNode):
            cartNd = CARTNode(
                nodeId=nd.nodeId,
                model=nd.model,
                dataLoader=nd.dataLoader,
                S=nd.S,
                nClasses=nClasses,
                distillStrength=getattr(nd, "distillStrength", 0.5),
                verifyThreshold=verifyThreshold,
                noiseModel=nd.noiseModel,
                sigma=nd.sigma,
                lr0=nd.lr0,
                localEpochs=nd.localEpochs,
                ebmLambda=nd.ebmLambda,
                momentum=nd.momentum,
            )
            cartNd.neighborMemory = nd.neighborMemory
            cartNd.round = nd.round
            cartNodes.append(cartNd)
        else:
            cartNodes.append(nd)
    nodes = cartNodes

    # Configure EBM/WCM
    mitigation = noiseModel if noiseModel in ("ebm", "wcm") else "none"
    for nd in nodes:
        if mitigation != "none" and sigma > 0:
            nd.noiseModel = noiseModel
            nd.sigma = float(sigma)
        else:
            nd.noiseModel = "none"
            nd.sigma = 0.0
        nd.lr0 = float(lr0)

    lrSched = makeLrScheduler(lr0, alpha=lrAlpha, useBasilSchedule=True, useLrDecay=useLrDecay)

    # Plateau LR state
    _plateauLr = float(lr0)
    _plateauBestAcc = -1.0
    _plateauPatience = 0
    _plateauCooldown = 0

    avgAccHist, worstAccHist = [], []

    # Each node stores the registry it will send clockwise
    # Initialise: each node has its own freshly zeroed registry
    nodeRegistries = [nd.classRegistry.clone() for nd in nodes]

    for r in range(rounds):
        if stopCallback is not None and stopCallback():
            print(f"[CART round {r}] STOP REQUESTED", flush=True)
            break

        channelNoiseActive = (r >= channelNoiseStart) and noiseModel not in ("none", "clean") and sigma > 0
        lr = _plateauLr if usePlateauLr else lrSched(r)

        mitigationStr = f" {mitigation.upper()}" if mitigation != "none" else ""
        noiseStatus = f" channel_noise(σ={sigma})" if channelNoiseActive else ""
        atk = attackTypes[r % len(attackTypes)]
        attackStr = f" | attack={atk}({len(attackers)} nodes)" if attackers and atk != "none" else ""
        ssStr = " SS" if useSnapshots else ""
        print(f"[CART round {r}] lr={lr:.6f}{noiseStatus}{ssStr}{mitigationStr}{attackStr}...", flush=True)

        # EBM uses a stricter per-hop budget so the effective round-level
        # ring noise remains learnable.
        if channelNoiseActive and noiseModel == "ebm":
            commSigma = sigma / n
        else:
            commSigma = sigma if channelNoiseActive else 0.0

        # ===== SEQUENTIAL RING =====
        for i in range(n):
            nd = nodes[i]

            # --- Step 1: Select starting model ---
            if nd.neighborMemory:
                if useSnapshots:
                    refClassAcc = nd.selectBestModelClassAware(maxBatches=classEvalMaxBatches)
                elif aggregationMode == "consensus":
                    paramsList = [getParams(nd.model)]
                    paramsList.extend(
                        [p.copy() for p in params]
                        for params in nd.neighborMemory.values()
                    )
                    setParams(nd.model, averageParams(paramsList))
                    refClassAcc = evaluatePerClass(
                        nd.model, nd.dataLoader, nClasses=nClasses, maxBatches=classEvalMaxBatches
                    )
                else:
                    predecessorId = (i - 1) % n
                    if predecessorId in nd.neighborMemory:
                        setParams(
                            nd.model,
                            [p.copy() for p in nd.neighborMemory[predecessorId]],
                        )
                    refClassAcc = evaluatePerClass(
                        nd.model, nd.dataLoader, nClasses=nClasses, maxBatches=classEvalMaxBatches
                    )
            else:
                refClassAcc = evaluatePerClass(
                    nd.model, nd.dataLoader, nClasses=nClasses, maxBatches=classEvalMaxBatches
                )

            # --- Step 2: Merge received registry (with Byzantine verification) ---
            receivedRegistry = nodeRegistries[i]
            nd.classRegistry.verifyAndMerge(receivedRegistry, refClassAcc, verifyThreshold)

            # --- Step 3: Compute trust weights ---
            trustWeights = nd.classRegistry.trustWeights(nd.classAcc)

            # --- Step 4: Class-aware proximal local training ---
            nd.cartLocalTrain(lr=lr, trustWeights=trustWeights, stepsPerEpoch=stepsPerEpoch)

            # --- Step 5: Evaluate own per-class accuracy after training ---
            nd.classAcc = evaluatePerClass(
                nd.model, nd.dataLoader, nClasses=nClasses, maxBatches=classEvalMaxBatches
            )
            nd.classRegistry.update(nd.nodeId, nd.classAcc)

            # --- Step 6: Get params, apply noise, apply attack ---
            params = getParams(nd.model)
            noisyParams = addChannelNoiseToParams(params, sigma=commSigma)

            if i in attackers:
                if atk == "hidden" and r < hiddenStartRound:
                    pass
                elif atk == "model_poison":
                    setParams(nd.model, noisyParams)
                    trainedParams = [p.copy() for p in noisyParams]
                    noisyParams = modelPoisonAttack(nd.model, nd.dataLoader)
                    setParams(nd.model, trainedParams)
                else:
                    noisyParams = applyAttack(noisyParams, atk)

            # --- Step 7: Multicast (model + registry) to next S clockwise nodes ---
            outRegistry = nd.classRegistry.clone()
            for offset in range(1, fanout + 1):
                j = (i + offset) % n
                nodes[j].receiveModel(i, noisyParams)
                # Merge registry into the destination node's incoming registry
                # (each node in the ring window gets the sender's best knowledge)
                nodeRegistries[j].merge(outRegistry)

        # ===== EVALUATION =====
        if testLoader is not None:
            avg, worst, _ = evaluateAll(nodes, testLoader, maxBatches=evalMaxBatches)
            print(f"[CART round {r}] eval avg={avg:.4f}", flush=True)
            avgAccHist.append(avg)
            worstAccHist.append(worst)
            if roundCallback is not None:
                roundCallback(r + 1, avg, worst, rounds)

            # Plateau LR
            if usePlateauLr:
                if avg <= 0.25:
                    _plateauBestAcc = -1.0
                    _plateauPatience = 0
                elif _plateauCooldown > 0:
                    _plateauCooldown -= 1
                elif avg > _plateauBestAcc + plateauThreshold:
                    _plateauBestAcc = avg
                    _plateauPatience = 0
                else:
                    _plateauPatience += 1
                    if _plateauPatience >= plateauPatience:
                        new_lr = max(plateauMinLr, _plateauLr * plateauFactor)
                        if new_lr < _plateauLr:
                            print(f"[CART round {r}] ReduceLROnPlateau: {_plateauLr:.6f} → {new_lr:.6f}", flush=True)
                            _plateauLr = new_lr
                        _plateauPatience = 0
                        _plateauCooldown = plateauPatience // 2

        # Reset incoming registries for next round (nodes start fresh from what
        # they accumulated during this round's ring pass)
        for i in range(n):
            nodeRegistries[i] = ClassRegistry(nClasses=nClasses)

    return avgAccHist, worstAccHist
