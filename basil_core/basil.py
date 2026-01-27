import sys
from copy import deepcopy

from .attacks import applyAttack
from .trainer import (
    localUpdate,
    evaluateBatchLoss,
    addChannelNoiseToParams,
    getParams,
    setParams,
    makeLrScheduler,
    evaluateAll,
)


class BasilNode:
    """
    Node with memory (last S snapshots). Selects best snapshot via local batch loss,
    runs local update, then forwards current weights to next neighbor in ring.
    noiseModel in {"none", "noisy", "ebm", "wcm"}
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
        ebmLambda=1.0,
        wcmLambda=0.1,
        wcmSamples=5,
        wcmRho=0.5,
        **kwargs,  # accept legacy/extra kwargs (e.g., lr)
    ):
        # Backward compatibility: map legacy 'lr' -> 'lr0'
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

        # Memory of parameter snapshots (list[list[np.ndarray]])
        self.memory = []
        self.round = 0

    def snapshot(self):
        """Store a copy of current params; keep only last S."""
        self.memory.append(getParams(self.model))
        if len(self.memory) > self.S:
            self.memory.pop(0)

    def selectBestSnapshot(self):
        """
        Choose the params (from {current} ∪ memory) that minimize local batch loss.
        Restores the model to the best params.
        """
        candidates = [getParams(self.model)] + self.memory
        bestParams = candidates[0]
        bestLoss = evaluateBatchLoss(self.model, self.dataLoader)  # current

        for p in self.memory:
            orig = getParams(self.model)
            setParams(self.model, p)
            lossP = evaluateBatchLoss(self.model, self.dataLoader)
            if lossP < bestLoss:
                bestLoss = lossP
                bestParams = [x.copy() for x in p]
            # restore
            setParams(self.model, orig)

        setParams(self.model, bestParams)

    def localTrain(self, lr, stepsPerEpoch=100):
        """
        Local training bounded by stepsPerEpoch to avoid infinite loops when the
        tf.data pipeline uses `.repeat()` without a take/epoch cap.
        """
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
    lr0=0.05,
    lrAlpha=0.6,
    stepsPerEpoch=100,  # safe bound for local updates
    useSnapshots=True,  # whether to use BASIL snapshot selection
    stopCallback=None,  # callable that returns True when training should stop
    **kwargs,  # accept legacy kw like attack_type
):
    """
    Basil on a ring: each node forwards params to the next neighbor each round.
    - attackTypes: list or str (e.g., ["gaussian", "sign-flip", "hidden"])
      rotated per round; legacy single 'attack_type' kw is mapped to [attack_type].
    - stepsPerEpoch: limits localUpdate iterations to avoid hangs with infinite datasets.
    - useSnapshots: if True, use BASIL snapshot selection; if False, skip it (for baseline comparison)
    - stopCallback: optional callable that returns True when training should stop (for GUI stop button)
    Returns: (avgAccHistory, worstAccHistory)
    """
    # Backward compatibility: allow single 'attack_type' kw
    if "attack_type" in kwargs and kwargs["attack_type"] is not None:
        attackTypes = [kwargs["attack_type"]]

    # Normalize attackTypes
    if isinstance(attackTypes, str):
        attackTypes = [attackTypes]
    if not attackTypes:
        attackTypes = ["none"]

    n = len(nodes)
    # Push global noise / lr settings into nodes
    for nd in nodes:
        nd.noiseModel = noiseModel
        nd.sigma = float(sigma)
        nd.lr0 = float(lr0)

    # Dynamic LR: lr_t = max(minLr, lr0 * (t+1)^(-alpha))
    lrSched = makeLrScheduler(lr0, alpha=lrAlpha)

    avgAccHist, worstAccHist = [], []

    # Pre-train evaluation (round -1)
    if testLoader is not None:
        avg, worst, _ = evaluateAll(nodes, testLoader)
        print(f"[round -1] pre-train avg={avg:.4f} worst={worst:.4f}", flush=True)
        avgAccHist.append(avg)
        worstAccHist.append(worst)

    for r in range(rounds):
        # Check if we should stop
        if stopCallback is not None and stopCallback():
            print(f"[round {r}] STOP REQUESTED - terminating training early", flush=True)
            break

        lr = lrSched(r)
        print(f"[round {r}] lr={lr:.6f} training...", flush=True)

        # Local step per node
        for nd in nodes:
            if useSnapshots:
                nd.selectBestSnapshot()
            nd.localTrain(lr=lr, stepsPerEpoch=stepsPerEpoch)
            if useSnapshots:
                nd.snapshot()

        # Communication: add channel noise and apply attacks, then deliver to next neighbor
        outParams = []
        for nd in nodes:
            params = getParams(nd.model)
            # Add sender-side channel noise when using noisy/ebm/wcm models
            noisyParams = addChannelNoiseToParams(
                params, sigma=sigma if noiseModel in ("noisy", "ebm", "wcm") else 0.0
            )
            outParams.append(noisyParams)

        atk = attackTypes[r % len(attackTypes)]
        attackers = set(attackerIds or [])

        newParamsAfterComm = [None] * n
        for i in range(n):
            send = deepcopy(outParams[i])
            if i in attackers:
                if atk == "hidden" and r < hiddenStartRound:
                    # delay hidden/backdoor until threshold
                    pass
                else:
                    send = applyAttack(send, atk)
            j = (i + 1) % n  # ring neighbor
            newParamsAfterComm[j] = send

        # Apply received params
        for j in range(n):
            if newParamsAfterComm[j] is not None:
                setParams(nodes[j].model, newParamsAfterComm[j])

        # Evaluation/logging
        if testLoader is not None:
            avg, worst, _ = evaluateAll(nodes, testLoader)
            print(f"[round {r}] eval avg={avg:.4f} worst={worst:.4f}", flush=True)
            avgAccHist.append(avg)
            worstAccHist.append(worst)

    return avgAccHist, worstAccHist
