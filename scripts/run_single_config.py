#!/usr/bin/env python3
"""
Run a single JSON config for a quick check.
Usage: python scripts/run_single_config.py <path_to_config.json> [--rounds N]
"""
import atexit
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common import cleanupTensorflowMemory, setupGpu
setupGpu()
atexit.register(lambda: cleanupTensorflowMemory(logger=print, context="run_single_config exit"))

import numpy as np
from basil_core.data.cifar  import loadCifar10, makeLoaders as makeCifarLoaders
from basil_core.data.mnist  import loadMnist,   makeLoaders as makeMnistLoaders
from basil_core.models      import CIFARModel, MNISTModel
from basil_core.basil       import BasilNode, basilRingTrainingWithAttack, fedAvgTrainingWithNoise
from basil_core.cart        import CARTNode, cartRingTraining
from basil_core.trainer     import getParams, setParams, evaluateAll

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to JSON config file")
parser.add_argument("--rounds", type=int, default=None, help="Override nRounds")
parser.add_argument(
    "--set",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Override a JSON config field for this run only. Supports bool/int/float/string values.",
)
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

def _parseOverrideValue(raw):
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "none":
        return None
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw

for item in args.set:
    if "=" not in item:
        raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
    key, rawValue = item.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit(f"--set key cannot be empty: {item}")
    cfg[key] = _parseOverrideValue(rawValue.strip())

nRounds      = args.rounds if args.rounds else cfg.get("nRounds", 100)
nNodes       = cfg.get("nNodes", 10)
lr           = cfg.get("learningRate", 0.05)
batchSize    = cfg.get("batchSize", 512)
localEpochs  = cfg.get("localEpochs", 5)
stepsPerEpoch = cfg.get("stepsPerEpoch", 5)
momentum     = cfg.get("momentum", 0.9)
nonIID       = cfg.get("nonIID", False)
alpha        = cfg.get("dirichletAlpha", 0.5)
dataset      = cfg.get("dataset", "cifar10")
approach     = cfg.get("approach", "basil")
cartAlgo     = cfg.get("cartAlgorithm", "basil")
useBasil     = cfg.get("useBasil", False)
aggregationMode = cfg.get("aggregationMode")
if not aggregationMode:
    aggregationMode = "handoff" if useBasil else ("consensus" if approach in ("merged", "cart") else "handoff")
cfg["aggregationMode"] = aggregationMode
S            = cfg.get("basilMemorySize", 1)
sigma        = cfg.get("channelNoiseSigma", 0.0) if cfg.get("useChannelNoise") else 0.0
ebmLambda    = cfg.get("ebmLambda", 25.0)
distillStr   = cfg.get("distillStrength", 0.4)
verifyThr    = cfg.get("verifyThreshold", 0.05)

# Mirror GUI's getNoiseModel logic
if not cfg.get("useChannelNoise") or sigma == 0:
    noiseModel = "none"
elif cfg.get("noiseMitigation") == "ebm":
    noiseModel = "ebm"
else:
    noiseModel = "noisy"

attackerIds = [int(x) for x in cfg.get("attackerIds","").split(",") if x.strip()]

attackTypes = []
for k, name in [("attackGaussian","gaussian"),("attackSignFlip","signFlip"),
                ("attackHidden","hidden"),("attackModelPoison","model_poison"),
                ("attackScaling","scaling"),("attackAlie","alie"),
                ("attackIpm","ipm"),("attackNoiseAmp","noiseAmp")]:
    if cfg.get(k):
        attackTypes.append(name)
if not attackTypes:
    attackTypes = ["none"]

print(f"\nConfig: {cfg.get('experimentName','?')}")
print(f"  approach={approach}, cartAlgo={cartAlgo}, nonIID={nonIID}")
print(f"  sigma={sigma}, noiseModel={noiseModel}, useBasil={useBasil}, aggregationMode={aggregationMode}")
print(f"  S={S}, attackers={attackerIds}, attacks={attackTypes}, rounds={nRounds}\n")

# Load data
if dataset == "cifar10":
    train, test = loadCifar10()
    loaders, testLoader = makeCifarLoaders(train, test, nClients=nNodes,
                                           batchSize=batchSize, iid=(not nonIID),
                                           dirichletAlpha=alpha)
    ModelClass = CIFARModel
else:
    train, test = loadMnist()
    loaders, testLoader = makeMnistLoaders(train, test, nClients=nNodes,
                                           batchSize=batchSize, iid=(not nonIID),
                                           dirichletAlpha=alpha)
    ModelClass = MNISTModel

print(f"Data split: {'non-IID (Dirichlet α=' + str(alpha) + ')' if nonIID else 'IID'}")

commonKw = dict(
    rounds=nRounds, testLoader=testLoader,
    attackTypes=attackTypes, attackerIds=attackerIds,
    sigma=sigma, noiseModel=noiseModel,
    lr0=lr, stepsPerEpoch=stepsPerEpoch,
    useLrDecay=cfg.get("useLrDecay", False),
    usePlateauLr=cfg.get("usePlateauLr", True),
    plateauPatience=cfg.get("plateauPatience", 8),
    plateauFactor=cfg.get("plateauFactor", 0.7),
    plateauMinLr=cfg.get("plateauMinLr", 0.001),
    plateauThreshold=cfg.get("plateauThreshold", 0.01),
)

if approach == "cart" and cartAlgo == "cart":
    nodes = [CARTNode(nodeId=i, model=ModelClass(), dataLoader=loaders[i],
                      S=S, nClasses=10, distillStrength=distillStr,
                      verifyThreshold=verifyThr, noiseModel=noiseModel,
                      sigma=sigma, lr0=lr, localEpochs=localEpochs,
                      ebmLambda=ebmLambda, momentum=momentum)
             for i in range(nNodes)]
    p0 = getParams(nodes[0].model)
    for nd in nodes[1:]:
        setParams(nd.model, p0)
    avgAcc, _ = cartRingTraining(
        nodes, useSnapshots=useBasil, aggregationMode=aggregationMode, **commonKw)

elif approach == "noisy":
    nodes = [BasilNode(nodeId=i, model=ModelClass(), dataLoader=loaders[i],
                       S=S, noiseModel=noiseModel, sigma=sigma, lr0=lr,
                       localEpochs=localEpochs, ebmLambda=ebmLambda, momentum=momentum)
             for i in range(nNodes)]
    p0 = getParams(nodes[0].model)
    for nd in nodes[1:]:
        setParams(nd.model, p0)
    avgAcc, _ = fedAvgTrainingWithNoise(
        nodes=nodes,
        channelNoiseStart=cfg.get("channelNoiseStart", 0),
        **commonKw)

else:
    # basil / merged / cart (non-cart sub-algorithm)
    nodes = [BasilNode(nodeId=i, model=ModelClass(), dataLoader=loaders[i],
                       S=S, noiseModel=noiseModel, sigma=sigma, lr0=lr,
                       localEpochs=localEpochs, ebmLambda=ebmLambda, momentum=momentum)
             for i in range(nNodes)]
    p0 = getParams(nodes[0].model)
    for nd in nodes[1:]:
        setParams(nd.model, p0)
    avgAcc, _ = basilRingTrainingWithAttack(
        nodes, useSnapshots=useBasil, S=S,
        aggregationMode=aggregationMode,
        channelNoiseStart=cfg.get("channelNoiseStart", 0), **commonKw)

print(f"\n=== FINAL ACCURACY ===")
print(f"  Round {nRounds}: avg={avgAcc[-1]:.4f} ({avgAcc[-1]*100:.1f}%)")
print(f"  Best: {max(avgAcc):.4f} ({max(avgAcc)*100:.1f}%) at round {avgAcc.index(max(avgAcc))}")
