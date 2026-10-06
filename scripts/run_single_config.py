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
from basil_core.experiment_engine import run_campaign_three
from basil_core.trainer     import getParams, setParams, evaluateAll
from gui.baseline_study import make_config as makeCampaign3Config

parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    help=(
        "Path to a JSON config, or "
        "campaign3:<merged|cart>:<environment>:<mitigation>[:sigma]"
    ),
)
parser.add_argument("--rounds", type=int, default=None, help="Override nRounds")
parser.add_argument(
    "--set",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Override a JSON config field for this run only. Supports bool/int/float/string values.",
)
args = parser.parse_args()

if args.config.startswith("campaign3:"):
    parts = args.config.split(":")
    if len(parts) not in (4, 5):
        raise SystemExit(
            "Campaign spec must be "
            "campaign3:<merged|cart>:<environment>:<mitigation>[:sigma]"
        )
    _, campaignApproach, campaignEnvironment, campaignMitigation, *sigmaPart = parts
    campaignSigma = float(sigmaPart[0]) if sigmaPart else 0.0
    cfg = makeCampaign3Config(
        split="nonIID",
        approach=campaignApproach,
        environment=campaignEnvironment,
        mitigation=campaignMitigation,
        sigma=campaignSigma,
        seed=2025,
        rounds=args.rounds or 5,
        phase="pilot",
        gamma=0.2,
    )
else:
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
cfg["nRounds"] = nRounds
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
print(
    f"  effective gamma={distillStr:g}, ebmLambda={ebmLambda:g}, "
    f"lambda*sigma^2={float(ebmLambda) * float(sigma) * float(sigma):g}"
)
print(f"  S={S}, attackers={attackerIds}, attacks={attackTypes}, rounds={nRounds}\n")

# Load data
if dataset == "cifar10":
    train, test = loadCifar10()
    if int(cfg.get("campaignVersion", 0)) == 3:
        loaders, testLoader, dataMetadata = makeCifarLoaders(
            train,
            test,
            nClients=nNodes,
            batchSize=batchSize,
            iid=(not nonIID),
            dirichletAlpha=alpha,
            seed=int(cfg["seed"]),
            returnMetadata=True,
        )
    else:
        loaders, testLoader = makeCifarLoaders(
            train,
            test,
            nClients=nNodes,
            batchSize=batchSize,
            iid=(not nonIID),
            dirichletAlpha=alpha,
        )
        dataMetadata = None
    ModelClass = CIFARModel
else:
    train, test = loadMnist()
    loaders, testLoader = makeMnistLoaders(train, test, nClients=nNodes,
                                           batchSize=batchSize, iid=(not nonIID),
                                           dirichletAlpha=alpha)
    ModelClass = MNISTModel

print(f"Data split: {'non-IID (Dirichlet α=' + str(alpha) + ')' if nonIID else 'IID'}")

if int(cfg.get("campaignVersion", 0)) == 3:
    if dataset != "cifar10":
        raise SystemExit("Campaign 3 currently supports CIFAR-10 only.")
    result = run_campaign_three(
        config=cfg,
        model_class=ModelClass,
        train_loaders=loaders,
        test_loader=testLoader,
        data_metadata=dataMetadata,
    )
    if result.get("stopped"):
        raise SystemExit("Campaign run stopped before completion.")
    avgAcc = np.asarray(result["avg_history"], dtype=np.float64)
    print("\n=== CAMPAIGN 3 PILOT ACCURACY ===")
    print(
        f"  Final full-test average={float(result['final_avg']):.4f} "
        f"({float(result['final_avg']) * 100:.1f}%)"
    )
    print(
        f"  Final full-test worst={float(result['final_worst']):.4f} "
        f"({float(result['final_worst']) * 100:.1f}%)"
    )
    print(
        f"  Tracked best={float(np.max(avgAcc)):.4f} "
        f"({float(np.max(avgAcc)) * 100:.1f}%) at round "
        f"{int(np.argmax(avgAcc)) + 1}"
    )
    if approach == "cart":
        print(
            f"  Mean CART mu={float(np.mean(result['mu_history'])):.6f}, "
            f"final registry coverage={float(result['registry_coverage'][-1]):.1%}"
        )
    if cfg.get("snapshotSelection"):
        sources = np.asarray(result["selected_sources"], dtype=np.int32)
        allReceivers = np.arange(sources.shape[1])[None, :]
        allSelfSelections = sources == allReceivers
        print(
            f"  All-round selected self="
            f"{float(np.mean(allSelfSelections)):.1%}, "
            f"selected neighbor={float(np.mean(~allSelfSelections)):.1%}"
        )
        active = sources[int(cfg.get("attackHiddenStart", 0)) :]
        attackerSet = {
            int(value)
            for value in str(cfg.get("attackerIds", "")).split(",")
            if value.strip()
        }
        if active.size and attackerSet:
            receivers = np.arange(active.shape[1])[None, :]
            selfSelections = active == receivers
            maliciousSelections = (
                np.isin(active, sorted(attackerSet))
                & ~selfSelections
            )
            print(
                f"  Post-activation selected external attackers="
                f"{float(np.mean(maliciousSelections)):.1%}, "
                f"selected self={float(np.mean(selfSelections)):.1%}"
            )
    print(f"  Runtime={float(result['runtime_seconds']):.1f}s")
    raise SystemExit(0)

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
