#!/usr/bin/env python3
"""Generate 48 CART configs matching the same attack × mitigation matrix as basil/noisy/merged."""

import json
import os

OUT_DIR = "/home/ojaswi/PaperMerge/gui/configs/cart"
os.makedirs(OUT_DIR, exist_ok=True)

BASE = {
    "dataset": "cifar10",
    "approach": "cart",
    "useBasil": True,           # SS=ON (class-aware SS)
    "basilMemorySize": 1,
    "noiseMitigation": "none",
    "ebmLambda": 25.0,
    "momentum": 0.9,
    "nNodes": 10,
    "nRounds": 100,
    "localEpochs": 5,
    "learningRate": 0.05,
    "batchSize": 512,
    "useLrDecay": False,
    "stepsPerEpoch": 5,
    "usePlateauLr": True,
    "plateauPatience": 8,
    "plateauFactor": 0.7,
    "plateauMinLr": 0.001,
    "plateauThreshold": 0.01,
    "nonIID": True,
    "dirichletAlpha": 0.2,
    "attackerIds": "",
    # CART-specific
    "distillStrength": 0.5,
    "verifyThreshold": 0.05,
    # Attacks all off by default
    "attackGaussian": False, "attackGaussianStart": 0,
    "attackSignFlip": False, "attackSignFlipStart": 0,
    "attackHidden": False,   "attackHiddenStart": 0,
    "attackModelPoison": False, "attackModelPoisonStart": 0,
    "attackScaling": False,  "attackScalingStart": 0,
    "attackAlie": False,     "attackAlieStart": 0,
    "attackIpm": False,      "attackIpmStart": 0,
    "attackNoiseAmp": False, "attackNoiseAmpStart": 0,
}

# 8 attack scenarios (matches basil matrix)
# prefix index, name used in filename, attack field, approach description
ATTACKS = [
    (0,  "Byzantine Nodes",              None,               None),
    (4,  "Byzantine Attack (ALIE)",      "attackAlie",       "ALIE"),
    (4,  "Byzantine Attack (Hidden)",    "attackHidden",     "Hidden"),
    (4,  "Byzantine Attack (IPM)",       "attackIpm",        "IPM"),
    (4,  "Byzantine Attack (ModelPoison)", "attackModelPoison", "ModelPoison"),
    (4,  "Byzantine Attack (NoiseAmp)",  "attackNoiseAmp",   "NoiseAmp"),
    (4,  "Byzantine Attack (Scaling)",   "attackScaling",    "Scaling"),
    (4,  "Byzantine Attack (SignFlip)",  "attackSignFlip",   "SignFlip"),
]

# 6 mitigation combos (same as basil):
# (useBasil, useChannelNoise, noiseMitigation, label)
MITIGATIONS = [
    (False, False, "none",  "No Channel Noise + No Mitigation"),
    (True,  False, "none",  "No Channel Noise + SS Mitigation"),
    (False, True,  "none",  "Channel Noise + No Mitigation"),
    (False, True,  "ebm",   "Channel Noise + EBM Mitigation"),
    (True,  True,  "none",  "Channel Noise + SS Mitigation"),
    (True,  True,  "ebm",   "Channel Noise + SS & EBM Mitigation"),
]

written = 0
for prefix, atkName, atkField, atkTag in ATTACKS:
    for useBasil, useNoise, noiseMit, mitLabel in MITIGATIONS:
        cfg = dict(BASE)

        # Set attack
        if atkField:
            cfg[atkField] = True
            cfg[f"{atkField}Start"] = 0
            cfg["attackerIds"] = "1,4,6,8"
            cfg["basilMemorySize"] = 5
        else:
            cfg["attackerIds"] = ""
            cfg["basilMemorySize"] = 1

        # Set mitigation
        cfg["useBasil"] = useBasil
        cfg["useChannelNoise"] = useNoise
        cfg["noiseMitigation"] = noiseMit
        cfg["usePlateauLr"] = not (useNoise and noiseMit == "ebm")
        cfg["channelNoiseStart"] = 0
        cfg["channelNoiseSigma"] = 0.2 if useNoise else 0.0

        # Experiment name
        name = f"{prefix} - {atkName} + {mitLabel}"
        cfg["experimentName"] = name

        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        written += 1
        print(f"  Written: {name}.json")

print(f"\nTotal written: {written}")
