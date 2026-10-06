#!/usr/bin/env python3
"""
Update all JSON config files in gui/configs/:
- nRounds: 100
- momentum: 0.9 (all configs)
- learningRate: 0.05
- usePlateauLr: true, except channel-noise EBM configs where noisy eval makes plateau reductions harmful
- plateauPatience: 8   (less trigger-happy — was 5)
- plateauFactor: 0.7   (gentler reduction — was 0.5)
- plateauMinLr: 0.001  (keep LR productive — was 0.0001)
- plateauThreshold: 0.01
- attackerIds: clean/no-attack configs → ""; "0,3,5,7" → "1,4,6,8" for attack configs
- basilMemorySize: S = b+1 where b = number of Byzantine nodes in attackerIds
    e.g. 4 attackers → S=5, 2 attackers → S=3, 0 attackers → S=1 (min 1)
    (Paper BASIL: S=b+1 guarantees honest subgraph remains connected)
- distillStrength: 0.4 for cart configs (was 0.3 — stronger proximal = less forgetting)

For EBM configs (noiseMitigation=ebm), leave useLrDecay as-is (already false).
For non-EBM configs, leave useLrDecay as-is (true).
"""

import glob
import json

config_dir = "/home/ojaswi/PaperMerge/gui/configs"
# Walk nonIID/ and IID/ subdirectories (new structure)
# Also pick up any loose configs at the old flat level for backward compat
files = glob.glob(f"{config_dir}/**/*.json", recursive=True)
files.sort()

updated = 0
for path in files:
    with open(path, "r") as f:
        config = json.load(f)

    changed = False

    if config.get("nRounds") != 100:
        config["nRounds"] = 100
        changed = True

    if config.get("momentum") != 0.9:
        config["momentum"] = 0.9
        changed = True

    if config.get("learningRate") != 0.05:
        config["learningRate"] = 0.05
        changed = True

    if not config.get("useChannelNoise", False):
        if config.get("channelNoiseSigma", 0.0) != 0.0:
            config["channelNoiseSigma"] = 0.0
            changed = True
        name_allows_no_noise_ebm = (
            "No Channel Noise + EBM Mitigation"
            in config.get("experimentName", "")
        )
        if config.get("noiseMitigation") == "ebm" and not name_allows_no_noise_ebm:
            config["noiseMitigation"] = "none"
            changed = True
    else:
        if float(config.get("channelNoiseSigma", 0.0) or 0.0) <= 0.0:
            config["channelNoiseSigma"] = 0.2
            changed = True

    is_ebm_channel_noise = (
        bool(config.get("useChannelNoise"))
        and config.get("noiseMitigation") == "ebm"
    )
    expected_plateau = not is_ebm_channel_noise
    if config.get("usePlateauLr") != expected_plateau:
        config["usePlateauLr"] = expected_plateau
        changed = True

    if config.get("plateauPatience") != 8:
        config["plateauPatience"] = 8
        changed = True

    if config.get("plateauFactor") != 0.7:
        config["plateauFactor"] = 0.7
        changed = True

    if config.get("plateauMinLr") != 0.001:
        config["plateauMinLr"] = 0.001
        changed = True

    if config.get("plateauThreshold") != 0.01:
        config["plateauThreshold"] = 0.01
        changed = True

    attack_flags = [
        "attackGaussian",
        "attackSignFlip",
        "attackHidden",
        "attackModelPoison",
        "attackScaling",
        "attackAlie",
        "attackIpm",
        "attackNoiseAmp",
    ]
    has_attack = any(bool(config.get(flag)) for flag in attack_flags)

    if not has_attack and config.get("attackerIds", "") != "":
        config["attackerIds"] = ""
        changed = True
    elif has_attack and config.get("attackerIds") == "0,3,5,7":
        config["attackerIds"] = "1,4,6,8"
        changed = True

    # basilMemorySize = S = b+1 (paper BASIL formula)
    # b = actual number of Byzantine nodes (length of attackerIds)
    attacker_ids_str = config.get("attackerIds", "")
    b = len([x for x in attacker_ids_str.split(",") if x.strip()]) if attacker_ids_str.strip() else 0
    correct_S = max(1, b + 1)
    if config.get("basilMemorySize") != correct_S:
        config["basilMemorySize"] = correct_S
        changed = True

    # distillStrength: CART configs only — 0.4 (was 0.3)
    if config.get("approach") == "cart" and config.get("distillStrength") != 0.4:
        config["distillStrength"] = 0.4
        changed = True

    # Set nonIID based on which folder the file lives in (IID/ → false, nonIID/ → true)
    import os as _os
    parts = path.replace("\\", "/").split("/")
    if "IID" in parts and "nonIID" not in parts:
        expected_noniid = False
    else:
        expected_noniid = True
    if config.get("nonIID") != expected_noniid:
        config["nonIID"] = expected_noniid
        changed = True

    if expected_noniid and config.get("dirichletAlpha") != 0.2:
        config["dirichletAlpha"] = 0.2
        changed = True

    if changed:
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        updated += 1
        print(f"  Updated: {path.split('/')[-1]}")

print(f"\nTotal files found: {len(files)}")
print(f"Total files updated: {updated}")
print(f"Files already correct: {len(files) - updated}")
