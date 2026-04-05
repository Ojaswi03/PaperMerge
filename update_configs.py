#!/usr/bin/env python3
"""
Update all JSON config files in gui/configs/:
- nRounds: 100
- momentum: 0.9 (all configs)
- learningRate: 0.05
- usePlateauLr: true
- plateauPatience: 5
- plateauThreshold: 0.01
- attackerIds: "0,3,5,7" → "1,4,6,8" (more spread, node 0 kept honest)

For EBM configs (noiseMitigation=ebm), leave useLrDecay as-is (already false).
For non-EBM configs, leave useLrDecay as-is (true).
"""

import glob
import json

config_dir = "/home/ojaswi/PaperMerge/gui/configs"
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

    if config.get("usePlateauLr") != True:
        config["usePlateauLr"] = True
        changed = True

    if config.get("plateauPatience") != 5:
        config["plateauPatience"] = 5
        changed = True

    if config.get("plateauThreshold") != 0.01:
        config["plateauThreshold"] = 0.01
        changed = True

    if config.get("attackerIds") == "0,3,5,7":
        config["attackerIds"] = "1,4,6,8"
        changed = True

    if changed:
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        updated += 1
        print(f"  Updated: {path.split('/')[-1]}")

print(f"\nTotal files found: {len(files)}")
print(f"Total files updated: {updated}")
print(f"Files already correct: {len(files) - updated}")
