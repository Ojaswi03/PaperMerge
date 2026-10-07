#!/usr/bin/env python3
"""Channel-only norm propagation: no SGD, attacks, selection, or test tuning."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from basil_core.research_protocol import apply_channel_noise, build_model, keyed_rng, parameter_stats


def main():
    initial = [v.numpy() for v in build_model("basil_paper_cnn", 2025).trainable_variables]
    dimension, norm = parameter_stats(initial)
    results = []
    for sigma in (.2, .4, .6):
        ratios = []
        for replicate in range(20):
            state = initial
            for activation in range(30):
                sender, round_id = activation % 10, activation // 10
                state, _ = apply_channel_noise(state, semantics="relative_l2_gaussian", sigma=sigma,
                    rng=keyed_rng(2025 + replicate, "channel_noise", round_id, sender, (sender + 1) % 10))
            ratios.append(parameter_stats(state)[1] / norm)
        results.append({"sigmaRelative": sigma, "transmissions": 30, "replicates": 20,
            "expectedSquaredNormRatio": (1 + sigma ** 2) ** 30,
            "sqrtExpectedSquaredNormRatio": (1 + sigma ** 2) ** 15,
            "observedNormRatioMean": float(np.mean(ratios)), "observedNormRatioStd": float(np.std(ratios))})
    output = {"purpose": "channel_only_numerical_diagnostic", "researchValid": False,
        "parameterCount": dimension, "initialNorm": norm, "rows": results}
    (ROOT / "docs/validation_noise_growth.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
