"""Versioned, individually selectable GUI configuration library."""

from __future__ import annotations

from pathlib import Path

from gui.campaign3 import (
    CONFIRMATION_SEEDS,
    NOISE_LEVELS,
    build_approach_confirmation,
    frozen_cart_schedule,
)


CURRENT_LIBRARY_VERSION = "hidden-cifar10-current-2026-07-26"
CURRENT_CONFIG_ROOT = Path(__file__).resolve().parent / "configs" / "current"
APPROACHES = ("basil", "noisy", "merged", "cart")
SPLITS = ("nonIID", "IID")
HIDDEN_ATTACKERS = (1, 4, 6, 8)

_LABELS = {
    "none": "No Mitigation",
    "ss": "SS",
    "ebm": "EBM",
    "ss_ebm": "SS+EBM",
}


def _condition_id(environment: str, mitigation: str, sigma: float) -> str:
    if environment == "clean":
        return "clean_no_mitigation"
    if environment == "hidden":
        return f"hidden_{mitigation}"
    sigma_label = f"{float(sigma):.1f}".replace(".", "_")
    prefix = "hidden_noise" if environment == "hidden_noise" else "noise"
    return f"{prefix}_sigma_{sigma_label}_{mitigation}"


def _standalone_config(
    *,
    split: str,
    approach: str,
    environment: str,
    mitigation: str,
    seed: int,
    sigma: float = 0.0,
) -> dict:
    has_hidden = environment in ("hidden", "hidden_noise")
    has_noise = environment in ("noise", "hidden_noise")
    use_ss = mitigation in ("ss", "ss_ebm")
    use_ebm = mitigation in ("ebm", "ss_ebm")
    if use_ss and not has_hidden:
        raise ValueError("SS requires an active Hidden attack.")
    if use_ebm and not has_noise:
        raise ValueError("EBM requires active channel noise.")
    if environment == "clean" and mitigation != "none":
        raise ValueError("The clean environment cannot use mitigation.")

    sigma = float(sigma) if has_noise else 0.0
    ebm_lambda = 1.0 / (sigma * sigma) if use_ebm else 0.0
    condition_id = _condition_id(environment, mitigation, sigma)
    environment_label = {
        "clean": "Clean Environment",
        "hidden": "Hidden Attack",
        "noise": f"Channel Noise sigma={sigma:.1f}",
        "hidden_noise": f"Hidden Attack + Channel Noise sigma={sigma:.1f}",
    }[environment]

    return {
        "schemaVersion": 2,
        "configLibraryVersion": CURRENT_LIBRARY_VERSION,
        "executionProtocol": "standalone_paper_baseline",
        "conditionId": condition_id,
        "environment": environment,
        "experimentName": (
            f"Current | {split} | {approach.upper()} | "
            f"{environment_label} | {_LABELS[mitigation]} | seed={seed}"
        ),
        "dataset": "cifar10",
        "approach": approach,
        "cartAlgorithm": "cart",
        "useBasil": use_ss,
        "snapshotSelection": use_ss,
        "basilMemorySize": 5,
        "aggregationMode": (
            "handoff" if approach == "basil" and use_ss else "consensus"
        ),
        "useChannelNoise": has_noise,
        "channelNoiseStart": 0,
        "channelNoiseSigma": sigma,
        "channelNoiseSemantics": "relative_l2_per_link",
        "noiseMitigation": "ebm" if use_ebm else "none",
        "ebmLambda": ebm_lambda,
        "ebmTargetScale": 2.0 if use_ebm else 1.0,
        "ebmImplementation": "paper_gradient_scaling",
        "attackGaussian": False,
        "attackGaussianStart": 0,
        "attackSignFlip": False,
        "attackSignFlipStart": 0,
        "attackHidden": has_hidden,
        "attackHiddenStart": 20 if has_hidden else 0,
        "attackWarmupRounds": 20 if has_hidden else 0,
        "attackModelPoison": False,
        "attackModelPoisonStart": 0,
        "attackScaling": False,
        "attackScalingStart": 0,
        "attackAlie": False,
        "attackAlieStart": 0,
        "attackIpm": False,
        "attackIpmStart": 0,
        "attackNoiseAmp": False,
        "attackNoiseAmpStart": 0,
        "attackerIds": (
            ",".join(str(node) for node in HIDDEN_ATTACKERS)
            if has_hidden
            else ""
        ),
        "nonIID": split == "nonIID",
        "dirichletAlpha": 0.2,
        "nNodes": 10,
        "nRounds": 100,
        "localEpochs": 5,
        "stepsPerEpoch": 5,
        "learningRate": 0.05,
        "momentum": 0.9,
        "batchSize": 512,
        "useLrDecay": True,
        "usePlateauLr": False,
        "plateauPatience": 8,
        "plateauFactor": 0.7,
        "plateauMinLr": 0.001,
        "plateauThreshold": 0.01,
        "distillStrength": 0.0,
        "verifyThreshold": 0.05,
        "seed": int(seed),
    }


def _standalone_matrix(split: str, approach: str) -> list[dict]:
    if approach == "basil":
        environment_mitigations = {
            "clean": ("none",),
            "hidden": ("none", "ss"),
            "noise": ("none",),
            "hidden_noise": ("none", "ss"),
        }
    elif approach == "noisy":
        environment_mitigations = {
            "clean": ("none",),
            "hidden": ("none",),
            "noise": ("none", "ebm"),
            "hidden_noise": ("none", "ebm"),
        }
    else:
        raise ValueError(f"Unsupported standalone approach: {approach}")

    configs = []
    for seed in CONFIRMATION_SEEDS:
        for environment, mitigations in environment_mitigations.items():
            sigma_values = NOISE_LEVELS if "noise" in environment else (0.0,)
            for sigma in sigma_values:
                for mitigation in mitigations:
                    configs.append(
                        _standalone_config(
                            split=split,
                            approach=approach,
                            environment=environment,
                            mitigation=mitigation,
                            sigma=sigma,
                            seed=seed,
                        )
                    )
    return configs


def build_current_configs(
    split: str,
    approach: str,
    *,
    gamma: float | dict[str, float] | None = None,
) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if approach not in APPROACHES:
        raise ValueError(f"Unknown approach: {approach}")
    if approach in ("basil", "noisy"):
        return _standalone_matrix(split, approach)

    if approach == "merged":
        return build_approach_confirmation(
            "merged",
            split=split,
            gamma=0.0,
        )

    selected_gamma = frozen_cart_schedule() if gamma is None else gamma
    if selected_gamma is None:
        raise RuntimeError("CART calibration must be frozen first.")
    return build_approach_confirmation(
        "cart",
        split=split,
        gamma=selected_gamma,
    )


def config_filename(config: dict) -> str:
    return f"{int(config['seed'])} - {config['conditionId']}.json"
