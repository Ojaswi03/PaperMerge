"""Campaign-three configuration and result contracts.

This module deliberately has no GUI or TensorFlow imports.  It is the single
source of truth for the hidden-attack experiment matrix, stable run IDs, and
the isolated ``experiments/results3/r2`` directory layout.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


CAMPAIGN_ID = "hidden_cifar10_v3_r2"
CAMPAIGN_VERSION = 3
PROTOCOL_REVISION = "campaign3-2026-07-25-r2"
RESULT_ROOT = Path("experiments") / "results3" / "r2" / "gui"
CALIBRATION_STATE = (
    Path("experiments") / "results3" / "r2" / "campaign_state.json"
)
PLOT_ROOT = Path("plots3") / "r2"

CONFIRMATION_SEEDS = (2026, 2027, 2028)
CALIBRATION_SEED = 2025
NOISE_LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6)
REPRESENTATIVE_NOISE_LEVELS = (0.2, 0.4, 0.6)
CART_GAMMA_CANDIDATES = (0.0, 0.00025, 0.0005, 0.001)
CART_LOW_NOISE_REFINEMENT_GAMMAS = (0.00030, 0.00035, 0.00040)
HIDDEN_ATTACKERS = (1, 4, 6, 8)
EBM_OBJECTIVE_COEFFICIENT_BY_SIGMA = {
    0.2: 0.00100,
    0.3: 0.00025,
    0.4: 0.00025,
    0.5: 0.00010,
    0.6: 0.00010,
}
LOW_NOISE_EBM_PATCH = "sigma_0_2_ebm_coefficient_0_001"
CART_LOW_NOISE_REFINEMENT_PATCH = (
    "cart_sigma_0_2_ebm_coefficient_0_00025_refinement"
)
CART_LOW_NOISE_CONFIRMATION_PATCH = (
    "cart_sigma_0_2_ebm_coefficient_0_00025_confirmation"
)
CART_LOW_NOISE_EBM_COEFFICIENT = 0.00025
MERGED_JOINT_NONINFERIORITY_MARGIN = 0.01

PRESET_LABELS = {
    "cart_low_noise_refinement": "CART Low-Noise Refinement R2",
    "repair": "Low-Noise Repair R2",
    "calibration": "Calibration R2",
    "core_confirmation": "Core Confirmation R2",
    "merged_core": "Merged Core R2",
    "cart_addon": "CART Add-on R2",
    "cart_controls": "CART Non-IID Controls R2",
    "iid_controls": "IID Controls R2",
    "full_campaign": "Full Campaign R2",
}

_MITIGATION_LABELS = {
    "none": "No Mitigation",
    "ss": "SS",
    "ebm": "EBM",
    "ss_ebm": "SS+EBM",
}


def _canonical_payload(config: dict) -> dict:
    ignored = {
        "experimentName",
        "runId",
        "_cleanReferenceNote",
        "_autoTuningNote",
        "status",
        "startedAt",
        "completedAt",
    }
    return {key: config[key] for key in sorted(config) if key not in ignored}


def config_hash(config: dict) -> str:
    payload = json.dumps(
        _canonical_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def make_run_id(config: dict) -> str:
    return f"v3r2-{config_hash(config)[:20]}"


def _sigma_label(sigma: float) -> str:
    return f"{float(sigma):.1f}".replace(".", "_")


def ebm_objective_coefficient(sigma: float) -> float:
    sigma = round(float(sigma), 1)
    try:
        return EBM_OBJECTIVE_COEFFICIENT_BY_SIGMA[sigma]
    except KeyError as exc:
        raise ValueError(f"No predeclared EBM coefficient for sigma={sigma:.1f}") from exc


def _condition_id(environment: str, mitigation: str, sigma: float) -> str:
    if environment == "clean":
        return "clean_no_mitigation"
    if environment == "hidden":
        return f"hidden_{mitigation}"
    if environment == "noise":
        return f"noise_sigma_{_sigma_label(sigma)}_{mitigation}"
    return f"hidden_noise_sigma_{_sigma_label(sigma)}_{mitigation}"


def _experiment_name(
    split: str,
    approach: str,
    environment: str,
    mitigation: str,
    sigma: float,
    seed: int,
    gamma: float,
) -> str:
    environment_labels = {
        "clean": "Clean Environment",
        "hidden": "Hidden Attack",
        "noise": f"Channel Noise sigma={sigma:.1f}",
        "hidden_noise": f"Hidden Attack + Channel Noise sigma={sigma:.1f}",
    }
    cart_suffix = f" | gamma={gamma:g}" if approach == "cart" else ""
    return (
        f"V3-R2 | {split} | {approach.upper()} | "
        f"{environment_labels[environment]} | {_MITIGATION_LABELS[mitigation]}"
        f"{cart_suffix} | seed={seed}"
    )


def make_config(
    *,
    split: str,
    approach: str,
    environment: str,
    mitigation: str,
    seed: int,
    sigma: float = 0.0,
    rounds: int = 100,
    phase: str = "confirmation",
    gamma: float = 0.4,
    ebm_coefficient_override: float | None = None,
    protocol_patch: str | None = None,
) -> dict:
    if split not in ("nonIID", "IID"):
        raise ValueError(f"Unknown split: {split}")
    if approach not in ("merged", "cart"):
        raise ValueError(f"Campaign three supports merged/cart, got: {approach}")
    if environment not in ("clean", "hidden", "noise", "hidden_noise"):
        raise ValueError(f"Unknown environment: {environment}")
    if mitigation not in _MITIGATION_LABELS:
        raise ValueError(f"Unknown mitigation: {mitigation}")

    has_hidden = environment in ("hidden", "hidden_noise")
    has_noise = environment in ("noise", "hidden_noise")
    use_ss = mitigation in ("ss", "ss_ebm")
    use_ebm = mitigation in ("ebm", "ss_ebm")

    if environment == "clean" and (mitigation != "none" or sigma):
        raise ValueError("Clean Environment must be noiseless and unmitigated.")
    if not has_noise and use_ebm:
        raise ValueError("EBM is valid only when channel noise is active.")
    if not has_hidden and use_ss:
        raise ValueError("SS is valid only when the Hidden attack is active.")
    if has_noise and float(sigma) <= 0:
        raise ValueError("A noisy environment requires sigma > 0.")
    if ebm_coefficient_override is not None and not use_ebm:
        raise ValueError("An EBM coefficient override requires active EBM.")

    sigma = float(sigma) if has_noise else 0.0
    # Lambda remains a declared hyperparameter of the same Equation (13)
    # objective. R2 uses a predeclared noise-dependent schedule because the
    # fixed coefficient regressed at sigma=0.6 in paired pilots.
    ebm_coefficient = (
        float(ebm_coefficient_override)
        if ebm_coefficient_override is not None
        else ebm_objective_coefficient(sigma)
        if use_ebm
        else 0.0
    )
    if ebm_coefficient < 0:
        raise ValueError("The EBM objective coefficient cannot be negative.")
    ebm_lambda = (
        ebm_coefficient / (sigma * sigma)
        if use_ebm
        else 0.0
    )
    if protocol_patch == CART_LOW_NOISE_REFINEMENT_PATCH:
        ebm_schedule = "cart_low_noise_refinement"
    elif ebm_coefficient_override is not None:
        ebm_schedule = "cart_low_noise_method_specific_confirmation"
    else:
        ebm_schedule = "predeclared_piecewise_by_sigma"
    condition_id = _condition_id(environment, mitigation, sigma)
    hidden_start = 20 if has_hidden else 0

    config = {
        "schemaVersion": 3,
        "campaignVersion": CAMPAIGN_VERSION,
        "campaignId": CAMPAIGN_ID,
        "protocolRevision": PROTOCOL_REVISION,
        "phase": phase,
        "conditionId": condition_id,
        "environment": environment,
        "experimentName": _experiment_name(
            split, approach, environment, mitigation, sigma, seed, gamma
        ),
        "dataset": "cifar10",
        "approach": approach,
        "cartAlgorithm": "cart",
        "snapshotSelection": use_ss,
        "useBasil": use_ss,
        "basilMemorySize": 5,
        "snapshotSelectionRule": "plausibility_guard_then_lowest_loss_received_neighbor",
        "snapshotPlausibilityGuard": "relative_l2_channel_budget",
        "snapshotMaxRelativeDistance": (sigma if has_noise else 0.0) + 0.35,
        "aggregationMode": (
            "full_consensus" if environment == "clean" else "pairwise_consensus"
        ),
        "useChannelNoise": has_noise,
        "channelNoiseStart": 0,
        "channelNoiseSigma": sigma,
        "channelNoiseSemantics": "relative_l2_per_link",
        "noiseMitigation": "ebm" if use_ebm else "none",
        "ebmLambda": ebm_lambda,
        "ebmTargetCoefficient": ebm_coefficient,
        "ebmCoefficientSchedule": ebm_schedule,
        "ebmObjective": "F_plus_lambda_sigma2_gradient_norm_squared",
        "ebmImplementation": "bounded_second_order_microbatch_gradient_accumulation",
        "attackGaussian": False,
        "attackGaussianStart": 0,
        "attackSignFlip": False,
        "attackSignFlipStart": 0,
        "attackHidden": has_hidden,
        "attackHiddenStart": hidden_start,
        "attackWarmupRounds": hidden_start,
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
        "attackerIds": ",".join(str(node) for node in HIDDEN_ATTACKERS) if has_hidden else "",
        "nonIID": split == "nonIID",
        "dirichletAlpha": 0.2,
        "nNodes": 10,
        "nRounds": int(rounds),
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
        "distillStrength": float(gamma) if approach == "cart" else 0.0,
        "cartEmaBeta": 0.85,
        "cartGapRule": "registry_target_reproduced_by_consensus_reference",
        "verifyThreshold": 0.05,
        "seed": int(seed),
        "sharedGpuWorker": True,
        "autoPlotCampaign3": True,
    }
    if protocol_patch is None and use_ebm and abs(sigma - 0.2) < 1e-12:
        protocol_patch = LOW_NOISE_EBM_PATCH
    if protocol_patch:
        config["protocolPatch"] = protocol_patch
    config["runId"] = make_run_id(config)
    return config


def _gamma_for_sigma(gamma: float | dict[str, float], sigma: float) -> float:
    if not isinstance(gamma, dict):
        return float(gamma)

    sigma = float(sigma)
    if sigma <= 0:
        key = "default"
    elif sigma <= 0.3:
        key = "0.2"
    elif sigma <= 0.4:
        key = "0.4"
    else:
        key = "0.6"
    if key in gamma:
        return float(gamma[key])
    if key == "default" and "0.4" in gamma:
        return float(gamma["0.4"])
    raise ValueError(f"CART gamma schedule is missing the {key!r} bucket.")


def _full_approach_matrix(
    approach: str,
    *,
    split: str,
    seeds: Iterable[int],
    rounds: int,
    phase: str,
    gamma: float | dict[str, float],
) -> list[dict]:
    configs: list[dict] = []
    for seed in seeds:
        configs.append(
            make_config(
                split=split,
                approach=approach,
                environment="clean",
                mitigation="none",
                seed=seed,
                rounds=rounds,
                phase=phase,
                gamma=_gamma_for_sigma(gamma, 0.0),
            )
        )
        for mitigation in ("none", "ss"):
            configs.append(
                make_config(
                    split=split,
                    approach=approach,
                    environment="hidden",
                    mitigation=mitigation,
                    seed=seed,
                    rounds=rounds,
                    phase=phase,
                    gamma=_gamma_for_sigma(gamma, 0.0),
                )
            )
        for sigma in NOISE_LEVELS:
            for mitigation in ("none", "ebm"):
                cart_low_noise_ebm = (
                    approach == "cart"
                    and abs(sigma - 0.2) < 1e-12
                    and mitigation == "ebm"
                )
                configs.append(
                    make_config(
                        split=split,
                        approach=approach,
                        environment="noise",
                        mitigation=mitigation,
                        sigma=sigma,
                        seed=seed,
                        rounds=rounds,
                        phase=phase,
                        gamma=_gamma_for_sigma(gamma, sigma),
                        ebm_coefficient_override=(
                            CART_LOW_NOISE_EBM_COEFFICIENT
                            if cart_low_noise_ebm
                            else None
                        ),
                        protocol_patch=(
                            CART_LOW_NOISE_CONFIRMATION_PATCH
                            if cart_low_noise_ebm
                            else None
                        ),
                    )
                )
            for mitigation in ("none", "ss", "ebm", "ss_ebm"):
                cart_low_noise_ebm = (
                    approach == "cart"
                    and abs(sigma - 0.2) < 1e-12
                    and mitigation in ("ebm", "ss_ebm")
                )
                configs.append(
                    make_config(
                        split=split,
                        approach=approach,
                        environment="hidden_noise",
                        mitigation=mitigation,
                        sigma=sigma,
                        seed=seed,
                        rounds=rounds,
                        phase=phase,
                        gamma=_gamma_for_sigma(gamma, sigma),
                        ebm_coefficient_override=(
                            CART_LOW_NOISE_EBM_COEFFICIENT
                            if cart_low_noise_ebm
                            else None
                        ),
                        protocol_patch=(
                            CART_LOW_NOISE_CONFIRMATION_PATCH
                            if cart_low_noise_ebm
                            else None
                        ),
                    )
                )
    return configs


def build_cart_low_noise_refinement() -> list[dict]:
    """Return the paired CART-only low-noise calibration refinement."""
    configs = []
    for gamma in CART_LOW_NOISE_REFINEMENT_GAMMAS:
        for mitigation in ("ss", "ss_ebm"):
            configs.append(
                make_config(
                    split="nonIID",
                    approach="cart",
                    environment="hidden_noise",
                    mitigation=mitigation,
                    sigma=0.2,
                    seed=CALIBRATION_SEED,
                    rounds=30,
                    phase="calibration",
                    gamma=gamma,
                    ebm_coefficient_override=(
                        CART_LOW_NOISE_EBM_COEFFICIENT
                        if mitigation == "ss_ebm"
                        else None
                    ),
                    protocol_patch=CART_LOW_NOISE_REFINEMENT_PATCH,
                )
            )
    return configs


def build_calibration() -> list[dict]:
    configs: list[dict] = []
    configs.append(
        make_config(
            split="nonIID",
            approach="merged",
            environment="clean",
            mitigation="none",
            seed=CALIBRATION_SEED,
            rounds=30,
            phase="calibration",
            gamma=0.0,
        )
    )
    for mitigation in ("none", "ss"):
        configs.append(
            make_config(
                split="nonIID",
                approach="merged",
                environment="hidden",
                mitigation=mitigation,
                seed=CALIBRATION_SEED,
                rounds=30,
                phase="calibration",
                gamma=0.0,
            )
        )
    for sigma in REPRESENTATIVE_NOISE_LEVELS:
        for environment, mitigations in (
            ("noise", ("none", "ebm")),
            ("hidden_noise", ("ss", "ss_ebm")),
        ):
            for mitigation in mitigations:
                configs.append(
                    make_config(
                        split="nonIID",
                        approach="merged",
                        environment=environment,
                        mitigation=mitigation,
                        sigma=sigma,
                        seed=CALIBRATION_SEED,
                        rounds=30,
                        phase="calibration",
                        gamma=0.0,
                    )
                )

    for gamma in CART_GAMMA_CANDIDATES:
        configs.append(
            make_config(
                split="nonIID",
                approach="cart",
                environment="clean",
                mitigation="none",
                seed=CALIBRATION_SEED,
                rounds=30,
                phase="calibration",
                gamma=gamma,
            )
        )
        for sigma in REPRESENTATIVE_NOISE_LEVELS:
            for mitigation in ("ss", "ss_ebm"):
                configs.append(
                    make_config(
                        split="nonIID",
                        approach="cart",
                        environment="hidden_noise",
                        mitigation=mitigation,
                        sigma=sigma,
                        seed=CALIBRATION_SEED,
                        rounds=30,
                        phase="calibration",
                        gamma=gamma,
                    )
                )
    configs.extend(build_cart_low_noise_refinement())
    return configs


def build_repair() -> list[dict]:
    """Return only calibration runs superseded by the low-noise EBM patch."""
    return [
        config
        for config in build_calibration()
        if config.get("protocolPatch") == LOW_NOISE_EBM_PATCH
    ]


def build_merged_core() -> list[dict]:
    return build_approach_confirmation(
        "merged",
        split="nonIID",
        gamma=0.0,
    )


def build_cart_addon(gamma: float | dict[str, float]) -> list[dict]:
    return build_approach_confirmation(
        "cart",
        split="nonIID",
        gamma=gamma,
    )


def build_approach_confirmation(
    approach: str,
    *,
    split: str,
    gamma: float | dict[str, float],
) -> list[dict]:
    """Return the complete three-seed matrix for one supported approach."""
    return _full_approach_matrix(
        approach,
        split=split,
        seeds=CONFIRMATION_SEEDS,
        rounds=100,
        phase="confirmation",
        gamma=gamma,
    )


def build_cart_controls(gamma: float | dict[str, float]) -> list[dict]:
    """Return direct clean and hidden-only CART confirmation controls."""
    return [
        config
        for config in build_cart_addon(gamma)
        if config["environment"] in ("clean", "hidden")
    ]


def build_iid_controls(gamma: float | dict[str, float]) -> list[dict]:
    configs: list[dict] = []
    for seed in CONFIRMATION_SEEDS:
        for approach in ("merged", "cart"):
            configs.append(
                make_config(
                    split="IID",
                    approach=approach,
                    environment="clean",
                    mitigation="none",
                    seed=seed,
                    gamma=_gamma_for_sigma(gamma, 0.0),
                )
            )
        for mitigation in ("none", "ss"):
            configs.append(
                make_config(
                    split="IID",
                    approach="merged",
                    environment="hidden",
                    mitigation=mitigation,
                    seed=seed,
                    gamma=0.0,
                )
            )
        for sigma in REPRESENTATIVE_NOISE_LEVELS:
            for mitigation in ("none", "ebm"):
                configs.append(
                    make_config(
                        split="IID",
                        approach="merged",
                        environment="noise",
                        mitigation=mitigation,
                        sigma=sigma,
                        seed=seed,
                        gamma=0.0,
                    )
                )
            for approach in ("merged", "cart"):
                cart_low_noise_ebm = (
                    approach == "cart" and abs(sigma - 0.2) < 1e-12
                )
                configs.append(
                    make_config(
                        split="IID",
                        approach=approach,
                        environment="hidden_noise",
                        mitigation="ss_ebm",
                        sigma=sigma,
                        seed=seed,
                        gamma=_gamma_for_sigma(gamma, sigma),
                        ebm_coefficient_override=(
                            CART_LOW_NOISE_EBM_COEFFICIENT
                            if cart_low_noise_ebm
                            else None
                        ),
                        protocol_patch=(
                            CART_LOW_NOISE_CONFIRMATION_PATCH
                            if cart_low_noise_ebm
                            else None
                        ),
                    )
                )
    return configs


def build_core_confirmation(gamma: float | dict[str, float]) -> list[dict]:
    """Return the 105-run non-IID matrix needed for the paper's core claims.

    The full 246-run preset remains available for supplementary experiments.
    This matrix keeps every noise level for the central SS versus SS+EBM and
    Merged versus CART comparisons, while using the representative noise
    levels for component controls.
    """
    configs: list[dict] = []
    for seed in CONFIRMATION_SEEDS:
        configs.append(
            make_config(
                split="nonIID",
                approach="merged",
                environment="clean",
                mitigation="none",
                seed=seed,
                gamma=0.0,
            )
        )
        for mitigation in ("none", "ss"):
            configs.append(
                make_config(
                    split="nonIID",
                    approach="merged",
                    environment="hidden",
                    mitigation=mitigation,
                    seed=seed,
                    gamma=0.0,
                )
            )
        for sigma in REPRESENTATIVE_NOISE_LEVELS:
            for mitigation in ("none", "ebm"):
                configs.append(
                    make_config(
                        split="nonIID",
                        approach="merged",
                        environment="noise",
                        mitigation=mitigation,
                        sigma=sigma,
                        seed=seed,
                        gamma=0.0,
                    )
                )
        for sigma in NOISE_LEVELS:
            for approach in ("merged", "cart"):
                for mitigation in ("ss", "ss_ebm"):
                    cart_low_noise_ebm = (
                        approach == "cart"
                        and abs(sigma - 0.2) < 1e-12
                        and mitigation == "ss_ebm"
                    )
                    configs.append(
                        make_config(
                            split="nonIID",
                            approach=approach,
                            environment="hidden_noise",
                            mitigation=mitigation,
                            sigma=sigma,
                            seed=seed,
                            gamma=(
                                _gamma_for_sigma(gamma, sigma)
                                if approach == "cart"
                                else 0.0
                            ),
                            ebm_coefficient_override=(
                                CART_LOW_NOISE_EBM_COEFFICIENT
                                if cart_low_noise_ebm
                                else None
                            ),
                            protocol_patch=(
                                CART_LOW_NOISE_CONFIRMATION_PATCH
                                if cart_low_noise_ebm
                                else None
                            ),
                        )
                    )
        for sigma in REPRESENTATIVE_NOISE_LEVELS:
            for mitigation in ("none", "ebm"):
                configs.append(
                    make_config(
                        split="nonIID",
                        approach="merged",
                        environment="hidden_noise",
                        mitigation=mitigation,
                        sigma=sigma,
                        seed=seed,
                        gamma=0.0,
                    )
                )
    return configs


def load_calibration_state(path: Path | str = CALIBRATION_STATE) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else None
    except (OSError, ValueError):
        return None


def frozen_cart_gamma(path: Path | str = CALIBRATION_STATE) -> float | None:
    state = load_calibration_state(path)
    if state and state.get("status") == "frozen":
        return float(state["cartGamma"])
    return None


def frozen_cart_schedule(
    path: Path | str = CALIBRATION_STATE,
) -> dict[str, float] | None:
    state = load_calibration_state(path)
    if not state or state.get("status") != "frozen":
        return None

    raw_schedule = state.get("cartGammaBySigma")
    if isinstance(raw_schedule, dict):
        schedule = {
            str(key): float(value)
            for key, value in raw_schedule.items()
        }
        schedule["default"] = float(
            state.get("cartGamma", schedule.get("0.4", 0.0))
        )
        return schedule

    gamma = float(state["cartGamma"])
    return {
        "0.2": gamma,
        "0.4": gamma,
        "0.6": gamma,
        "default": gamma,
    }


def freeze_calibration_if_ready(
    *,
    result_root: Path | str = RESULT_ROOT,
    state_path: Path | str = CALIBRATION_STATE,
) -> dict | None:
    """Freeze the predeclared per-noise CART schedule after calibration."""
    expected = build_calibration()
    records = {}
    for config in expected:
        metrics_path, run_path = result_paths(config, root=result_root)
        if not metrics_path.exists() or not run_path.exists():
            return None
        try:
            with run_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("status") != "completed":
                return None
            with np.load(metrics_path, allow_pickle=False) as metrics:
                final_average = float(metrics["final_avg"])
        except (OSError, ValueError, KeyError):
            return None
        records[config["runId"]] = final_average

    def final_for(config):
        return records[config["runId"]]

    cart_configs = [
        config for config in expected if config.get("approach") == "cart"
    ]
    merged_configs = [
        config for config in expected if config.get("approach") == "merged"
    ]
    merged_noise_gain = {}
    merged_joint_gain = {}
    merged_joint_score = {}
    merged_clean = next(
        config for config in merged_configs if config["environment"] == "clean"
    )
    merged_hidden_none = next(
        config
        for config in merged_configs
        if config["environment"] == "hidden"
        and not config["snapshotSelection"]
    )
    merged_hidden_ss = next(
        config
        for config in merged_configs
        if config["environment"] == "hidden"
        and config["snapshotSelection"]
    )
    merged_hidden_gain = final_for(merged_hidden_ss) - final_for(merged_hidden_none)
    merged_clean_score = final_for(merged_clean)
    for sigma in REPRESENTATIVE_NOISE_LEVELS:
        noise_none = next(
            config
            for config in merged_configs
            if config["environment"] == "noise"
            and config["noiseMitigation"] == "none"
            and abs(float(config["channelNoiseSigma"]) - sigma) < 1e-12
        )
        noise_ebm = next(
            config
            for config in merged_configs
            if config["environment"] == "noise"
            and config["noiseMitigation"] == "ebm"
            and abs(float(config["channelNoiseSigma"]) - sigma) < 1e-12
        )
        joint_ss = next(
            config
            for config in merged_configs
            if config["environment"] == "hidden_noise"
            and config["snapshotSelection"]
            and config["noiseMitigation"] == "none"
            and abs(float(config["channelNoiseSigma"]) - sigma) < 1e-12
        )
        joint_both = next(
            config
            for config in merged_configs
            if config["environment"] == "hidden_noise"
            and config["snapshotSelection"]
            and config["noiseMitigation"] == "ebm"
            and abs(float(config["channelNoiseSigma"]) - sigma) < 1e-12
        )
        merged_noise_gain[str(sigma)] = final_for(noise_ebm) - final_for(noise_none)
        merged_joint_gain[str(sigma)] = final_for(joint_both) - final_for(joint_ss)
        merged_joint_score[str(sigma)] = final_for(joint_both)

    all_cart_gamma_candidates = (
        CART_GAMMA_CANDIDATES + CART_LOW_NOISE_REFINEMENT_GAMMAS
    )
    zero_clean_config = next(
        config
        for config in cart_configs
        if config["environment"] == "clean"
        and abs(float(config.get("distillStrength", 0.0))) < 1e-12
    )
    zero_clean = final_for(zero_clean_config)
    scores = {}
    clean_scores = {}
    clean_score_sources = {}
    composition_deltas = {}
    minimum_composition_deltas = {}
    cart_lifts = {}
    minimum_cart_lifts = {}
    cart_final_by_sigma = {}
    cart_composition_by_sigma = {}
    cart_lift_by_sigma = {}
    cart_ceiling_ok = {}
    for gamma in all_cart_gamma_candidates:
        gamma_configs = [
            config
            for config in cart_configs
            if abs(float(config.get("distillStrength", 0.0)) - gamma) < 1e-12
        ]
        clean_config = next(
            (
                config
                for config in gamma_configs
                if config["environment"] == "clean"
            ),
            None,
        )
        if clean_config is None:
            clean_scores[str(gamma)] = zero_clean
            clean_score_sources[str(gamma)] = (
                "gamma_zero_clean_control_cart_mu_is_exactly_zero"
            )
        else:
            clean_scores[str(gamma)] = final_for(clean_config)
            clean_score_sources[str(gamma)] = clean_config["runId"]
        target = [
            final_for(config)
            for config in gamma_configs
            if config["environment"] == "hidden_noise"
            and config["noiseMitigation"] == "ebm"
            and config["snapshotSelection"]
        ]
        ss_lookup = {
            float(config["channelNoiseSigma"]): final_for(config)
            for config in gamma_configs
            if config["environment"] == "hidden_noise"
            and config["snapshotSelection"]
            and config["noiseMitigation"] == "none"
        }
        both_lookup = {
            float(config["channelNoiseSigma"]): final_for(config)
            for config in gamma_configs
            if config["environment"] == "hidden_noise"
            and config["snapshotSelection"]
            and config["noiseMitigation"] == "ebm"
        }
        available_sigmas = sorted(set(ss_lookup) & set(both_lookup))
        composition_values = [
            both_lookup[sigma] - ss_lookup[sigma]
            for sigma in available_sigmas
        ]
        lift_values = [
            both_lookup[sigma] - merged_joint_score[str(sigma)]
            for sigma in available_sigmas
        ]
        composition_deltas[str(gamma)] = float(np.mean(composition_values))
        minimum_composition_deltas[str(gamma)] = float(np.min(composition_values))
        cart_lifts[str(gamma)] = float(np.mean(lift_values))
        minimum_cart_lifts[str(gamma)] = float(np.min(lift_values))
        scores[str(gamma)] = float(np.mean(target))
        cart_final_by_sigma[str(gamma)] = {
            str(sigma): float(both_lookup[sigma])
            for sigma in available_sigmas
        }
        cart_composition_by_sigma[str(gamma)] = {
            str(sigma): float(both_lookup[sigma] - ss_lookup[sigma])
            for sigma in available_sigmas
        }
        cart_lift_by_sigma[str(gamma)] = {
            str(sigma): float(
                both_lookup[sigma] - merged_joint_score[str(sigma)]
            )
            for sigma in available_sigmas
        }
        cart_ceiling_ok[str(gamma)] = bool(
            max(final_for(config) for config in gamma_configs)
            <= merged_clean_score
        )

    core_passed = (
        merged_hidden_gain >= 0.0
        and min(merged_noise_gain.values()) >= 0.0
        and min(merged_joint_gain.values())
        >= -MERGED_JOINT_NONINFERIORITY_MARGIN
        and max(final_for(config) for config in merged_configs) <= merged_clean_score
    )
    eligible_by_sigma = {}
    selected_by_sigma = {}
    for sigma in REPRESENTATIVE_NOISE_LEVELS:
        sigma_key = str(sigma)
        eligible = [
            gamma
            for gamma in all_cart_gamma_candidates
            if gamma > 0
            and sigma_key in cart_final_by_sigma[str(gamma)]
            and clean_scores[str(gamma)] >= zero_clean - 0.02
            and cart_composition_by_sigma[str(gamma)][sigma_key] >= 0.0
            and cart_lift_by_sigma[str(gamma)][sigma_key] >= 0.0
            and cart_ceiling_ok[str(gamma)]
        ]
        eligible_by_sigma[sigma_key] = [float(gamma) for gamma in eligible]
        if eligible:
            selected_by_sigma[sigma_key] = float(
                max(
                    eligible,
                    key=lambda gamma: cart_final_by_sigma[str(gamma)][sigma_key],
                )
            )

    schedule_complete = len(selected_by_sigma) == len(
        REPRESENTATIVE_NOISE_LEVELS
    )
    default_gamma = float(selected_by_sigma.get("0.4", 0.0))
    status = "frozen" if core_passed and schedule_complete else "blocked"
    if status != "frozen":
        selected_by_sigma = {}
        default_gamma = 0.0
    blocked_reasons = []
    if not core_passed:
        blocked_reasons.append(
            "Merged failed at least one component/ceiling gate: hidden-only SS, "
            "noise-only EBM, joint SS+EBM non-inferiority, or clean upper "
            "reference."
        )
    if not schedule_complete:
        blocked_reasons.append(
            "At least one representative noise level has no non-zero CART "
            "gamma meeting the clean, SS+EBM-over-SS, CART-over-Merged, and "
            "clean-ceiling constraints."
        )
    state = {
        "schemaVersion": 3,
        "campaignId": CAMPAIGN_ID,
        "status": status,
        "cartGamma": default_gamma,
        "cartGammaBySigma": selected_by_sigma,
        "selectionRule": (
            "At each representative sigma, select the non-zero gamma with the "
            "highest final CART SS+EBM accuracy among candidates with <=0.02 "
            "clean regression, non-negative SS+EBM-over-SS and "
            "CART-over-Merged deltas, and no result above the clean upper "
            "reference. Merged hidden-only SS and noise-only EBM gains must be "
            "non-negative; Merged joint SS+EBM must be within the declared "
            "non-inferiority margin of SS."
        ),
        "blockedReasons": blocked_reasons,
        "scores": scores,
        "cleanScores": clean_scores,
        "cleanScoreSources": clean_score_sources,
        "compositionDeltas": composition_deltas,
        "minimumCompositionDeltas": minimum_composition_deltas,
        "cartLifts": cart_lifts,
        "minimumCartLifts": minimum_cart_lifts,
        "cartFinalBySigma": cart_final_by_sigma,
        "cartCompositionBySigma": cart_composition_by_sigma,
        "cartLiftBySigma": cart_lift_by_sigma,
        "cartEligibleGammaBySigma": eligible_by_sigma,
        "mergedNoiseOnlyGains": merged_noise_gain,
        "mergedJointGains": merged_joint_gain,
        "mergedJointNoninferiorityMargin": (
            MERGED_JOINT_NONINFERIORITY_MARGIN
        ),
        "mergedHiddenOnlySsGain": float(merged_hidden_gain),
        "mergedCleanScore": float(merged_clean_score),
    }
    write_json_atomic(state_path, state)
    return state


def build_preset(
    name: str,
    *,
    gamma: float | dict[str, float] | None = None,
) -> list[dict]:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key == "cart_low_noise_refinement":
        return build_cart_low_noise_refinement()
    if key == "repair":
        return build_repair()
    if key == "calibration":
        return build_calibration()
    if key == "merged_core":
        return build_merged_core()

    selected_gamma = frozen_cart_schedule() if gamma is None else gamma
    if selected_gamma is None:
        raise RuntimeError(
            "Run and complete Calibration R2 before loading CART confirmation presets."
        )
    if key == "cart_addon":
        return build_cart_addon(selected_gamma)
    if key == "cart_controls":
        return build_cart_controls(selected_gamma)
    if key == "iid_controls":
        return build_iid_controls(selected_gamma)
    if key == "core_confirmation":
        return build_core_confirmation(selected_gamma)
    if key == "full_campaign":
        return (
            build_merged_core()
            + build_cart_addon(selected_gamma)
            + build_iid_controls(selected_gamma)
        )
    raise ValueError(f"Unknown campaign-three preset: {name}")


def result_run_dir(config: dict, root: Path | str = RESULT_ROOT) -> Path:
    split = "nonIID" if config.get("nonIID", True) else "IID"
    attack_key = "hidden" if config.get("attackHidden", False) else "none"
    approach = config.get("approach", "merged")
    if config.get("useChannelNoise", False):
        bucket = f"sigma_{_sigma_label(float(config.get('channelNoiseSigma', 0.0)))}"
    else:
        bucket = "no_channel_noise"
    condition = config.get("conditionId", "unclassified")
    seed = int(config.get("seed", 0))
    path = Path(root) / split / "cifar10" / attack_key / approach / bucket / condition
    protocol_patch = config.get("protocolPatch")
    if protocol_patch:
        safe_patch = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in str(protocol_patch)
        )
        path = path / f"patch_{safe_patch}"
    if approach == "cart":
        gamma = f"{float(config.get('distillStrength', 0.0)):g}".replace(".", "_")
        path = path / f"gamma_{gamma}"
    return path / f"seed_{seed}"


def result_paths(config: dict, root: Path | str = RESULT_ROOT) -> tuple[Path, Path]:
    run_dir = result_run_dir(config, root=root)
    return run_dir / "metrics.npz", run_dir / "run.json"


def is_completed(config: dict, root: Path | str = RESULT_ROOT) -> bool:
    metrics_path, run_path = result_paths(config, root=root)
    if not metrics_path.exists() or not run_path.exists():
        return False
    try:
        with run_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return (
            metadata.get("status") == "completed"
            and metadata.get("runId") == config.get("runId")
            and metadata.get("configHash") == config_hash(config)
        )
    except (OSError, ValueError):
        return False


def write_json_atomic(path: Path | str, payload: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_npz_atomic(path: Path | str, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
