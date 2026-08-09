"""Versioned Campaign 4 experiment contracts and isolated output paths."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from gui.campaign3 import write_json_atomic, write_npz_atomic


CAMPAIGN_ID = "hidden_cifar10_v4"
CAMPAIGN_VERSION = 4
PROTOCOL_REVISION = "campaign4-2026-08-03-r2"

CONFIG_ROOT = Path("gui") / "configs" / "campaign4"
RESULT_BASE = Path("experiments") / "results4" / "campaign4"
RESULT_ROOT = RESULT_BASE / "gui"
WORKER_ROOT = RESULT_BASE / "workers"
PERFORMANCE_PROFILE = RESULT_BASE / "performance_profile.json"
CAMPAIGN_STATE = RESULT_BASE / "campaign_state.json"
PLOT_ROOT = Path("plots4") / "campaign4"

CONFIRMATION_SEEDS = (2026, 2027, 2028)
DIAGNOSTIC_SEED = 2025
NOISE_LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6)
DIAGNOSTIC_NOISE_LEVELS = (0.4, 0.5, 0.6)
HIDDEN_ATTACKERS = (1, 4, 6, 8)
ALLOWED_PHASES = frozenset(
    {
        "confirmation",
        "diagnostic",
        "static_control",
        "performance_benchmark",
        "precision_validation",
    }
)
PROVENANCE_FILES = (
    "basil_core/basil.py",
    "basil_core/campaign4_engine.py",
    "basil_core/class_registry.py",
    "basil_core/models.py",
    "basil_core/trainer.py",
    "basil_core/data/cifar.py",
    "gui/campaign4.py",
    "scripts/run_campaign4_worker.py",
)

# These reproduce the Campaign 3 static objective coefficients. They remain
# controls; adaptive EBM has a distinct method identifier and state.
STATIC_EBM_COEFFICIENT_BY_SIGMA = {
    0.2: 0.00100,
    0.3: 0.00025,
    0.4: 0.00025,
    0.5: 0.00010,
    0.6: 0.00010,
}

DEFAULT_CART_GAMMA = {
    "default": 0.0005,
    "0.2": 0.0003,
    "0.4": 0.0010,
    "0.6": 0.0005,
}

PRESET_LABELS = {
    "diagnostic_non_iid_merged": "Campaign 4 Diagnosis: non-IID Merged",
    "diagnostic_non_iid_cart": "Campaign 4 Diagnosis: non-IID CART",
    "static_controls": "Campaign 4 Static EBM Controls",
    "main_confirmation": "Campaign 4 Main Confirmation",
    "non_iid_confirmation": "Campaign 4 non-IID Confirmation",
    "iid_confirmation": "Campaign 4 IID Confirmation",
    "performance_benchmark": "Campaign 4 GPU Benchmark",
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
        "status",
        "startedAt",
        "completedAt",
        "_runtimeEstimate",
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
    return f"v4-{config_hash(config)[:24]}"


def apply_performance_profile(config: dict, profile: dict | None) -> dict:
    """Apply one validated machine profile without changing method semantics."""
    result = dict(config)
    profile = profile if isinstance(profile, dict) else {}
    selected = profile.get("selectedProfile")
    if (
        int(result.get("campaignVersion", 0)) != 4
        or profile.get("status") != "validated"
        or not isinstance(selected, dict)
        or result.get("phase") == "performance_benchmark"
        or result.get("performanceProfileFallback", False)
    ):
        return result
    for key in (
        "internalMicroBatchSize",
        "precisionProfile",
        "jitCompile",
        "gpuAllocator",
    ):
        if key in selected:
            result[key] = selected[key]
    if (
        result.get("noiseMitigation") == "ebm"
        and result.get("phase") != "performance_benchmark"
    ):
        result["internalMicroBatchSize"] = int(result.get("batchSize", 512))
        result["precisionProfile"] = str(
            selected.get("precisionProfile", "float32")
        )
    result["performanceProfileId"] = selected.get("profileId", "validated")
    result["runId"] = make_run_id(result)
    return result


def _sigma_key(sigma: float) -> float:
    return round(float(sigma), 1)


def _sigma_label(sigma: float) -> str:
    return f"{float(sigma):.1f}".replace(".", "_")


def static_ebm_coefficient(sigma: float) -> float:
    try:
        return STATIC_EBM_COEFFICIENT_BY_SIGMA[_sigma_key(sigma)]
    except KeyError as error:
        raise ValueError(f"No static EBM coefficient for sigma={sigma:.1f}") from error


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
    return float(gamma.get("default", 0.0))


def _condition_id(
    environment: str,
    mitigation: str,
    sigma: float,
    ebm_mode: str,
    weight_decay: float = 0.0,
) -> str:
    method = ""
    if mitigation in ("ebm", "ss_ebm"):
        method = f"_{ebm_mode}_ebm"
    suffix = (
        f"_wd_{weight_decay:g}".replace(".", "_")
        if float(weight_decay) > 0.0
        else ""
    )
    if environment == "clean":
        return "clean_pairwise_no_mitigation" + suffix
    if environment == "clean_ceiling":
        return "clean_full_consensus_ceiling" + suffix
    if environment == "hidden":
        return f"hidden_{mitigation}" + suffix
    prefix = f"sigma_{_sigma_label(sigma)}"
    if environment == "noise":
        return f"noise_{prefix}_{mitigation}{method}" + suffix
    return f"hidden_noise_{prefix}_{mitigation}{method}" + suffix


def _experiment_name(config: dict) -> str:
    sigma = float(config["channelNoiseSigma"])
    environment = config["environment"]
    environment_label = {
        "clean": "Protocol-Matched Clean Ring",
        "clean_ceiling": "Ideal Full-Consensus Clean Ceiling",
        "hidden": "Hidden Attack",
        "noise": f"Channel Noise sigma={sigma:.1f}",
        "hidden_noise": f"Hidden Attack + Channel Noise sigma={sigma:.1f}",
    }[environment]
    method = ""
    if config["ebmMode"] != "none":
        method = f" ({config['ebmMode'].title()} EBM)"
    cart = (
        f" | gamma={float(config['distillStrength']):g}"
        if config["approach"] == "cart"
        else ""
    )
    optimizer = (
        " | optimizer=persistent"
        if config.get("optimizerStateMode") == "persistent"
        else ""
    )
    weight_decay = (
        f" | weight_decay={float(config.get('weightDecayCoefficient', 0.0)):g}"
        if float(config.get("weightDecayCoefficient", 0.0)) > 0.0
        else ""
    )
    return (
        f"V4 | {config['split']} | {config['approach'].upper()} | "
        f"{environment_label} | {_MITIGATION_LABELS[config['mitigation']]}"
        f"{method}{cart}{optimizer}{weight_decay} | seed={config['seed']}"
    )


def make_config(
    *,
    split: str,
    approach: str,
    environment: str,
    mitigation: str,
    seed: int,
    sigma: float = 0.0,
    ebm_mode: str = "adaptive",
    rounds: int = 100,
    phase: str = "confirmation",
    gamma: float | dict[str, float] = DEFAULT_CART_GAMMA,
    optimizer_state_mode: str = "visit_reset",
    internal_micro_batch: int = 512,
    jit_compile: bool = False,
    precision: str = "float32",
    weight_decay_coefficient: float = 0.0,
    adaptive_weight_decay_mode: str = "none",
    adaptive_weight_decay_target_ratio: float = 2.0,
    adaptive_weight_decay_gain: float = 0.025,
    adaptive_weight_decay_coefficient_min: float = 1e-6,
    adaptive_weight_decay_coefficient_max: float = 0.05,
    adaptive_weight_decay_beta: float = 0.9,
    adaptive_weight_decay_max_change_factor: float = 2.0,
) -> dict:
    """Build one explicit Campaign 4 configuration.

    ``mitigation`` identifies which defenses are active. ``ebm_mode`` only has
    an effect when mitigation includes EBM, and is always stored explicitly.
    """
    if split not in ("IID", "nonIID"):
        raise ValueError(f"Unknown split: {split}")
    if approach not in ("merged", "cart"):
        raise ValueError(f"Campaign 4 supports merged/cart, got: {approach}")
    if environment not in (
        "clean",
        "clean_ceiling",
        "hidden",
        "noise",
        "hidden_noise",
    ):
        raise ValueError(f"Unknown environment: {environment}")
    if mitigation not in _MITIGATION_LABELS:
        raise ValueError(f"Unknown mitigation: {mitigation}")
    if ebm_mode not in ("none", "static", "adaptive"):
        raise ValueError(f"Unknown EBM mode: {ebm_mode}")
    if optimizer_state_mode not in ("visit_reset", "persistent"):
        raise ValueError(f"Unknown optimizer state mode: {optimizer_state_mode}")
    if int(internal_micro_batch) not in (128, 256, 512):
        raise ValueError("Internal microbatch must be 128, 256, or 512.")
    if precision != "float32":
        raise ValueError(
            "Generated Campaign 4 configs must use the float32 baseline; a "
            "validated machine execution profile is applied separately."
        )
    phase = str(phase)
    if phase not in ALLOWED_PHASES:
        raise ValueError(f"Unsupported Campaign 4 phase: {phase}")

    has_hidden = environment in ("hidden", "hidden_noise")
    has_noise = environment in ("noise", "hidden_noise")
    use_ss = mitigation in ("ss", "ss_ebm")
    use_ebm = mitigation in ("ebm", "ss_ebm")
    is_clean = environment in ("clean", "clean_ceiling")

    if is_clean and (mitigation != "none" or float(sigma) != 0.0):
        raise ValueError("Clean references must be noiseless and unmitigated.")
    if use_ss and not has_hidden:
        raise ValueError("SS is valid only when the Hidden attack is active.")
    if use_ebm and not has_noise:
        raise ValueError("EBM is valid only when channel noise is active.")
    if adaptive_weight_decay_mode not in ("none", "adaptive"):
        raise ValueError(
            f"Unknown adaptive weight decay mode: {adaptive_weight_decay_mode}"
        )
    if adaptive_weight_decay_mode == "adaptive" and not has_noise:
        raise ValueError(
            "Adaptive weight decay is valid only when channel noise is active."
        )
    if has_noise and float(sigma) <= 0.0:
        raise ValueError("A noisy environment requires sigma > 0.")
    if not use_ebm:
        ebm_mode = "none"
    elif ebm_mode == "none":
        raise ValueError("An EBM mitigation requires static or adaptive mode.")
    if (
        use_ebm
        and int(internal_micro_batch) != 512
        and phase != "performance_benchmark"
    ):
        raise ValueError(
            "Official Campaign 4 EBM configs require internal_micro_batch=512 "
            "for the exact configured-batch gradient-norm objective."
        )

    sigma = float(sigma) if has_noise else 0.0
    initial_coefficient = static_ebm_coefficient(sigma) if use_ebm else 0.0
    gamma_value = _gamma_for_sigma(gamma, sigma) if approach == "cart" else 0.0
    hidden_start = 20 if has_hidden else 0

    config = {
        "schemaVersion": 4,
        "campaignVersion": CAMPAIGN_VERSION,
        "campaignId": CAMPAIGN_ID,
        "protocolRevision": PROTOCOL_REVISION,
        "phase": phase,
        "split": split,
        "dataset": "cifar10",
        "approach": approach,
        "environment": environment,
        "mitigation": mitigation,
        "aggregationMode": (
            "full_consensus" if environment == "clean_ceiling" else "pairwise_consensus"
        ),
        "protocolMatchedClean": environment == "clean",
        "snapshotSelection": use_ss,
        "useBasil": use_ss,
        "basilMemorySize": 5,
        "snapshotSelectionRule": "plausibility_guard_then_lowest_loss_received_neighbor",
        "snapshotPlausibilityGuard": "relative_l2_channel_budget",
        "snapshotMaxRelativeDistance": (sigma if has_noise else 0.0) + 0.35,
        "useChannelNoise": has_noise,
        "channelNoiseStart": 0,
        "channelNoiseSigma": sigma,
        "channelNoiseSemantics": "relative_l2_per_link_tf_stateless_v1",
        "noiseMitigation": "ebm" if use_ebm else "none",
        "ebmMode": ebm_mode,
        "ebmObjective": "F_plus_coefficient_gradient_norm_squared",
        "ebmImplementation": "full_batch_second_order_device_gradient_v2",
        "ebmGradientBatchSemantics": "exact_configured_batch",
        "ebmInitialCoefficient": initial_coefficient,
        "ebmLambda": initial_coefficient / (sigma * sigma) if use_ebm else 0.0,
        # Retuned 2026-08-05 from the sigma=0.4/0.5/0.6 noise-only diagnostic
        # (seed 2025): the prior bounds drove the applied coefficient to
        # ~20-60x the static schedule's calibrated values (active EBM ratio
        # ~0.20-0.30 vs static's incidental ~0.02-0.04), producing pre-clip
        # gradient-norm spikes as high as 1484 at sigma=0.6 and no accuracy
        # benefit over no mitigation. These bounds target the same order of
        # magnitude as the proven-safe static schedule while preserving a
        # genuine (if narrower) adaptive range.
        "adaptiveEbmTargetRatioBase": 0.015,
        "adaptiveEbmStressGain": 0.05,
        "adaptiveEbmRatioMin": 0.005,
        "adaptiveEbmRatioMax": 0.08,
        "adaptiveEbmCoefficientMin": 1e-6,
        "adaptiveEbmCoefficientMax": 0.0025,
        "adaptiveEbmBeta": 0.90,
        "adaptiveEbmMaxChangeFactor": 2.0,
        "attackHidden": has_hidden,
        "attackHiddenStart": hidden_start,
        "attackWarmupRounds": hidden_start,
        "attackerIds": (
            ",".join(str(node) for node in HIDDEN_ATTACKERS) if has_hidden else ""
        ),
        "attackGaussian": False,
        "attackGaussianStart": 0,
        "attackSignFlip": False,
        "attackSignFlipStart": 0,
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
        "nonIID": split == "nonIID",
        "dirichletAlpha": 0.2,
        "nNodes": 10,
        "nRounds": int(rounds),
        "localEpochs": 5,
        "stepsPerEpoch": 5,
        "learningRate": 0.05,
        "momentum": 0.9,
        "weightDecayCoefficient": float(weight_decay_coefficient),
        "adaptiveWeightDecayMode": str(adaptive_weight_decay_mode),
        "adaptiveWeightDecayTargetRatio": float(adaptive_weight_decay_target_ratio),
        "adaptiveWeightDecayGain": float(adaptive_weight_decay_gain),
        "adaptiveWeightDecayCoefficientMin": float(adaptive_weight_decay_coefficient_min),
        "adaptiveWeightDecayCoefficientMax": float(adaptive_weight_decay_coefficient_max),
        "adaptiveWeightDecayBeta": float(adaptive_weight_decay_beta),
        "adaptiveWeightDecayMaxChangeFactor": float(adaptive_weight_decay_max_change_factor),
        "optimizerStateMode": optimizer_state_mode,
        "batchSize": 512,
        "internalMicroBatchSize": int(internal_micro_batch),
        "precisionProfile": precision,
        "jitCompile": bool(jit_compile),
        "gpuStateMode": "device_tensors",
        "useLrDecay": True,
        "distillStrength": gamma_value,
        "cartEmaBeta": 0.85,
        "cartGapRule": "registry_target_reproduced_by_consensus_reference",
        "verifyThreshold": 0.05,
        "seed": int(seed),
        "telemetrySchemaVersion": 1,
        "autoPlotCampaign4": True,
    }
    config["conditionId"] = _condition_id(
        environment, mitigation, sigma, ebm_mode, weight_decay_coefficient
    )
    config["experimentName"] = _experiment_name(config)
    config["runId"] = make_run_id(config)
    return config


def _main_matrix(
    *,
    splits: Iterable[str],
    approaches: Iterable[str],
    seeds: Iterable[int],
    rounds: int = 100,
) -> list[dict]:
    configs: list[dict] = []
    for split in splits:
        for approach in approaches:
            for seed in seeds:
                configs.append(
                    make_config(
                        split=split,
                        approach=approach,
                        environment="clean",
                        mitigation="none",
                        seed=seed,
                        rounds=rounds,
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
                        )
                    )
                for sigma in NOISE_LEVELS:
                    for mitigation in ("none", "ebm"):
                        configs.append(
                            make_config(
                                split=split,
                                approach=approach,
                                environment="noise",
                                mitigation=mitigation,
                                ebm_mode="adaptive",
                                sigma=sigma,
                                seed=seed,
                                rounds=rounds,
                            )
                        )
                    for mitigation in ("none", "ss", "ebm", "ss_ebm"):
                        configs.append(
                            make_config(
                                split=split,
                                approach=approach,
                                environment="hidden_noise",
                                mitigation=mitigation,
                                ebm_mode="adaptive",
                                sigma=sigma,
                                seed=seed,
                                rounds=rounds,
                            )
                        )
    return configs


def build_main_confirmation() -> list[dict]:
    return _main_matrix(
        splits=("nonIID", "IID"),
        approaches=("merged", "cart"),
        seeds=CONFIRMATION_SEEDS,
    )


def build_split_confirmation(split: str) -> list[dict]:
    return _main_matrix(
        splits=(split,),
        approaches=("merged", "cart"),
        seeds=CONFIRMATION_SEEDS,
    )


def build_static_controls() -> list[dict]:
    configs: list[dict] = []
    for approach in ("merged", "cart"):
        for seed in CONFIRMATION_SEEDS:
            for sigma in NOISE_LEVELS:
                for environment, mitigation in (
                    ("noise", "ebm"),
                    ("hidden_noise", "ss_ebm"),
                ):
                    configs.append(
                        make_config(
                            split="nonIID",
                            approach=approach,
                            environment=environment,
                            mitigation=mitigation,
                            ebm_mode="static",
                            sigma=sigma,
                            seed=seed,
                            phase="static_control",
                        )
                    )
    return configs


def build_diagnostic(approach: str) -> list[dict]:
    configs = [
        make_config(
            split="nonIID",
            approach=approach,
            environment="clean",
            mitigation="none",
            seed=DIAGNOSTIC_SEED,
            phase="diagnostic",
        )
    ]
    for mitigation in ("none", "ss"):
        configs.append(
            make_config(
                split="nonIID",
                approach=approach,
                environment="hidden",
                mitigation=mitigation,
                seed=DIAGNOSTIC_SEED,
                phase="diagnostic",
            )
        )
    for sigma in DIAGNOSTIC_NOISE_LEVELS:
        for mode in ("static", "adaptive"):
            configs.append(
                make_config(
                    split="nonIID",
                    approach=approach,
                    environment="noise",
                    mitigation="ebm",
                    ebm_mode=mode,
                    sigma=sigma,
                    seed=DIAGNOSTIC_SEED,
                    phase="diagnostic",
                )
            )
            configs.append(
                make_config(
                    split="nonIID",
                    approach=approach,
                    environment="noise",
                    mitigation="ebm",
                    ebm_mode=mode,
                    sigma=sigma,
                    seed=DIAGNOSTIC_SEED,
                    phase="diagnostic",
                    optimizer_state_mode="persistent",
                )
            )
        configs.append(
            make_config(
                split="nonIID",
                approach=approach,
                environment="noise",
                mitigation="none",
                sigma=sigma,
                seed=DIAGNOSTIC_SEED,
                phase="diagnostic",
            )
        )
        for mitigation in ("none", "ss", "ebm"):
            configs.append(
                make_config(
                    split="nonIID",
                    approach=approach,
                    environment="hidden_noise",
                    mitigation=mitigation,
                    ebm_mode="adaptive",
                    sigma=sigma,
                    seed=DIAGNOSTIC_SEED,
                    phase="diagnostic",
                )
            )
        for mode in ("static", "adaptive"):
            configs.append(
                make_config(
                    split="nonIID",
                    approach=approach,
                    environment="hidden_noise",
                    mitigation="ss_ebm",
                    ebm_mode=mode,
                    sigma=sigma,
                    seed=DIAGNOSTIC_SEED,
                    phase="diagnostic",
                )
            )
    return configs


def build_performance_benchmark(rounds: int = 2) -> list[dict]:
    cases = (
        ("merged", "noise", "none", "none"),
        ("merged", "hidden", "ss", "none"),
        ("merged", "noise", "ebm", "adaptive"),
        ("cart", "hidden_noise", "ss_ebm", "adaptive"),
    )
    return [
        make_config(
            split="nonIID",
            approach=approach,
            environment=environment,
            mitigation=mitigation,
            ebm_mode=mode,
            sigma=0.5 if environment in ("noise", "hidden_noise") else 0.0,
            seed=9100 + index,
            rounds=rounds,
            phase="performance_benchmark",
        )
        for index, (approach, environment, mitigation, mode) in enumerate(cases)
    ]


def build_preset(key: str) -> list[dict]:
    if key == "diagnostic_non_iid_merged":
        return build_diagnostic("merged")
    if key == "diagnostic_non_iid_cart":
        return build_diagnostic("cart")
    if key == "static_controls":
        return build_static_controls()
    if key == "main_confirmation":
        return build_main_confirmation()
    if key == "non_iid_confirmation":
        return build_split_confirmation("nonIID")
    if key == "iid_confirmation":
        return build_split_confirmation("IID")
    if key == "performance_benchmark":
        return build_performance_benchmark()
    raise ValueError(f"Unknown Campaign 4 preset: {key}")


def result_run_dir(config: dict, root: Path | str = RESULT_ROOT) -> Path:
    root = Path(root)
    split = str(config.get("split") or ("nonIID" if config.get("nonIID") else "IID"))
    attack_key = "hidden" if config.get("attackHidden") else "none"
    approach = str(config["approach"])
    sigma = float(config.get("channelNoiseSigma", 0.0))
    noise_dir = f"sigma_{_sigma_label(sigma)}" if sigma > 0 else "no_channel_noise"
    condition = str(config["conditionId"])
    seed = int(config["seed"])
    return (
        root
        / split
        / str(config.get("dataset", "cifar10"))
        / attack_key
        / approach
        / noise_dir
        / condition
        / f"seed_{seed}"
        / str(config["runId"])
    )


def result_paths(
    config: dict,
    root: Path | str = RESULT_ROOT,
) -> tuple[Path, Path, Path]:
    directory = result_run_dir(config, root=root)
    return directory / "metrics.npz", directory / "run.json", directory / "telemetry.npz"


def is_completed(config: dict, root: Path | str = RESULT_ROOT) -> bool:
    metrics_path, run_path, telemetry_path = result_paths(config, root=root)
    if not (metrics_path.exists() and run_path.exists() and telemetry_path.exists()):
        return False
    try:
        metadata = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        metadata.get("status") == "completed"
        and metadata.get("campaignId") == CAMPAIGN_ID
        and metadata.get("configHash") == config_hash(config)
        and metadata.get("runId") == config.get("runId")
    )


def _json_hash(payload: dict | list) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def current_source_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    hashes = {}
    for relative in PROVENANCE_FILES:
        path = project_root / relative
        try:
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError(
                f"Cannot establish Campaign 4 provenance without reading {path}."
            ) from error
    return hashes


def adaptive_method_contract(execution_profile: dict | None = None) -> dict:
    """Return the exact Campaign 4 method and experiment contract to freeze."""
    merged = apply_performance_profile(
        make_config(
            split="nonIID",
            approach="merged",
            environment="hidden_noise",
            mitigation="ss_ebm",
            sigma=0.6,
            seed=DIAGNOSTIC_SEED,
            phase="diagnostic",
        ),
        execution_profile,
    )
    cart = apply_performance_profile(
        make_config(
            split="nonIID",
            approach="cart",
            environment="hidden_noise",
            mitigation="ss_ebm",
            sigma=0.6,
            seed=DIAGNOSTIC_SEED,
            phase="diagnostic",
        ),
        execution_profile,
    )
    method_keys = (
        "ebmObjective",
        "ebmImplementation",
        "ebmGradientBatchSemantics",
        "adaptiveEbmTargetRatioBase",
        "adaptiveEbmStressGain",
        "adaptiveEbmRatioMin",
        "adaptiveEbmRatioMax",
        "adaptiveEbmCoefficientMin",
        "adaptiveEbmCoefficientMax",
        "adaptiveEbmBeta",
        "adaptiveEbmMaxChangeFactor",
        "snapshotSelectionRule",
        "snapshotPlausibilityGuard",
        "basilMemorySize",
        "channelNoiseSemantics",
        "channelNoiseStart",
        "attackHiddenStart",
        "attackerIds",
        "aggregationMode",
        "optimizerStateMode",
        "batchSize",
        "internalMicroBatchSize",
        "precisionProfile",
        "gpuStateMode",
        "localEpochs",
        "stepsPerEpoch",
        "learningRate",
        "momentum",
        "weightDecayCoefficient",
    )
    cart_keys = (
        "distillStrength",
        "cartEmaBeta",
        "cartGapRule",
        "verifyThreshold",
    )
    confirmations = [
        apply_performance_profile(config, execution_profile)
        for config in build_main_confirmation()
    ]
    diagnostics = [
        apply_performance_profile(config, execution_profile)
        for config in build_diagnostic("merged") + build_diagnostic("cart")
    ]
    selected_profile = (
        execution_profile.get("selectedProfile")
        if isinstance(execution_profile, dict)
        and execution_profile.get("status") == "validated"
        else None
    )
    return {
        "schemaVersion": 1,
        "campaignId": CAMPAIGN_ID,
        "protocolRevision": PROTOCOL_REVISION,
        "sourceHashes": current_source_hashes(),
        "executionProfile": (
            dict(selected_profile)
            if isinstance(selected_profile, dict)
            else {
                "profileId": "generated_default",
                "internalMicroBatchSize": 512,
                "precisionProfile": "float32",
                "jitCompile": False,
                "gpuAllocator": "bfc",
            }
        ),
        "adaptiveMethod": {key: merged[key] for key in method_keys},
        "cartMethod": {key: cart[key] for key in cart_keys},
        "staticEbmCoefficientBySigma": {
            f"{sigma:.1f}": coefficient
            for sigma, coefficient in sorted(
                STATIC_EBM_COEFFICIENT_BY_SIGMA.items()
            )
        },
        "cartGammaSchedule": dict(DEFAULT_CART_GAMMA),
        "diagnosticRunCount": len(diagnostics),
        "diagnosticRunIdDigest": _json_hash(
            sorted(config["runId"] for config in diagnostics)
        ),
        "confirmationRunCount": len(confirmations),
        "confirmationRunIdDigest": _json_hash(
            sorted(config["runId"] for config in confirmations)
        ),
    }


def load_campaign_state(
    state_path: Path | str = CAMPAIGN_STATE,
) -> dict:
    path = Path(state_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def diagnostic_completion(
    result_root: Path | str = RESULT_ROOT,
    execution_profile: dict | None = None,
) -> dict:
    expected = [
        apply_performance_profile(config, execution_profile)
        for config in build_diagnostic("merged") + build_diagnostic("cart")
    ]
    expected_sources = current_source_hashes()
    completed = []
    source_mismatches = []
    for config in expected:
        if not is_completed(config, root=result_root):
            continue
        _, run_path, _ = result_paths(config, root=result_root)
        try:
            metadata = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if metadata.get("sourceHashes") != expected_sources:
            source_mismatches.append(config["runId"])
            continue
        completed.append(config["runId"])
    completed_set = set(completed)
    missing = [
        config["runId"]
        for config in expected
        if config["runId"] not in completed_set
    ]
    return {
        "expected": len(expected),
        "completed": len(completed),
        "complete": len(completed) == len(expected),
        "completedRunIds": sorted(completed),
        "missingRunIds": sorted(missing),
        "sourceMismatchRunIds": sorted(source_mismatches),
    }


def is_confirmation_frozen(
    *,
    result_root: Path | str = RESULT_ROOT,
    state_path: Path | str = CAMPAIGN_STATE,
    execution_profile: dict | None = None,
) -> bool:
    state = load_campaign_state(state_path)
    if not (
        state.get("status") == "frozen"
        and state.get("campaignId") == CAMPAIGN_ID
        and state.get("protocolRevision") == PROTOCOL_REVISION
        and isinstance(state.get("rationale"), str)
        and state.get("rationale", "").strip()
    ):
        return False
    try:
        margin = float(state["nonInferiorityMargin"])
    except (KeyError, TypeError, ValueError):
        return False
    if not 0.0 <= margin <= 1.0:
        return False
    contract = adaptive_method_contract(execution_profile)
    if state.get("adaptiveMethodContractHash") != _json_hash(contract):
        return False
    completion = diagnostic_completion(
        result_root=result_root,
        execution_profile=execution_profile,
    )
    return bool(
        completion["complete"]
        and state.get("diagnosticRunIds") == completion["completedRunIds"]
    )


def freeze_campaign_state(
    *,
    rationale: str,
    noninferiority_margin: float,
    result_root: Path | str = RESULT_ROOT,
    state_path: Path | str = CAMPAIGN_STATE,
    execution_profile: dict | None = None,
) -> dict:
    rationale = str(rationale).strip()
    if not rationale:
        raise ValueError("A written freeze rationale is required.")
    margin = float(noninferiority_margin)
    if not 0.0 <= margin <= 1.0:
        raise ValueError("The non-inferiority margin must be between 0 and 1.")
    completion = diagnostic_completion(
        result_root=result_root,
        execution_profile=execution_profile,
    )
    if not completion["complete"]:
        raise RuntimeError(
            "Campaign 4 cannot be frozen until all diagnostics are complete: "
            f"{completion['completed']}/{completion['expected']} completed."
        )
    contract = adaptive_method_contract(execution_profile)
    state = {
        "schemaVersion": 1,
        "campaignId": CAMPAIGN_ID,
        "protocolRevision": PROTOCOL_REVISION,
        "status": "frozen",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "rationale": rationale,
        "nonInferiorityMargin": margin,
        "adaptiveMethodContract": contract,
        "adaptiveMethodContractHash": _json_hash(contract),
        "diagnosticRunIds": completion["completedRunIds"],
    }
    write_json_atomic(state_path, state)
    return state


__all__ = [
    "CAMPAIGN_ID",
    "CAMPAIGN_STATE",
    "CAMPAIGN_VERSION",
    "CONFIG_ROOT",
    "CONFIRMATION_SEEDS",
    "DIAGNOSTIC_NOISE_LEVELS",
    "NOISE_LEVELS",
    "PERFORMANCE_PROFILE",
    "PLOT_ROOT",
    "PRESET_LABELS",
    "PROTOCOL_REVISION",
    "RESULT_BASE",
    "RESULT_ROOT",
    "WORKER_ROOT",
    "adaptive_method_contract",
    "apply_performance_profile",
    "build_diagnostic",
    "build_main_confirmation",
    "build_performance_benchmark",
    "build_preset",
    "build_split_confirmation",
    "build_static_controls",
    "config_hash",
    "current_source_hashes",
    "diagnostic_completion",
    "freeze_campaign_state",
    "is_completed",
    "is_confirmation_frozen",
    "load_campaign_state",
    "make_config",
    "make_run_id",
    "result_paths",
    "result_run_dir",
    "static_ebm_coefficient",
    "write_json_atomic",
    "write_npz_atomic",
]
