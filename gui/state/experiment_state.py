from __future__ import annotations

from dataclasses import dataclass, field, fields
from copy import deepcopy
from typing import Any, Mapping
from basil_core.protocol_compatibility import normalize_config


SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3, 4, 5}


@dataclass(slots=True)
class ValidationIssue:
    field: str
    message: str


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str = "Untitled experiment"
    dataset: str = "cifar10"
    approach: str = "basil"
    non_iid: bool = True
    partition_strategy: str = "dirichlet"
    dirichlet_alpha: float = 0.2
    node_count: int = 10
    rounds: int = 100
    local_epochs: int = 5
    learning_rate: float = 0.05
    batch_size: int = 512
    use_channel_noise: bool = False
    channel_noise_sigma: float = 0.2
    channel_noise_semantics: str = "relative_l2_gaussian"
    channel_noise_sigma_rel: float = 0.2
    channel_noise_sigma_absolute: float = 0.01
    channel_noise_start: int = 0
    noise_mitigation: str = "none"
    ebm_lambda: float = 25.0
    ebm_mode: str = "none"
    snapshot_selection: bool = False
    basil_memory_size: int = 5
    cart_algorithm: str = "cart"
    distill_strength: float = 0.5
    verify_threshold: float = 0.05
    momentum: float = 0.9
    model_architecture: str = "legacy_vgg_cnn"
    aggregation_mode: str = "handoff"
    epoch_semantics: str = "fixed_steps"
    attacker_ids: str = ""
    attacks: dict[str, bool] = field(default_factory=dict)
    attack_starts: dict[str, int] = field(default_factory=dict)
    schema_version: int = 4
    extra: dict[str, Any] = field(default_factory=dict)
    form_errors: dict[str, str] = field(default_factory=dict, repr=False)
    original_keys: frozenset[str] | None = field(default=None, repr=False)
    original_values: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def split(self) -> str:
        return "Non-IID" if self.non_iid else "IID"

    @property
    def dirty_summary(self) -> str:
        attackers = [name for name, enabled in self.attacks.items() if enabled]
        parts = [f"{self.dataset.upper()} • {self.split}",
                 f"{self.node_count}-node directed ring • {self.approach.upper()}"]
        if self.extra.get('iidCampaignId'):
            parts += [self.extra['experimentProtocol'],f"BASIL memory S={self.basil_memory_size} • actual attackers={self.extra.get('actualAttackerCount',0)} • assumed bound=4",
                f"Objective: {self.extra.get('localObjective')} • {self.extra.get('executionStartMode')} R0→R99"]
        if attackers:
            start = min(self.attack_starts.get(name, 0) for name in attackers)
            parts.append(f"{len(self.attacker_list() or self.extra.get('resolvedAttackerIds',[]))} Byzantine nodes • {', '.join(attackers)} from round {start}")
        if self.use_channel_noise:
            sigma=self.channel_noise_sigma_rel if self.channel_noise_semantics=="relative_l2_gaussian" else self.channel_noise_sigma_absolute
            parts.append(f"Channel noise {self.channel_noise_semantics} σ={sigma:g}")
        if self.snapshot_selection or self.noise_mitigation != "none" or self.ebm_mode!="none":
            defenses = (["Snapshot Selection"] if self.snapshot_selection else []) + (
                [self.ebm_mode.replace('_',' ')] if self.ebm_mode!="none" else [self.noise_mitigation.upper()] if self.noise_mitigation != "none" else [])
            parts.append(" + ".join(defenses) + " enabled")
        parts.append(f"{self.rounds} rounds • batch {self.batch_size} • {self.local_epochs} local epoch(s)")
        return "\n".join(parts)

    def attacker_list(self) -> list[int]:
        try:
            return [int(value.strip()) for value in self.attacker_ids.split(",") if value.strip()]
        except ValueError:
            return []


_KEYS = {
    "experimentName": "experiment_name", "dataset": "dataset", "approach": "approach",
    "nonIID": "non_iid", "partitionStrategy": "partition_strategy", "dirichletAlpha": "dirichlet_alpha", "nNodes": "node_count",
    "nRounds": "rounds", "localEpochs": "local_epochs", "learningRate": "learning_rate",
    "batchSize": "batch_size", "useChannelNoise": "use_channel_noise",
    "channelNoiseSigma": "channel_noise_sigma", "channelNoiseSigmaRel": "channel_noise_sigma_rel", "channelNoiseSigmaRelative": "channel_noise_sigma_rel", "channelNoiseSigmaAbsolute": "channel_noise_sigma_absolute", "channelNoiseSemantics": "channel_noise_semantics", "channelNoiseStart": "channel_noise_start", "channelNoiseStartRound": "channel_noise_start",
    "noiseMitigation": "noise_mitigation", "ebmLambda": "ebm_lambda", "ebmMode": "ebm_mode",
    "snapshotSelection": "snapshot_selection", "basilMemorySize": "basil_memory_size",
    "cartAlgorithm": "cart_algorithm", "distillStrength": "distill_strength",
    "verifyThreshold": "verify_threshold", "momentum": "momentum",
    "modelArchitecture": "model_architecture", "aggregationMode": "aggregation_mode", "epochSemantics": "epoch_semantics",
    "attackerIds": "attacker_ids", "schemaVersion": "schema_version",
}
_REVERSE_KEYS = {value: key for key, value in _KEYS.items()}
ATTACK_NAMES = ("Gaussian", "SignFlip", "Hidden", "ModelPoison", "Scaling", "Alie", "Ipm", "NoiseAmp")


def from_persisted(data: Mapping[str, Any]) -> ExperimentConfig:
    if not isinstance(data, Mapping):
        raise ValueError("Configuration must be a JSON object.")
    data = normalize_config(data)
    version = int(data.get("schemaVersion", 1))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schemaVersion {version}; supported versions are 1–5.")
    known, extra = {}, dict(data)
    for persisted, internal in _KEYS.items():
        if persisted in data:
            known[internal] = data[persisted]
            extra.pop(persisted, None)
    attacks, starts = {}, {}
    for name in ATTACK_NAMES:
        enabled_key, start_key = f"attack{name}", f"attack{name}Start"
        attacks[name] = bool(data.get(enabled_key, False))
        starts[name] = int(data.get(start_key, 0))
        extra.pop(enabled_key, None); extra.pop(start_key, None)
    if "snapshotSelection" not in data and "useBasil" in data:
        known["snapshot_selection"] = bool(data["useBasil"])
        extra.pop("useBasil", None)
    known["attacks"], known["attack_starts"], known["extra"] = attacks, starts, extra
    known["schema_version"]=version
    if data.get("experimentProtocol") in {"sequential_basil_one_class_v1","sequential_basil_iid_v1"}:
        starts["Hidden"]=int(data.get("attackStartRound",20))
        known["non_iid"]=data.get("partitionStrategy")!="iid"
        if known.get("channel_noise_semantics")=="absolute_coordinate_gaussian":known["channel_noise_semantics"]="paper_absolute_gaussian"
    allowed = {item.name for item in fields(ExperimentConfig)}
    config=ExperimentConfig(**{key: value for key, value in known.items() if key in allowed})
    config.original_keys=frozenset(data)
    config.original_values={key:deepcopy(getattr(config,key)) for key in (*_REVERSE_KEYS,"attacks","attack_starts")}
    return config


def to_persisted(config: ExperimentConfig) -> dict[str, Any]:
    data = dict(config.extra)
    for internal, persisted in _REVERSE_KEYS.items():
        value=getattr(config,internal)
        original=[key for key,target in _KEYS.items() if target==internal and config.original_keys is not None and key in config.original_keys]
        if original:
            for key in original:data[key]=value
        elif config.original_keys is None or value!=config.original_values.get(internal):data[persisted]=value
    if config.original_keys is None or "useBasil" in config.original_keys:
        data["useBasil"] = config.snapshot_selection
    if data.get("experimentProtocol") in {"sequential_basil_one_class_v1","sequential_basil_iid_v1"}:
        data["attackStartRound"]=config.attack_starts.get("Hidden",20)
    for name in ATTACK_NAMES:
        for key,values,original in ((f"attack{name}",config.attacks,config.original_values.get("attacks",{})),(f"attack{name}Start",config.attack_starts,config.original_values.get("attack_starts",{}))):
            value=values.get(name,False if key==f"attack{name}" else 0)
            if config.original_keys is None or key in config.original_keys or value!=original.get(name):data[key]=value
    return normalize_config(data)


def validate_experiment(config: ExperimentConfig) -> list[ValidationIssue]:
    if config.form_errors:return [ValidationIssue(key,value) for key,value in config.form_errors.items()]
    if config.extra.get("experimentProtocol") in {"sequential_basil_one_class_v1","sequential_basil_iid_v1"}:
        from basil_core.research_protocol import validate_config
        try:validate_config(to_persisted(config))
        except (ValueError,TypeError,KeyError) as error:return [ValidationIssue("protocol",str(error))]
        return []
    issues: list[ValidationIssue] = []
    def positive(field_name: str, value: int | float, label: str) -> None:
        if value <= 0: issues.append(ValidationIssue(field_name, f"{label} must be greater than 0."))
    if not config.experiment_name.strip(): issues.append(ValidationIssue("experiment_name", "Experiment name is required."))
    if config.dataset not in {"mnist", "cifar10", "nmnist"}: issues.append(ValidationIssue("dataset", "Select a supported dataset."))
    if config.approach not in {"basil", "noisy", "merged", "cart"}: issues.append(ValidationIssue("approach", "Select a supported approach."))
    positive("node_count", config.node_count, "Node count"); positive("rounds", config.rounds, "Rounds")
    positive("local_epochs", config.local_epochs, "Local epochs"); positive("learning_rate", config.learning_rate, "Learning rate")
    positive("batch_size", config.batch_size, "Batch size")
    if config.non_iid and config.dirichlet_alpha <= 0: issues.append(ValidationIssue("dirichlet_alpha", "Dirichlet alpha must be greater than 0 for Non-IID data."))
    active_sigma=config.channel_noise_sigma_rel if config.channel_noise_semantics=="relative_l2_gaussian" else config.channel_noise_sigma_absolute
    if config.use_channel_noise and active_sigma <= 0: issues.append(ValidationIssue("channel_noise_sigma", "Noise sigma must be greater than 0 when channel noise is enabled."))
    if config.channel_noise_start < 0 or config.channel_noise_start >= max(config.rounds, 1): issues.append(ValidationIssue("channel_noise_start", "Channel noise start must be within the experiment rounds."))
    if (config.noise_mitigation == "ebm" or config.ebm_mode != "none") and not config.use_channel_noise: issues.append(ValidationIssue("noise_mitigation", "EBM requires channel noise to be enabled."))
    active_attacks = [name for name, enabled in config.attacks.items() if enabled]
    if active_attacks and not config.attacker_ids.strip() and not config.extra.get("resolvedAttackerIds"): issues.append(ValidationIssue("attacker_ids", "Enabled attacks require at least one attacker ID."))
    if config.snapshot_selection and not active_attacks: issues.append(ValidationIssue("snapshot_selection", "Snapshot Selection requires an enabled Byzantine attack."))
    ids = config.attacker_list()
    if config.attacker_ids.strip() and not ids: issues.append(ValidationIssue("attacker_ids", "Attacker IDs must be comma-separated integers."))
    if any(item < 0 or item >= config.node_count for item in ids): issues.append(ValidationIssue("attacker_ids", "Attacker IDs must identify existing nodes."))
    for attack in active_attacks:
        start = config.attack_starts.get(attack, 0)
        if start < 0 or start >= max(config.rounds, 1): issues.append(ValidationIssue(f"attack_{attack}", f"{attack} start round must be within the experiment rounds."))
    return issues
