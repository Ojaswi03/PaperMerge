"""Paired research configurations; generation never starts research runs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from basil_core.research_protocol import PROTOCOL, resolve_attackers

CONFIG_ROOT = Path(__file__).resolve().parent / "configs" / "sequential_basil"
RESULT_ROOT = Path("experiments/research_protocol_results")
PLOT_ROOT = Path("plots/research_protocol")
SCHEMA_VERSION = 5
REVISION = "five_epoch_variance_v2"
ABSOLUTE_SIGMAS = (0.005, 0.01, 0.02)

def base_config(*, seed=2025) -> dict:
    return {
        "schemaVersion":5,"experimentProtocol":PROTOCOL,"protocolRevision":REVISION,
        "experimentName":"Sequential BASIL protocol","dataset":"cifar10","nNodes":10,
        "partitionStrategy":"one_class_per_node","classAssignment":"node_id_equals_class_id",
        "dirichletAlpha":0.2,"modelArchitecture":"basil_paper_cnn",
        "aggregationMode":"strict_sequential_handoff","ringOrder":"fixed_0_to_9",
        "nRounds":100,"localEpochs":5,"epochSemantics":"full_local_dataset","batchSize":512,
        "optimizer":"sgd","momentum":0.0,"optimizerStateMode":"reset_each_activation",
        "learningRate":0.05,"learningRateSchedule":"basil_round_decay","gradientClipNorm":None,
        "weightDecayCoefficient":0.0,"adaptiveWeightDecayMode":"off","assumedByzantineCount":4,
        "authorizedByzantineCount":4,"basilMemorySize":5,"snapshotSelection":False,
        "snapshotSelectionRule":"lowest_receiver_local_loss","snapshotPlausibilityGuard":"none",
        "attackType":"delayed_hidden_parameter_attack_v1","attackerCount":4,
        "attackerSelection":"seeded_random_without_replacement","attackerSelectionSeed":int(seed),
        "resolvedAttackerIds":list(resolve_attackers(seed)),"attackHidden":False,"attackStartRound":20,
        "channelNoiseStartRound":0,"useChannelNoise":False,"channelNoiseSemantics":"relative_l2_gaussian",
        "channelNoiseSigmaRelative":0.0,"channelNoiseSigmaAbsolute":0.0,
        "ebmMode":"gradient_norm_objective","ebmCoefficientSemantics":"effective_channel_coordinate_variance",
        "augmentation":"seeded_flip_pad_crop","seed":int(seed),"researchValid":True,
        "purpose":"research_protocol_production",
    }

def named(config: dict, condition: str) -> dict:
    config=dict(config); config.pop("runId",None); config["conditionId"]=condition
    config["experimentName"]=f"Sequential BASIL | {condition} | seed {config['seed']}"
    if config["partitionStrategy"]!="one_class_per_node":config["classAssignment"]="partition_rng"
    payload=json.dumps(config,sort_keys=True,separators=(",",":"))
    config["runId"]="basil-"+hashlib.sha256(payload.encode()).hexdigest()[:20]
    return config

def production_configs(seed=2025) -> list[dict]:
    configs=[]
    for strategy in ("iid","dirichlet","one_class_per_node"):
        cfg=base_config(seed=seed);cfg.update(partitionStrategy=strategy,ebmMode="none")
        configs.append(named(cfg,f"clean_{strategy}"))
    for ss in (False,True):
        cfg=base_config(seed=seed);cfg.update(attackHidden=True,snapshotSelection=ss,ebmMode="none")
        configs.append(named(cfg,f"hidden_{'ss' if ss else 'none'}"))
    for semantics,sigmas,key,prefix in (
        ("relative_l2_gaussian",(0.2,0.4,0.6),"channelNoiseSigmaRelative","rel"),
        ("paper_absolute_gaussian",ABSOLUTE_SIGMAS,"channelNoiseSigmaAbsolute","abs"),
    ):
        for sigma in sigmas:
            for mode,suffix in (("none","none"),("gradient_norm_objective","ebm"),("legacy_gradient_scale","legacy")):
                cfg=base_config(seed=seed);cfg.update(useChannelNoise=True,channelNoiseSemantics=semantics,ebmMode=mode)
                cfg[key]=sigma;cfg["ebmCoefficientSemantics"]="channel_coordinate_variance" if prefix=="abs" else "effective_channel_coordinate_variance"
                if mode=="legacy_gradient_scale":cfg.update(legacyEbmLambda=25.0,purpose="legacy_compatibility_ablation")
                configs.append(named(cfg,f"noise_{prefix}_{sigma:g}_{suffix}"))
            for ss,ebm,label in ((False,False,"none"),(True,False,"ss"),(False,True,"ebm"),(True,True,"ss_ebm")):
                cfg=base_config(seed=seed);cfg.update(attackHidden=True,snapshotSelection=ss,useChannelNoise=True,channelNoiseSemantics=semantics,ebmMode="gradient_norm_objective" if ebm else "none")
                cfg[key]=sigma;cfg["ebmCoefficientSemantics"]="channel_coordinate_variance" if prefix=="abs" else "effective_channel_coordinate_variance"
                configs.append(named(cfg,f"hidden_noise_{prefix}_{sigma:g}_{label}"))
    return configs

def smoke_configs(seed=2025) -> list[dict]:
    configs=[]
    for production in production_configs(seed):
        cfg=dict(production);cfg.update(nRounds=3,diagnosticSamplesPerClass=512,diagnosticTestSamplesPerClass=100,researchValid=False,purpose="implementation_smoke_test",attackStartRound=1)
        configs.append(named(cfg,"smoke_"+production["conditionId"]))
    return configs

def preflight_configs(seed=2025) -> list[dict]:
    configs=[]
    for strategy in ("iid","dirichlet","one_class_per_node"):
        cfg=base_config(seed=seed);cfg.update(nRounds=5,partitionStrategy=strategy,ebmMode="none",researchValid=False,purpose="full_data_preflight")
        configs.append(named(cfg,"preflight_clean_"+strategy))
    return configs

def write_config_library(root=CONFIG_ROOT) -> tuple[int,int]:
    from basil_core.artifact_paths import writable_output
    root=writable_output(Path(root));families={"production":production_configs(),"smoke":smoke_configs(),"preflight":preflight_configs()}
    for family,configs in families.items():
        folder=root/family;folder.mkdir(parents=True,exist_ok=True)
        for config in configs:(folder/f"{config['conditionId']}.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    manifest={"schemaVersion":SCHEMA_VERSION,"experimentProtocol":PROTOCOL,"protocolRevision":REVISION,"families":{k:[c["conditionId"] for c in v] for k,v in families.items()}}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return len(families["production"]),len(families["smoke"])
