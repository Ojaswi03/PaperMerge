"""Explicit, independently switched CIFAR experiments and seed-level summaries.

This is an opt-in development protocol, separate from the published-paper
reproduction. A high accuracy target is not an asserted property of a config.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

REVISION = 'factorial_cifar_v1'
SEEDS = (2025, 2026, 2027)
PARTITIONS = ('iid', 'dirichlet')
MODEL = 'cifar_resnet20_groupnorm_v1'


@dataclass(frozen=True)
class Condition:
    name: str
    attack: bool = False
    noise: bool = False
    byzantine_mitigation: bool = False
    noise_mitigation: bool = False
    cart: bool = False


# Include untreated controls as well as the user's seven requested comparisons.
CONDITIONS = (
    Condition('clean'),
    Condition('attack_only', attack=True),
    Condition('attack_basil', attack=True, byzantine_mitigation=True),
    Condition('noise_only', noise=True),
    Condition('noise_ebm', noise=True, noise_mitigation=True),
    Condition('attack_noise', attack=True, noise=True),
    Condition('attack_noise_basil', attack=True, noise=True, byzantine_mitigation=True),
    Condition('attack_noise_ebm', attack=True, noise=True, noise_mitigation=True),
    Condition('attack_noise_basil_ebm', attack=True, noise=True,
              byzantine_mitigation=True, noise_mitigation=True),
    Condition('attack_noise_basil_ebm_cart', attack=True, noise=True,
              byzantine_mitigation=True, noise_mitigation=True, cart=True),
)


def noise_levels(values):
    values = tuple(float(v) for v in values)
    if (len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values)
            or tuple(sorted(set(values))) != values):
        raise ValueError('Exactly three distinct, increasing, finite positive sigmas required.')
    return values


def configuration(condition, partition, seed, sigma, *, stage='smoke', device='GPU'):
    from gui.research_protocol import base_config
    from basil_core.research_protocol import resolve_attackers
    if condition not in CONDITIONS or partition not in PARTITIONS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unknown condition, partition or seed.')
    if stage not in ('smoke', 'pilot', 'comparison') or device not in ('CPU', 'GPU'):
        raise ValueError('Unknown execution stage/device.')
    if not math.isfinite(sigma) or (sigma <= 0 if condition.noise else sigma != 0):
        raise ValueError('Only noisy conditions have a positive absolute sigma.')
    rounds = {'smoke': 2, 'pilot': 10, 'comparison': 100}[stage]
    cfg = base_config(seed=seed)
    cfg.update(lossStudyRevision=REVISION, protocolRevision=REVISION,
        conditionId=condition.name, factorialSwitches=asdict(condition),
        factorialStage=stage, partitionStrategy=partition, classAssignment='partition_rng',
        modelArchitecture=MODEL, nRounds=rounds, batchSize=128,
        learningRate=.1, learningRateSchedule='cosine_round_decay',
        minimumLearningRate=.001, momentum=.9, weightDecayCoefficient=.0005,
        attackHidden=condition.attack, attackerCount=4 if condition.attack else 0,
        resolvedAttackerIds=list(resolve_attackers(seed, 4)) if condition.attack else [],
        snapshotSelection=condition.byzantine_mitigation,
        useChannelNoise=condition.noise, channelNoiseSemantics='paper_absolute_gaussian',
        channelNoiseSigmaAbsolute=float(sigma), channelNoiseSigmaRelative=0.,
        ebmMode='gradient_norm_objective' if condition.noise_mitigation else 'none',
        ebmCoefficientSemantics='channel_coordinate_variance',
        snapshotAnchorMu=0., logitCalibrationEnabled=False,
        cartEnabled=condition.cart, cartBeta=1. if condition.cart else 0.,
        cartMemoryMode='budgeted', cartCorrectionRatioCap=.1, cartBarrierSlack=.001,
        cartStatus='candidate_not_performance_validated',
        attackStartRound=1 if stage == 'smoke' else 20,
        localObjective='ordinary_ce_full_ebm_optional_class_memory',
        executionDevice=device, roundEvaluationMetrics=True,
        evaluationDataOrigin='heldout_TRAIN', officialTestUsed=False,
        diagnosticTestSamplesPerClass=100,
        diagnosticSamplesPerClass=64 if stage == 'smoke' else None,
        researchValid=False, fullResearchReady=False,
        purpose='factorial_development_comparison')
    payload=json.dumps(cfg,sort_keys=True,separators=(',',':'))
    cfg['runId']=f'{partition}-{seed}-{condition.name}-s{sigma:g}-'+hashlib.sha256(payload.encode()).hexdigest()[:12]
    return cfg


def matrix(sigmas, *, stage='comparison', device='GPU'):
    sigmas=noise_levels(sigmas)
    return [configuration(c,p,s,sigma,stage=stage,device=device)
        for p in PARTITIONS for s in SEEDS for c in CONDITIONS
        for sigma in (sigmas if c.noise else (0.,))]


def validate_config(cfg):
    condition=next((c for c in CONDITIONS if c.name == cfg.get('conditionId')),None)
    if condition is None:
        raise ValueError('Unknown factorial condition.')
    expected=configuration(condition,cfg.get('partitionStrategy'),cfg.get('seed'),
        cfg.get('channelNoiseSigmaAbsolute',float('nan')),
        stage=cfg.get('factorialStage'),device=cfg.get('executionDevice'))
    if cfg != expected:
        raise ValueError('Factorial config differs from its explicit immutable definition.')


def learning_rate(cfg, round_id):
    if not 0 <= round_id < cfg['nRounds']:
        raise ValueError('Round outside configured horizon.')
    progress=round_id/max(cfg['nRounds']-1,1)
    return cfg['minimumLearningRate']+(cfg['learningRate']-cfg['minimumLearningRate'])*(1+math.cos(math.pi*progress))/2


def aggregate_seed_results(rows, sigmas, *, endpoint_start, expected_rounds):
    """Nodes -> fixed round window within seed -> three seeds, never across sigma.

    Require the complete rectangular study. Failed/incomplete runs must never
    silently disappear from the denominator or become a different experiment.
    """
    import numpy as np
    expected=matrix(sigmas)
    keys=lambda c:(c['partitionStrategy'],c['conditionId'],c['seed'],c['channelNoiseSigmaAbsolute'])
    lookup={}
    for row in rows:
        key=keys(row['config'])
        validate_config(row['config'])
        if row['config']['nRounds']!=expected_rounds:
            raise ValueError('Result horizon differs from the declared endpoint horizon.')
        if key in lookup:raise ValueError('Duplicate condition/seed/sigma result.')
        if row.get('status')!='COMPLETED':raise ValueError('Incomplete study: all expected seeds must complete.')
        values=np.asarray(row['roundNodeAccuracy'],dtype=float)
        if (values.shape!=(expected_rounds,10) or not np.isfinite(values).all()
                or np.any(values<0) or np.any(values>1)):
            raise ValueError('Expected finite round-by-node accuracies in [0, 1].')
        lookup[key]=values
    if set(lookup)!={keys(c) for c in expected}:
        raise ValueError('Missing or unexpected condition/seed/noise results.')
    if type(endpoint_start) is not int or not 0 <= endpoint_start < expected_rounds:
        raise ValueError('Endpoint window must be fixed before evaluation.')
    result=[]
    for p in PARTITIONS:
        for c in CONDITIONS:
            for sigma in (noise_levels(sigmas) if c.noise else (0.,)):
                curves=np.stack([lookup[p,c.name,s,sigma].mean(axis=1) for s in SEEDS])
                endpoints=curves[:,endpoint_start:].mean(axis=1)
                result.append(dict(partition=p,condition=c.name,sigma=sigma,seeds=list(SEEDS),
                    seedEndpoints=endpoints.tolist(),mean=float(endpoints.mean()),
                    sampleStd=float(endpoints.std(ddof=1)),
                    curveMean=curves.mean(axis=0).tolist(),curveSampleStd=curves.std(axis=0,ddof=1).tolist(),
                    endpointStart=endpoint_start,endpointEndExclusive=expected_rounds))
    return result
