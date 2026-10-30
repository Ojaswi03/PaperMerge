"""Explicit professor-SHARP experiment matrix: datasets, splits, seeds, noise.

Historical configs are preserved. Five local SGD steps are NOT five epochs.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math

from basil_core.factorial_study import CONDITIONS

REVISION = 'sharp_professor_20261027_v1'
DATASETS = ('mnist', 'cifar10', 'cifar100')
SEEDS = (2025, 2026, 2027)
SIGMAS = (.005, .01, .02)
CONDITION_NAMES = tuple(c.name.replace('basil', 'defense').replace('ebm', 'rla') for c in CONDITIONS)


def configuration(dataset, partition, seed, condition, sigma, *, nodes=100, rounds=100,
                  method='sharp', alpha=.5, local_steps=5, model='paper',
                  train_per_class=None, validation_per_class=None, attack_start=20,
                  denominator='actual_selected', curvature=1., proximal_mu=1.,
                  initial_rate=.03, attack_type='hidden', budget=None, basil_window=None):
    if dataset not in DATASETS or partition not in ('iid', 'dirichlet') or method not in ('sharp', 'basil'):
        raise ValueError('Unknown dataset, partition or selection method.')
    if type(seed) is not int or seed not in SEEDS or condition not in CONDITION_NAMES:
        raise ValueError('Unknown study seed or condition.')
    if type(nodes) is not int or nodes < 4 or type(rounds) is not int or rounds < 1:
        raise ValueError('At least four nodes and a positive round horizon required.')
    b = (nodes - 1) // 3 if budget is None else budget
    if type(b) is not int or not 1 <= b or 2 * b >= nodes:
        raise ValueError('Design budget must satisfy 1 <= b and 2b < N.')
    bw=b+1 if basil_window is None else basil_window
    if type(bw) is not int or not 1<=bw<nodes:raise ValueError('BASIL window must be between one and N-1.')
    if (not all(math.isfinite(v) for v in (sigma, alpha, curvature, proximal_mu, initial_rate))
            or alpha <= 0 or curvature <= 0 or proximal_mu < 0 or initial_rate <= 0
            or type(local_steps) is not int or local_steps < 1 or model not in ('paper', 'resnet20')
            or denominator not in ('actual_selected', 'budget') or attack_type not in ('hidden', 'spike')
            or type(attack_start) is not int or attack_start < 0):
        raise ValueError('Invalid explicit SHARP settings.')
    for limit in (train_per_class, validation_per_class):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('Optional per-class limits must be positive integers.')
    switch = CONDITIONS[CONDITION_NAMES.index(condition)]
    if (switch.noise and sigma <= 0) or (not switch.noise and sigma != 0):
        raise ValueError('Only noisy conditions have positive absolute sigma.')
    cfg = dict(revision=REVISION, dataset=dataset, classes=100 if dataset == 'cifar100' else 10,
        partition=partition, seed=seed, condition=condition, switches=asdict(switch),
        nodes=nodes, rounds=rounds, byzantineBudget=b,
        window=2*b if method == 'sharp' else bw, basilWindow=bw, method=method,
        dirichletAlpha=float(alpha), localSteps=local_steps, localEpochs=None,
        batchSize=14 if dataset == 'mnist' else 80, model=model,
        learningRate=float(initial_rate), learningRateSchedule='eta0_over_1_plus_eta0_round',
        momentum=0., weightDecay=0., gradientClipping=None,
        sigma=float(sigma), channel='absolute_independent_gaussian_per_link',
        hopDenominator=denominator, curvature=float(curvature),
        proximalMu=float(proximal_mu) if method == 'sharp' and (switch.byzantine_mitigation or switch.noise_mitigation) else 0.,
        cart=bool(switch.cart), cartMode='coherent_classifier_residual_transport',
        calibrationDraws=19, cartProbePerClass=16, cartRidgeFraction=.01,
        trainPerClass=train_per_class, validationPerClass=validation_per_class,
        attackStart=attack_start, attackType=attack_type, actualAttackers=b if switch.attack else 0,
        startup='empty_real_windows_with_initial_model_fallback',
        allFailPolicy='keep_all_and_record_pdf_algorithm_1',
        officialTestUsed=False, evaluationOrigin='stratified_heldout_TRAIN',
        evaluationSamples=1000, primaryEndpoint='mean_node_validation_accuracy_last_10_rounds',
        researchValid=False, performanceClaim='not_established')
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    cfg['runId'] = f'{dataset}-{partition}-{seed}-{condition}-s{sigma:g}-{method}-{digest}'
    return cfg


def validate_config(cfg):
    fields = dict(nodes=cfg['nodes'], rounds=cfg['rounds'], method=cfg['method'],
        alpha=cfg['dirichletAlpha'], local_steps=cfg['localSteps'], model=cfg['model'],
        train_per_class=cfg['trainPerClass'], validation_per_class=cfg['validationPerClass'],
        attack_start=cfg['attackStart'], denominator=cfg['hopDenominator'],
        curvature=cfg['curvature'], proximal_mu=cfg['proximalMu'] if cfg['proximalMu'] else 1.,
        initial_rate=cfg['learningRate'], attack_type=cfg['attackType'], budget=cfg['byzantineBudget'],basil_window=cfg['basilWindow'])
    # Zero is an explicit proximal ablation; preserve it when reconstructing.
    if cfg['method']=='sharp' and (cfg['switches']['byzantine_mitigation'] or cfg['switches']['noise_mitigation']):
        fields['proximal_mu']=cfg['proximalMu']
    expected=configuration(cfg['dataset'], cfg['partition'], cfg['seed'], cfg['condition'], cfg['sigma'], **fields)
    if cfg != expected:
        raise ValueError('SHARP configuration differs from its declared matrix definition.')


def matrix(*, datasets=DATASETS, partitions=('iid','dirichlet'), seeds=SEEDS,
           conditions=CONDITION_NAMES, sigmas=SIGMAS, methods=('sharp',), **kwargs):
    levels=tuple(float(v) for v in sigmas)
    if len(levels)!=3 or levels != tuple(sorted(set(levels))) or any(not math.isfinite(v) or v<=0 for v in levels):
        raise ValueError('Exactly three distinct increasing positive noise levels required.')
    return [configuration(d,p,s,c,v,method=m,**kwargs)
        for d in datasets for p in partitions for s in seeds for m in methods for c in conditions
        for v in (levels if CONDITIONS[CONDITION_NAMES.index(c)].noise else (0.,))]
