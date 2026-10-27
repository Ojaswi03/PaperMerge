"""Fixed, opt-in paired pilots of coherent CART; no novelty/performance claim."""
from __future__ import annotations

import hashlib
import json

PROTOCOL_REVISION='cart_coherent_channel_pilot_v1'
COHERENT_VARIANTS=('coherent','coherent_refresh','coherent_commit_ce','coherent_commit_error','coherent_qualified_error','coherent_projected_error','coherent_head_transport')
VARIANTS=('none','budgeted')+COHERENT_VARIANTS
SIGMAS=(.005,.01,.02)
SEEDS=(2025,2026,2027)


def configuration(partition,seed,sigma,variant,*,stage='pilot',device='GPU',case='joint',partition_hash=None,initial_hash=None,training_objective='raw_ce',compiled_optimizer=False):
    from basil_core.factorial_study import configuration as previous,CONDITIONS
    if (partition not in ('iid','dirichlet') or type(seed) is not int or seed not in SEEDS
            or variant not in VARIANTS or stage not in ('smoke','pilot','mini','confirmation')
            or device not in ('CPU','GPU') or case not in ('clean','joint')
            or training_objective not in ('raw_ce','local_prior') or type(compiled_optimizer) is not bool):
        raise ValueError('Unknown fixed coherent-CART pilot selection.')
    if sigma not in (SIGMAS if case=='joint' else (0.,)):
        raise ValueError('Use the frozen three Gaussian levels, or zero for a clean control.')
    for digest in (partition_hash,initial_hash):
        if digest is not None and (not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Runtime identities must be SHA-256 digests.')
    condition=CONDITIONS[0] if case=='clean' else CONDITIONS[-2]
    cfg=previous(condition,partition,seed,sigma,stage='smoke' if stage=='smoke' else 'pilot',device=device)
    coherent=variant in COHERENT_VARIANTS
    boundary=variant in ('coherent_commit_ce','coherent_commit_error','coherent_qualified_error','coherent_projected_error')
    cfg.update(lossStudyRevision=PROTOCOL_REVISION,protocolRevision=PROTOCOL_REVISION,
        coherentStage=stage,coherentCase=case,cartVariant=variant,
        partitionHash=partition_hash,initialModelHash=initial_hash,
        factorialStage=stage,nRounds={'smoke':2,'pilot':3,'mini':5,'confirmation':10}[stage],
        attackStartRound=2 if stage=='confirmation' else 1,
        conditionId=f'{case}_{variant}',cartEnabled=variant!='none',cartBeta=0. if variant=='none' else 1.,
        cartMemoryMode=variant,cartCorrectionRatioCap=.1 if variant=='budgeted' else None,
        cartReferencePolicy=('breadth_qualified_receiver_class_error_then_ce' if variant in ('coherent_qualified_error','coherent_projected_error')
            else 'whole_receiver_trained_model_class_error_then_ce_within_previous_envelope' if variant=='coherent_commit_error'
            else 'whole_receiver_trained_model_supported_class_dominance' if variant=='coherent'
            else 'whole_receiver_trained_model_mean_improvement_within_previous_envelope' if coherent else None),
        cartChannelCalibrationDraws=19 if coherent else 0,
        cartReferenceScaleFloor=.1 if coherent else None,
        cartFeasibilityTolerance=1e-6 if coherent else None,
        cartBridgeScales=[1.,.5,.25,.125,.0625,.03125,.015625,0.] if coherent else None,
        cartMomentumPolicy='ordinary_sgd_reset_per_activation' if boundary else
            'scale_proposed_velocity_by_accepted_step' if coherent else 'ordinary_sgd',
        purpose='bounded_coherent_cart_development',researchValid=False,fullResearchReady=False)
    if boundary:
        cfg.update(cartGuardGranularity='activation_boundary',cartInputRestoration=False,
            cartRiskMetric='cross_entropy' if variant=='coherent_commit_ce' else 'classification_error')
    if variant in ('coherent_qualified_error','coherent_projected_error'):
        cfg.update(cartReferenceQualification='majority_supported_classes_above_uniform_guess_with_minimum_two',
            cartChanceRecall=.1,cartMinimumAboveChanceClasses=2)
    if variant=='coherent_projected_error':
        cfg.update(cartCommitGeometry='violated_class_endpoint_gradient_projection_with_scalar_candidate',
            cartProjectionTolerance=1e-8,cartProjectionMaxSweeps=4000,
            cartProjectionGradientEndpoints=['reference','proposal'])
    if variant=='coherent_head_transport':
        cfg.update(cartGuardGranularity='classifier_transport',cartInputRestoration=False,
            cartRiskMetric='cross_entropy',cartReferencePolicy='whole_receiver_mean_probe_ce_improvement',
            cartMomentumPolicy='ordinary_sgd_reset_per_activation',cartTransportRidgeFraction=.01,
            cartTransportScales=[0.,.25,.5,.75,1.],cartTransportCheckPerClass=16,
            cartTransportGate='positive_class_ce_gap_over_gap_plus_reference_channel_allowance',
            cartTransportAcceptance='disjoint_local_train_probe_macro_ce_nonincrease')
    if compiled_optimizer:
        cfg.update(compiledOptimizerStep=True,optimizerExecution='tf_function_sgd_apply_v1')
    if training_objective=='local_prior':
        cfg.update(coherentTrainingObjective='local_prior',logitCalibrationEnabled=True,
            logitCalibrationMethod='prior_adjustment',logitCalibrationTau=1.,logitCalibrationCountFloor=1,
            localObjective='local_prior_adjusted_ce_full_ebm_optional_class_memory')
    cfg['factorialSwitches']=dict(name=cfg['conditionId'],attack=cfg['attackHidden'],
        noise=cfg['useChannelNoise'],byzantine_mitigation=cfg['snapshotSelection'],
        noise_mitigation=cfg['ebmMode']=='gradient_norm_objective',cart=cfg['cartEnabled'])
    cfg.pop('runId',None)
    payload=json.dumps(cfg,sort_keys=True,separators=(',',':'))
    cfg['runId']=f'{partition}-{seed}-{case}-{variant}-s{sigma:g}-'+hashlib.sha256(payload.encode()).hexdigest()[:12]
    return cfg


def validate_config(config):
    expected=configuration(config.get('partitionStrategy'),config.get('seed'),
        config.get('channelNoiseSigmaAbsolute'),config.get('cartVariant'),
        stage=config.get('coherentStage'),device=config.get('executionDevice'),case=config.get('coherentCase'),
        partition_hash=config.get('partitionHash'),initial_hash=config.get('initialModelHash'),
        training_objective=config.get('coherentTrainingObjective','raw_ce'),compiled_optimizer=config.get('compiledOptimizerStep',False))
    if config!=expected:raise ValueError('Coherent-CART pilot differs from its fixed definition.')


def make_worker(model,config):
    import tensorflow as tf
    from basil_core.factorial_worker import PlainWorker,CartWorker
    from basil_core.cart_feasible_worker import FeasibleCartWorker
    validate_config(config)
    kind=config['cartVariant']
    if kind=='coherent_head_transport':
        from basil_core.cart_head_worker import HeadTransportWorker
        worker=HeadTransportWorker(model,seed=config['seed'],momentum=config['momentum'],weight_decay=config['weightDecayCoefficient'])
    elif kind=='coherent_projected_error':
        from basil_core.cart_projected_commit import ProjectedCommitWorker
        worker=ProjectedCommitWorker(model,seed=config['seed'],momentum=config['momentum'],weight_decay=config['weightDecayCoefficient'])
    elif kind in ('coherent_commit_ce','coherent_commit_error','coherent_qualified_error'):
        from basil_core.cart_boundary_worker import BoundaryCartWorker
        worker=BoundaryCartWorker(model,seed=config['seed'],momentum=config['momentum'],
            weight_decay=config['weightDecayCoefficient'],restore_input=config['cartInputRestoration'],
            risk_metric=config['cartRiskMetric'],require_breadth=kind=='coherent_qualified_error')
    elif kind in ('coherent','coherent_refresh'):
        worker=FeasibleCartWorker(model,seed=config['seed'],momentum=config['momentum'],
            weight_decay=config['weightDecayCoefficient'],refresh_policy='envelope' if kind=='coherent_refresh' else 'dominance')
    elif kind=='budgeted':
        worker=CartWorker(model,beta=1.,seed=config['seed'],device=config['executionDevice'],correction_ratio_cap=.1)
        worker.mu=0.;worker.calibration_enabled=config['logitCalibrationEnabled']
    else:worker=PlainWorker(model)
    worker.factorial_config=dict(config)
    worker.diagnostic_revision=PROTOCOL_REVISION
    if kind not in COHERENT_VARIANTS:
        worker.optimizer_factory=lambda rate:tf.keras.optimizers.SGD(rate,momentum=config['momentum'])
    if config.get('compiledOptimizerStep'):
        from basil_core.compiled_optimizer import CompiledSGD
        if kind in ('coherent','coherent_refresh'):
            from basil_core.cart_feasible_worker import GuardedOptimizer
            worker.optimizer_factory=lambda rate:GuardedOptimizer(CompiledSGD(model,rate,config['momentum']),worker)
        else:worker.optimizer_factory=lambda rate:CompiledSGD(model,rate,config['momentum'])
    return worker
