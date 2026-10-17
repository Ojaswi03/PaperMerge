"""Separate, bounded local-logit calibration diagnostic, not the old baseline.

FedLC-inspired margins (Zhang et al., ICML 2022, Eqs. 6/7) and a separate local
training-prior adjustment (Menon et al., ICLR 2021, Eq. 10). Our explicitly
declared adaptations are floor-one missing counts, receiver-local priors,
strict BASIL handoff and a frozen snapshot proximal term. No sharing or tuning.
"""
from __future__ import annotations

import math

from basil_core.noniid_study import SnapshotAnchorWorker

REVISION = 'dirichlet_logit_calibration_diagnostic_v1'
GATE_REVISION = 'dirichlet_prior_readiness_diagnostic_v1'
REVISIONS = (REVISION, GATE_REVISION)
TAU = 1.0


def calibration_offsets(counts, tau=TAU, *, method='fedlc'):
    import numpy as np
    counts = np.asarray(counts)
    if (counts.shape != (10,) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0) or counts.sum() <= 0 or not math.isfinite(tau) or tau < 0):
        raise ValueError('Ten nonnegative local TRAIN counts and finite nonnegative tau required.')
    floored=np.maximum(counts,1).astype(np.float64)
    # Loss evaluates logits-offset: negative log prior gives logits+log prior.
    if method=='prior_adjustment':return (-tau*np.log(floored/floored.sum())).astype(np.float32)
    if method!='fedlc':raise ValueError('Unknown predeclared calibration method.')
    # Finite absent-class adaptation; no invented training examples or test priors.
    return (tau*floored**(-.25)).astype(np.float32)


def calibrated_anchor_gradients(model, x, y, reference, offsets, *, mu=.01, coefficient=0.):
    """F_cal + c||grad F_cal||² + mu/2||theta-selected||², no lambda.

    Return the same first thirteen scalar metrics as SnapshotAnchorWorker,
    plus ordinary (uncalibrated) CE and the magnitude of the calibration offsets.
    When enabled in future, EBM regularizes F_cal, not the old baseline CE.
    """
    import tensorflow as tf
    weights = model.trainable_variables
    if len(reference) != len(weights):
        raise ValueError('Frozen reference parameter count mismatch.')
    offsets = tf.stop_gradient(tf.cast(offsets, tf.float32))
    coefficient = tf.stop_gradient(tf.cast(coefficient, tf.float32))
    tf.debugging.assert_all_finite(tf.cast(mu,tf.float32), 'Non-finite anchor strength')
    tf.debugging.assert_non_negative(tf.cast(mu,tf.float32))
    for variable, anchor in zip(weights,reference):
        tf.debugging.assert_equal(tf.shape(variable),tf.shape(anchor))
        tf.debugging.assert_all_finite(variable,'Non-finite incoming calibration parameter')
    tf.debugging.assert_equal(tf.shape(offsets), [10])
    tf.debugging.assert_all_finite(offsets, 'Non-finite logit calibration')
    tf.debugging.assert_non_negative(coefficient)
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            logits = model(x, training=True)
            tf.debugging.assert_all_finite(logits, 'Non-finite calibrated-training logits')
            raw_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
            base = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits-offsets))
        first = inner.gradient(base, weights)
        if any(g is None for g in first):
            raise ValueError('Disconnected calibrated CE gradient.')
        norm_sq = tf.add_n([tf.reduce_sum(g**2) for g in first])
        source = base + coefficient * norm_sq
    robust = outer.gradient(source, weights)
    if any(g is None for g in robust):
        raise ValueError('Disconnected calibrated robust gradient.')
    drift = [w-tf.stop_gradient(tf.cast(a, w.dtype)) for w, a in zip(weights, reference)]
    distance = tf.add_n([tf.reduce_sum(v**2) for v in drift])
    anchor = [tf.cast(mu, g.dtype)*d for g, d in zip(robust, drift)]
    applied = [g+a for g, a in zip(robust, anchor)]
    base_norm = tf.linalg.global_norm(first)
    correction = tf.sqrt(tf.add_n([tf.reduce_sum(tf.cast(r-g, tf.float64)**2) for r, g in zip(robust, first)]))
    anchor_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.cast(g, tf.float64)**2) for g in anchor]))
    penalty = tf.cast(mu, tf.float32)*distance/2
    accuracy = tf.reduce_mean(tf.cast(tf.argmax(logits, axis=1, output_type=tf.int32)==y, tf.float32))
    values = (base, norm_sq, source+penalty, coefficient, base_norm,
        tf.linalg.global_norm(applied), correction, accuracy, distance, penalty,
        anchor_norm, tf.cast(mu, tf.float32), anchor_norm/(tf.cast(base_norm, tf.float64)+1e-12),
        raw_ce, tf.linalg.norm(offsets))
    for value in values:
        tf.debugging.assert_all_finite(value, 'Non-finite calibrated objective metric')
    for gradient in applied:
        tf.debugging.assert_all_finite(gradient, 'Non-finite calibrated update gradient')
    return applied, values


class DirichletCalibrationWorker(SnapshotAnchorWorker):
    """Same local SGD loop and ordinary-CE Snapshot Selection, explicit new loss."""
    extra_metric_names = SnapshotAnchorWorker.extra_metric_names + ('ordinaryCrossEntropy', 'calibrationOffsetNorm')
    diagnostic_revision = REVISION

    def __init__(self, model, *, calibration, device='CPU', method='fedlc', revision=REVISION):
        import tensorflow as tf
        self.calibration_enabled = bool(calibration)
        if method not in {'fedlc','prior_adjustment'}:raise ValueError('Unsupported calibration method.')
        self.calibration_method=method
        if revision not in REVISIONS:raise ValueError('Unsupported diagnostic revision.')
        self.diagnostic_revision=revision
        self.offsets = tf.Variable(tf.zeros(10), trainable=False)
        super().__init__(model, mu=.01, device=device)

    def _batch_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
        import tensorflow as tf
        from basil_core.research_audit import augment_batch
        if not self.calibration_enabled:
            gradients, values = super()._batch_gradients(x, y, key, mode, semantics, sigma, legacy_lambda)
            return gradients, tuple(values)+(values[0], tf.constant(0.))
        if semantics != 'paper_absolute_gaussian' or mode not in {'none','gradient_norm_objective'}:
            raise ValueError('Calibration uses explicit absolute noise and CE/full second-order modes only.')
        if self.diagnostic_revision==REVISION:
            if mode!='none':raise ValueError('Initial calibration diagnostic must be clean, without EBM.')
            tf.debugging.assert_equal(sigma, 0.)
        else:
            tf.debugging.assert_equal(tf.logical_or(tf.equal(sigma,0.),tf.equal(sigma,.01)),True)
        with tf.device('/CPU:0'):
            images = augment_batch(x, key)
        return calibrated_anchor_gradients(self.model, images, y, self.reference, self.offsets,
            mu=self.mu,coefficient=sigma**2 if mode=='gradient_norm_objective' else 0.)

    def train(self, params, x, y, **kwargs):
        import numpy as np
        counts = np.bincount(y, minlength=10)
        self.offsets.assign(calibration_offsets(counts,method=self.calibration_method) if self.calibration_enabled else np.zeros(10, np.float32))
        output, metrics = super().train(params, x, y, **kwargs)
        metrics.update(localClassCounts=counts.tolist(), logitCalibrationOffsets=self.offsets.numpy().tolist())
        return output, metrics


def diagnostic_configs(partition_hash, initial_hash):
    from basil_core.dirichlet_study import configurations
    common = configurations(partition_hash, initial_hash)[0]
    configs = []
    for name in ('baseline', 'calibrated'):
        cfg = dict(common, lossStudyRevision=REVISION, nRounds=3,
            protocolRevision=REVISION, noniidStudyId='dirichlet_logit_calibration_3r_v1',
            displayName=f'Clean {name} — Dirichlet α=0.2 — 3 rounds, 5 epochs — diagnostic only',
            researchValid=False, purpose='paired_clean_three_round_local_calibration',
            localObjective='snapshot_anchor_only' if name == 'baseline' else 'fedlc_inspired_snapshot_anchor',
            logitCalibrationEnabled=name=='calibrated', logitCalibrationTau=TAU,
            logitCalibrationCountFloor=1, calibrationSelectionRule='ordinary_receiver_local_ce',
            experimentName=f'clean_{name}_3r', runId=f'clean_{name}_3r',
            resultDirectory=f'newResults/nonIID/dirichlet_logit_calibration_3r_v1/{name}',
            plotDirectory=f'newPlots/nonIID/dirichlet_logit_calibration_3r_v1/{name}')
        validate_diagnostic_config(cfg)
        configs.append(cfg)
    return configs


def validate_diagnostic_config(config):
    if config.get('horizonStudyRevision') == 'dirichlet_prior_fixed_horizons_v1':
        from basil_core.dirichlet_horizon_study import validate_configuration
        validate_configuration(config)
        return
    from basil_core.research_protocol import validate_config
    gate=config.get('lossStudyRevision')==GATE_REVISION
    envelope = dict(config, lossStudyRevision='dirichlet_local_snapshot_anchor_v1',
        localObjective='source_ebm_snapshot_anchor' if config.get('ebmMode')=='gradient_norm_objective' else 'snapshot_anchor_only')
    validate_config(envelope)
    if gate:
        stage=config.get('readinessStage')
        expected=readiness_configs(config['partitionHash'],config['initialModelHash'],stage,
            _validate=False)
        match=next((c for c in expected if c['runId']==config.get('runId')),None)
        # Execution device and routed output paths may change, not science.
        routed={'executionDevice','resultDirectory','plotDirectory'}
        if match is None or any(config.get(k)!=v for k,v in match.items() if k not in routed):
            raise ValueError('Unapproved readiness diagnostic configuration.')
        if any(k.startswith('diagnostic') for k in config):
            raise ValueError('Readiness diagnostics retain the entire dataset.')
        return
    required = dict(lossStudyRevision=REVISION, nRounds=3, researchValid=False,
        purpose='paired_clean_three_round_local_calibration', partitionStrategy='dirichlet', dirichletAlpha=.2,
        attackerCount=0, resolvedAttackerIds=[], attackHidden=False, useChannelNoise=False,
        channelNoiseSigmaAbsolute=0., ebmMode='none', snapshotAnchorMu=.01, liveTrackingEnabled=False,
        logitCalibrationTau=TAU, logitCalibrationCountFloor=1,
        calibrationSelectionRule='ordinary_receiver_local_ce', executionMode='deterministic_single_process')
    if any(config.get(k)!=v for k,v in required.items()):
        raise ValueError('Calibration is restricted to this predeclared clean three-round diagnostic.')
    enabled = config.get('logitCalibrationEnabled')
    method=config.get('logitCalibrationMethod','fedlc')
    objective='logit_prior_snapshot_anchor' if method=='prior_adjustment' else 'fedlc_inspired_snapshot_anchor'
    if method not in {'fedlc','prior_adjustment'}:raise ValueError('Unsupported calibration study method.')
    if not isinstance(enabled, bool) or config['localObjective'] != (
            objective if enabled else 'snapshot_anchor_only'):
        raise ValueError('Calibration objective and worker flag mismatch.')
    if any(k.startswith('diagnostic') for k in config):
        raise ValueError('No reduced data or epoch changes in this diagnostic.')


def prior_diagnostic_config(partition_hash, initial_hash):
    cfg=diagnostic_configs(partition_hash,initial_hash)[1]
    cfg.update(localObjective='logit_prior_snapshot_anchor',logitCalibrationMethod='prior_adjustment',
        displayName='Clean local-prior adjustment — Dirichlet α=0.2 — 3 rounds, 5 epochs — diagnostic only',
        runId='clean_prior_adjusted_3r',experimentName='clean_prior_adjusted_3r',
        resultDirectory='newResults/nonIID/dirichlet_logit_calibration_3r_v1/prior_adjusted',
        plotDirectory='newPlots/nonIID/dirichlet_logit_calibration_3r_v1/prior_adjusted')
    validate_diagnostic_config(cfg)
    return cfg


def readiness_configs(partition_hash, initial_hash, stage, *, _validate=True):
    """Fixed prospective gates, not a hyperparameter search or production queue.

    Clean: 10 rounds, original vs prior correction. Noise: three rounds, paired
    prior CE vs prior EBM. Attack gate: 23 rounds B/C/D, unchanged R20 attack.
    A separate 30-round A/B/C/D feasibility profile is prepared, not automatic.
    EBM differentiates the adjusted F, explicitly a separate method adaptation.
    """
    from basil_core.dirichlet_study import configurations
    stages={'clean':(10,('baseline','prior_adjusted')),
            'noise':(3,('noise_ce','noise_ebm')),
            'attack_gate':(23,tuple('BCD')),'threat':(30,tuple('ABCD'))}
    if stage not in stages:raise ValueError('Unknown fixed readiness stage.')
    rounds,names=stages[stage]
    canonical=configurations(partition_hash,initial_hash)
    output=[]
    for name in names:
        letter=name if stage in {'threat','attack_gate'} else 'A' if stage=='clean' else 'C' if name=='noise_ce' else 'D'
        cfg=dict(canonical[ord(letter)-ord('A')])
        if stage=='noise':cfg.update(attackerCount=0,resolvedAttackerIds=[],attackHidden=False)
        enabled=name!='baseline'
        ebm=cfg['ebmMode']=='gradient_norm_objective'
        study=f'dirichlet_prior_readiness_{stage}_{rounds}r_v1'
        cfg.update(lossStudyRevision=GATE_REVISION,protocolRevision=GATE_REVISION,
            readinessStage=stage,noniidStudyId=study,nRounds=rounds,researchValid=False,
            purpose='bounded_prior_readiness_diagnostic',fullResearchReady=False,
            logitCalibrationEnabled=enabled,logitCalibrationMethod='prior_adjustment',
            logitCalibrationTau=TAU,logitCalibrationCountFloor=1,
            calibrationSelectionRule='ordinary_receiver_local_ce',
            localObjective='logit_prior_ebm_snapshot_anchor' if ebm else 'logit_prior_snapshot_anchor' if enabled else 'snapshot_anchor_only',
            runId=name,experimentName=name,
            displayName=f'{name} — local-prior {stage} diagnostic — {rounds} rounds, five full epochs',
            resultDirectory=f'newResults/nonIID/{study}/{name}',
            plotDirectory=f'newPlots/nonIID/{study}/{name}')
        if _validate:validate_diagnostic_config(cfg)
        output.append(cfg)
    return output
