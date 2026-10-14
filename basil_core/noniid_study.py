"""One-class BASIL snapshot anchoring; IID/historical objectives stay separate."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


from basil_core.iid_study import IidWorker
from basil_core.research_audit import augment_batch
from basil_core.research_protocol import gradient_norm_objective_gradients, params_hash

REVISION = 'source_grounded_no_lambda_v1'
ROOT = Path(__file__).resolve().parents[1]


def study_configs(mu=0.01, *, rounds=50, seed=2025, device='GPU'):
    from gui.research_protocol import base_config
    if not math.isfinite(mu) or mu < 0:
        raise ValueError('Snapshot mu must be explicitly finite and nonnegative.')
    configs = []
    for letter in 'ABCD':
        attack = letter != 'A'
        noise = letter in 'CD'
        ebm = letter == 'D'
        name = f'{letter}_one_class_{"attack" if attack else "clean"}_{"noise" if noise else "clean_channel"}_{"ebm_anchor" if ebm else "anchor"}_{rounds}r_{device.lower()}'
        cfg = base_config(seed=seed)
        cfg.update(protocolRevision='one_class_fixed_anchor_v1', lossStudyRevision=REVISION,
            conditionId=letter, runId=name, experimentName=name, nRounds=rounds,
            purpose='one_class_fixed_anchor_exploratory', snapshotSelection=True,
            attackerCount=4 if attack else 0, resolvedAttackerIds=cfg['resolvedAttackerIds'] if attack else [],
            attackHidden=attack, channelNoiseSemantics='paper_absolute_gaussian',
            channelNoiseSigmaAbsolute=.010 if noise else 0., useChannelNoise=noise,
            ebmCoefficientSemantics='channel_coordinate_variance',
            ebmMode='gradient_norm_objective' if ebm else 'none',
            localObjective='source_ebm_snapshot_anchor' if ebm else 'snapshot_anchor_only',
            snapshotAnchorMu=mu, snapshotAnchorMuMax=0., monteCarloNoiseSamples=0,
            executionDevice=device, executionMode='deterministic_single_process',
            roundEvaluationMetrics=True, liveTrackingEnabled=False,
            attackSourceStatus='project_transformation_retained_by_explicit_operator_choice',
            resultDirectory=f'newResults/nonIID/{name}', plotDirectory=f'newPlots/nonIID/{name}')
        configs.append(cfg)
    verify_study(configs)
    return configs


def verify_study(configs):
    from basil_core.research_protocol import validate_config
    if len(configs) != 4 or [c['conditionId'] for c in configs] != list('ABCD'):
        raise ValueError('Exactly A/B/C/D in order are required.')
    for cfg in configs:
        validate_config(cfg)
        if cfg['lossStudyRevision'] != REVISION or cfg.get('liveTrackingEnabled') is not False:
            raise ValueError('Non-IID loss study requires this revision and disabled cloud tracking.')
        for field, tree in [('resultDirectory', 'newResults/nonIID'), ('plotDirectory', 'newPlots/nonIID')]:
            if not (ROOT / cfg[field]).resolve().is_relative_to((ROOT / tree).resolve()):
                raise ValueError('Non-IID outputs must not enter IID evidence.')
    ignored = {'conditionId', 'runId', 'experimentName', 'resultDirectory', 'plotDirectory'}
    contrasts = {}
    allowed = {'A/B': {'attackerCount', 'resolvedAttackerIds', 'attackHidden'},
               'B/C': {'channelNoiseSigmaAbsolute', 'useChannelNoise'},
               'C/D': {'localObjective', 'ebmMode'}}
    for label, a, b in [('A/B', 0, 1), ('B/C', 1, 2), ('C/D', 2, 3)]:
        diff = {k for k in configs[a].keys() | configs[b].keys()
                if configs[a].get(k) != configs[b].get(k)} - ignored
        if diff != allowed[label]:
            raise ValueError(f'Unpaired {label}: {sorted(diff)}')
        contrasts[label] = sorted(diff)
    return contrasts


def partition_audit(train_labels, test_labels):
    import numpy as np
    train = np.asarray(train_labels)
    test = np.asarray(test_labels)
    if not np.array_equal(np.bincount(train, minlength=10), [5000]*10):
        raise ValueError('Full 50,000-image one-class training set is required.')
    if not np.array_equal(np.bincount(test, minlength=10), [1000]*10):
        raise ValueError('Full 10,000-image evaluation-only test set is required.')
    indices = [np.flatnonzero(train == i) for i in range(10)]
    joined = np.concatenate(indices)
    if not np.array_equal(np.sort(joined), np.arange(50000)):
        raise ValueError('Class partition overlap or coverage failure.')
    digest = hashlib.sha256()
    nodes = []
    for i, chunk in enumerate(indices):
        digest.update(np.asarray(chunk, dtype='<i8').tobytes())
        nodes.append(dict(nodeId=i, samples=len(chunk), classCounts=np.bincount(train[chunk], minlength=10).tolist()))
    return dict(partitionHash=digest.hexdigest(), assignedSamples=50000, uniqueSamples=50000,
                nodes=nodes, testClassCounts=[1000]*10, testUsage='evaluation_only'), indices


def anchored_gradients(model, x, y, reference, *, mu, sigma_e, ebm):
    """Differentiate CE/EBM and add the exact frozen-reference proximal gradient.

    EBM is exclusively sigma_e² ||grad CE||²: the anchor is NOT included inside
    that inner gradient, and legacy lambda is never read. References are detached.
    """
    from basil_core.trainer import lossFn
    import tensorflow as tf
    weights = model.trainable_variables
    if len(reference) != len(weights):
        raise ValueError('Anchor shape/count differs from model.')
    if ebm:
        source, values = gradient_norm_objective_gradients(model, x, y, sigma_e**2, details=True)
        ce, norm_sq, source_loss, coefficient, base_norm, _, correction_norm, accuracy = values
    else:
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            tf.debugging.assert_all_finite(logits, 'Non-finite training logits')
            ce = lossFn(y, logits)
        source = tape.gradient(ce, weights)
        if any(g is None for g in source):
            raise ValueError('Disconnected CE gradient.')
        base_norm = tf.linalg.global_norm(source)
        norm_sq = base_norm**2
        source_loss = ce
        coefficient = tf.constant(0., tf.float32)
        correction_norm = tf.constant(0., tf.float64)
        accuracy = tf.reduce_mean(tf.cast(tf.argmax(logits, axis=1, output_type=tf.int32) == y, tf.float32))
    drift = [w - tf.stop_gradient(tf.cast(a, w.dtype)) for w, a in zip(weights, reference)]
    distance_sq = tf.add_n([tf.reduce_sum(v**2) for v in drift])
    anchor_penalty = tf.cast(mu, tf.float32) * distance_sq / 2
    anchor_grads = [tf.cast(mu, g.dtype) * v for g, v in zip(source, drift)]
    applied = [g + a for g, a in zip(source, anchor_grads)]
    anchor_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.cast(g, tf.float64)**2) for g in anchor_grads]))
    values = (ce, norm_sq, source_loss + anchor_penalty, coefficient, base_norm,
              tf.linalg.global_norm(applied), correction_norm, accuracy, distance_sq,
              anchor_penalty, anchor_norm, tf.cast(mu, tf.float32),
              anchor_norm / (tf.cast(base_norm, tf.float64) + 1e-12))
    for value in values:
        tf.debugging.assert_all_finite(value, 'Non-finite anchored objective telemetry')
    for value in applied:
        tf.debugging.assert_all_finite(value, 'Non-finite anchored applied gradient')
    return applied, values


class SnapshotAnchorWorker(IidWorker):
    extra_metric_names = ('ebmCorrectionNorm', 'trainingBatchAccuracy', 'snapshotDistanceSquared',
        'snapshotAnchorPenalty', 'anchorGradientNorm', 'snapshotAnchorMu', 'anchorCorrectionRatio')

    def __init__(self, model, *, mu, device='CPU'):
        import tensorflow as tf
        self.mu = float(mu)
        self.device = device
        self.reference = [tf.Variable(tf.zeros_like(w), trainable=False) for w in model.trainable_variables]
        super().__init__(model)

    def _batch_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
        import tensorflow as tf
        if semantics != 'paper_absolute_gaussian' or mode not in {'none', 'gradient_norm_objective'}:
            raise ValueError('Anchored study requires absolute channel semantics and CE/source EBM.')
        with tf.device('/CPU:0'):
            images = augment_batch(x, key)
        return anchored_gradients(self.model, images, y, self.reference, mu=self.mu,
                                  sigma_e=sigma, ebm=mode == 'gradient_norm_objective')

    def train(self, params, x, y, **kwargs):
        for anchor, value in zip(self.reference, params):
            anchor.assign(value)
        expected = params_hash(params)
        result, telemetry = super().train(params, x, y, **kwargs)
        if params_hash([a.numpy() for a in self.reference]) != expected:
            raise ValueError('Frozen snapshot anchor changed during local training.')
        telemetry['snapshotAnchorReferenceHash'] = expected
        return result, telemetry
