"""Training-only class-conditioned gradient importance; NOT exact Fisher/EWC.

This separately labelled experiment transmits diagonal summaries, never images.
It does not alter BASIL selection and makes no Byzantine-trust guarantee.
"""
from __future__ import annotations

import hashlib
import math

from basil_core.noniid_study import SnapshotAnchorWorker, anchored_gradients
from basil_core.research_audit import augment_batch
from basil_core.research_protocol import params_hash

REVISION = 'class_gradient_importance_anchor_v1'
IMPORTANCE_CAP = 100.0


def validate_state(state, params):
    """Copy and validate the bounded, latest-per-class auxiliary snapshot state."""
    import numpy as np
    if state is None:
        return {'revision': REVISION, 'classes': {}, 'visits': {}}
    if set(state) != {'revision', 'classes', 'visits'} or state['revision'] != REVISION:
        raise ValueError('Invalid class-retention state revision/fields.')
    if set(state['classes']) != set(state['visits']) or len(state['classes']) > 10:
        raise ValueError('Invalid class-retention coverage.')
    result = {'revision': REVISION, 'classes': {}, 'visits': {}}
    for key, arrays in state['classes'].items():
        if not isinstance(key, str) or key not in {str(c) for c in range(10)}:
            raise ValueError('Class-retention labels must be CIFAR class IDs.')
        visits = state['visits'][key]
        if isinstance(visits, bool) or not isinstance(visits, int) or visits <= 0:
            raise ValueError('Class-retention sample visits must be positive integers.')
        if len(arrays) != len(params):
            raise ValueError('Class-retention parameter count differs.')
        copies = []
        for value, parameter in zip(arrays, params):
            a = np.asarray(value, dtype=np.float32)
            if a.shape != np.asarray(parameter).shape or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > IMPORTANCE_CAP):
                raise ValueError('Invalid class-retention importance shape/range.')
            copies.append(a.copy())
        result['classes'][key] = copies
        result['visits'][key] = visits
    return result


def state_hash(state):
    digest = hashlib.sha256(REVISION.encode())
    if state:
        for label in sorted(state['classes'], key=int):
            digest.update(f"{label}:{state['visits'][label]}:".encode())
            digest.update(params_hash(state['classes'][label]).encode())
    return digest.hexdigest()


def normalized_importance(moments):
    """Global coordinate-mean normalization and predeclared importance saturation.

    Squared minibatch-mean gradients along training are NOT per-example Fisher.
    The cap bounds the penalty's stiffness, not the applied SGD gradient.
    """
    import numpy as np
    arrays = [np.asarray(a, np.float64) for a in moments]
    if not arrays or any(not np.isfinite(a).all() or np.any(a < 0) for a in arrays):
        raise ValueError('Non-finite/negative gradient-importance accumulator.')
    mean = sum(float(a.sum()) for a in arrays) / sum(a.size for a in arrays)
    return [np.minimum(a / mean, IMPORTANCE_CAP).astype(np.float32) if mean > 0 else np.zeros_like(a, np.float32) for a in arrays]


def protection_weights(state, params, current_class):
    import numpy as np
    state = validate_state(state, params)
    previous = sorted(int(c) for c in state['classes'] if int(c) != current_class)
    weights = [np.ones_like(p, np.float32) for p in params]
    if previous:
        for index in range(len(params)):
            weights[index] += np.mean([state['classes'][str(c)][index] for c in previous], axis=0, dtype=np.float32)
    return weights, previous


def weighted_anchor_gradients(model, x, y, reference, importance, *, mu):
    """CE + mu/2 sum_q Omega_q (theta_q - selected_q)^2, Omega detached."""
    import numpy as np
    import tensorflow as tf
    if len(importance) != len(model.trainable_variables):
        raise ValueError('Importance shape/count differs from model.')
    base, raw = anchored_gradients(model, x, y, reference, mu=0., sigma_e=0., ebm=False)
    drift = [w - tf.stop_gradient(tf.cast(a, w.dtype)) for w, a in zip(model.trainable_variables, reference)]
    diagonal = [tf.stop_gradient(tf.cast(v, w.dtype)) for v, w in zip(importance, model.trainable_variables)]
    for value, variable in zip(diagonal, model.trainable_variables):
        tf.debugging.assert_equal(tf.shape(value), tf.shape(variable))
        tf.debugging.assert_all_finite(value, 'Non-finite importance diagonal')
        tf.debugging.assert_greater_equal(value, tf.ones_like(value))
        tf.debugging.assert_less_equal(value, tf.ones_like(value) * (1 + IMPORTANCE_CAP))
    weighted_distance = tf.add_n([tf.reduce_sum(v * d**2) for v, d in zip(diagonal, drift)])
    penalty = tf.cast(mu, tf.float32) / 2 * weighted_distance
    anchor = [tf.cast(mu, g.dtype) * v * d for g, v, d in zip(base, diagonal, drift)]
    applied = [g + a for g, a in zip(base, anchor)]
    norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.cast(a, tf.float64)**2) for a in anchor]))
    values = list(raw)
    values[2] = raw[0] + penalty
    values[5] = tf.linalg.global_norm(applied)
    values[9:13] = [penalty, norm, tf.cast(mu, tf.float32), norm / (tf.cast(raw[4], tf.float64) + 1e-12)]
    count = sum(int(np.prod(w.shape)) for w in model.trainable_variables)
    values.extend([tf.add_n([tf.reduce_sum(v) for v in diagonal]) / count,
                   tf.reduce_max(tf.stack([tf.reduce_max(v) for v in diagonal])), weighted_distance])
    for value in values:
        tf.debugging.assert_all_finite(value, 'Non-finite class-aware objective')
    for gradient in applied:
        tf.debugging.assert_all_finite(gradient, 'Non-finite class-aware gradient')
    return applied, tuple(values), base


class ClassAwareRetentionWorker(SnapshotAnchorWorker):
    extra_metric_names = SnapshotAnchorWorker.extra_metric_names + ('importanceWeightMean', 'importanceWeightMax', 'weightedSnapshotDistanceSquared')

    def __init__(self, model, *, mu=.01, device='CPU'):
        import tensorflow as tf
        if not math.isfinite(mu) or mu < 0:
            raise ValueError('Finite nonnegative retention mu is required.')
        self.diagonal = [tf.Variable(tf.ones_like(w), trainable=False) for w in model.trainable_variables]
        self.moments = [tf.Variable(tf.zeros_like(w, dtype=tf.float64), trainable=False) for w in model.trainable_variables]
        self.input_state = None
        self.output_state = None
        super().__init__(model, mu=mu, device=device)

    def receive_retention_state(self, state):
        self.input_state = validate_state(state, self.export())

    def export_retention_state(self):
        return self.output_state

    def _batch_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
        import tensorflow as tf
        if mode != 'none' or semantics != 'paper_absolute_gaussian':
            raise ValueError('Initial class-aware diagnostic is clean CE only; no EBM/threat mixing.')
        tf.debugging.assert_equal(sigma, 0.)
        with tf.device('/CPU:0'):
            images = augment_batch(x, key)
        applied, values, base = weighted_anchor_gradients(self.model, images, y, self.reference, self.diagonal, mu=self.mu)
        for accumulator, gradient in zip(self.moments, base):
            update = tf.cast(tf.shape(y)[0], tf.float64) * tf.cast(gradient, tf.float64)**2
            tf.debugging.assert_all_finite(update, 'Non-finite importance update')
            accumulator.assign_add(update)
        return applied, values

    def train(self, params, x, y, **kwargs):
        import numpy as np
        import tensorflow as tf
        current = kwargs['node_id']
        if np.any(np.asarray(y) != current):
            raise ValueError('Class-aware diagnostic requires actual receiver-only class labels.')
        incoming = validate_state(self.input_state, params)
        mask, previous = protection_weights(incoming, params, current)
        for variable, value in zip(self.diagonal, mask):
            variable.assign(value)
        for variable in self.moments:
            variable.assign(tf.zeros_like(variable))
        frozen = state_hash(incoming)
        trained, record = super().train(params, x, y, **kwargs)
        if state_hash(incoming) != frozen:
            raise ValueError('Incoming class importance changed during training.')
        outgoing = validate_state(incoming, params)
        outgoing['classes'][str(current)] = normalized_importance([a.numpy() for a in self.moments])
        outgoing['visits'][str(current)] = len(y) * kwargs['epochs']
        self.output_state = outgoing
        record.update(classRetention=dict(revision=REVISION, inputStateHash=frozen,
            outputStateHash=state_hash(outgoing), inputClasses=sorted(map(int, incoming['classes'])),
            protectedClasses=previous, outputClasses=sorted(map(int, outgoing['classes'])),
            rawDataShared=False, estimator='sample_weighted_squared_minibatch_CE_gradient_along_training',
            sampleVisits=len(y)*kwargs['epochs'], importanceCap=IMPORTANCE_CAP))
        return trained, record


def diagnostic_config():
    from scripts.prepare_retention_diagnostic import diagnostic_configs
    cfg = diagnostic_configs(30)[1]
    name = 'clean_class_gradient_importance_anchor_30r_gpu'
    cfg.update(lossStudyRevision=REVISION, protocolRevision=REVISION,
        localObjective='class_gradient_importance_anchor', effectiveLocalObjective='class_gradient_importance_anchor',
        controlId='clean_class_gradient_importance_anchor', runId=name, experimentName=name,
        purpose='full_data_class_aware_retention_diagnostic', classImportanceCap=IMPORTANCE_CAP,
        classImportanceEstimator='squared_minibatch_gradient_along_training',
        resultDirectory=f'newResults/nonIID/class_aware_retention/{name}',
        plotDirectory=f'newPlots/nonIID/class_aware_retention/{name}')
    return cfg
