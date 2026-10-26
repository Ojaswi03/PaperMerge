"""Receiver-local class prediction memory; separate from historical CART.

Training-prior CE, full EBM and the selected-snapshot anchor retain their
existing definitions. Only the additional retention term is new. Heavy imports
are local so inspection and queue planning do not initialize TensorFlow.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from basil_core.dirichlet_calibration import DirichletCalibrationWorker, GATE_REVISION

REVISION = 'cart_local_class_memory_v1'
PROBE_PER_CLASS = 16
MAX_PROBES = 160
EPSILON = 1e-8


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def _digest(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def regression_weights(current, best):
    import numpy as np
    current, best = np.asarray(current, np.float64), np.asarray(best, np.float64)
    if current.shape != best.shape or not np.isfinite(current).all() or not np.isfinite(best).all():
        raise ValueError('Finite matching class losses required.')
    if np.any(current < 0) or np.any(best < 0):
        raise ValueError('Class CE must be nonnegative.')
    return np.clip((current - best) / (current + EPSILON), 0., 1.).astype(np.float32)


class LocalClassMemory:
    def __init__(self, seed=2025, state=None):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError('Integer memory seed required.')
        self.seed = seed
        self.nodes = {}
        if state is not None:
            self.restore(state)

    def export(self):
        return copy.deepcopy(dict(revision=REVISION, seed=self.seed, nodes=self.nodes))

    def restore(self, state):
        import numpy as np
        value = copy.deepcopy(state)
        if (not isinstance(value, dict) or set(value) != {'revision', 'seed', 'nodes'} or
                value['revision'] != REVISION or value['seed'] != self.seed or not isinstance(value['nodes'], dict)):
            raise ValueError('CART memory revision/seed/schema mismatch.')
        for key, node in value['nodes'].items():
            if key not in {str(i) for i in range(10)}:
                raise ValueError('Invalid memory receiver.')
            if set(node) != {'dataHash', 'labelHash', 'classCounts', 'probeIndices', 'classes'}:
                raise ValueError('Invalid node memory schema.')
            counts = np.asarray(node['classCounts'])
            indices = np.asarray(node['probeIndices'])
            if (counts.shape != (10,) or not np.issubdtype(counts.dtype, np.integer) or
                    np.any(counts < 0) or counts.sum() <= 0 or
                    indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or
                    len(indices) != sum(min(PROBE_PER_CLASS, int(n)) for n in counts) or
                    len(np.unique(indices)) != len(indices) or
                    np.any(indices < 0) or np.any(indices >= counts.sum()) or
                    not _sha(node['dataHash']) or not _sha(node['labelHash'])):
                raise ValueError('Invalid memory support or probe identity.')
            if not isinstance(node['classes'], dict):
                raise ValueError('Invalid class memory.')
            for c, record in node['classes'].items():
                if c not in {str(i) for i in np.flatnonzero(counts)}:
                    raise ValueError('Memory cannot invent absent classes.')
                if set(record) != {'logProbabilities', 'loss', 'modelHash', 'activation', 'boundary'}:
                    raise ValueError('Invalid teacher record schema.')
                logs = np.asarray(record['logProbabilities'], np.float32)
                if (logs.shape != (min(PROBE_PER_CLASS, int(counts[int(c)])), 10) or
                        not np.isfinite(logs).all() or np.any(logs > 1e-6) or
                        not np.allclose(np.exp(logs).sum(axis=1), 1., atol=2e-6) or
                        not math.isfinite(record['loss']) or record['loss'] < 0 or
                        not math.isclose(record['loss'], float(-logs[:, int(c)].mean(dtype=np.float64)), abs_tol=1e-7) or
                        not _sha(record['modelHash']) or type(record['activation']) is not int or
                        record['activation'] < 0 or record['activation'] % 10 != int(key) or
                        record['boundary'] not in {'selected', 'trained'}):
                    raise ValueError('Invalid or corrupt class teacher targets.')
        self.nodes = value['nodes']

    def prepare(self, node_id, x, y):
        """Bind each cache to the exact receiver-local data; TEST is never an input."""
        import numpy as np
        from basil_core.research_protocol import keyed_rng
        labels = np.asarray(y)
        if (type(node_id) is not int or not 0 <= node_id < 10 or
                labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or
                not len(labels) or np.any(labels < 0) or np.any(labels >= 10) or
                len(x) != len(labels) or not np.isfinite(x).all()):
            raise ValueError('Valid receiver-local TRAIN images and labels required.')
        counts = np.bincount(labels, minlength=10)
        probes = np.concatenate([keyed_rng(self.seed, 'cart_memory_probe', node_id, int(c)).choice(
            np.flatnonzero(labels == c), min(PROBE_PER_CLASS, int(counts[c])), replace=False)
            for c in np.flatnonzero(counts)]).astype(np.int64)
        identity = dict(dataHash=_digest(x), labelHash=_digest(labels),
            classCounts=counts.tolist(), probeIndices=probes.tolist())
        previous = self.nodes.get(str(node_id))
        if previous is not None and any(previous[k] != v for k, v in identity.items()):
            raise ValueError('Receiver TRAIN data/probe identity changed since checkpoint.')
        if previous is None:
            self.nodes[str(node_id)] = dict(identity, classes={})
        return probes

    def admit(self, node_id, labels, log_probabilities, model_hash, activation, boundary):
        import numpy as np
        node = self.nodes[str(node_id)]
        logs = np.asarray(log_probabilities, np.float32)
        labels = np.asarray(labels)
        if (logs.shape != (len(labels), 10) or not np.isfinite(logs).all() or
                np.any(logs > 1e-6) or not np.allclose(np.exp(logs).sum(axis=1), 1., atol=2e-6) or
                not _sha(model_hash) or type(activation) is not int or activation < 0 or
                activation % 10 != node_id or boundary not in {'selected', 'trained'} or
                not np.array_equal(np.bincount(labels, minlength=10), np.minimum(node['classCounts'], PROBE_PER_CLASS))):
            raise ValueError('Invalid locally measured teacher candidate.')
        accepted = []
        for c in np.unique(labels):
            class_logs = logs[labels == c]
            loss = float(-class_logs[:, c].mean(dtype=np.float64))
            old = node['classes'].get(str(c))
            if old is None or loss < old['loss']:
                node['classes'][str(c)] = dict(logProbabilities=class_logs.tolist(), loss=loss,
                    modelHash=model_hash, activation=activation, boundary=boundary)
                accepted.append(int(c))
        return accepted

    def controls(self, node_id, labels, current_logs):
        import numpy as np
        node = self.nodes[str(node_id)]
        labels = np.asarray(labels)
        current_logs = np.asarray(current_logs, np.float32)
        if current_logs.shape != (len(labels), 10) or not np.isfinite(current_logs).all():
            raise ValueError('Invalid current class-probe predictions.')
        teacher = np.full((len(labels), 10), -np.log(10.), np.float32)
        row_weights = np.zeros(len(labels), np.float32)
        records = []
        class_count = len(node['classes'])
        for key, record in node['classes'].items():
            c = int(key)
            mask = labels == c
            current = float(-current_logs[mask, c].mean(dtype=np.float64))
            weight = float(regression_weights([current], [record['loss']])[0])
            teacher[mask] = record['logProbabilities']
            row_weights[mask] = weight / (class_count * int(mask.sum()))
            records.append(dict(classId=c, currentLoss=current, bestLoss=record['loss'],
                weight=weight, support=int(mask.sum()), teacherActivation=record['activation']))
        return teacher, row_weights, records


def retained_gradients(model, x, y, reference, offsets, probe_x, teacher_logs,
                      row_weights, *, beta=1., mu=.01, coefficient=0.):
    """Exact base/EBM/anchor gradient plus beta times raw prediction KL gradient."""
    import tensorflow as tf
    from basil_core.dirichlet_calibration import calibrated_anchor_gradients
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('Finite nonnegative retention coefficient required.')
    base, values = calibrated_anchor_gradients(model, x, y, reference, offsets,
        mu=mu, coefficient=coefficient)
    weights = tf.stop_gradient(tf.cast(row_weights, tf.float32))
    teacher = tf.stop_gradient(tf.cast(teacher_logs, tf.float32))
    tf.debugging.assert_non_negative(weights)
    tf.debugging.assert_all_finite(weights, 'Invalid CART class weights')
    tf.debugging.assert_all_finite(teacher, 'Invalid CART teacher')
    tf.debugging.assert_equal(tf.shape(teacher), [tf.shape(probe_x)[0], 10])
    tf.debugging.assert_equal(tf.shape(weights), [tf.shape(probe_x)[0]])
    def retention():
        with tf.GradientTape() as tape:
            current = tf.nn.log_softmax(model(probe_x, training=False))
            loss = tf.reduce_sum(weights * tf.reduce_sum(tf.exp(teacher) * (teacher-current), axis=1))
        gradients = tape.gradient(loss, model.trainable_variables)
        if any(g is None for g in gradients):
            raise ValueError('Disconnected CART retention gradient.')
        return loss, gradients
    if beta == 0:
        loss, correction = tf.constant(0.), [tf.zeros_like(g) for g in base]
    else:
        loss, correction = tf.cond(tf.reduce_any(weights > 0), retention,
            lambda: (tf.constant(0.), [tf.zeros_like(g) for g in base]))
    correction = [beta*g for g in correction]
    applied = [g+r for g, r in zip(base, correction)] if beta else base
    result = list(values)
    result[2] = values[2] + beta*loss
    result[5] = tf.linalg.global_norm(applied) if beta else values[5]
    result += [loss, tf.linalg.global_norm(correction), tf.reduce_sum(weights)]
    for value in applied + result:
        tf.debugging.assert_all_finite(value, 'Non-finite CART objective/gradient')
    return applied, tuple(result)


class LocalMemoryCartWorker(DirichletCalibrationWorker):
    extra_metric_names = DirichletCalibrationWorker.extra_metric_names + (
        'cartRetentionLoss', 'cartRetentionGradientNorm', 'cartMeanRegressionWeight')

    def __init__(self, model, *, beta=1., seed=2025, device='CPU'):
        import tensorflow as tf
        if beta not in (0., 1.):
            raise ValueError('This bounded revision predeclares beta 0 or 1.')
        self.beta = float(beta)
        self.local_memory = LocalClassMemory(seed)
        self.probe_images = tf.Variable(tf.zeros((MAX_PROBES, 32, 32, 3)), trainable=False)
        self.teacher_logs = tf.Variable(tf.fill((MAX_PROBES, 10), -math.log(10.)), trainable=False)
        self.row_weights = tf.Variable(tf.zeros(MAX_PROBES), trainable=False)
        super().__init__(model, calibration=True, device=device, method='prior_adjustment', revision=GATE_REVISION)
        self.diagnostic_revision = REVISION

    def export_local_memory(self):
        return self.local_memory.export()

    def restore_local_memory(self, state):
        self.local_memory.restore(state)

    def _set_retention_controls(self, probe_y, rows):
        """Separate revisions may configure a different local retention loss."""

    def _batch_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
        import tensorflow as tf
        from basil_core.research_audit import augment_batch
        if semantics != 'paper_absolute_gaussian' or mode not in {'none', 'gradient_norm_objective'}:
            raise ValueError('CART uses the existing absolute-channel CE/full EBM modes.')
        with tf.device('/CPU:0'):
            images = augment_batch(x, key)
        return retained_gradients(self.model, images, y, self.reference, self.offsets,
            self.probe_images, self.teacher_logs, self.row_weights, beta=self.beta, mu=self.mu,
            coefficient=sigma**2 if mode == 'gradient_norm_objective' else 0.)

    def train(self, params, x, y, **kwargs):
        import numpy as np
        import tensorflow as tf
        from basil_core.research_protocol import params_hash
        if kwargs['seed'] != self.local_memory.seed:
            raise ValueError('Worker/config memory seed mismatch.')
        old = self.export_local_memory()
        try:
            node_id = kwargs['node_id']
            activation = kwargs['round_id']*10 + node_id
            probe = self.local_memory.prepare(node_id, x, y)
            probe_x, probe_y = np.asarray(x[probe]), np.asarray(y[probe])
            padded = np.zeros(self.probe_images.shape, np.float32)
            padded[:len(probe)] = probe_x
            self.probe_images.assign(padded)
            def logs():
                return tf.nn.log_softmax(self.predict(tf.convert_to_tensor(probe_x))).numpy()
            self.load(params)
            incoming = self.local_memory.admit(node_id, probe_y, logs(), params_hash(params), activation, 'selected')
            controls = []
            external = kwargs.pop('evaluation', None)
            def evaluate():
                teacher, weights, rows = self.local_memory.controls(node_id, probe_y, logs())
                q = np.full(self.teacher_logs.shape, -math.log(10.), np.float32)
                w = np.zeros(self.row_weights.shape, np.float32)
                q[:len(probe)], w[:len(probe)] = teacher, weights
                self.teacher_logs.assign(q)
                self.row_weights.assign(w)
                self._set_retention_controls(probe_y, rows)
                controls.append(dict(stage=len(controls), classes=rows,
                    positiveWeights=sum(r['weight'] > 0 for r in rows)))
                # Control is finalized before evaluation-only reporting.
                return external() if external else {}
            trained, record = super().train(params, x, y, evaluation=evaluate, **kwargs)
            outgoing = self.local_memory.admit(node_id, probe_y, logs(), params_hash(trained), activation, 'trained')
            record['cartMemory'] = dict(revision=self.diagnostic_revision, beta=self.beta, probeCount=len(probe),
                probeHash=_digest(probe), beforeHash=state_hash(old),
                afterHash=state_hash(self.export_local_memory()), incomingAdmissions=incoming,
                trainedAdmissions=outgoing, epochControls=controls,
                probeControlForwardImages=(len(controls)+2)*len(probe),
                retentionForwardImages=sum(MAX_PROBES for s in record['optimizerTelemetry']
                    if self.beta and s.get('cartMeanRegressionWeight', s.get('cartActiveClassFraction', 0.)) > 0))
            return trained, record
        except BaseException:
            # An interrupted activation must not leak future teacher knowledge
            # into an otherwise earlier model/memory checkpoint.
            self.restore_local_memory(old)
            raise
