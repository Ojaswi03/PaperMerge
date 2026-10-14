"""GPM-inspired, bias-aware single-head adaptation; NOT exact paper GPM.

Only orthonormal layer-input bases travel with a selected snapshot. No images,
test metrics, model averaging, proximal penalty or noise objective is added.
The previous scientific implementations are intentionally left byte-unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


from basil_core.iid_study import IidWorker
from basil_core.research_audit import augment_batch
from basil_core.research_protocol import Snapshot, initialize_memories, keyed_rng, params_hash

REVISION = 'bias_aware_layer_input_projection_v1'
ENERGY = .99
REPRESENTATION_IMAGES = 32
MAX_PATCHES = 2048


def validate_state(state, dimensions):
    import numpy as np
    if state is None:
        return dict(revision=REVISION, seenClasses=[],
                    bases=[np.zeros((d, 0), np.float32) for d in dimensions])
    if set(state) != {'revision', 'seenClasses', 'bases'} or state['revision'] != REVISION:
        raise ValueError('Invalid projection state revision/fields.')
    labels = state['seenClasses']
    if (not isinstance(labels, list) or any(type(c) is not int or not 0 <= c < 10 for c in labels)
            or labels != sorted(set(labels)) or len(state['bases']) != len(dimensions)):
        raise ValueError('Invalid projection state coverage/layer count.')
    bases = []
    for basis, dim in zip(state['bases'], dimensions):
        q = np.asarray(basis, np.float32)
        if q.ndim != 2 or q.shape[0] != dim or q.shape[1] > dim or not np.isfinite(q).all():
            raise ValueError('Invalid projection basis shape/values.')
        if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=2e-5, rtol=2e-5):
            raise ValueError('Projection basis is not orthonormal.')
        bases.append(q.copy())
    return dict(revision=REVISION, seenClasses=list(labels), bases=bases)


def state_hash(state):
    digest = hashlib.sha256(json.dumps([REVISION, state['seenClasses']]).encode())
    digest.update(params_hash(state['bases']).encode())
    return digest.hexdigest()


def update_basis(old, representation, energy=ENERGY):
    """Eq. 8/9 residual expansion using SVD-equivalent covariance eigenvectors.

    Representation columns are augmented layer inputs, including a bias one.
    No rank truncation discards previously protected directions.
    """
    import numpy as np
    if not 0 < energy <= 1:
        raise ValueError('Energy threshold must be in (0, 1].')
    r = np.asarray(representation, np.float64)
    q = np.asarray(old, np.float64)
    if r.ndim != 2 or q.ndim != 2 or r.shape[0] != q.shape[0] or not np.isfinite(r).all():
        raise ValueError('Invalid training representation.')
    validate_state(dict(revision=REVISION, seenClasses=[], bases=[q]), [r.shape[0]])
    total = float(np.sum(r*r))
    residual = r - q @ (q.T @ r)
    captured = float(np.sum((q.T @ r)**2))
    deficit = max(0., energy * total - captured)
    if deficit > max(total * 1e-12, 1e-20) and q.shape[1] < q.shape[0]:
        values, vectors = np.linalg.eigh(residual @ residual.T)
        values = np.maximum(values[::-1], 0.)
        vectors = vectors[:, ::-1]
        positive = values > max(total * 1e-12, 1e-20)
        rank = min(int(np.searchsorted(np.cumsum(values), deficit) + 1),
                   int(positive.sum()), q.shape[0] - q.shape[1])
        added = vectors[:, :rank]
        # Reorthogonalize against old state, preserving its exact span.
        added -= q @ (q.T @ added)
        added, _ = np.linalg.qr(added, mode='reduced')
        for col in range(added.shape[1]):
            pivot = int(np.argmax(np.abs(added[:, col])))
            if added[pivot, col] < 0:
                added[:, col] *= -1
        q = np.concatenate([q, added], axis=1)
    result = q.astype(np.float32)
    validate_state(dict(revision=REVISION, seenClasses=[], bases=[result]), [r.shape[0]])
    fraction = float(np.sum((q.T @ r)**2) / total) if total else 1.
    if fraction + 1e-6 < energy:
        raise ValueError('Projection basis failed its declared representation-energy contract.')
    return result, fraction


def project_affine_gradient(kernel_gradient, bias_gradient, basis):
    """Keras input-by-output convention: G_perp = G - Q(Q^T G).

    Bias is the final row of an affine matrix, not an unprotected shortcut.
    This is constrained CE optimization, NOT the gradient of CE + a penalty.
    """
    import tensorflow as tf
    shape = tf.shape(kernel_gradient)
    matrix = tf.concat([tf.reshape(kernel_gradient, [-1, shape[-1]]),
                        tf.reshape(bias_gradient, [1, shape[-1]])], axis=0)
    q = tf.stop_gradient(tf.cast(basis, matrix.dtype))
    tf.debugging.assert_equal(tf.shape(q)[0], tf.shape(matrix)[0])
    result = matrix - tf.matmul(q, tf.matmul(q, matrix, transpose_a=True))
    tf.debugging.assert_all_finite(result, 'Non-finite projected affine gradient')
    residual = tf.linalg.norm(tf.matmul(q, result, transpose_a=True)) / (tf.linalg.norm(matrix) + 1e-12)
    return tf.reshape(result[:-1], shape), result[-1], residual


class GradientProjectionWorker(IidWorker):
    extra_metric_names = ('ebmCorrectionNorm', 'trainingBatchAccuracy',
                          'projectionCorrectionNorm', 'projectionRemovedRatio',
                          'projectionOrthogonalityResidual')

    def __init__(self, model, *, enabled=True):
        import numpy as np
        import tensorflow as tf
        network = getattr(model, 'model', model)
        self.affine = [layer for layer in network.layers
                       if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))]
        if not self.affine or any(not layer.use_bias for layer in self.affine):
            raise ValueError('Bias-aware projection requires affine kernels and biases.')
        covered = {id(v) for layer in self.affine for v in (layer.kernel, layer.bias)}
        if covered != {id(v) for v in model.trainable_variables}:
            raise ValueError('Unsupported trainable variables: none may escape projection.')
        self.dimensions = [int(np.prod(layer.kernel.shape[:-1])) + 1 for layer in self.affine]
        self.bases = [tf.Variable(np.zeros((d, 0), np.float32), trainable=False,
                                 shape=tf.TensorShape([d, None])) for d in self.dimensions]
        self.features = tf.keras.Model(network.inputs, [layer.input for layer in self.affine])
        self.enabled = bool(enabled)
        self.input_state = None
        self.output_state = None
        super().__init__(model)

    def receive_retention_state(self, state):
        self.input_state = validate_state(state, self.dimensions)

    def _batch_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
        from basil_core.trainer import lossFn
        import tensorflow as tf
        if mode != 'none' or semantics != 'paper_absolute_gaussian':
            raise ValueError('Initial projection gate is clean CE only.')
        tf.debugging.assert_equal(sigma, 0.)
        with tf.device('/CPU:0'):
            images = augment_batch(x, key)
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            tf.debugging.assert_all_finite(logits, 'Non-finite projection training logits')
            ce = lossFn(y, logits)
        base = tape.gradient(ce, self.model.trainable_variables)
        if any(g is None for g in base):
            raise ValueError('Disconnected CE gradient in projection method.')
        indices = {id(v): i for i, v in enumerate(self.model.trainable_variables)}
        applied = list(base)
        residuals = []
        for layer, q in zip(self.affine, self.bases):
            k, b = indices[id(layer.kernel)], indices[id(layer.bias)]
            if self.enabled:
                applied[k], applied[b], residual = project_affine_gradient(base[k], base[b], q)
                residuals.append(residual)
        correction = tf.linalg.global_norm([a-g for a, g in zip(applied, base)])
        norm = tf.linalg.global_norm(base)
        accuracy = tf.reduce_mean(tf.cast(tf.argmax(logits, axis=1, output_type=tf.int32) == y, tf.float32))
        values = (ce, norm**2, ce, tf.constant(0.), norm, tf.linalg.global_norm(applied),
                  tf.constant(0., tf.float64), accuracy, correction,
                  correction/(norm+1e-12), tf.reduce_max(tf.stack(residuals)) if residuals else tf.constant(0.))
        for value in (*values, *applied):
            tf.debugging.assert_all_finite(value, 'Non-finite projection loss/gradient/telemetry')
        return applied, values

    def train(self, params, x, y, **kwargs):
        import numpy as np
        import tensorflow as tf
        current = kwargs['node_id']
        if np.any(np.asarray(y) != current):
            raise ValueError('Projection gate requires the receiver-only training class.')
        state = validate_state(self.input_state, self.dimensions)
        frozen = state_hash(state)
        for variable, value in zip(self.bases, state['bases']):
            variable.assign(value)
        # Parent loads ALL selected weights/biases and resets plain SGD.
        trained, record = super().train(params, x, y, **kwargs)
        if state_hash(state) != frozen:
            raise ValueError('Frozen incoming projection state changed during training.')
        fractions = []
        if self.enabled:
            chosen = keyed_rng(kwargs['seed'], 'projection_representations', current).permutation(len(y))[:REPRESENTATION_IMAGES]
            inputs = self.features(tf.convert_to_tensor(x[chosen]), training=False)
            if not isinstance(inputs, (tuple, list)):
                inputs = [inputs]
            for index, (layer, value, old) in enumerate(zip(self.affine, inputs, state['bases'])):
                with tf.device('/CPU:0'):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        value = tf.image.extract_patches(value, [1, *layer.kernel_size, 1],
                            [1, *layer.strides, 1], [1, *layer.dilation_rate, 1], layer.padding.upper())
                    array = np.asarray(value).reshape(-1, self.dimensions[index]-1)
                if len(array) > MAX_PATCHES:
                    visits = keyed_rng(kwargs['seed'], 'projection_patches', current, index).choice(len(array), MAX_PATCHES, replace=False)
                    array = array[visits]
                representation = np.concatenate([array, np.ones((len(array), 1), np.float32)], axis=1).T
                state['bases'][index], fraction = update_basis(old, representation)
                fractions.append(fraction)
            state['seenClasses'] = sorted(set(state['seenClasses']) | {current})
        self.output_state = validate_state(state, self.dimensions)
        record['gradientProjection'] = dict(revision=REVISION, enabled=self.enabled,
            inputStateHash=frozen, outputStateHash=state_hash(self.output_state),
            inputClasses=validate_state(self.input_state, self.dimensions)['seenClasses'],
            outputClasses=state['seenClasses'], ranks=[q.shape[1] for q in state['bases']],
            capturedEnergy=fractions, rawDataShared=False, biasProtected=True)
        record['inputHash'] = params_hash(params)
        return trained, record


def save_memory(path, memories, dimensions):
    """Lossless, atomic selected-snapshot+projection-state transport checkpoint.

    Gate-only memory checkpoint: NOT a complete resumable research checkpoint.
    No claim of exact research-run resume is made for this new method yet.
    """
    import numpy as np
    arrays, records = {}, []
    for receiver, memory in enumerate(memories):
        snapshots = []
        for index, snapshot in enumerate(memory.candidates()):
            prefix = f'r{receiver}_s{index}'
            state = validate_state(snapshot.retention_state, dimensions)
            for j, p in enumerate(snapshot.params):
                arrays[f'{prefix}_w{j}'] = p
            for j, q in enumerate(state['bases']):
                arrays[f'{prefix}_q{j}'] = q
            snapshots.append(dict(prefix=prefix, sender=snapshot.sender_id, round=snapshot.round_id,
                count=len(snapshot.params), paramsHash=params_hash(snapshot.params),
                seenClasses=state['seenClasses'], stateHash=state_hash(state)))
        records.append(snapshots)
    arrays['metadata'] = np.asarray(json.dumps(dict(revision=REVISION, dimensions=dimensions, memories=records)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_memory(path, dimensions):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved['metadata']))
        if meta['revision'] != REVISION or meta['dimensions'] != list(dimensions) or len(meta['memories']) != 10:
            raise ValueError('Invalid projection memory checkpoint.')
        memories = initialize_memories([np.zeros(1, np.float32)])
        for memory, records in zip(memories, meta['memories']):
            if len(records) != 5 or [r['sender'] for r in records] != list(memory.allowed_senders):
                raise ValueError('Projection checkpoint lost its five distinct predecessors.')
            memory.snapshots.clear()
            for r in records:
                prefix = r['prefix']
                params = [saved[f'{prefix}_w{j}'].copy() for j in range(r['count'])]
                state = validate_state(dict(revision=REVISION, seenClasses=r['seenClasses'],
                    bases=[saved[f'{prefix}_q{j}'].copy() for j in range(len(dimensions))]), dimensions)
                if params_hash(params) != r['paramsHash'] or state_hash(state) != r['stateHash'] or any(not np.isfinite(p).all() for p in params):
                    raise ValueError('Projection memory checksum/finite validation failed.')
                memory.receive(Snapshot(r['sender'], r['round'], params, retention_state=state))
        return memories
