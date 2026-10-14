"""Experimental, activation-bound CART feature rehearsal; no automatic wiring.

Credit: PASS/feature rehearsal, FedProto/class means, FedProK and FeTrIL for
related feature-space retention, PearFL for decentralized exchange. This is not
an exact reproduction: no translated features, frozen encoder or averaging.
Features can leak information. Provenance checks do not establish honesty.
Numerical/ML dependencies are imported only on use.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from basil_core.strict_cart import StrictCartWorker

REVISION = 'strict_cart_activation_feature_replay_v1'
MAX_FEATURES_PER_CLASS = 16
MAX_OWNER_PROBE = 512
MAX_FEATURE_DIM = 192


def _integer(value, lower, upper, name):
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f'Invalid {name}.')


def _hash(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Expected an exact SHA-256 identity.')


def encoder_hash(params):
    """Canonical hash of all parameters preceding the final dense kernel/bias."""
    from basil_core.research_protocol import params_hash
    if len(params) < 4:
        raise ValueError('Separate encoder and final dense head required.')
    return params_hash(params[:-2])


@dataclass(frozen=True)
class FeaturePacket:
    owner: int
    class_id: int
    round_id: int
    receiver: int
    model_hash: str
    encoder_hash: str
    support: int
    prototype: tuple
    features: tuple

    def __post_init__(self):
        for name in ('owner', 'class_id', 'receiver'):
            _integer(getattr(self, name), 0, 9, name)
        _integer(self.round_id, 0, 10**9, 'round')
        _integer(self.support, 1, MAX_OWNER_PROBE, 'support')
        if self.owner != self.class_id:
            raise ValueError('This revision requires node ID == class ID.')
        _hash(self.model_hash); _hash(self.encoder_hash)
        if type(self.prototype) is not tuple or type(self.features) is not tuple:
            raise ValueError('Immutable prototype/feature tuples required.')
        width = len(self.prototype)
        if not 1 <= width <= MAX_FEATURE_DIM:
            raise ValueError('Invalid embedding dimension; raw images are not packets.')
        if not 1 <= len(self.features) <= min(self.support, MAX_FEATURES_PER_CLASS):
            raise ValueError('Feature packet exceeds its support or bounded memory.')
        rows = (self.prototype, *self.features)
        if any(type(row) is not tuple or len(row) != width for row in rows):
            raise ValueError('Matching immutable feature vectors required.')
        if any(type(v) is not float or not math.isfinite(v) for row in rows for v in row):
            raise ValueError('Finite feature coordinates required.')

    def to_dict(self):
        return dict(owner=self.owner, classId=self.class_id, round=self.round_id,
            receiver=self.receiver, modelHash=self.model_hash, encoderHash=self.encoder_hash,
            support=self.support, prototype=list(self.prototype),
            features=[list(row) for row in self.features], dataUsage='training_only')

    @classmethod
    def from_dict(cls, value):
        required = {'owner', 'classId', 'round', 'receiver', 'modelHash', 'encoderHash',
                    'support', 'prototype', 'features', 'dataUsage'}
        if set(value) != required or value['dataUsage'] != 'training_only':
            raise ValueError('Unsupported packet schema or non-training provenance.')
        return cls(value['owner'], value['classId'], value['round'], value['receiver'],
            value['modelHash'], value['encoderHash'], value['support'],
            tuple(float(v) for v in value['prototype']),
            tuple(tuple(float(v) for v in row) for row in value['features']))

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()


def packet_from_features(features, *, owner, round_id, receiver, model_hash,
                         encoder_id, data_usage='training_only'):
    import numpy as np
    if data_usage != 'training_only':
        raise ValueError('Test/validation data cannot supply feature rehearsal.')
    z = np.asarray(features, dtype=np.float32)
    if z.ndim != 2 or not 1 <= len(z) <= MAX_OWNER_PROBE or not np.isfinite(z).all():
        raise ValueError('Finite bounded owner embeddings required, not raw images.')
    prototype = z.astype(np.float64).mean(axis=0)
    return FeaturePacket(owner, owner, round_id, receiver, model_hash, encoder_id,
        len(z), tuple(map(float, prototype)),
        tuple(tuple(map(float, row)) for row in z[:MAX_FEATURES_PER_CLASS]))


@dataclass(frozen=True)
class ActivationFeatureMemory:
    """Fresh selected-model coordinates, fixed for exactly one local activation."""
    receiver: int
    round_id: int
    model_hash: str
    encoder_hash: str
    feature_dim: int
    allowed_classes: tuple
    packets: tuple = ()

    def __post_init__(self):
        _integer(self.receiver, 0, 9, 'receiver')
        _integer(self.round_id, 0, 10**9, 'round')
        _integer(self.feature_dim, 1, MAX_FEATURE_DIM, 'feature dimension')
        _hash(self.model_hash); _hash(self.encoder_hash)
        if type(self.allowed_classes) is not tuple or type(self.packets) is not tuple:
            raise ValueError('Immutable memory required.')
        for c in self.allowed_classes: _integer(c, 0, 9, 'old class')
        if len(set(self.allowed_classes)) != len(self.allowed_classes) or self.receiver in self.allowed_classes:
            raise ValueError('Distinct previously visited other classes required.')
        present = []
        for packet in self.packets:
            if not isinstance(packet, FeaturePacket): raise ValueError('Invalid packet.')
            if (packet.receiver, packet.round_id, packet.model_hash, packet.encoder_hash) != (
                self.receiver, self.round_id, self.model_hash, self.encoder_hash):
                raise ValueError('Stale, wrong-model, wrong-encoder or wrong-activation features.')
            if packet.class_id not in self.allowed_classes or len(packet.prototype) != self.feature_dim:
                raise ValueError('Unexpected class or feature coordinates.')
            present.append(packet.class_id)
        if len(set(present)) != len(present):
            raise ValueError('Only one owner packet per class is permitted.')

    def tensors(self):
        """Fixed bounded arrays; each available old class gets equal total weight."""
        import numpy as np
        count = len(self.packets)
        z = np.zeros((10*MAX_FEATURES_PER_CLASS, self.feature_dim), np.float32)
        y = np.zeros(len(z), np.int32)
        weights = np.zeros(len(z), np.float32)
        offset = 0
        for packet in sorted(self.packets, key=lambda p: p.class_id):
            n = len(packet.features)
            z[offset:offset+n] = packet.features
            y[offset:offset+n] = packet.class_id
            weights[offset:offset+n] = 1./(count*n)
            offset += n
        return z, y, weights

    def to_dict(self):
        present = sorted(p.class_id for p in self.packets)
        return dict(revision=REVISION, receiver=self.receiver, round=self.round_id,
            modelHash=self.model_hash, encoderHash=self.encoder_hash, featureDim=self.feature_dim,
            allowedClasses=list(self.allowed_classes), presentClasses=present,
            unavailableClasses=sorted(set(self.allowed_classes)-set(present)),
            packets=[p.to_dict() for p in self.packets], packetHashes=[p.digest for p in self.packets],
            featureRows=sum(len(p.features) for p in self.packets),
            coordinateBytes=sum((1+len(p.features))*self.feature_dim*4 for p in self.packets),
            trustStatus='clean diagnostic only; provenance is not Byzantine verification')


def owner_refresh_memory(model, train_images, train_labels, training_indices, *,
                         receiver, round_id, seen_classes, seed):
    """Simulate owner-local encoding; return embeddings, never another owner's images.

    Additional protocol: send the selected encoder to previously visited class
    owners, who encode a deterministic training subset. This is explicit new
    communication, not BASIL metadata and not privacy/Byzantine-safe by default.
    """
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import keyed_rng, params_hash
    network = getattr(model, 'model', model)
    params = [w.numpy() for w in model.trainable_variables]
    model_id, encoder_id = params_hash(params), encoder_hash(params)
    width = int(network.layers[-1].kernel.shape[0])
    allowed = tuple(sorted(set(seen_classes)-{receiver}))
    # Validate even when no class has yet been visited.
    memory = ActivationFeatureMemory(receiver, round_id, model_id, encoder_id, width, allowed)
    if not allowed: return memory
    embedding = tf.keras.Model(network.inputs, network.layers[-2].output)
    packets = []
    for owner in allowed:
        indices = np.asarray(training_indices[owner])
        if not np.issubdtype(indices.dtype, np.integer) or len(indices) == 0:
            raise ValueError('Owner training indices required.')
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(train_labels)):
            raise ValueError('Invalid owner training assignment.')
        if np.any(np.asarray(train_labels)[indices] != owner):
            raise ValueError('One-class owner assignment differs from labels.')
        selected = keyed_rng(seed, 'cart_owner_feature_batch', owner).permutation(indices)[:MAX_OWNER_PROBE]
        chunks = []
        for start in range(0, len(selected), 128):
            z = embedding(tf.convert_to_tensor(train_images[selected[start:start+128]]), training=False)
            tf.debugging.assert_all_finite(z, 'Non-finite owner embedding')
            chunks.append(z.numpy())
        packets.append(packet_from_features(np.concatenate(chunks), owner=owner,
            round_id=round_id, receiver=receiver, model_hash=model_id, encoder_id=encoder_id))
    if params_hash([w.numpy() for w in model.trainable_variables]) != model_id:
        raise ValueError('Owner refresh changed selected model parameters.')
    return ActivationFeatureMemory(receiver, round_id, model_id, encoder_id, width, allowed, tuple(packets))


def feature_replay_gradients(model, x, y, reference, features, feature_labels, row_weights,
                             *, mu, sigma_e, ebm, beta):
    """CE + exact source EBM + frozen anchor + beta * old-class feature CE.

    EBM remains the gradient norm of local CE only. Detached feature replay
    differentiates the final head only; absent encoder gradients are expected
    for THIS term, never a fallback for disconnected local CE gradients.
    """
    import tensorflow as tf
    from basil_core.noniid_study import anchored_gradients
    network = getattr(model, 'model', model)
    head = network.layers[-1]
    if not isinstance(head, tf.keras.layers.Dense) or head.activation != tf.keras.activations.linear:
        raise ValueError('A separate final Dense logits head is required.')
    beta = tf.cast(beta, tf.float32)
    tf.debugging.assert_all_finite(beta, 'Non-finite replay beta')
    tf.debugging.assert_greater_equal(beta, 0.)
    tf.debugging.assert_all_finite(features, 'Non-finite replay features')
    tf.debugging.assert_all_finite(row_weights, 'Non-finite replay weights')
    tf.debugging.assert_greater_equal(row_weights, 0.)
    weight_sum = tf.reduce_sum(row_weights)
    tf.debugging.assert_near(weight_sum, tf.where(weight_sum > 0, 1., 0.), atol=1e-6)
    source, values = anchored_gradients(model, x, y, reference, mu=mu, sigma_e=sigma_e, ebm=ebm)
    with tf.GradientTape() as tape:
        logits = head(tf.stop_gradient(features), training=True)
        tf.debugging.assert_all_finite(logits, 'Non-finite replay logits')
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=feature_labels, logits=logits)
        replay = tf.reduce_sum(losses*tf.stop_gradient(row_weights))
        penalty = beta*replay
    replay_head = tape.gradient(penalty, head.trainable_variables)
    if any(g is None for g in replay_head): raise ValueError('Disconnected replay head gradient.')
    head_map = {id(w): g for w, g in zip(head.trainable_variables, replay_head)}
    replay_grads = [head_map.get(id(w), tf.zeros_like(w)) for w in model.trainable_variables]
    applied = [g+r for g, r in zip(source, replay_grads)]
    norm = tf.linalg.global_norm(replay_grads)
    result = list(values)
    result[2] = values[2]+penalty
    result[5] = tf.linalg.global_norm(applied)
    result.extend((replay, penalty, norm, norm/(tf.cast(values[4], tf.float32)+1e-12), beta))
    for v in result: tf.debugging.assert_all_finite(v, 'Non-finite feature objective telemetry')
    for g in applied: tf.debugging.assert_all_finite(g, 'Non-finite feature applied gradient')
    return applied, tuple(result)


class FeatureReplayWorker(StrictCartWorker):
    """Explicit opt-in; inherits full epochs, strict load, registry and frozen anchor."""
    extra_metric_names = StrictCartWorker.extra_metric_names + (
        'featureReplayLoss', 'featureReplayPenalty', 'featureReplayGradientNorm',
        'featureReplayCorrectionRatio', 'featureReplayBeta')

    def __init__(self, model, *, beta, mu_max=.01, device='CPU'):
        import tensorflow as tf
        if not math.isfinite(beta) or beta < 0: raise ValueError('Explicit finite nonnegative beta required.')
        self.beta = float(beta)
        network = getattr(model, 'model', model)
        width = int(network.layers[-1].kernel.shape[0])
        if not 1 <= width <= MAX_FEATURE_DIM: raise ValueError('Unsupported feature width.')
        self.feature_rows = tf.Variable(tf.zeros((160, width)), trainable=False)
        self.feature_labels = tf.Variable(tf.zeros(160, tf.int32), trainable=False)
        self.feature_weights = tf.Variable(tf.zeros(160), trainable=False)
        self._memory = None
        super().__init__(model, mu_max=mu_max, device=device)

    def set_memory(self, memory):
        from basil_core.research_protocol import params_hash
        params = self.export()
        if not isinstance(memory, ActivationFeatureMemory) or (
            memory.model_hash != params_hash(params) or memory.encoder_hash != encoder_hash(params)):
            raise ValueError('Replay memory must match the exact selected starting model.')
        z, labels, weights = memory.tensors()
        if z.shape != self.feature_rows.shape: raise ValueError('Replay shape mismatch.')
        self.feature_rows.assign(z); self.feature_labels.assign(labels); self.feature_weights.assign(weights)
        self._memory = memory

    def _batch_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
        import tensorflow as tf
        from basil_core.research_audit import augment_batch
        if semantics != 'paper_absolute_gaussian' or mode not in {'none', 'gradient_norm_objective'}:
            raise ValueError('Feature study requires CE/full EBM and absolute semantics.')
        with tf.device('/CPU:0'): images = augment_batch(x, key)
        return feature_replay_gradients(self.model, images, y, self.reference,
            self.feature_rows, self.feature_labels, self.feature_weights, mu=self.anchor_mu,
            sigma_e=sigma, ebm=mode == 'gradient_norm_objective', beta=self.beta)

    def train(self, params, x, y, *, feature_memory, **kwargs):
        from basil_core.research_protocol import params_hash
        if (feature_memory.receiver, feature_memory.round_id) != (kwargs['node_id'], kwargs['round_id']):
            raise ValueError('Memory does not belong to this activation.')
        self.load(params); self.set_memory(feature_memory)
        expected = feature_memory.to_dict()
        result, record = super().train(params, x, y, **kwargs)
        if feature_memory.to_dict() != expected: raise ValueError('Frozen feature memory changed.')
        record.update(featureMemory=expected, methodRevision=REVISION, featureReplayBeta=self.beta,
            featureMemoryFrozenForActivation=True, featureMemorySource='training_only_owner_refresh',
            selectedEncoderHash=feature_memory.encoder_hash, finalEncoderHash=encoder_hash(result),
            backboneChangedDuringActivation=encoder_hash(result) != feature_memory.encoder_hash,
            selectedStartHash=params_hash(params))
        for step in record['optimizerTelemetry']:
            step['totalObjective'] = step['ebmObjective']
            step['ebmObjective'] = step['baseCrossEntropy'] + step['ebmCoefficient']*step['gradientNormSquared']
        return result, record
