"""Unwired clean CART candidate: balanced, owner-local supervised gradients.

This changes information exchange, not just a coefficient. Helpers compute
gradients at the active trainer's EXACT current parameters; only that trainer
applies SGD. No raw images or model averaging are part of a response. This is
not plain BASIL, a privacy guarantee, or Byzantine-safe gradient aggregation.
There is deliberately no experiment runner or automatic baseline/queue wiring.
Heavy dependencies are imported only when called.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

REVISION = 'cart_balanced_owner_gradient_candidate_v1'


def _node(value):
    if type(value) is not int or not 0 <= value < 10:
        raise ValueError('One-class owner IDs must be integers 0..9.')


def feature_supervision_prior(receiver, older_classes, *, beta):
    """The existing grouping's bias-only stationary prior; not a CNN prediction.

    Local CE has mass 1, ALL older classes together have mass beta, and each
    older class has mass beta/K. This diagnosis assumes constant logits and
    makes no claim that weighting alone explains a nonlinear CNN failure.
    """
    _node(receiver)
    older = tuple(older_classes)
    for c in older: _node(c)
    if len(set(older)) != len(older) or receiver in older:
        raise ValueError('Distinct older classes excluding the receiver required.')
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('Finite nonnegative replay beta required.')
    masses = {receiver: 1.}
    if older: masses.update({c: beta/len(older) for c in older})
    total = sum(masses.values())
    return {c: value/total for c, value in sorted(masses.items())}


@dataclass(frozen=True)
class OwnerGradientResponse:
    owner: int
    receiver: int
    round_id: int
    optimizer_step: int
    model_hash: str
    support: int
    cross_entropy: float
    gradients: tuple
    data_usage: str = 'training_only'

    def __post_init__(self):
        import numpy as np
        _node(self.owner); _node(self.receiver)
        if any(type(v) is not int or v < 0 for v in (self.round_id, self.optimizer_step)):
            raise ValueError('Nonnegative round/step identities required.')
        if not isinstance(self.model_hash, str) or re.fullmatch('[0-9a-f]{64}', self.model_hash) is None:
            raise ValueError('Exact current parameter hash required.')
        if type(self.support) is not int or not 1 <= self.support <= 512:
            raise ValueError('Bounded positive owner training support required.')
        if self.data_usage != 'training_only':
            raise ValueError('Test/validation data cannot supply owner gradients.')
        if not math.isfinite(self.cross_entropy) or self.cross_entropy < 0:
            raise ValueError('Finite nonnegative owner CE required.')
        if type(self.gradients) is not tuple or not self.gradients:
            raise ValueError('Complete gradient tensor sequence required.')
        immutable = []
        for gradient in self.gradients:
            array = np.asarray(gradient)
            if array.dtype != np.float32 or not array.size or not np.isfinite(array).all():
                raise ValueError('Finite float32 owner gradients required.')
            # Bytes backing makes the copy immutable, not just write-protected
            # on a mutable array that another holder can re-enable writing to.
            immutable.append(np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape))
        object.__setattr__(self, 'gradients', tuple(immutable))


def compute_owner_response(model, params, images, labels, training_indices, *,
                           owner, receiver, round_id, optimizer_step,
                           data_usage='training_only'):
    """Owner-side CE backprop only: never apply SGD or return images.

    A future engine must execute this at the class owner. Training provenance
    is a caller contract plus checked indices/labels, not cryptographic proof.
    No helper model is created here; the caller must reuse a synchronized model.
    """
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import params_hash
    _node(owner); _node(receiver)
    if data_usage != 'training_only':
        raise ValueError('Test/validation data cannot supply owner gradients.')
    ids, y = np.asarray(training_indices), np.asarray(labels)
    if (not np.issubdtype(ids.dtype, np.integer) or ids.ndim != 1 or
        not 1 <= len(ids) <= 512 or len(np.unique(ids)) != len(ids) or
        np.any((ids < 0) | (ids >= 50000)) or y.shape != ids.shape or
        not np.issubdtype(y.dtype, np.integer) or np.any(y != owner) or
        len(images) != len(ids)):
        raise ValueError('Distinct owner-local training indices and one-class labels required.')
    before = params_hash([w.numpy() for w in model.trainable_variables])
    network = getattr(model, 'model', model)
    if network.non_trainable_variables:
        raise ValueError('Stateful helper layers are outside this stateless owner-gradient candidate.')
    if before != params_hash(params):
        raise ValueError('Owner helper is not synchronized to the exact current model.')
    with tf.GradientTape() as tape:
        logits = model(tf.convert_to_tensor(images), training=True)
        tf.debugging.assert_all_finite(logits, 'Non-finite owner CE logits')
        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.convert_to_tensor(y, tf.int32), logits=logits))
    gradients = tape.gradient(ce, model.trainable_variables)
    if any(g is None for g in gradients):
        raise ValueError('Unexpected disconnected owner supervised gradient.')
    tf.debugging.assert_all_finite(ce, 'Non-finite owner CE')
    for gradient in gradients:
        tf.debugging.assert_all_finite(gradient, 'Non-finite owner supervised gradient')
    if params_hash([w.numpy() for w in model.trainable_variables]) != before:
        raise ValueError('Owner helper changed parameters; only the active trainer may update.')
    return OwnerGradientResponse(owner, receiver, round_id, optimizer_step,
        before, len(ids), float(ce), tuple(g.numpy() for g in gradients))


def aggregate_balanced_owner_gradients(params, responses, *, receiver, round_id,
                                     optimizer_step, available_classes):
    """Gradient of mean_c CE_c at ONE parameter state, not model averaging.

    The local class must be present. Missing/future classes are never invented.
    All tensors, source identities, state hashes and contexts must match. This
    candidate has no EBM, lambda, clipping, anchoring or implicit optimizer.
    """
    import numpy as np
    from basil_core.research_protocol import params_hash
    _node(receiver)
    classes = tuple(available_classes)
    for c in classes: _node(c)
    if len(set(classes)) != len(classes) or receiver not in classes:
        raise ValueError('Distinct available classes including the local class required.')
    if any(type(v) is not int or v < 0 for v in (round_id, optimizer_step)):
        raise ValueError('Nonnegative round/step identities required.')
    responses = tuple(responses)
    if any(not isinstance(r, OwnerGradientResponse) for r in responses):
        raise ValueError('Validated owner gradient responses required.')
    if len(responses) != len(classes) or {r.owner for r in responses} != set(classes):
        raise ValueError('Missing, duplicate or unrequested owner responses.')
    current = [np.asarray(p) for p in params]
    if not current or any(p.dtype != np.float32 or not p.size or not np.isfinite(p).all() for p in current):
        raise ValueError('Finite float32 current parameters required.')
    model_hash = params_hash(current)
    for response in responses:
        if (response.receiver, response.round_id, response.optimizer_step, response.model_hash) != (
            receiver, round_id, optimizer_step, model_hash):
            raise ValueError('Stale or wrong-context owner gradient response.')
        if len(response.gradients) != len(current) or any(g.shape != p.shape for g, p in zip(response.gradients, current)):
            raise ValueError('Owner gradient tensor layout differs from current parameters.')
    ordered = sorted(responses, key=lambda r: r.owner)
    gradients = tuple((sum((r.gradients[j].astype(np.float64) for r in ordered),
        np.zeros_like(p, dtype=np.float64))/len(ordered)).astype(np.float32) for j, p in enumerate(current))
    if any(not np.isfinite(g).all() for g in gradients):
        raise ValueError('Non-finite balanced owner aggregate.')
    return gradients, dict(revision=REVISION, modelHash=model_hash,
        availableClasses=sorted(classes), classWeights={str(c): 1./len(classes) for c in sorted(classes)},
        crossEntropy=sum(r.cross_entropy for r in ordered)/len(ordered),
        responseCoordinateBytes=sum(g.nbytes for r in ordered if r.owner != receiver for g in r.gradients),
        extraOwnerResponses=len(ordered)-1, ownerOptimizerSteps=0,
        rawImagesReturned=False, modelAveraging=False, cleanCandidateOnly=True)
