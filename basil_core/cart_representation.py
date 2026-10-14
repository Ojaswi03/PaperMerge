"""Opt-in selected-encoder preservation and analysis-only same-image audits.

Feature/logit distillation has established precedent (LwF, related feature
retention work). This selected-snapshot feature-MSE integration is our
adaptation, not a reproduction, novelty claim or guarantee of class retention.
No raw-image sharing, frozen student, averaging or baseline changes.
"""
from __future__ import annotations

import math

from basil_core.cart_feature_memory import FeatureReplayWorker, encoder_hash

REVISION = 'strict_cart_selected_encoder_preservation_v1'


class FrozenSelectedEncoder:
    """One reusable detached teacher; synchronized only before an activation."""
    def __init__(self, model):
        import tensorflow as tf
        network = getattr(model, 'model', model)
        self.student = tf.keras.Model(network.inputs, network.layers[-2].output)
        self.teacher = tf.keras.models.clone_model(self.student)
        self.teacher.trainable = False
        self.source_hash = None

    def freeze(self, params):
        import numpy as np
        if len(params) != len(self.teacher.weights) + 2:
            raise ValueError('Encoder/head split differs from selected parameters.')
        if any(np.asarray(p).shape != tuple(w.shape) for p, w in zip(params[:-2], self.teacher.weights)):
            raise ValueError('Teacher parameter shape mismatch.')
        if not all(np.isfinite(p).all() for p in params):
            raise ValueError('Non-finite selected teacher state.')
        self.teacher.set_weights(params[:-2])
        self.source_hash = encoder_hash(params)
        self.assert_frozen()

    def assert_frozen(self):
        from basil_core.research_protocol import params_hash
        if self.source_hash is None:
            raise ValueError('Selected teacher was not synchronized for this activation.')
        if self.teacher.trainable_variables or params_hash(self.teacher.get_weights()) != self.source_hash:
            raise ValueError('Frozen selected encoder changed.')


def preservation_gradients(model, selected_encoder, x, *, gamma):
    """gamma * mean ||h_current(x)-stop_gradient(h_selected(x))||².

    Student and teacher see exactly the same already-augmented local batch.
    The final head is absent from this term, so its zero contribution is
    expected. An unexpected disconnected encoder variable is an error.
    """
    import tensorflow as tf
    gamma = tf.cast(gamma, tf.float32)
    tf.debugging.assert_all_finite(gamma, 'Non-finite preservation gamma')
    tf.debugging.assert_greater_equal(gamma, 0.)
    target = tf.stop_gradient(selected_encoder.teacher(x, training=False))
    tf.debugging.assert_all_finite(target, 'Non-finite selected-encoder target')
    with tf.GradientTape() as tape:
        current = selected_encoder.student(x, training=True)
        tf.debugging.assert_all_finite(current, 'Non-finite current-encoder features')
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(current-target), axis=1))
        penalty = gamma*loss
    encoder_variables = selected_encoder.student.trainable_variables
    encoder_grads = tape.gradient(penalty, encoder_variables)
    if any(g is None for g in encoder_grads):
        raise ValueError('Unexpected disconnected preservation encoder gradient.')
    mapping = {id(w): g for w, g in zip(encoder_variables, encoder_grads)}
    grads = [mapping.get(id(w), tf.zeros_like(w)) for w in model.trainable_variables]
    for value in (loss, penalty, *grads):
        tf.debugging.assert_all_finite(value, 'Non-finite representation objective/gradient')
    return grads, (loss, penalty, tf.linalg.global_norm(grads), gamma)


def representation_replay_gradients(model, selected_encoder, x, y, reference,
                                   features, feature_labels, row_weights,
                                   *, mu, sigma_e, ebm, beta, gamma):
    import tensorflow as tf
    from basil_core.cart_feature_memory import feature_replay_gradients
    source, values = feature_replay_gradients(model, x, y, reference, features,
        feature_labels, row_weights, mu=mu, sigma_e=sigma_e, ebm=ebm, beta=beta)
    preserve, extra = preservation_gradients(model, selected_encoder, x, gamma=gamma)
    applied = [a+b for a, b in zip(source, preserve)]
    result = list(values)
    result[2] += extra[1]
    result[5] = tf.linalg.global_norm(applied)
    result.extend((*extra, extra[2]/(tf.cast(values[4], tf.float32)+1e-12)))
    for value in (*result, *applied):
        tf.debugging.assert_all_finite(value, 'Non-finite full representation-replay objective')
    return applied, tuple(result)


class RepresentationReplayWorker(FeatureReplayWorker):
    extra_metric_names = FeatureReplayWorker.extra_metric_names + (
        'representationLoss', 'representationPenalty', 'representationGradientNorm',
        'representationGamma', 'representationCorrectionRatio')

    def __init__(self, model, *, beta, gamma, mu_max=.01, device='CPU'):
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError('Explicit finite nonnegative gamma required.')
        self.gamma = float(gamma)
        self.selected_encoder = FrozenSelectedEncoder(model)
        super().__init__(model, beta=beta, mu_max=mu_max, device=device)

    def _batch_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
        import tensorflow as tf
        from basil_core.research_audit import augment_batch
        if semantics != 'paper_absolute_gaussian' or mode not in {'none', 'gradient_norm_objective'}:
            raise ValueError('Representation study requires CE/full source EBM and absolute semantics.')
        with tf.device('/CPU:0'):
            images = augment_batch(x, key)
        return representation_replay_gradients(self.model, self.selected_encoder,
            images, y, self.reference, self.feature_rows, self.feature_labels,
            self.feature_weights, mu=self.anchor_mu, sigma_e=sigma,
            ebm=mode == 'gradient_norm_objective', beta=self.beta, gamma=self.gamma)

    def train(self, params, x, y, **kwargs):
        self.selected_encoder.freeze(params)
        # Preserve the teacher even when an external observation callback fails.
        try:
            result, record = super().train(params, x, y, **kwargs)
        except Exception:
            if self.audit is not None:
                # Retain the extra state needed to reproduce this objective,
                # alongside the inherited failing model/data/batch checkpoint.
                import numpy as np
                np.savez_compressed(self.audit.directory/'representation_failure_state.npz',
                    **{f'teacher_{j}': w.numpy() for j, w in enumerate(self.selected_encoder.teacher.weights)},
                    **{f'anchor_{j}': w.numpy() for j, w in enumerate(self.reference)},
                    feature_rows=self.feature_rows.numpy(), feature_labels=self.feature_labels.numpy(),
                    feature_weights=self.feature_weights.numpy(), gamma=self.gamma, beta=self.beta,
                    anchor_mu=float(self.anchor_mu.numpy()))
            raise
        finally:
            self.selected_encoder.assert_frozen()
        record.update(representationRevision=REVISION, representationGamma=self.gamma,
            selectedTeacherEncoderHash=self.selected_encoder.source_hash,
            selectedTeacherFrozenForActivation=True, representationTrainingData='receiver_local_only',
            representationObjective='mean_sample_sum_coordinate_squared_feature_drift')
        return result, record


def same_image_audit(model, selected_encoder, x, labels, indices, selected_params, *, batch_size=128):
    """Factor head/encoder effects on IDENTICAL training images, observation only.

    No assignment to the live model, no gradients, no coefficient feedback.
    Counterfactuals characterize this state; they are not trained alternatives
    or proof that encoder drift is the only cause of collapse.
    """
    import numpy as np
    import tensorflow as tf
    from basil_core.forgetting_audit import classification_metrics, representation_metrics
    from basil_core.research_protocol import params_hash
    from basil_core.research_audit import array_hash
    y, ids = np.asarray(labels), np.asarray(indices)
    if len(y) != len(ids) or len(np.unique(ids)) != len(ids) or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= 50000)):
        raise ValueError('Distinct matching training image identities required.')
    if not np.array_equal(np.bincount(y, minlength=10), [len(y)//10]*10) or len(y) == 0 or len(y)%10:
        raise ValueError('Balanced ten-class training-only audit required.')
    selected_encoder.assert_frozen()
    if encoder_hash(selected_params) != selected_encoder.source_hash:
        raise ValueError('Selected audit reference differs from frozen teacher.')
    current_params = [w.numpy() for w in model.trainable_variables]
    before = params_hash(current_params)
    current, target = [], []
    for start in range(0, len(y), batch_size):
        batch = tf.convert_to_tensor(x[start:start+batch_size])
        a = selected_encoder.student(batch, training=False)
        b = selected_encoder.teacher(batch, training=False)
        tf.debugging.assert_all_finite(a, 'Non-finite same-image current features')
        tf.debugging.assert_all_finite(b, 'Non-finite same-image reference features')
        current.append(a.numpy()); target.append(b.numpy())
    current, target = np.concatenate(current), np.concatenate(target)
    network = getattr(model, 'model', model)
    # Approved separate Dense linear head; four factorized observations.
    if len(network.layers[-1].weights) != 2:
        raise ValueError('Expected final dense kernel/bias for head audit.')
    logits = {
        'current_encoder_current_head': current@current_params[-2]+current_params[-1],
        'selected_encoder_current_head': target@current_params[-2]+current_params[-1],
        'current_encoder_selected_head': current@selected_params[-2]+selected_params[-1],
        'selected_encoder_selected_head': target@selected_params[-2]+selected_params[-1],
    }
    per_class = []
    for c in range(10):
        a, b = current[y == c].astype(np.float64), target[y == c].astype(np.float64)
        drift_sq = np.sum((a-b)**2, axis=1)
        reference_sq = np.sum(b*b, axis=1)
        denominator = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
        valid = denominator > 0
        per_class.append(dict(classId=c, support=int((y == c).sum()),
            meanSquaredFeatureDrift=float(drift_sq.mean()),
            relativeRmsFeatureDrift=float(np.sqrt(drift_sq.mean()/max(reference_sq.mean(), 1e-24))),
            meanCosine=float((np.sum(a[valid]*b[valid], axis=1)/denominator[valid]).mean()) if valid.any() else None,
            zeroNormPairs=int((~valid).sum())))
    selected_encoder.assert_frozen()
    if params_hash([w.numpy() for w in model.trainable_variables]) != before:
        raise ValueError('Same-image analysis mutated training parameters.')
    return dict(trainingOnly=True, heldOut=False, analysisOnly=True,
        imageIndexHash=array_hash(np.asarray(ids, dtype=np.int64)),
        imageHash=array_hash(np.asarray(x)), labelsHash=array_hash(y),
        currentParameterHash=before, selectedParameterHash=params_hash(selected_params),
        currentEncoderHash=encoder_hash(current_params), selectedEncoderHash=selected_encoder.source_hash,
        representation=representation_metrics(current, target, y), perClassDrift=per_class,
        interventions={key: classification_metrics(value, y) for key, value in logits.items()})


def calibrate_gamma(model, selected_encoder, x, y, *, learning_rate=.05, target_ratio=.1):
    """Training-only, one-time scale rule; NEVER a test-score search.

    At handoff preservation gradient is zero. Measure after one virtual CE
    update, compute gamma so its norm is target_ratio * CE norm at that state,
    then restore all parameters. No optimizer is run, no state is transmitted.
    Returned gamma remains fixed in every activation of the diagnostic.
    """
    import numpy as np
    import tensorflow as tf
    from basil_core.trainer import lossFn
    from basil_core.research_protocol import params_hash
    from basil_core.research_audit import array_hash
    if not math.isfinite(target_ratio) or not 0 < target_ratio <= 1 or learning_rate != .05:
        raise ValueError('Explicit calibration ratio in (0,1] and approved LR .05 required.')
    initial = [w.numpy().copy() for w in model.trainable_variables]
    expected = params_hash(initial)
    if encoder_hash(initial) != selected_encoder.source_hash:
        raise ValueError('Calibration must begin at exact selected encoder.')
    def ce_grads():
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            tf.debugging.assert_all_finite(logits, 'Non-finite gamma-calibration logits')
            ce = lossFn(y, logits)
        grads = tape.gradient(ce, model.trainable_variables)
        if any(g is None for g in grads):
            raise ValueError('Disconnected calibration CE gradient.')
        for g in grads: tf.debugging.assert_all_finite(g, 'Non-finite calibration CE gradient')
        return grads, ce
    try:
        base, _ = ce_grads()
        for w, p, g in zip(model.trainable_variables, initial, base):
            w.assign(p-learning_rate*g)
        base, ce = ce_grads()
        preserve, values = preservation_gradients(model, selected_encoder, x, gamma=1.)
        base_norm, unit_norm = float(tf.linalg.global_norm(base)), float(values[2])
        if not all(math.isfinite(v) and v > 1e-12 for v in (base_norm, unit_norm)):
            raise ValueError('Unavailable calibration: zero/non-finite CE or preservation norm.')
        gamma = target_ratio*base_norm/unit_norm
        result = dict(gamma=gamma, targetRatio=target_ratio, baseGradientNorm=base_norm,
            unitPreservationGradientNorm=unit_norm, virtualCrossEntropy=float(ce),
            virtualFeatureDrift=float(values[0]), trainingOnly=True, testSetUsed=False,
            parameterHash=expected, inputHash=array_hash(np.asarray(x)),
            labelHash=array_hash(np.asarray(y)), virtualLearningRate=learning_rate,
            calibrationScope='once on canonical Node-0 training batch; fixed thereafter')
    finally:
        for w, p in zip(model.trainable_variables, initial): w.assign(p)
        selected_encoder.assert_frozen()
    if params_hash([w.numpy() for w in model.trainable_variables]) != expected:
        raise ValueError('Virtual calibration failed to restore initial model.')
    return result
