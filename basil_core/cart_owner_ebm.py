"""Exact local-CE EBM within class-balanced owner-gradient CART; opt-in only.

L = mean_c CE_c + sigma_e² ||grad CE_receiver||² + mu/2 ||theta-reference||².
EBM is NOT applied to the class mean or anchor. No legacy lambda is used.
Owner transport/selection/Byzantine safety are separate protocol contracts.
"""
from __future__ import annotations

import math


def local_ebm_terms(model, images, labels, sigma_e, *, enabled):
    """Return CE gradients and the actual Hessian-gradient correction separately."""
    import tensorflow as tf
    from .trainer import lossFn
    coefficient = tf.stop_gradient(tf.cast(sigma_e, tf.float32)**2) if enabled else tf.constant(0.)
    weights = model.trainable_variables
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            logits = model(images, training=True)
            tf.debugging.assert_all_finite(logits, 'Non-finite owner EBM logits')
            ce = lossFn(labels, logits)
        base = inner.gradient(ce, weights)
        if any(g is None for g in base):
            raise ValueError('Unexpected disconnected receiver CE gradient.')
        norm_sq = tf.add_n([tf.reduce_sum(g*g) for g in base])
        penalty = coefficient*norm_sq
    correction = outer.gradient(penalty, weights) if enabled else [tf.zeros_like(g) for g in base]
    if any(g is None for g in correction):
        raise ValueError('Unexpected disconnected EBM correction.')
    accuracy = tf.reduce_mean(tf.cast(tf.argmax(logits, axis=1, output_type=tf.int32) ==
                                     tf.cast(labels, tf.int32), tf.float32))
    for value in (*base, *correction, ce, norm_sq, penalty):
        tf.debugging.assert_all_finite(value, 'Non-finite local EBM objective/gradient')
    return base, correction, (ce, norm_sq, penalty, coefficient, accuracy)


def owner_objective_gradients(model, images, labels, reference, helper_responses, *,
                              receiver, round_id, optimizer_step, mu, sigma_e, ebm,
                              legacy_lambda=None):
    """Use fresh full-network helper gradients, exact EBM and a detached anchor.

    legacy_lambda is deliberately unused, including in gradients and metrics.
    Scalar coefficients must be finite; no clipping or optimizer is hidden here.
    The caller must enforce the declared helper threat boundary before calling.
    """
    import numpy as np
    import tensorflow as tf
    from .cart_owner_gradients import OwnerGradientResponse, aggregate_balanced_owner_gradients
    from .research_protocol import params_hash
    if not math.isfinite(float(mu)) or float(mu) < 0 or not math.isfinite(float(sigma_e)) or float(sigma_e) < 0:
        raise ValueError('Finite nonnegative anchor mu and coordinate sigma required.')
    if type(ebm) is not bool:
        raise ValueError('Explicit EBM boolean required.')
    weights = model.trainable_variables
    if len(reference) != len(weights) or any(tuple(a.shape) != tuple(w.shape) for a, w in zip(reference, weights)):
        raise ValueError('Frozen anchor tensor layout differs.')
    if not len(labels) or not np.all(np.asarray(labels) == receiver):
        raise ValueError('Receiver-local one-class CE batch required.')
    params = [w.numpy().astype(np.float32) for w in weights]
    base, correction, terms = local_ebm_terms(model, images, labels, sigma_e, enabled=ebm)
    ce, norm_sq, penalty, coefficient, accuracy = terms
    local = OwnerGradientResponse(receiver, receiver, round_id, optimizer_step, params_hash(params),
        len(labels), float(ce), tuple(g.numpy() for g in base))
    helpers = tuple(helper_responses)
    balanced, metadata = aggregate_balanced_owner_gradients(params, (*helpers, local),
        receiver=receiver, round_id=round_id, optimizer_step=optimizer_step,
        available_classes=tuple(r.owner for r in helpers)+(receiver,))
    drift = [w-tf.stop_gradient(tf.cast(a, w.dtype)) for w, a in zip(weights, reference)]
    distance = tf.add_n([tf.reduce_sum(d*d) for d in drift])
    anchor = [tf.cast(mu, d.dtype)*d for d in drift]
    anchor_penalty = tf.cast(mu, tf.float32)*distance/2
    applied = [tf.convert_to_tensor(g)+c+a for g, c, a in zip(balanced, correction, anchor)]
    objective = tf.cast(metadata['crossEntropy'], tf.float32)+penalty+anchor_penalty
    base_norm = tf.linalg.global_norm(base)
    correction_norm = tf.linalg.global_norm(correction)
    anchor_norm = tf.linalg.global_norm(anchor)
    values = (ce, norm_sq, objective, coefficient, base_norm, tf.linalg.global_norm(applied),
              correction_norm, accuracy, distance, anchor_penalty, anchor_norm, tf.cast(mu, tf.float32),
              anchor_norm/(base_norm+1e-12), tf.cast(metadata['crossEntropy'], tf.float32),
              tf.cast(metadata['responseCoordinateBytes'], tf.float64),
              tf.constant(float(np.mean([np.sqrt(sum(np.sum(g.astype(np.float64)**2)
                  for g in r.gradients[:-2])) for r in helpers])) if helpers else 0., tf.float64))
    for value in (*applied, *values):
        tf.debugging.assert_all_finite(value, 'Non-finite owner EBM/anchor update')
    metadata.update(ebmInnerObjective='receiver_local_ce', ebmCoefficient=float(coefficient),
        ebmPenalty=float(penalty), ebmCorrectionNorm=float(correction_norm),
        ebmCorrectionRatio=float(correction_norm/(base_norm+1e-12)), legacyLambdaIgnored=True)
    return applied, values, metadata


def make_owner_ebm_worker(model, helper_model, *, helper_scope, attacker_ids=(),
                         diagnostic_rounds=3, device='CPU'):
    """Lazy factory for the existing five-epoch CART worker with the new objective.

    New selection and snapshot transport must be used by an explicitly approved
    future runner. This factory never starts a runner or authorizes a horizon.
    """
    from .cart_owner_protocol import require_helper_boundary
    require_helper_boundary(helper_scope, attacker_ids)
    from .cart_owner_gradient_worker import OwnerGradientCartWorker

    class OwnerEbmWorker(OwnerGradientCartWorker):
        def _owner_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
            import numpy as np
            import tensorflow as tf
            from .cart_owner_gradients import compute_owner_response
            from .research_protocol import keyed_rng, params_hash
            from .research_audit import augment_batch
            if mode not in ('none', 'gradient_norm_objective') or semantics != 'paper_absolute_gaussian':
                raise ValueError('Explicit absolute channel and CE/full local EBM only.')
            params = self.export()
            digest = params_hash(params)
            step = len(self.owner_records)
            epoch, batch = divmod(step, 10)
            record = dict(round=self.round_id, receiver=self.receiver, optimizerStep=step,
                          epoch=epoch+1, batch=batch, parameterHash=digest, owners=[])
            if self.audit:
                self.audit.current['ownerGradientContext'] = record
            for w, p in zip(self.helper_model.trainable_variables, params):
                w.assign(p)
            responses = []
            for owner in self.older_classes:
                ids = keyed_rng(self.seed, 'cart_owner_gradient_samples', self.round_id,
                    self.receiver, epoch, batch, owner).choice(self.owner_pools[owner], size=16, replace=False)
                aug = keyed_rng(self.seed, 'cart_owner_gradient_augmentation', self.round_id,
                    self.receiver, epoch, batch, owner).integers(0, 2**30, size=2, dtype=np.int32)
                record['pendingOwner'] = dict(owner=owner, trainingIndices=ids.tolist(), augmentationKey=aug.tolist())
                with tf.device('/CPU:0'):
                    images = augment_batch(self.source_x[ids], aug)
                response = compute_owner_response(self.helper_model, params, images,
                    self.source_y[ids], ids, owner=owner, receiver=self.receiver,
                    round_id=self.round_id, optimizer_step=step)
                responses.append(response)
                record['owners'].append(dict(owner=owner, trainingIndices=ids.tolist(),
                    augmentationKey=aug.tolist(), crossEntropy=response.cross_entropy,
                    parameterHash=response.model_hash, support=response.support))
                record.pop('pendingOwner')
            with tf.device('/CPU:0'):
                images = augment_batch(x, key)
            gradients, values, metadata = owner_objective_gradients(self.model, images, y,
                self.reference, responses, receiver=self.receiver, round_id=self.round_id,
                optimizer_step=step, mu=self.anchor_mu, sigma_e=sigma,
                ebm=mode == 'gradient_norm_objective')
            record.update(metadata)
            self.owner_records.append(record)
            return gradients, values

    return OwnerEbmWorker(model, helper_model, device=device, diagnostic_rounds=diagnostic_rounds)
