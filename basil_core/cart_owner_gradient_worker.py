"""Isolated clean CART diagnostic with fresh, class-balanced owner gradients.

This simulates owner-side computation in one process, not deployed RPC. Only
gradient DTOs cross the logical helper boundary; helpers never run optimizers.
The existing receiver-supported CART anchor is kept separately from the CE.
"""
from __future__ import annotations

from .strict_cart import StrictCartWorker


def available_owner_classes(round_id, node, *, diagnostic_rounds=1):
    if type(diagnostic_rounds) is not int or diagnostic_rounds not in (1, 3, 30):
        raise ValueError('Only the approved one-, three-, or thirty-round diagnostic is supported.')
    if type(round_id) is not int or not 0 <= round_id < diagnostic_rounds or type(node) is not int or not 0 <= node < 10:
        raise ValueError('Only the approved diagnostic horizon and already visited owners are allowed.')
    return tuple(range(node)) if round_id == 0 else tuple(c for c in range(10) if c != node)


class OwnerGradientCartWorker(StrictCartWorker):
    extra_metric_names = StrictCartWorker.extra_metric_names + (
        'balancedClassCrossEntropy', 'ownerResponseBytes', 'olderEncoderGradientNorm')

    def __init__(self, model, helper_model, *, device='CPU', diagnostic_rounds=1):
        import tensorflow as tf
        if type(diagnostic_rounds) is not int or diagnostic_rounds not in (1, 3, 30):
            raise ValueError('Only the approved one-, three-, or thirty-round diagnostic is supported.')
        self.diagnostic_rounds = diagnostic_rounds
        if model is helper_model:
            raise ValueError('Separate active and reusable helper models required.')
        self.helper_model = helper_model
        super().__init__(model, mu_max=.01, device=device)
        # Hash/context validation and owner simulation intentionally run outside
        # tf.function. Only the local CE kernel is compiled, once per batch shape.
        self._local_ce = tf.function(self._local_gradients, reduce_retracing=True)
        self._gradients = self._owner_gradients
        self.owner_records = []

    def _local_gradients(self, x, y, key):
        import tensorflow as tf
        from .research_audit import augment_batch
        with tf.device('/CPU:0'):
            images = augment_batch(x, key)
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            tf.debugging.assert_all_finite(logits, 'Non-finite local CE logits')
            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        gradients = tape.gradient(ce, self.model.trainable_variables)
        if any(g is None for g in gradients):
            raise ValueError('Disconnected local CE gradient.')
        tf.debugging.assert_all_finite(ce, 'Non-finite local CE')
        for g in gradients:
            tf.debugging.assert_all_finite(g, 'Non-finite local CE gradient')
        accuracy = tf.reduce_mean(tf.cast(tf.argmax(logits, axis=1, output_type=tf.int32) == y, tf.float32))
        return gradients, ce, accuracy

    def _owner_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
        import numpy as np
        import tensorflow as tf
        from .cart_owner_gradients import OwnerGradientResponse, compute_owner_response, aggregate_balanced_owner_gradients
        from .research_protocol import keyed_rng, params_hash
        from .research_audit import augment_batch
        if mode != 'none' or semantics != 'paper_absolute_gaussian' or float(sigma) != 0.:
            raise ValueError('Owner-gradient diagnostic is clean CE only: no noise or EBM.')
        params = self.export()
        digest = params_hash(params)
        step = len(self.owner_records)
        epoch, batch = divmod(step, 10)
        local, ce, accuracy = self._local_ce(x, y, key)
        responses = [OwnerGradientResponse(self.receiver, self.receiver, self.round_id,
            step, digest, int(y.shape[0]), float(ce), tuple(g.numpy() for g in local))]
        # One helper serves all requested owners serially at this exact step.
        for w, p in zip(self.helper_model.trainable_variables, params): w.assign(p)
        record = dict(round=self.round_id, receiver=self.receiver, optimizerStep=step,
            epoch=epoch+1, batch=batch, parameterHash=digest, owners=[])
        if self.audit: self.audit.current['ownerGradientContext'] = record
        for owner in self.older_classes:
            rng = keyed_rng(self.seed, 'cart_owner_gradient_samples', self.round_id,
                self.receiver, epoch, batch, owner)
            ids = rng.choice(self.owner_pools[owner], size=16, replace=False)
            aug = keyed_rng(self.seed, 'cart_owner_gradient_augmentation', self.round_id,
                self.receiver, epoch, batch, owner).integers(0, 2**30, size=2, dtype=np.int32)
            with tf.device('/CPU:0'):
                images = augment_batch(self.source_x[ids], aug)
            record['pendingOwner'] = dict(owner=owner, trainingIndices=ids.tolist(),
                augmentationKey=aug.tolist())
            # Raw helper images are not returned to the active trainer. This
            # call is the explicitly simulated, owner-local gradient service.
            response = compute_owner_response(self.helper_model, params, images,
                self.source_y[ids], ids, owner=owner, receiver=self.receiver,
                round_id=self.round_id, optimizer_step=step)
            responses.append(response)
            record['owners'].append(dict(owner=owner, trainingIndices=ids.tolist(),
                augmentationKey=aug.tolist(), crossEntropy=response.cross_entropy,
                parameterHash=response.model_hash, support=response.support))
            record.pop('pendingOwner')
        balanced, metadata = aggregate_balanced_owner_gradients(params, responses,
            receiver=self.receiver, round_id=self.round_id, optimizer_step=step,
            available_classes=(*self.older_classes, self.receiver))
        base_norm = tf.linalg.global_norm(local)
        drift = [w-tf.stop_gradient(a) for w, a in zip(self.model.trainable_variables, self.reference)]
        anchor = [self.anchor_mu*d for d in drift]
        applied = [tf.convert_to_tensor(g)+a for g, a in zip(balanced, anchor)]
        distance = tf.add_n([tf.reduce_sum(d*d) for d in drift])
        penalty = self.anchor_mu*distance/2
        anchor_norm = tf.linalg.global_norm(anchor)
        old_norm = float(np.mean([np.sqrt(sum(np.sum(g.astype(np.float64)**2)
            for g in r.gradients[:-2])) for r in responses[1:]])) if len(responses)>1 else 0.
        values = (ce, base_norm**2, tf.constant(metadata['crossEntropy'], tf.float32)+penalty,
            tf.constant(0.), base_norm, tf.linalg.global_norm(applied), tf.constant(0.), accuracy,
            distance, penalty, anchor_norm, self.anchor_mu.read_value(), anchor_norm/(base_norm+1e-12),
            tf.constant(metadata['crossEntropy'], tf.float32),
            tf.constant(metadata['responseCoordinateBytes'], tf.float64), tf.constant(old_norm, tf.float64))
        for value in (*applied, *values):
            tf.debugging.assert_all_finite(value, 'Non-finite balanced owner/CART objective')
        record.update(metadata)
        self.owner_records.append(record)
        return applied, values

    def train(self, params, x, y, *, source_x, source_y, owner_pools, older_classes, **kwargs):
        import numpy as np
        older = tuple(sorted(older_classes))
        node = kwargs['node_id']
        round_id = kwargs['round_id']
        expected = available_owner_classes(round_id,node,diagnostic_rounds=self.diagnostic_rounds)
        if older != expected:
            raise ValueError('Only the approved diagnostic horizon and already visited owners are allowed.')
        if len(y) != 5000 or kwargs['batch_size'] != 512 or not np.all(np.asarray(y) == node):
            raise ValueError('Full one-class 5000-sample activation with batch 512 required.')
        if len(source_y) != 50000 or len(owner_pools) != 10:
            raise ValueError('Complete owner-local training provenance required.')
        for owner in older:
            ids = np.asarray(owner_pools[owner])
            if len(ids)<16 or len(np.unique(ids))!=len(ids) or np.any((ids<0)|(ids>=50000)) or not np.all(source_y[ids]==owner):
                raise ValueError('Invalid owner-local training pool.')
        self.source_x, self.source_y, self.owner_pools = source_x, source_y, owner_pools
        self.older_classes, self.receiver = older, node
        self.seed, self.round_id = kwargs['seed'], kwargs['round_id']
        self.owner_records = []
        result, record = super().train(params, x, y, **kwargs)
        record.update(ownerGradients=self.owner_records, olderClasses=list(older),
            balancedObjective='mean of current-class CE and fresh older-owner CEs plus CART anchor',
            helperOptimizerSteps=0, rawImagesTransferred=False,
            responseCoordinateBytes=sum(r['responseCoordinateBytes'] for r in self.owner_records))
        return result, record
