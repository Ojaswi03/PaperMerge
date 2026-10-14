"""Receiver-supported CART adaptation for strict BASIL; legacy cart is unchanged.

Historical registry records are claims, not certificates of current knowledge.
Only receiver-local training support controls the anchor. Test metrics never do.
The positive local-loss confidence avoids the zero-gap-at-handoff degeneracy.
This confidence-plus-gap rule is our adaptation, not a BASIL/source-paper rule.
"""
from __future__ import annotations

import math

from .class_registry import ClassRegistry
from .noniid_study import SnapshotAnchorWorker, anchored_gradients
from .research_audit import augment_batch

REVISION = 'strict_cart_receiver_supported_confidence_gap_v1'


def anchor_strength(mu_max, selected_loss, reference_accuracy, current_accuracy, support):
    """mu_e = mu_max * min(1, 1/(1+L_SS) + supported reference gap).

    Scalar mu is held fixed during an epoch; only local training probes update it.
    A missing class is unavailable, never a zero-accuracy observation.
    """
    import numpy as np
    if not math.isfinite(mu_max) or mu_max < 0 or not math.isfinite(selected_loss) or selected_loss < 0:
        raise ValueError('Finite nonnegative mu and receiver-local CE required.')
    ref, cur, counts = map(np.asarray, (reference_accuracy, current_accuracy, support))
    if ref.shape != (10,) or cur.shape != (10,) or counts.shape != (10,):
        raise ValueError('Ten-class support/accuracy vectors required.')
    mask = counts > 0
    if np.any(counts < 0) or not np.isfinite(ref[mask]).all() or not np.isfinite(cur[mask]).all():
        raise ValueError('Invalid locally supported class evidence.')
    if np.any((ref[mask] < 0) | (ref[mask] > 1) | (cur[mask] < 0) | (cur[mask] > 1)):
        raise ValueError('Supported accuracies must be in [0, 1].')
    confidence = 1. / (1. + selected_loss)
    gap = float(np.mean(np.maximum(0., ref[mask] - cur[mask]))) if mask.any() else 0.
    mu = mu_max * min(1., confidence + gap) if mask.any() else 0.
    return dict(mu=mu, confidence=confidence, supportedGap=gap,
                supportedClasses=np.flatnonzero(mask).tolist())


class StrictCartWorker(SnapshotAnchorWorker):
    """Frozen selected snapshot, full CE/second-order EBM, epoch-wise local control.

    Legacy lambda is never read. No clipping, averaging, optimizer transmission,
    fixed-step epochs, or class-importance/GPM modifications are introduced.
    """
    def __init__(self, model, *, mu_max=.01, device='CPU'):
        import tensorflow as tf
        if not math.isfinite(mu_max) or mu_max < 0:
            raise ValueError('Finite nonnegative CART cap required.')
        self.mu_max = float(mu_max)
        self.anchor_mu = tf.Variable(0., trainable=False, dtype=tf.float32)
        super().__init__(model, mu=0., device=device)

    def _batch_gradients(self, x, y, key, mode, semantics, sigma, legacy_lambda):
        import tensorflow as tf
        if semantics != 'paper_absolute_gaussian' or mode not in {'none', 'gradient_norm_objective'}:
            raise ValueError('Strict CART requires CE or full EBM and explicit absolute semantics.')
        with tf.device('/CPU:0'):
            images = augment_batch(x, key)
        return anchored_gradients(self.model, images, y, self.reference, mu=self.anchor_mu,
                                  sigma_e=sigma, ebm=mode == 'gradient_norm_objective')

    def _anchor_control(self, selected_loss, target, current, support):
        """Overridable policy; the historical default is unchanged."""
        return anchor_strength(self.mu_max, selected_loss, target, current, support)

    def train(self, params, x, y, *, local_registry, received_registry, selection_x,
              selection_y, selected_loss, **kwargs):
        # Exactly the received/selected parameters are the immutable reference.
        import numpy as np
        import tensorflow as tf
        self.load(params)
        support = np.bincount(np.asarray(selection_y, np.int32), minlength=10)
        def local_accuracy():
            logits = self.predict(tf.convert_to_tensor(selection_x))
            tf.debugging.assert_all_finite(logits, 'Non-finite CART training-side probe')
            predicted = tf.argmax(logits, axis=1).numpy()
            correct = np.bincount(np.asarray(selection_y)[predicted == selection_y], minlength=10)
            return np.divide(correct, support, out=np.full(10, np.nan), where=support > 0)
        reference_accuracy = local_accuracy()
        before_registry = local_registry.toDict()
        accepted, rejected = local_registry.verifyAndMerge(received_registry, reference_accuracy,
                                                           receivedSupport=support)
        # A historical best cannot claim more than this exact reference reproduces.
        target = np.minimum(local_registry.class_best_acc, reference_accuracy)
        local_registry.update(kwargs['node_id'], reference_accuracy, support=support,
                              roundId=kwargs['round_id'])
        target = np.maximum(target, np.where(support > 0, reference_accuracy, np.nan))
        controls = []
        external_evaluation = kwargs.pop('evaluation', None)
        def evaluate_and_control():
            current = local_accuracy()
            control = self._anchor_control(selected_loss, target, current, support)
            self.anchor_mu.assign(control['mu'])
            control.update(stage=len(controls), referenceAccuracy=[None if not np.isfinite(a) else float(a) for a in target],
                currentLocalAccuracy=[None if not np.isfinite(a) else float(a) for a in current])
            controls.append(control)
            # Evaluation-only callback cannot feed back into the coefficient above.
            return external_evaluation() if external_evaluation else {}
        result, record = super().train(params, x, y, evaluation=evaluate_and_control, **kwargs)
        final_local = local_accuracy()
        local_registry.update(kwargs['node_id'], final_local, support=support, roundId=kwargs['round_id'])
        carried = received_registry.clone()
        carried.merge(local_registry)
        record.update(cartRevision=REVISION, cartControl=controls,
            registryBefore=before_registry, registryAfter=local_registry.toDict(),
            receivedRegistry=received_registry.toDict(), verifiedClaimsAccepted=accepted,
            verifiedClaimsRejected=rejected, unverifiedLocalClasses=np.flatnonzero(support == 0).tolist(),
            registryTrainingSupport=support.tolist(), selectedReceiverLocalLoss=float(selected_loss),
            outputRegistry=carried.toDict())
        return result, record
