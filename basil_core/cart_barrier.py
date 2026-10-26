"""Separate class-loss barrier candidate; no change to the v1 KL objective.

The registry supplies locally measured class CE ceilings. Penalizing current
violations avoids multiplying two small historical prediction discrepancies.
This is a proposed constrained-retention adaptation, not a novelty guarantee.
"""
from __future__ import annotations

from basil_core.cart_local_memory import LocalMemoryCartWorker, MAX_PROBES
from basil_core.dirichlet_calibration import DirichletCalibrationWorker

REVISION='cart_local_class_loss_barrier_v1'
SLACK=.001


def barrier_gradients(model,x,y,reference,offsets,probe_x,probe_labels,class_average,
                      best_losses,support,*,beta=1.,mu=.01,coefficient=0.,correction_ratio_cap=None):
    import tensorflow as tf
    from basil_core.dirichlet_calibration import calibrated_anchor_gradients
    base,values=calibrated_anchor_gradients(model,x,y,reference,offsets,mu=mu,coefficient=coefficient)
    average=tf.stop_gradient(tf.cast(class_average,tf.float32))
    best=tf.stop_gradient(tf.cast(best_losses,tf.float32))
    mask=tf.stop_gradient(tf.cast(support,tf.float32))
    tf.debugging.assert_non_negative(best)
    tf.debugging.assert_non_negative(average)
    tf.debugging.assert_all_finite(best,'Non-finite CART barrier targets')
    if beta not in (0.,1.):raise ValueError('This candidate predeclares beta zero or one.')
    if beta:
        with tf.GradientTape() as tape:
            ce=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=probe_labels,logits=model(probe_x,training=False))
            losses=tf.linalg.matvec(average,ce)
            violation=mask*tf.nn.relu(losses-best-SLACK)
            count=tf.maximum(tf.reduce_sum(mask),1.)
            barrier=tf.reduce_sum(violation)/count
            active=tf.reduce_sum(tf.cast(violation>0,tf.float32))/count
        correction=tape.gradient(barrier,model.trainable_variables)
        if any(g is None for g in correction):raise ValueError('Disconnected class-loss barrier.')
    else:
        barrier,active=tf.constant(0.),tf.constant(0.)
        correction=[tf.zeros_like(g) for g in base]
    raw_norm=tf.linalg.global_norm(correction)
    scale=tf.constant(1. if beta else 0.)
    if correction_ratio_cap is not None:
        if correction_ratio_cap!=.1:raise ValueError('The budgeted candidate predeclares a 0.1 gradient-ratio cap.')
        scale=tf.stop_gradient(tf.minimum(scale,.1*tf.cast(values[4],tf.float32)/(raw_norm+1e-12)))
    applied=[g+beta*scale*r for g,r in zip(base,correction)] if beta else base
    result=list(values)
    result[2]=values[2]+beta*scale*barrier
    result[5]=tf.linalg.global_norm(applied) if beta else values[5]
    result += [barrier,beta*scale*raw_norm,active,scale,raw_norm]
    for value in applied+result:tf.debugging.assert_all_finite(value,'Non-finite CART barrier update')
    return applied,tuple(result)


class BarrierCartWorker(LocalMemoryCartWorker):
    extra_metric_names=DirichletCalibrationWorker.extra_metric_names+(
        'cartBarrierLoss','cartBarrierGradientNorm','cartActiveClassFraction',
        'cartBarrierEffectiveScale','cartBarrierUnscaledGradientNorm')

    def __init__(self,model,*,beta=1.,seed=2025,device='CPU',correction_ratio_cap=None):
        import tensorflow as tf
        self.probe_labels=tf.Variable(tf.zeros(MAX_PROBES,tf.int32),trainable=False)
        self.class_average=tf.Variable(tf.zeros((10,MAX_PROBES)),trainable=False)
        self.best_losses=tf.Variable(tf.zeros(10),trainable=False)
        self.class_support=tf.Variable(tf.zeros(10),trainable=False)
        self.correction_ratio_cap=correction_ratio_cap
        super().__init__(model,beta=beta,seed=seed,device=device)
        self.diagnostic_revision=REVISION

    def _set_retention_controls(self,probe_y,rows):
        import numpy as np
        labels=np.zeros(MAX_PROBES,np.int32);labels[:len(probe_y)]=probe_y
        average=np.zeros((10,MAX_PROBES),np.float32)
        best=np.zeros(10,np.float32);support=np.zeros(10,np.float32)
        for row in rows:
            c=row['classId'];mask=np.flatnonzero(probe_y==c)
            average[c,mask]=1./len(mask)
            best[c]=row['bestLoss'];support[c]=1.
        self.probe_labels.assign(labels);self.class_average.assign(average)
        self.best_losses.assign(best);self.class_support.assign(support)

    def _batch_gradients(self,x,y,key,mode,semantics,sigma,legacy_lambda):
        import tensorflow as tf
        from basil_core.research_audit import augment_batch
        if semantics!='paper_absolute_gaussian' or mode not in {'none','gradient_norm_objective'}:
            raise ValueError('Barrier CART retains the explicit absolute-channel base modes.')
        with tf.device('/CPU:0'):images=augment_batch(x,key)
        return barrier_gradients(self.model,images,y,self.reference,self.offsets,self.probe_images,
            self.probe_labels,self.class_average,self.best_losses,self.class_support,
            beta=self.beta,mu=self.mu,coefficient=sigma**2 if mode=='gradient_norm_objective' else 0.,
            correction_ratio_cap=self.correction_ratio_cap)

    def train(self,params,x,y,**kwargs):
        trained,record=super().train(params,x,y,**kwargs)
        record['cartMemory'].update(lossRevision=REVISION,barrierSlack=SLACK,
            correctionRatioCap=self.correction_ratio_cap,
            retentionForwardImages=MAX_PROBES*record['steps'] if self.beta else 0)
        return trained,record
