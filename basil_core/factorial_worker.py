"""Workers for explicitly independent defenses; historical workers stay fixed."""
from __future__ import annotations

from basil_core.research_protocol import SequentialWorker
from basil_core.cart_barrier import BarrierCartWorker


def regularize(model, gradients, values, coefficient):
    """Coupled L2 on convolution/dense kernels only, common to all conditions."""
    import tensorflow as tf
    terms=[tf.reduce_sum(v*v) for v in model.trainable_variables if len(v.shape)>1]
    penalty=coefficient*.5*tf.add_n(terms)
    applied=[g+coefficient*v if len(v.shape)>1 else g for g,v in zip(gradients,model.trainable_variables)]
    values=list(values)
    values[2]=values[2]+penalty
    values[5]=tf.linalg.global_norm(applied)
    for v in applied+[penalty]:tf.debugging.assert_all_finite(v,'Non-finite factorial update')
    return applied,tuple(values)


class PlainWorker(SequentialWorker):
    def __init__(self,model):
        import tensorflow as tf
        super().__init__(model)
        self.training_offsets=tf.Variable(tf.zeros(10),trainable=False)

    def train(self,params,x,y,**kwargs):
        import numpy as np
        from basil_core.dirichlet_calibration import calibration_offsets
        enabled=self.factorial_config.get('logitCalibrationEnabled',False)
        counts=np.bincount(y,minlength=10)
        self.training_offsets.assign(calibration_offsets(counts,method='prior_adjustment')
            if enabled else np.zeros(10,np.float32))
        trained,record=super().train(params,x,y,**kwargs)
        record.update(localClassCounts=counts.tolist(),logitCalibrationOffsets=self.training_offsets.numpy().tolist())
        return trained,record

    def _batch_gradients(self,x,y,key,mode,semantics,sigma,legacy_lambda):
        import tensorflow as tf
        from basil_core.research_audit import augment_batch
        from basil_core.research_protocol import gradient_norm_objective_gradients
        from basil_core.trainer import lossFn
        with tf.device('/CPU:0'):images=augment_batch(x,key)
        if self.factorial_config.get('logitCalibrationEnabled',False):
            # Established training-time logit adjustment, shared by paired arms;
            # neither receiver selection nor raw-logit inference is changed.
            with tf.GradientTape() as outer:
                with tf.GradientTape() as inner:
                    ce=lossFn(y,self.model(images,training=True)-self.training_offsets)
                first=inner.gradient(ce,self.model.trainable_variables)
                squared=tf.add_n([tf.reduce_sum(g*g) for g in first])
                coefficient=sigma**2 if mode=='gradient_norm_objective' else tf.constant(0.)
                objective=ce+coefficient*squared
            gradients=outer.gradient(objective,self.model.trainable_variables)
            values=(ce,squared,objective,coefficient,tf.linalg.global_norm(first),tf.linalg.global_norm(gradients))
            return regularize(self.model,gradients,values,self.factorial_config['weightDecayCoefficient'])
        if mode=='gradient_norm_objective':
            gradients,ce,squared,norm,robust=gradient_norm_objective_gradients(self.model,images,y,sigma**2)
            values=(ce,squared,ce+sigma**2*squared,sigma**2,norm,robust)
        elif mode=='none':
            with tf.GradientTape() as tape:ce=lossFn(y,self.model(images,training=True))
            gradients=tape.gradient(ce,self.model.trainable_variables)
            norm=tf.linalg.global_norm(gradients)
            values=(ce,norm**2,ce,tf.constant(0.),norm,norm)
        else:raise ValueError('Factorial supports only ordinary CE or full second-order EBM.')
        return regularize(self.model,gradients,values,self.factorial_config['weightDecayCoefficient'])


class CartWorker(BarrierCartWorker):
    """Existing bounded class-memory candidate, with prior/anchor explicitly off.

    Its inclusion is an ablation, not a claim that the earlier failed candidate
    now works. Acceptance requires a measured paired improvement.
    """
    def _batch_gradients(self,x,y,key,mode,semantics,sigma,legacy_lambda):
        import tensorflow as tf
        from basil_core.research_audit import augment_batch
        from basil_core.cart_barrier import barrier_gradients
        with tf.device('/CPU:0'):images=augment_batch(x,key)
        gradients,values=barrier_gradients(self.model,images,y,self.reference,self.offsets,
            self.probe_images,self.probe_labels,self.class_average,self.best_losses,
            self.class_support,beta=self.beta,mu=0.,
            coefficient=sigma**2 if mode=='gradient_norm_objective' else 0.,correction_ratio_cap=.1)
        return regularize(self.model,gradients,values,self.factorial_config['weightDecayCoefficient'])


def make_worker(model,config):
    import tensorflow as tf
    from basil_core.factorial_study import validate_config,REVISION
    validate_config(config)
    if config['cartEnabled']:
        worker=CartWorker(model,beta=1.,seed=config['seed'],device=config['executionDevice'],correction_ratio_cap=.1)
        worker.mu=0.
        worker.calibration_enabled=False
        worker.diagnostic_revision=REVISION
    else:worker=PlainWorker(model)
    worker.factorial_config=dict(config)
    worker.optimizer_factory=lambda rate:tf.keras.optimizers.SGD(learning_rate=rate,momentum=config['momentum'])
    return worker
