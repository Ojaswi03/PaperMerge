"""Experimental training-only local-prior adjustment for the CART worker.

Logit adjustment follows established imbalanced-classification work. This is
not a new loss or a Byzantine guarantee. Selection, repair and inference still
use raw logits; the gradient-norm penalty differentiates the adjusted CE.
"""
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.cart_batched_scoring import BatchedAdmittedWorker


def log_prior(labels, classes):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not len(labels) or labels.dtype.kind not in 'iu' or np.any(labels<0) or np.any(labels>=classes):
        raise ValueError('Nonempty local integer labels in the configured class range required.')
    counts=np.maximum(np.bincount(labels,minlength=classes),1).astype(np.float64)
    return np.log(counts/counts.sum()).astype(np.float32)


def objective_gradients(model,images,labels,anchor,coefficient,mu,prior,*,robust=True):
    import tensorflow as tf
    variables=model.trainable_variables
    with tf.GradientTape() as outer:
        if robust:
            with tf.GradientTape() as inner:
                ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,logits=model(images,training=True)+prior))
            base=inner.gradient(ce,variables)
            norm_sq=tf.add_n([tf.reduce_sum(g*g) for g in base])
        else:
            ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,logits=model(images,training=True)+prior))
            norm_sq=tf.zeros_like(ce)
        prox=mu*.5*tf.add_n([tf.reduce_sum((v-a)**2) for v,a in zip(variables,anchor)])
        objective=ce+coefficient*norm_sq+prox
    gradients=outer.gradient(objective,variables)
    if any(g is None for g in gradients):raise ValueError('Disconnected prior-adjusted objective.')
    for value in [*gradients,ce,norm_sq,objective]:tf.debugging.assert_all_finite(value,'Nonfinite prior-adjusted objective')
    return gradients,(ce,norm_sq,objective,prox,tf.linalg.global_norm(gradients))


class LocalPriorWorker(BatchedAdmittedWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        import tensorflow as tf
        self._training_prior=tf.Variable(tf.zeros(self.cfg['classes']),trainable=False)
        robust=self.cfg['switches']['noise_mitigation']
        @tf.function(reduce_retracing=True)
        def step(x,y,anchor,eta,coefficient,mu):
            gradients,values=objective_gradients(self.model,x,y,anchor,coefficient,mu,
                self._training_prior,robust=robust)
            for variable,gradient in zip(self.model.trainable_variables,gradients):variable.assign_sub(eta*gradient)
            return values
        self.step=step

    def train(self,flat,x,y,round_id,node,coefficient,first_batch,should_stop=None):
        prior=log_prior(y,self.cfg['classes']);self._training_prior.assign(prior)
        result,record=super().train(flat,x,y,round_id,node,coefficient,first_batch,should_stop)
        record.update(trainingLogPrior=prior.tolist(),trainingObjective='local_training_prior_adjusted_CE',
            priorSource='complete receiver-owned fitting labels; floor-one counts; strength one',
            selectionAndInference='raw logits; unchanged CART admission and commitment')
        return result,record
