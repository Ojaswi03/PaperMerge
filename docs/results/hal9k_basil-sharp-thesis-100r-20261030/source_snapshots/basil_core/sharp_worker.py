"""Five local SGD steps with exact nested differentiation of SHARP Eq. 17.

No hidden momentum, weight decay, prior adjustment, augmentation or clipping.
"""
from __future__ import annotations

from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')


def build_model(cfg, device='CPU'):
    import tensorflow as tf
    from basil_core.research_protocol import keyed_rng
    seed=int(keyed_rng(cfg['seed'],'sharp_model',cfg['dataset'],cfg['model']).integers(2**29))
    tf.keras.utils.set_random_seed(seed)
    def create():
        if cfg['dataset']=='mnist':
            model=tf.keras.Sequential([tf.keras.layers.Input((28,28,1)),tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64,activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
                tf.keras.layers.Dense(32,activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1)),
                tf.keras.layers.Dense(cfg['classes'],kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+2))])
        elif cfg['model']=='paper':
            from basil_core.models import BasilPaperCifarModel
            model=BasilPaperCifarModel(num_classes=cfg['classes'],seed=seed).model
        else:
            from basil_core.factorial_model import build_model as resnet
            original=resnet(seed)
            model=tf.keras.Model(original.inputs,tf.keras.layers.Dense(cfg['classes'],
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1))(original.layers[-2].output))
        return model
    with tf.device('/CPU:0'):
        model=create()
        initial=model.get_weights()
    if device=='GPU':
        with tf.device('/GPU:0'):
            model=create()
            model.set_weights(initial)
    return model


def objective_gradients(model, images, labels, anchor, coefficient, mu, *, robust=True):
    """Differentiate CE + lambda ||grad CE||² + mu/2 ||theta-anchor||²."""
    import tensorflow as tf
    variables=model.trainable_variables
    with tf.GradientTape() as outer:
        if robust:
            with tf.GradientTape() as inner:
                ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=model(images,training=True)))
            base=inner.gradient(ce,variables)
            norm_sq=tf.add_n([tf.reduce_sum(g*g) for g in base])
        else:
            ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=model(images,training=True)))
            norm_sq=tf.constant(0.,tf.float32)
        prox=mu*.5*tf.add_n([tf.reduce_sum((v-a)**2) for v,a in zip(variables,anchor)])
        objective=ce+coefficient*norm_sq+prox
    gradients=outer.gradient(objective,variables)
    if any(g is None for g in gradients):raise ValueError('Disconnected SHARP objective gradient.')
    for value in [*gradients,ce,norm_sq,objective]:tf.debugging.assert_all_finite(value,'Non-finite SHARP objective')
    return gradients,(ce,norm_sq,objective,prox,tf.linalg.global_norm(gradients))


class SharpWorker:
    def __init__(self,cfg,device='CPU',model=None):
        import tensorflow as tf
        self.cfg=cfg
        self.model=model if model is not None else build_model(cfg,device)
        self.shapes=[tuple(v.shape) for v in self.model.trainable_variables]
        self.sizes=[int(np.prod(s)) for s in self.shapes]
        self.dimension=sum(self.sizes)
        self.feature_model=tf.keras.Model(self.model.inputs,self.model.layers[-2].output)
        self.predict=tf.function(lambda x:self.model(x,training=False),reduce_retracing=True)
        self.feature_predict=tf.function(lambda x:self.feature_model(x,training=False),reduce_retracing=True)
        robust=cfg['switches']['noise_mitigation']
        @tf.function(reduce_retracing=True)
        def step(x,y,anchor,eta,coefficient,mu):
            gradients,values=objective_gradients(self.model,x,y,anchor,coefficient,mu,robust=robust)
            for variable,gradient in zip(self.model.trainable_variables,gradients):variable.assign_sub(eta*gradient)
            return values
        self.step=step

    def tensors(self,flat):
        if np.asarray(flat).shape!=(self.dimension,) or not np.isfinite(flat).all():
            raise ValueError('Invalid model vector.')
        return [v.reshape(shape) for v,shape in zip(np.split(flat,np.cumsum(self.sizes)[:-1]),self.shapes)]

    def load(self,flat):
        for variable,value in zip(self.model.trainable_variables,self.tensors(flat)):variable.assign(value)

    def export(self):
        return np.concatenate([v.numpy().ravel() for v in self.model.trainable_variables]).astype(np.float32)

    def outputs(self,flat,x,*,features=False,batch=128):
        self.load(flat)
        f=self.feature_predict if features else self.predict
        return np.concatenate([f(x[i:i+batch]).numpy() for i in range(0,len(x),batch)])

    def probe(self,flat,x,y):
        from basil_core.sharp_math import class_ce
        logits=self.outputs(flat,x)
        shifted=logits.astype(np.float64)-logits.max(axis=1,keepdims=True)
        losses=np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(y)),y]
        if not np.isfinite(losses).all():return float('inf')
        return float(losses.mean())

    def train(self,flat,x,y,round_id,node,coefficient,first_batch,should_stop=None):
        import tensorflow as tf
        from basil_core.research_protocol import keyed_rng
        self.load(flat)
        anchor=tuple(tf.convert_to_tensor(v) for v in self.tensors(flat))
        eta=self.cfg['learningRate']/(1+self.cfg['learningRate']*round_id)
        values=[]
        for step in range(self.cfg['localSteps']):
            if should_stop and should_stop():raise InterruptedError('Stop requested between local steps; activation discarded.')
            indices=first_batch if step==0 else keyed_rng(self.cfg['seed'],'sharp_batch',round_id,node,step).choice(
                len(y),size=min(self.cfg['batchSize'],len(y)),replace=False)
            result=self.step(tf.convert_to_tensor(x[indices]),tf.convert_to_tensor(y[indices],tf.int32),anchor,
                tf.constant(eta,tf.float32),tf.constant(coefficient,tf.float32),tf.constant(self.cfg['proximalMu'],tf.float32))
            values.append(dict(zip(('ce','gradientSquared','objective','proximalPenalty','updateGradientNorm'),map(float,result))))
        final=self.export()
        if not np.isfinite(final).all():raise FloatingPointError('Non-finite trained model.')
        return final,dict(optimizerSteps=len(values),learningRate=eta,rlaCoefficient=coefficient,
                         proximalMu=self.cfg['proximalMu'],steps=values,localTrainingLoss=float(np.mean([v['ce'] for v in values])))
