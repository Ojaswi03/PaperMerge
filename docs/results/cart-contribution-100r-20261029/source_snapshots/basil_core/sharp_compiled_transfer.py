"""Explicit runtime adapter: compile parameter transfer with existing inference.

No selection, loss, random stream, optimizer, CART or checkpoint changes.
The original SharpWorker and frozen research sources remain untouched.
"""
from basil_core.sharp_worker import SharpWorker,np


class CompiledTransferWorker(SharpWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        import tensorflow as tf
        def assign(flat):
            pieces=tf.split(flat,self.sizes)
            for variable,value,shape in zip(self.model.trainable_variables,pieces,self.shapes):
                variable.assign(tf.reshape(value,shape))
        self._assign=tf.function(assign,reduce_retracing=True)
        @tf.function(reduce_retracing=True)
        def forward(flat,images):
            assign(flat)
            return self.predict(images)
        @tf.function(reduce_retracing=True)
        def features(flat,images):
            assign(flat)
            return self.feature_predict(images)
        self._forward=forward;self._features=features

    def _vector(self,flat):
        value=np.asarray(flat)
        if value.shape!=(self.dimension,) or not np.isfinite(value).all():
            raise ValueError('Invalid model vector.')
        return np.asarray(value,np.float32)

    def load(self,flat):
        self._assign(self._vector(flat))

    def outputs(self,flat,x,*,features=False,batch=128):
        if not len(x) or batch<=0:return super().outputs(flat,x,features=features,batch=batch)
        flat=self._vector(flat)
        combined=self._features if features else self._forward
        predict=self.feature_predict if features else self.predict
        result=[combined(flat,x[:batch]).numpy()]
        result.extend(predict(x[i:i+batch]).numpy() for i in range(batch,len(x),batch))
        return np.concatenate(result)
