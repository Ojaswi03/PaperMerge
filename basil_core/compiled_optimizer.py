"""Execution-only SGD wrapper; gradients and optimizer mathematics are unchanged."""
from __future__ import annotations


class CompiledSGD:
    def __init__(self,model,learning_rate,momentum):
        import tensorflow as tf
        self.inner=tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum)
        self.model_variables=tuple(model.trainable_variables)
        self.inner.build(self.model_variables)
        self.compiled_apply=tf.function(self._apply,autograph=False,reduce_retracing=True)

    def __getattr__(self,name):return getattr(self.inner,name)

    def _apply(self,gradients):
        return self.inner.apply_gradients(zip(gradients,self.model_variables))

    def apply_gradients(self,pairs):
        pairs=list(pairs)
        if len(pairs)!=len(self.model_variables) or any(v is not expected for (_,v),expected in zip(pairs,self.model_variables)):
            raise ValueError('Compiled SGD must update its bound model variables in order.')
        return self.compiled_apply(tuple(g for g,_ in pairs))
