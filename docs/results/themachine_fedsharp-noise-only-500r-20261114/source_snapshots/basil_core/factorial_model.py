"""Opt-in CIFAR residual model with no untransmitted running statistics.

All GroupNorm scale/offset variables travel with each snapshot. There is no
BatchNorm state, dropout RNG state, pretraining, or official TEST dependence.
"""
from __future__ import annotations


def build_model(seed):
    import tensorflow as tf
    from tensorflow.keras import layers
    counter=iter(range(1000))
    class CpuInitializer(tf.keras.initializers.Initializer):
        """Sample once on CPU; variables themselves stay on the requested device.

        Truncated-normal sampling can differ across CPU/GPU implementations.
        A common initialization must not depend on the execution device.
        """
        def __init__(self,inner):self.inner=inner
        def __call__(self,shape,dtype=None):
            with tf.device('/CPU:0'):return self.inner(shape,dtype=dtype)
        def get_config(self):return {'inner':tf.keras.initializers.serialize(self.inner)}
    def conv(x,filters,stride=1,kernel=3):
        return layers.Conv2D(filters,kernel,strides=stride,padding='same',use_bias=False,
            kernel_initializer=CpuInitializer(tf.keras.initializers.HeNormal(seed=seed+next(counter))))(x)
    def norm(x):
        return layers.GroupNormalization(groups=8,axis=-1,epsilon=1e-5)(x)
    inputs=layers.Input((32,32,3))
    # Fixed affine image scaling; TRAIN pixels are normalized to [0, 1].
    x=layers.Rescaling(2.,offset=-1.)(inputs)
    x=layers.ReLU()(norm(conv(x,16)))
    for filters in (16,32,64):
        for block in range(3):
            stride=2 if filters!=16 and block==0 else 1
            shortcut=x
            h=layers.ReLU()(norm(conv(x,filters,stride)))
            h=norm(conv(h,filters))
            if shortcut.shape[-1]!=filters or stride!=1:
                shortcut=norm(conv(shortcut,filters,stride,kernel=1))
            x=layers.ReLU()(layers.Add()([h,shortcut]))
    x=layers.GlobalAveragePooling2D()(x)
    logits=layers.Dense(10,kernel_initializer=CpuInitializer(tf.keras.initializers.GlorotUniform(seed=seed+next(counter))))(x)
    return tf.keras.Model(inputs,logits,name='cifar_resnet20_groupnorm_v1')
