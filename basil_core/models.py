import tensorflow as tf
from tensorflow.keras import layers, models

class MNISTModel:
    """
    MNIST model (Basil paper style):
    3 fully connected layers: 784->100->100->10 (logits)
    """
    def __init__(self, input_shape=(28, 28), num_classes=10):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(num_classes)
        ])

    # ----- Basil convenience API -----
    def get_params(self):
        return [w.numpy() for w in self.model.trainable_weights]

    def set_params(self, params):
        for var, val in zip(self.model.trainable_weights, params):
            var.assign(val)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def __call__(self, x, training=False):
        return self.model(x, training=training)


class CIFARModel:
    """
    CIFAR-10 CNN targeting 85-90% clean accuracy.
    VGG-style: 3 conv blocks (64→128→256 filters) + 512-unit head.
    Uses He normal init and Dropout - no BatchNorm so federated
    parameter averaging (getParams/setParams) works on trainable
    weights only without BN running-stat issues.
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        ki = 'he_normal'
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            # Block 1 - 64 filters
            layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer=ki),
            layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer=ki),
            layers.MaxPool2D(2),
            layers.Dropout(0.25),
            # Block 2 - 128 filters
            layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer=ki),
            layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer=ki),
            layers.MaxPool2D(2),
            layers.Dropout(0.25),
            # Block 3 - 256 filters
            layers.Conv2D(256, 3, padding='same', activation='relu', kernel_initializer=ki),
            layers.Conv2D(256, 3, padding='same', activation='relu', kernel_initializer=ki),
            layers.MaxPool2D(2),
            layers.Dropout(0.4),
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_initializer=ki),
            layers.Dropout(0.5),
            layers.Dense(num_classes),
        ])

    # ----- Basil convenience API -----
    def get_params(self):
        return [w.numpy() for w in self.model.trainable_weights]

    def set_params(self, params):
        for var, val in zip(self.model.trainable_weights, params):
            var.assign(val)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def __call__(self, x, training=False):
        return self.model(x, training=training)


class BasilPaperCifarModel:
    """Small CIFAR-10 CNN used by the research sequential protocol.

    Architecture: 16x3x3 conv, 3x3 pool, 64x4x4 conv, 4x4 pool,
    and 384/192/10 dense layers. With biases this has 117,706 trainable
    parameters. Historical CIFAR configurations continue to use ``CIFARModel``.
    """

    def __init__(self, input_shape=(32, 32, 3), num_classes=10, seed=2025):
        def dense_initializer(fan_in, offset):
            bound = fan_in ** -0.5
            return tf.keras.initializers.RandomUniform(-bound, bound, seed=seed + offset)

        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, activation="relu", padding="valid",
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
            layers.MaxPool2D(pool_size=3, strides=3),
            layers.Conv2D(64, 4, activation="relu", padding="valid",
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1)),
            layers.MaxPool2D(pool_size=4, strides=4),
            layers.Flatten(),
            layers.Dense(384, activation="relu", kernel_initializer=dense_initializer(64, 2)),
            layers.Dense(192, activation="relu", kernel_initializer=dense_initializer(384, 3)),
            layers.Dense(num_classes, kernel_initializer=dense_initializer(192, 4)),
        ])

    def get_params(self):
        return [weight.numpy() for weight in self.model.trainable_weights]

    def set_params(self, params):
        for variable, value in zip(self.model.trainable_weights, params):
            variable.assign(value)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def __call__(self, x, training=False):
        return self.model(x, training=training)


class NMNISTModel:
    """
    Neuromorphic MNIST model (same architecture as regular MNIST):
    3 fully connected layers: 784->100->100->10 (logits)
    Since N-MNIST has same spatial dimensions (28x28), we use the same architecture.
    """
    def __init__(self, input_shape=(34, 34), num_classes=10):
        # N-MNIST sensor size is 34x34 (different from regular MNIST 28x28)
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(num_classes)  # logits
        ])

    # ----- Basil convenience API -----
    def get_params(self):
        return [w.numpy() for w in self.model.trainable_weights]

    def set_params(self, params):
        for var, val in zip(self.model.trainable_weights, params):
            var.assign(val)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def __call__(self, x, training=False):
        return self.model(x, training=training)
