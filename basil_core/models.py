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


class CIFARModel:
    """
    CIFAR-10 model matching BASIL paper Table II:
    - conv1: 3->16 filters, 3x3 kernel, ReLU, MaxPool(2x2)
    - conv2: 16->64 filters, 4x4 kernel, ReLU, MaxPool(2x2)
    - fc1: flattened -> 384, ReLU
    - fc2: 384 -> 192, ReLU
    - fc3: 192 -> 10 (logits)
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            # conv1: 3->16 filters, 3x3 kernel (paper Table II)
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            # conv2: 16->64 filters, 4x4 kernel (paper Table II)
            layers.Conv2D(64, 4, padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            # fc1: -> 384 (paper Table II)
            layers.Dense(384, activation="relu"),
            # fc2: 384 -> 192 (paper Table II)
            layers.Dense(192, activation="relu"),
            # fc3: 192 -> 10 logits (paper Table II)
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
