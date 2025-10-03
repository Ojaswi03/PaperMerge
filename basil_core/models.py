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
    A light CNN for CIFAR-10 (kept simple to match Basil experiments).
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPool2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
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
