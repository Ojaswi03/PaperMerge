import numpy as np
import tensorflow as tf

def loadCifar10():
    (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
    # y comes shape (N,1); flatten to (N,)
    yTrain = yTrain.reshape((-1,)).astype("int32")
    yTest  = yTest.reshape((-1,)).astype("int32")
    xTrain = (xTrain.astype("float32") / 255.0)
    xTest  = (xTest.astype("float32") / 255.0)
    return (list(zip(xTrain, yTrain)), list(zip(xTest, yTest)))

def _augment(x, y):
    """Random horizontal flip + random crop (pad 4px, then crop back to 32x32)."""
    x = tf.image.random_flip_left_right(x)
    x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
    x = tf.image.random_crop(x, [32, 32, 3])
    return x, y

def makeLoaders(train, test, batchSize=32, iid=True, nClients=10):
    idx = np.arange(len(train))
    np.random.shuffle(idx)
    chunks = np.array_split(idx, nClients)

    def toDataset(indices):
        data = [train[i] for i in indices]
        X = np.stack([xi for xi, _ in data]).astype(np.float32)
        y = np.array([yi for _, yi in data], dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.shuffle(buffer_size=len(indices), reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batchSize, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    trainLoaders = [toDataset(c) for c in chunks]

    Xt = np.stack([xi for xi, _ in test]).astype(np.float32)
    yt = np.array([yi for _, yi in test], dtype=np.int32)
    tLoader = tf.data.Dataset.from_tensor_slices((Xt, yt))
    tLoader = tLoader.batch(batchSize, drop_remainder=False)
    tLoader = tLoader.prefetch(tf.data.AUTOTUNE)

    return trainLoaders, tLoader
