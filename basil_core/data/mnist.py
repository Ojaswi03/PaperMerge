import numpy as np
import tensorflow as tf

def loadMnist():
    (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
    xTrain = (xTrain.astype("float32") / 255.0)
    xTest  = (xTest.astype("float32") / 255.0)
    yTrain = yTrain.astype("int32")
    yTest  = yTest.astype("int32")
    return (list(zip(xTrain, yTrain)), list(zip(xTest, yTest)))

def makeLoaders(train, test, batchSize=32, iid=True, nClients=10):
    # Simple IID split into nClients chunks
    idx = np.arange(len(train))
    np.random.shuffle(idx)
    chunks = np.array_split(idx, nClients)

    def toDataset(indices):
        data = [train[i] for i in indices]
        X = np.stack([xi for xi, _ in data]).astype(np.float32)
        y = np.array([yi for _, yi in data], dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.batch(batchSize, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    trainLoaders = [toDataset(c) for c in chunks]

    # single shared test loader
    Xt = np.stack([xi for xi, _ in test]).astype(np.float32)
    yt = np.array([yi for _, yi in test], dtype=np.int32)
    tLoader = tf.data.Dataset.from_tensor_slices((Xt, yt))
    tLoader = tLoader.batch(batchSize, drop_remainder=False)
    tLoader = tLoader.prefetch(tf.data.AUTOTUNE)

    return trainLoaders, tLoader
