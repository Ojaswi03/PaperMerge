import numpy as np
import tensorflow as tf

def loadMnist():
    (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
    xTrain = (xTrain.astype("float32") / 255.0)
    xTest  = (xTest.astype("float32") / 255.0)
    yTrain = yTrain.astype("int32")
    yTest  = yTest.astype("int32")
    return (list(zip(xTrain, yTrain)), list(zip(xTest, yTest)))

def _dirichletPartition(train, nClients, alpha):
    """Partition training data using Dirichlet distribution for non-IID splits."""
    labels = np.array([y for _, y in train])
    nClasses = int(labels.max()) + 1
    clientIndices = [[] for _ in range(nClients)]

    for c in range(nClasses):
        classIdx = np.where(labels == c)[0]
        np.random.shuffle(classIdx)
        proportions = np.random.dirichlet(alpha * np.ones(nClients))
        splits = (np.cumsum(proportions) * len(classIdx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(classIdx, splits)):
            clientIndices[cid].extend(chunk.tolist())

    return [np.array(indices) for indices in clientIndices]

def makeLoaders(train, test, batchSize=32, iid=True, nClients=10, dirichletAlpha=0.5):
    if iid:
        idx = np.arange(len(train))
        np.random.shuffle(idx)
        chunks = np.array_split(idx, nClients)
    else:
        chunks = _dirichletPartition(train, nClients, dirichletAlpha)

    def toDataset(indices):
        data = [train[i] for i in indices]
        X = np.stack([xi for xi, _ in data]).astype(np.float32)
        y = np.array([yi for _, yi in data], dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.shuffle(buffer_size=len(indices), reshuffle_each_iteration=True)
        ds = ds.batch(batchSize, drop_remainder=True)
        ds = ds.repeat()
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
