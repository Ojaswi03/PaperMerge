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

def makeLoaders(train, test, batchSize=32, iid=True, nClients=10):
    idx = np.arange(len(train))
    np.random.shuffle(idx)
    chunks = np.array_split(idx, nClients)

    def toBatches(indices):
        data = [train[i] for i in indices]
        X, y, out = [], [], []
        for xi, yi in data:
            X.append(xi)
            y.append(yi)
            if len(X) == batchSize:
                out.append((np.stack(X), np.array(y)))
                X, y = [], []
        if X:
            out.append((np.stack(X), np.array(y)))
        return out

    trainLoaders = [toBatches(c) for c in chunks]

    Xt, yt, tLoader = [], [], []
    for xi, yi in test:
        Xt.append(xi)
        yt.append(yi)
        if len(Xt) == batchSize:
            tLoader.append((np.stack(Xt), np.array(yt)))
            Xt, yt = [], []
    if Xt:
        tLoader.append((np.stack(Xt), np.array(yt)))

    return trainLoaders, tLoader
