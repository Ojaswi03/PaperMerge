import numpy as np
import tensorflow as tf

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # y comes shape (N,1); flatten to (N,)
    y_train = y_train.reshape((-1,)).astype("int32")
    y_test  = y_test.reshape((-1,)).astype("int32")
    x_train = (x_train.astype("float32") / 255.0)
    x_test  = (x_test.astype("float32") / 255.0)
    return (list(zip(x_train, y_train)), list(zip(x_test, y_test)))

def make_loaders(train, test, batch_size=32, iid=True, n_clients=10):
    idx = np.arange(len(train))
    np.random.shuffle(idx)
    chunks = np.array_split(idx, n_clients)

    def to_batches(indices):
        data = [train[i] for i in indices]
        X, y, out = [], [], []
        for xi, yi in data:
            X.append(xi)
            y.append(yi)
            if len(X) == batch_size:
                out.append((np.stack(X), np.array(y)))
                X, y = [], []
        if X:
            out.append((np.stack(X), np.array(y)))
        return out

    train_loaders = [to_batches(c) for c in chunks]

    Xt, yt, tloader = [], [], []
    for xi, yi in test:
        Xt.append(xi); yt.append(yi)
        if len(Xt) == batch_size:
            tloader.append((np.stack(Xt), np.array(yt)))
            Xt, yt = [], []
    if Xt:
        tloader.append((np.stack(Xt), np.array(yt)))

    return train_loaders, tloader
