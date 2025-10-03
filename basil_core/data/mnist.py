import numpy as np
import tensorflow as tf

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)
    x_test  = (x_test.astype("float32") / 255.0)
    y_train = y_train.astype("int32")
    y_test  = y_test.astype("int32")
    return (list(zip(x_train, y_train)), list(zip(x_test, y_test)))

def make_loaders(train, test, batch_size=32, iid=True, n_clients=10):
    """
    Convert list-of-tuples dataset into:
    - train_loaders: list of per-client batches
    - test_loader  : one shared test loader list
    """
    # Simple IID split into n_clients chunks
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

    # single shared test loader
    Xt, yt, tloader = [], [], []
    for xi, yi in test:
        Xt.append(xi); yt.append(yi)
        if len(Xt) == batch_size:
            tloader.append((np.stack(Xt), np.array(yt)))
            Xt, yt = [], []
    if Xt:
        tloader.append((np.stack(Xt), np.array(yt)))

    return train_loaders, tloader
