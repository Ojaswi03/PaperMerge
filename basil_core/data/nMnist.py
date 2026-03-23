import numpy as np
import tensorflow as tf
import os

# Optional tonic library for real N-MNIST data
try:
    import tonic
    import torch
    from torch.utils.data import DataLoader
    TONIC_AVAILABLE = True
except ImportError:
    TONIC_AVAILABLE = False
    tonic = None
    torch = None


def loadNMnist():
    # attempt to load real N-MNIST via tonic, fall back to simulated data
    xTrain, yTrain, xTest, yTest = None, None, None, None

    # Try to use tonic library for N-MNIST
    if TONIC_AVAILABLE:
        try:
            # Convert events to frames with time binning
            frameTransform = tonic.transforms.ToFrame(
                sensor_size=tonic.datasets.NMNIST.sensor_size,
                time_window=10000  # 10ms time bins
            )

            # Download and load training data
            trainDataset = tonic.datasets.NMNIST(
                save_to='./data/nmnist',
                train=True,
                transform=frameTransform
            )

            # Download and load test data
            testDataset = tonic.datasets.NMNIST(
                save_to='./data/nmnist',
                train=False,
                transform=frameTransform
            )

            # Convert to numpy arrays
            xTrainList, yTrainList = [], []
            for frames, label in trainDataset:
                # frames shape: (T, 2, H, W) where T=time_bins, 2=polarities
                # Sum across time and polarities, resulting in (H, W)
                frameSum = np.sum(frames, axis=(0, 1))  # Sum over time and polarity
                # Normalize to [0, 1]
                if frameSum.max() > 0:
                    frameSum = frameSum / frameSum.max()
                xTrainList.append(frameSum.astype("float32"))
                yTrainList.append(int(label))

            xTestList, yTestList = [], []
            for frames, label in testDataset:
                frameSum = np.sum(frames, axis=(0, 1))
                if frameSum.max() > 0:
                    frameSum = frameSum / frameSum.max()
                xTestList.append(frameSum.astype("float32"))
                yTestList.append(int(label))

            xTrain = np.array(xTrainList)
            yTrain = np.array(yTrainList, dtype="int32")
            xTest = np.array(xTestList)
            yTest = np.array(yTestList, dtype="int32")

            print("Successfully loaded real N-MNIST data from tonic library.")

        except ImportError as e:
            print(f"Warning: tonic library not found ({e}). Using simulated N-MNIST data based on regular MNIST.")
            print("Install tonic with: pip install tonic")
        except Exception as e:
            print(f"Warning: Error loading N-MNIST from tonic ({e}). Using simulated N-MNIST data.")
            print("Falling back to MNIST-based simulation.")

    # Fallback: Use regular MNIST with added event-like noise as placeholder
    if xTrain is None:
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()

        # Add Poisson noise to simulate event-based sensor
        # This is a simplified approximation for testing purposes
        def addEventNoise(images, scale=10.0):
            # Scale up intensities
            scaled = images.astype("float32") * scale
            # Add Poisson noise
            events = np.random.poisson(scaled).astype("float32")
            # Normalize back to [0, 1]
            maxVal = events.max(axis=(1, 2), keepdims=True)
            maxVal = np.where(maxVal > 0, maxVal, 1.0)  # Avoid division by zero
            return events / maxVal

        xTrain = addEventNoise(xTrain)
        xTest = addEventNoise(xTest)
        yTrain = yTrain.astype("int32")
        yTest = yTest.astype("int32")

    return (list(zip(xTrain, yTrain)), list(zip(xTest, yTest)))


def makeLoaders(train, test, batchSize=32, iid=True, nClients=10):
    # Simple IID split into nClients chunks
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

    # Single shared test loader
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
