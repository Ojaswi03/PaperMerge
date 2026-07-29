from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf

try:
    import fcntl
except ImportError:  # pragma: no cover - the project runs under Linux/WSL
    fcntl = None


_CACHE_SCHEMA_VERSION = 1
_MAX_PARTITION_CACHE_FILES = 24


@dataclass(frozen=True)
class CifarArrayDataset:
    """Read-only array-backed CIFAR split with legacy sequence behavior."""

    images: np.ndarray
    labels: np.ndarray

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __iter__(self):
        return iter(zip(self.images, self.labels))


def _writeNpyAtomic(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tempName = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npy",
    )
    os.close(fd)
    tempPath = Path(tempName)
    try:
        np.save(tempPath, array, allow_pickle=False)
        os.replace(tempPath, path)
    finally:
        if tempPath.exists():
            tempPath.unlink()


def _writeJsonAtomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tempPath = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tempPath, path)
    finally:
        if tempPath.exists():
            tempPath.unlink()


def _cacheLock(cacheDir):
    cacheDir = Path(cacheDir)
    cacheDir.mkdir(parents=True, exist_ok=True)
    handle = (cacheDir / ".cache.lock").open("a+")
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _cachedCifarArrays(cacheDir):
    cacheDir = Path(cacheDir)
    paths = {
        "xTrain": cacheDir / "x_train_float32.npy",
        "yTrain": cacheDir / "y_train_int32.npy",
        "xTest": cacheDir / "x_test_float32.npy",
        "yTest": cacheDir / "y_test_int32.npy",
    }
    manifestPath = cacheDir / "manifest.json"
    lock = _cacheLock(cacheDir)
    try:
        valid = False
        try:
            manifest = json.loads(manifestPath.read_text(encoding="utf-8"))
            valid = (
                manifest.get("schemaVersion") == _CACHE_SCHEMA_VERSION
                and all(path.exists() for path in paths.values())
            )
        except (OSError, ValueError):
            valid = False

        if not valid:
            (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
            arrays = {
                "xTrain": xTrain.astype("float32") / 255.0,
                "yTrain": yTrain.reshape((-1,)).astype("int32"),
                "xTest": xTest.astype("float32") / 255.0,
                "yTest": yTest.reshape((-1,)).astype("int32"),
            }
            for key, array in arrays.items():
                _writeNpyAtomic(paths[key], array)
            _writeJsonAtomic(
                manifestPath,
                {
                    "schemaVersion": _CACHE_SCHEMA_VERSION,
                    "normalization": "float32_divide_255",
                    "files": {
                        key: {
                            "name": path.name,
                            "shape": list(arrays[key].shape),
                            "dtype": str(arrays[key].dtype),
                        }
                        for key, path in paths.items()
                    },
                },
            )
    finally:
        lock.close()

    return {
        key: np.load(path, mmap_mode="r", allow_pickle=False)
        for key, path in paths.items()
    }


def loadCifar10(cacheDir=None):
    if cacheDir is not None:
        arrays = _cachedCifarArrays(cacheDir)
        return (
            CifarArrayDataset(arrays["xTrain"], arrays["yTrain"]),
            CifarArrayDataset(arrays["xTest"], arrays["yTest"]),
        )
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


def _augmentStateless(index, sample, baseSeed):
    """Deterministic CIFAR augmentation keyed by client seed and sample visit."""
    x, y = sample
    visit = tf.cast(tf.math.floormod(index, 2_147_483_647), tf.int32)
    seed = tf.stack([tf.cast(baseSeed, tf.int32), visit])
    x = tf.image.stateless_random_flip_left_right(x, seed=seed)
    x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
    cropSeed = seed + tf.constant([0, 1], dtype=tf.int32)
    x = tf.image.stateless_random_crop(x, [32, 32, 3], seed=cropSeed)
    return x, y


def _datasetArrays(dataset):
    if isinstance(dataset, CifarArrayDataset):
        return dataset.images, dataset.labels
    images = np.stack([x for x, _ in dataset]).astype(np.float32)
    labels = np.asarray([y for _, y in dataset], dtype=np.int32)
    return images, labels


def _dirichletPartition(labels, nClients, alpha, rng=None):
    """Partition training data using Dirichlet distribution for non-IID splits."""
    rng = rng if rng is not None else np.random
    labels = np.asarray(labels, dtype=np.int32)
    nClasses = int(labels.max()) + 1
    clientIndices = [[] for _ in range(nClients)]

    for c in range(nClasses):
        classIdx = np.where(labels == c)[0]
        rng.shuffle(classIdx)
        proportions = rng.dirichlet(alpha * np.ones(nClients))
        splits = (np.cumsum(proportions) * len(classIdx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(classIdx, splits)):
            clientIndices[cid].extend(chunk.tolist())

    return [np.array(indices) for indices in clientIndices]


def _partitionCachePath(cacheDir, *, iid, nClients, alpha, seed, dataSize):
    payload = json.dumps(
        {
            "schemaVersion": _CACHE_SCHEMA_VERSION,
            "iid": bool(iid),
            "nClients": int(nClients),
            "alpha": float(alpha),
            "seed": None if seed is None else int(seed),
            "dataSize": int(dataSize),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:20]
    return Path(cacheDir) / "partitions" / f"partition_{digest}.npz"


def _prunePartitionCache(directory, keep):
    candidates = sorted(
        (
            path
            for path in Path(directory).glob("partition_*.npz")
            if path != Path(keep)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in candidates[_MAX_PARTITION_CACHE_FILES - 1 :]:
        try:
            stale.unlink()
        except OSError:
            pass


def _loadOrCreatePartition(
    labels,
    *,
    iid,
    nClients,
    alpha,
    seed,
    rng,
    cacheDir,
):
    if cacheDir is None or seed is None:
        if iid:
            indices = np.arange(len(labels))
            rng.shuffle(indices)
            return [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(indices, nClients)]
        return [
            np.asarray(chunk, dtype=np.int64)
            for chunk in _dirichletPartition(labels, nClients, alpha, rng=rng)
        ]

    cachePath = _partitionCachePath(
        cacheDir,
        iid=iid,
        nClients=nClients,
        alpha=alpha,
        seed=seed,
        dataSize=len(labels),
    )
    cachePath.parent.mkdir(parents=True, exist_ok=True)
    lock = _cacheLock(cachePath.parent)
    try:
        if cachePath.exists():
            try:
                with np.load(cachePath, allow_pickle=False) as saved:
                    chunks = [
                        np.asarray(saved[f"client_{clientId}"], dtype=np.int64)
                        for clientId in range(nClients)
                    ]
                if sum(len(chunk) for chunk in chunks) == len(labels):
                    return chunks
            except (OSError, ValueError, KeyError):
                pass

        if iid:
            indices = np.arange(len(labels))
            rng.shuffle(indices)
            chunks = [
                np.asarray(chunk, dtype=np.int64)
                for chunk in np.array_split(indices, nClients)
            ]
        else:
            chunks = [
                np.asarray(chunk, dtype=np.int64)
                for chunk in _dirichletPartition(labels, nClients, alpha, rng=rng)
            ]

        fd, tempName = tempfile.mkstemp(
            dir=cachePath.parent,
            prefix=f".{cachePath.name}.",
            suffix=".npz",
        )
        os.close(fd)
        tempPath = Path(tempName)
        try:
            np.savez(
                tempPath,
                **{
                    f"client_{clientId}": chunk
                    for clientId, chunk in enumerate(chunks)
                },
            )
            os.replace(tempPath, cachePath)
            _prunePartitionCache(cachePath.parent, cachePath)
        finally:
            if tempPath.exists():
                tempPath.unlink()
        return chunks
    finally:
        lock.close()


def makeLoaders(
    train,
    test,
    batchSize=32,
    iid=True,
    nClients=10,
    dirichletAlpha=0.5,
    seed=None,
    returnMetadata=False,
    cacheDir=None,
):
    if seed is not None:
        tf.keras.utils.set_random_seed(int(seed))
    rng = np.random.default_rng(int(seed)) if seed is not None else np.random
    trainImages, labels = _datasetArrays(train)
    testImages, testLabels = _datasetArrays(test)
    chunks = _loadOrCreatePartition(
        labels,
        iid=iid,
        nClients=nClients,
        alpha=dirichletAlpha,
        seed=seed,
        rng=rng,
        cacheDir=cacheDir,
    )

    def toDataset(indices, clientId):
        X = np.asarray(trainImages[indices], dtype=np.float32)
        y = np.asarray(labels[indices], dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        clientSeed = None if seed is None else int(seed) + 10_007 * (clientId + 1)
        ds = ds.shuffle(
            buffer_size=len(indices),
            seed=clientSeed,
            reshuffle_each_iteration=True,
        )
        ds = ds.repeat()
        if clientSeed is None:
            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.enumerate()
            ds = ds.map(
                lambda index, sample: _augmentStateless(index, sample, clientSeed),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
        ds = ds.batch(batchSize, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        if seed is not None:
            options = tf.data.Options()
            options.experimental_deterministic = True
            ds = ds.with_options(options)
        return ds

    trainLoaders = [toDataset(c, clientId) for clientId, c in enumerate(chunks)]

    tLoader = tf.data.Dataset.from_tensor_slices(
        (
            np.asarray(testImages, dtype=np.float32),
            np.asarray(testLabels, dtype=np.int32),
        )
    )
    tLoader = tLoader.batch(batchSize, drop_remainder=False)
    tLoader = tLoader.prefetch(tf.data.AUTOTUNE)

    if not returnMetadata:
        return trainLoaders, tLoader

    classCounts = np.stack(
        [np.bincount(labels[chunk], minlength=10) for chunk in chunks]
    ).astype(np.int32)
    probeBatches = []
    for chunk in chunks:
        selected = []
        chunkLabels = labels[chunk]
        for classId in range(10):
            classPositions = np.flatnonzero(chunkLabels == classId)[:32]
            selected.extend(chunk[classPositions].tolist())
        if selected:
            probeIndices = np.asarray(selected, dtype=np.int64)
            probeX = np.asarray(trainImages[probeIndices], dtype=np.float32)
            probeY = np.asarray(labels[probeIndices], dtype=np.int32)
        else:
            probeX = np.empty((0, 32, 32, 3), dtype=np.float32)
            probeY = np.empty((0,), dtype=np.int32)
        probeBatches.append((probeX, probeY))

    metadata = {
        "clientIndices": [np.asarray(chunk, dtype=np.int64) for chunk in chunks],
        "classCounts": classCounts,
        "probeBatches": probeBatches,
    }
    return trainLoaders, tLoader, metadata
