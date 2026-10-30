"""Load original fitting archives into RAM; never read normalized cache files."""
from contextlib import contextmanager
from pathlib import Path
import pickle
from unittest.mock import patch

EXPECTED = {
    'mnist': dict(imagesSha256='8b40caca67f0698ef159ea1b4a502534192bc9bea42dfd329b3867fe044f61dc',
        labelsSha256='d2b17876ba99cc2047c707b3b06080e9940c6110997da3c0fd0f2f00394cfbe3',
        samples=60000, officialTestLoaded=False),
    'cifar10': dict(imagesSha256='f8d4e2cda7d5b07dab87dc18de073ebd26513ca2a097b6b7b8b97b991cc23462',
        labelsSha256='9c307df4e68bb31df7e92944444ba8d69473a952644f744b4aa44c721cb9a7ba',
        samples=50000, officialTestLoaded=False),
}


def training_data(dataset):
    import numpy as np
    from basil_core.sharp_data import digest
    base = Path.home() / '.keras/datasets'
    if dataset == 'mnist':
        with np.load(base / 'mnist.npz', allow_pickle=False) as archive:
            x = archive['x_train'].astype(np.float32)[..., None] / 255.
            y = archive['y_train'].astype(np.int32)
        shape = (28, 28, 1)
    elif dataset == 'cifar10':
        folder = base / 'cifar-10-batches-py-target/cifar-10-batches-py'
        images, labels = [], []
        for index in range(1, 6):
            with (folder / f'data_batch_{index}').open('rb') as handle:
                batch = pickle.load(handle, encoding='bytes')
            images.append(batch[b'data'])
            labels.extend(batch[b'labels'])
        x = np.concatenate(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.
        y = np.asarray(labels, np.int32)
        shape = (32, 32, 3)
    else:
        raise ValueError('Direct thesis inputs support MNIST and CIFAR-10.')
    expected = EXPECTED[dataset]
    if x.shape != (expected['samples'], *shape) or y.shape != (expected['samples'],):
        raise ValueError(f'{dataset}: raw training shape differs from the verified dataset.')
    actual = dict(imagesSha256=digest(x), labelsSha256=digest(y), samples=len(y), officialTestLoaded=False)
    if actual != expected:
        raise ValueError(f'{dataset}: original training archives do not match the verified data fingerprints.')
    x.flags.writeable = False
    y.flags.writeable = False
    return x, y


@contextmanager
def direct_inputs():
    from basil_core import sharp_data
    with patch.object(sharp_data, 'training_data', training_data):
        yield
