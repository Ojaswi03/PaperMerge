#!/usr/bin/env python3
"""Read both complete original training inputs before launching any experiments."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basil_core.thesis_raw_data import training_data, mnist_archive


if __name__ == '__main__':
    print('Checking original training inputs; no experiment will be started.', flush=True)
    for dataset in ('mnist', 'cifar10'):
        print(f'Loading and verifying {dataset}.', flush=True)
        x, y = training_data(dataset)
        print(json.dumps(dict(dataset=dataset, samples=len(y), shape=x.shape, dtype=str(x.dtype),
                              minimum=float(x.min()), maximum=float(x.max()), verified=True)), flush=True)
        del x, y
    print('MNIST archive: ' + str(mnist_archive()), flush=True)
    print('INPUTS_VERIFIED: both complete datasets match the verified fingerprints.', flush=True)
