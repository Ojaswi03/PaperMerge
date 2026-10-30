import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from basil_core import thesis_raw_data as raw
from basil_core.sharp_data import digest


@pytest.fixture
def archives(tmp_path, monkeypatch):
    base = tmp_path / '.keras/datasets'
    base.mkdir(parents=True)
    rng = np.random.default_rng(42)
    mnist = rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)
    labels = np.tile(np.arange(10, dtype=np.int32), 2)
    np.savez(base / 'mnist.npz', x_train=mnist, y_train=labels,
             x_test=np.array([object()], dtype=object))
    cifar = rng.integers(0, 256, (20, 3072), dtype=np.uint8)
    folder = base / 'cifar-10-batches-py-target/cifar-10-batches-py'
    folder.mkdir(parents=True)
    for i in range(5):
        with (folder / f'data_batch_{i+1}').open('wb') as handle:
            pickle.dump({b'data': cifar[4*i:4*i+4], b'labels': labels[4*i:4*i+4].tolist()}, handle)
    expected_images = dict(mnist=mnist.astype(np.float32)[..., None] / 255.,
        cifar10=cifar.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.)
    expected = {ds: dict(imagesSha256=digest(x), labelsSha256=digest(labels), samples=20,
                        officialTestLoaded=False) for ds, x in expected_images.items()}
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(raw, 'EXPECTED', expected)
    return tmp_path, expected_images, labels


@pytest.mark.parametrize('dataset', ['mnist', 'cifar10'])
def test_direct_archive_loading_is_exact_and_writes_no_cache(archives, dataset):
    root, expected, labels = archives
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    x, y = raw.training_data(dataset)
    np.testing.assert_array_equal(x, expected[dataset])
    np.testing.assert_array_equal(y, labels)
    assert not x.flags.writeable and not y.flags.writeable
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_changed_raw_pixels_are_rejected(archives):
    root, _, _ = archives
    path = root / '.keras/datasets/cifar-10-batches-py-target/cifar-10-batches-py/data_batch_1'
    with path.open('rb') as handle:
        value = pickle.load(handle)
    value[b'data'][0, 0] ^= np.uint8(1)
    with path.open('wb') as handle:
        pickle.dump(value, handle)
    with pytest.raises(ValueError, match='fingerprints'):
        raw.training_data('cifar10')


def test_worker_and_inference_scope_replaces_and_restores_legacy_loader(archives, monkeypatch):
    from basil_core import sharp_data
    from basil_core.basil_sharp_thesis import installed

    def forbidden(_):
        pytest.fail('The shared normalized cache must not be read')

    monkeypatch.setattr(sharp_data, 'training_data', forbidden)
    with installed():
        assert sharp_data.training_data is raw.training_data
        assert len(sharp_data.training_data('mnist')[0]) == 20
    assert sharp_data.training_data is forbidden


def test_input_identities_match_independently_verified_archive():
    archive = Path(__file__).resolve().parents[1] / 'docs/results/cart-contribution-100r-20261029/plan.json'
    assert raw.EXPECTED == json.loads(archive.read_text())['data']
