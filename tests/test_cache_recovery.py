import fcntl
import json
from pathlib import Path

import numpy as np
import pytest

from basil_core import cache_recovery as recovery
from basil_core.sharp_data import digest


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    output = root / 'run'
    output.mkdir(parents=True)
    images = np.random.default_rng(14).uniform(size=(20, 32, 32, 3)).astype(np.float32)
    labels = np.tile(np.arange(10, dtype=np.int32), 2)
    folder = root / 'experiments/cache/cifar10'
    folder.mkdir(parents=True)
    np.save(folder / 'x_train_float32.npy', images)
    np.save(folder / 'y_train_int32.npy', labels)
    identity = dict(imagesSha256=digest(images), labelsSha256=digest(labels), samples=20, officialTestLoaded=False)
    (output / 'plan.json').write_text(json.dumps(dict(data=dict(cifar10=identity))))
    monkeypatch.setattr(recovery, 'ROOT', root)

    def stage(staging):
        cache = staging / 'experiments/cache/cifar10'
        cache.mkdir(parents=True)
        np.save(cache / 'x_train_float32.npy', images)
        np.save(cache / 'y_train_int32.npy', labels)

    monkeypatch.setattr(recovery, 'stage_from_raw', stage)
    return output, folder, images, labels, stage


def corrupt(folder, images):
    bad = images.copy()
    bad[0, 0, 0, 0] = 1.654902
    np.save(folder / 'x_train_float32.npy', bad)
    return bad


def test_matching_cache_is_never_replaced(fixture, monkeypatch):
    output, folder, images, _, _ = fixture
    before = recovery.file_sha(folder / 'x_train_float32.npy')
    monkeypatch.setattr(recovery, 'stage_from_raw', lambda _: pytest.fail('Unnecessary rebuild'))
    assert recovery.recover(output, apply=True)['status'] == 'ALREADY_VERIFIED'
    assert recovery.file_sha(folder / 'x_train_float32.npy') == before


def test_repair_preserves_damaged_file_and_restores_exact_plan(fixture):
    output, folder, images, _, _ = fixture
    bad = corrupt(folder, images)
    plan_before = (output / 'plan.json').read_bytes()
    label_before = (folder / 'y_train_int32.npy').read_bytes()
    record = recovery.recover(output, apply=True)
    assert record['status'] == 'REPAIRED_AND_VERIFIED' and record['cacheModified']
    np.testing.assert_array_equal(np.load(record['backup']), bad)
    np.testing.assert_array_equal(np.load(folder / 'x_train_float32.npy'), images)
    assert (output / 'plan.json').read_bytes() == plan_before
    assert (folder / 'y_train_int32.npy').read_bytes() == label_before
    assert (folder / 'x_train_float32.npy').stat().st_mode & 0o222 == 0


def test_dry_run_does_not_change_cache(fixture):
    output, folder, images, _, _ = fixture
    bad = corrupt(folder, images)
    record = recovery.recover(output)
    assert record['status'] == 'VERIFIED_REBUILD_AVAILABLE' and not record['cacheModified']
    np.testing.assert_array_equal(np.load(folder / 'x_train_float32.npy'), bad)


def test_nonmatching_raw_rebuild_is_rejected(fixture, monkeypatch):
    output, folder, images, _, stage = fixture
    bad = corrupt(folder, images)

    def different(staging):
        stage(staging)
        changed = images.copy()
        changed[0, 0, 0, 0] = .4
        np.save(staging / 'experiments/cache/cifar10/x_train_float32.npy', changed)

    monkeypatch.setattr(recovery, 'stage_from_raw', different)
    with pytest.raises(ValueError, match='does not exactly match'):
        recovery.recover(output, apply=True)
    np.testing.assert_array_equal(np.load(folder / 'x_train_float32.npy'), bad)


def test_active_supervisor_blocks_repair(fixture):
    output, _, _, _, _ = fixture
    with (output / 'supervisor.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='in use'):
            recovery.recover(output, apply=True)


def test_detects_another_cache_change_before_installation(fixture, monkeypatch):
    output, folder, images, _, stage = fixture
    corrupt(folder, images)
    newer = images.copy()
    newer[0, 0, 0, 0] = 2.

    def concurrent(staging):
        stage(staging)
        np.save(folder / 'x_train_float32.npy', newer)

    monkeypatch.setattr(recovery, 'stage_from_raw', concurrent)
    with pytest.raises(RuntimeError, match='changed again'):
        recovery.recover(output, apply=True)
    np.testing.assert_array_equal(np.load(folder / 'x_train_float32.npy'), newer)


def test_nonfinite_cache_can_be_preserved_and_repaired(fixture):
    output, folder, images, _, _ = fixture
    bad = images.copy()
    bad[0, 0, 0, 0] = np.nan
    np.save(folder / 'x_train_float32.npy', bad)
    record = recovery.recover(output, apply=True)
    assert record['status'] == 'REPAIRED_AND_VERIFIED'
    assert not record['before']['finite'] and record['before']['maximum'] is None
    np.testing.assert_array_equal(np.load(record['backup']), bad)
    np.testing.assert_array_equal(np.load(folder / 'x_train_float32.npy'), images)


def test_label_damage_is_not_silently_replaced(fixture):
    output, folder, images, labels, _ = fixture
    corrupt(folder, images)
    labels = labels.copy()
    labels[0] = 9
    np.save(folder / 'y_train_int32.npy', labels)
    with pytest.raises(ValueError, match='Label cache also differs'):
        recovery.recover(output, apply=True)
    np.testing.assert_array_equal(np.load(folder / 'y_train_int32.npy'), labels)
