"""Restore damaged fitting images only when a rebuild matches a saved plan."""
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_cache(base, dataset, expected):
    import numpy as np
    from basil_core.sharp_data import digest
    folder = Path(base) / 'experiments/cache' / dataset
    x = np.load(folder / 'x_train_float32.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(folder / 'y_train_int32.npy', mmap_mode='r', allow_pickle=False)
    shape = (32, 32, 3)
    if x.shape != (expected['samples'], *shape) or y.shape != (expected['samples'],):
        raise ValueError('Cache shape differs from the saved plan sample count.')
    if x.dtype != np.float32 or y.dtype != np.int32:
        raise ValueError('Cache dtype differs from the fitting-data contract.')
    finite = bool(np.isfinite(x).all())
    low, high = float(x.min()), float(x.max())
    minimum = low if np.isfinite(low) else None
    maximum = high if np.isfinite(high) else None
    labels = np.unique(y).tolist()
    identity = dict(imagesSha256=digest(x), labelsSha256=digest(y), samples=len(y),
                    officialTestLoaded=False)
    return dict(identity=identity, finite=finite, minimum=minimum, maximum=maximum,
                labels=labels, valid=finite and minimum >= 0 and maximum <= 1 and labels == list(range(10)))


def stage_from_raw(staging):
    from basil_core import sharp_data
    # The existing loader reads the five raw training batches from ~/.keras.
    # Only its output root changes; the shared cache is not involved here.
    with patch.object(sharp_data, 'ROOT', staging):
        sharp_data.prepare_cache('cifar10')


def recover(output, *, apply=False):
    output = Path(output).resolve()
    plan_path = output / 'plan.json'
    plan = json.loads(plan_path.read_text())
    expected = plan['data']['cifar10']
    target = ROOT / 'experiments/cache/cifar10/x_train_float32.npy'
    with ExitStack() as stack:
        for name in ('preparation.lock', 'supervisor.lock'):
            handle = stack.enter_context((output / name).open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('The plan is in use; stop its supervisor before cache recovery.') from None
        cache_lock = stack.enter_context((target.parent / '.cache.lock').open('a'))
        try:
            fcntl.flock(cache_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The shared cache is in use by another cache operation.') from None
        before = inspect_cache(ROOT, 'cifar10', expected)
        if before['valid'] and before['identity'] == expected:
            return dict(status='ALREADY_VERIFIED', cache=str(target), cacheModified=False)
        if before['identity']['labelsSha256'] != expected['labelsSha256']:
            raise ValueError('Label cache also differs; image-only recovery refuses to change either file.')
        baseline_sha = file_sha(target)
        folder = Path(tempfile.mkdtemp(prefix='cache-recovery-', dir=output))
        record = dict(status='REBUILDING', planSha256=file_sha(plan_path), before=before,
                      originalFileSha256=baseline_sha, cacheModified=False,
                      recoveryDirectory=str(folder), toolSha256=file_sha(Path(__file__)))
        receipt = folder / 'recovery.json'

        def save():
            receipt.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

        save()
        try:
            stage_from_raw(folder)
            rebuilt = inspect_cache(folder, 'cifar10', expected)
            record['rebuilt'] = rebuilt
            if not rebuilt['valid'] or rebuilt['identity'] != expected:
                raise ValueError('Rebuilt data does not exactly match the saved plan; original cache preserved.')
            record['status'] = 'VERIFIED_REBUILD_AVAILABLE'
            save()
            if not apply:
                return record
            if file_sha(target) != baseline_sha or inspect_cache(ROOT, 'cifar10', expected) != before:
                raise RuntimeError('Shared cache changed again during recovery; installation cancelled.')
            backup = folder / 'cifar10_x_train_before.npy'
            shutil.copy2(target, backup)
            if file_sha(backup) != baseline_sha or file_sha(target) != baseline_sha:
                raise RuntimeError('Cache changed while preserving its backup; installation cancelled.')
            rebuilt_path = folder / 'experiments/cache/cifar10/x_train_float32.npy'
            # Discourage accidental in-place writes; replacement remains possible.
            rebuilt_path.chmod(0o444)
            with rebuilt_path.open('rb') as handle:
                os.fsync(handle.fileno())
            record.update(status='INSTALLING_VERIFIED_REBUILD', backup=str(backup))
            save()
            os.replace(rebuilt_path, target)
            record['cacheModified'] = True
            after = inspect_cache(ROOT, 'cifar10', expected)
            record['after'] = after
            if not after['valid'] or after['identity'] != expected:
                raise RuntimeError('Post-install cache identity changed; do not restart training.')
            record.update(status='REPAIRED_AND_VERIFIED', installedFileSha256=file_sha(target))
            save()
            return record
        except BaseException as error:
            record.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            save()
            raise
