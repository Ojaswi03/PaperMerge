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


def install_rebuild(output, rebuild, reference_plan, *, progress=None):
    """Install an existing rebuild against a checksummed archive, not a bad plan.

    The failed plan is retained verbatim and must never be resumed with different
    data. No payload is rewritten: preserve the old inode with a hard link and
    atomically rename the already-written rebuild over the shared cache.
    """
    log = progress or (lambda message: None)
    output, rebuild, reference_plan = map(lambda p: Path(p).resolve(), (output, rebuild, reference_plan))
    log('Checking the archived reference plan and its checksum.')
    manifest = json.loads((reference_plan.parent / 'SHA256.json').read_text())
    if manifest.get(reference_plan.name) != file_sha(reference_plan):
        raise ValueError('The archived reference plan does not match its checksum manifest.')
    expected = json.loads(reference_plan.read_text())['data']['cifar10']
    old_plan = output / 'plan.json'
    old_plan_sha = file_sha(old_plan)
    evidence = json.loads((rebuild / 'recovery.json').read_text())
    if evidence['planSha256'] != old_plan_sha or Path(evidence['recoveryDirectory']).resolve() != rebuild:
        raise ValueError('Staged rebuild does not belong to this failed plan.')
    target = ROOT / 'experiments/cache/cifar10/x_train_float32.npy'
    with ExitStack() as stack:
        for path in (output / 'preparation.lock', output / 'supervisor.lock', target.parent / '.cache.lock'):
            handle = stack.enter_context(path.open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('A supervisor or cache operation is active; installation cancelled.') from None
        plan = json.loads(old_plan.read_text())
        for view in plan.get('configs', []):
            folder = output / view['runId']
            if (folder / 'checkpoint.npz').exists():
                raise ValueError('The failed plan already has training checkpoints; do not change its input cache.')
            if (folder / 'run.json').exists():
                row = json.loads((folder / 'run.json').read_text())
                if row.get('completedActivations', 0) or row.get('status') == 'COMPLETED':
                    raise ValueError('The failed plan already has training measurements.')
        log('Inspecting the shared cache; the failed plan will remain unchanged.')
        before = inspect_cache(ROOT, 'cifar10', expected)
        if before['valid'] and before['identity'] == expected:
            return dict(status='ALREADY_VERIFIED_AGAINST_ARCHIVE', newPlanRequired=True,
                        cacheModified=False, oldPlanSha256=old_plan_sha)
        if before['identity']['labelsSha256'] != expected['labelsSha256']:
            raise ValueError('Shared labels differ from the verified archive; installation cancelled.')
        before_sha = file_sha(target)
        log('Verifying the existing rebuild against the archived dataset; no new rebuild is being written.')
        candidate = rebuild / 'experiments/cache/cifar10/x_train_float32.npy'
        rebuilt = inspect_cache(rebuild, 'cifar10', expected)
        if not rebuilt['valid'] or rebuilt['identity'] != expected:
            raise ValueError('Existing rebuild does not match the independently verified archive.')
        if file_sha(target) != before_sha or file_sha(old_plan) != old_plan_sha:
            raise RuntimeError('The shared cache or failed plan changed during installation checks.')
        backup = rebuild / 'cifar10_x_train_before_archive_install.npy'
        log('Preserving the damaged file with a hard link; no large backup copy is needed.')
        if backup.exists():
            if file_sha(backup) != before_sha:
                raise ValueError('An existing backup has a different checksum; it will not be overwritten.')
        else:
            os.link(target, backup)
        if file_sha(target) != before_sha or file_sha(backup) != before_sha:
            raise RuntimeError('Cache changed while preserving its backup; installation cancelled.')
        candidate.chmod(0o444)
        with candidate.open('rb') as handle:
            os.fsync(handle.fileno())
        record = dict(status='INSTALLING_ARCHIVE_VERIFIED_REBUILD', cacheModified=False,
            oldPlanSha256=old_plan_sha, referencePlan=str(reference_plan),
            referencePlanSha256=file_sha(reference_plan), before=before, rebuilt=rebuilt,
            backup=str(backup), newPlanRequired=True, toolSha256=file_sha(Path(__file__)))
        receipt = rebuild / 'archive_installation.json'

        def save():
            receipt.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

        save()
        log('Installing the verified file by atomic rename, then checking its identity.')
        os.replace(candidate, target)
        record['cacheModified'] = True
        after = inspect_cache(ROOT, 'cifar10', expected)
        record['after'] = after
        if not after['valid'] or after['identity'] != expected or file_sha(old_plan) != old_plan_sha:
            record['status'] = 'FAILED_POST_INSTALL_CHECK'
            save()
            raise RuntimeError('Post-install identity check failed; do not start training.')
        record.update(status='INSTALLED_ARCHIVE_VERIFIED_REBUILD', oldPlanUnmodified=True)
        save()
        return record
