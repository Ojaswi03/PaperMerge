"""A failed model download must not install or overwrite weights."""
import hashlib

import pytest

from scripts.download_research_models import download


def asset(path):
    return dict(url=path.as_uri(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_download_and_reuse_verified_model(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'known parameter bytes')
    target = tmp_path / 'models/run/model.npz'
    assert download(asset(source), target) == 'downloaded and verified'
    assert target.read_bytes() == source.read_bytes()
    assert download(asset(source), target) == 'already verified'


def test_bad_hash_leaves_no_installed_file(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'incomplete or corrupted download')
    description = dict(asset(source), sha256='0' * 64)
    target = tmp_path / 'model.npz'
    with pytest.raises(ValueError, match='verification'):
        download(description, target)
    assert not target.exists()
    assert not list(tmp_path.glob('.download-*'))


def test_existing_conflict_is_preserved(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'expected')
    target = tmp_path / 'model.npz'
    target.write_bytes(b'preserve these different weights')
    with pytest.raises(ValueError, match='preserved'):
        download(asset(source), target)
    assert target.read_bytes() == b'preserve these different weights'
