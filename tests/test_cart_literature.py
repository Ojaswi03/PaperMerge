"""Literature downloader checks; no network, TensorFlow, data or training."""
from io import BytesIO
import hashlib
import json
import sys

import pytest

from scripts import download_cart_literature as literature


class Response(BytesIO):
    url = 'https://example.org/paper.pdf'


def entry(name='paper_one'):
    return dict(id=name, pdfUrls=['https://example.org/paper.pdf'])


def test_manifest_is_unique_and_points_to_primary_https_sources():
    manifest = json.loads(literature.MANIFEST.read_text())
    entries = manifest['papers']
    assert len(entries) == 12
    assert len({e['id'] for e in entries}) == len(entries)
    allowed = {'arxiv.org', 'openreview.net', 'openaccess.thecvf.com',
               'www.mdpi.com', 'mdpi-res.com', 'ojs.aaai.org'}
    from urllib.parse import urlparse
    for paper in entries:
        for url in [paper['source'], *paper['pdfUrls']]:
            parsed = urlparse(url)
            assert parsed.scheme == 'https'
            assert parsed.netloc in allowed


def test_default_plan_never_downloads(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['download_cart_literature.py'])
    def unexpected(*args, **kwargs):
        raise AssertionError('Default plan must not access the network.')
    monkeypatch.setattr(literature, 'urlopen', unexpected)
    literature.main()
    assert json.loads(capsys.readouterr().out) == dict(papers=12, networkRequests=0, trainingRuns=0)


def test_valid_download_has_hash_and_no_partial_file(monkeypatch, tmp_path):
    monkeypatch.setattr(literature, 'DIRECTORY', tmp_path)
    monkeypatch.setattr(literature, 'ROOT', tmp_path)
    payload = b'%PDF-1.7\nTest stream\n%%EOF\n'
    monkeypatch.setattr(literature, 'urlopen', lambda *a, **kw: Response(payload))
    result = literature.download(entry())
    assert result['status'] == 'downloaded'
    assert result['bytes'] == len(payload)
    assert result['sha256'] == hashlib.sha256(payload).hexdigest()
    assert (tmp_path/'paper_one.pdf').read_bytes() == payload
    assert not list(tmp_path.glob('*.part'))


def test_refuses_to_overwrite_existing_papers(monkeypatch, tmp_path):
    monkeypatch.setattr(literature, 'DIRECTORY', tmp_path)
    destination = tmp_path/'paper_one.pdf'
    destination.write_bytes(b'original preserved paper')
    def unexpected(*args, **kwargs):
        raise AssertionError('Existing paper must not trigger network access.')
    monkeypatch.setattr(literature, 'urlopen', unexpected)
    with pytest.raises(FileExistsError):
        literature.download(entry())
    assert destination.read_bytes() == b'original preserved paper'


@pytest.mark.parametrize('payload', [b'<html>error</html>', b'%PDF-1.7\ntruncated'])
def test_rejects_non_pdf_or_incomplete_response(monkeypatch, tmp_path, payload):
    monkeypatch.setattr(literature, 'DIRECTORY', tmp_path)
    monkeypatch.setattr(literature, 'urlopen', lambda *a, **kw: Response(payload))
    result = literature.download(entry())
    assert result['status'] == 'failed'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('name', ['../escape', 'a/b', 'UPPER', 'name with space'])
def test_rejects_unsafe_paper_identifiers(name):
    with pytest.raises(ValueError, match='Unsafe'):
        literature.download(entry(name))


def test_rejects_insecure_url_before_accessing_network(monkeypatch, tmp_path):
    monkeypatch.setattr(literature, 'DIRECTORY', tmp_path)
    paper = entry()
    paper['pdfUrls'] = ['http://example.org/paper.pdf']
    with pytest.raises(ValueError, match='HTTPS'):
        literature.download(paper)
    assert not list(tmp_path.iterdir())


def test_fallback_keeps_primary_failure_in_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(literature, 'DIRECTORY', tmp_path)
    monkeypatch.setattr(literature, 'ROOT', tmp_path)
    paper = entry()
    paper['pdfUrls'].append('https://example.org/fallback.pdf')
    def respond(request, **kwargs):
        if request.full_url.endswith('/paper.pdf'):
            raise OSError('Source unavailable')
        return Response(b'%PDF-1.7\ntest\n%%EOF\n')
    monkeypatch.setattr(literature, 'urlopen', respond)
    result = literature.download(paper)
    assert result['status'] == 'downloaded'
    assert result['requestedURL'].endswith('/fallback.pdf')
    assert result['previousSourceAttempts'] == [dict(url=paper['pdfUrls'][0], error='Source unavailable')]
