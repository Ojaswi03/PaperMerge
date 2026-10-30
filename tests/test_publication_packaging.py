import json
from pathlib import Path
import pytest
from scripts.package_plot_formats import package,sha


def fixture(root):
    files={'figures/curve.pdf':b'plot pdf','figures/curve.svg':b'vector svg',
        'figures/curve.png':b'png','source_snapshots/papers/spec.pdf':b'scientific reference',
        'run/metrics.npz':b'original measurements'}
    for name,value in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    p=root/'figures/figures_manifest.json'
    p.write_text(json.dumps(dict(outputSha256={name:sha(root/'figures'/name) for name in ('curve.pdf','curve.svg','curve.png')})))
    files['figures/figures_manifest.json']=p.read_bytes()
    (root/'SHA256.json').write_text(json.dumps({name:sha(root/name) for name in files}))
    return files


def test_packaging_removes_only_plot_pdf_and_keeps_valid_checksums(tmp_path):
    original=fixture(tmp_path)
    assert package(tmp_path)==1 and package(tmp_path)==0
    assert not (tmp_path/'figures/curve.pdf').exists()
    for name in ('figures/curve.png','figures/curve.svg','run/metrics.npz','source_snapshots/papers/spec.pdf'):
        assert (tmp_path/name).read_bytes()==original[name]
    manifest=json.loads((tmp_path/'SHA256.json').read_text())
    assert all(sha(tmp_path/name)==value for name,value in manifest.items())
    assert 'curve.pdf' not in json.loads((tmp_path/'figures/figures_manifest.json').read_text())['outputSha256']


def test_bad_export_does_not_delete_anything(tmp_path):
    fixture(tmp_path);(tmp_path/'run/metrics.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum'):package(tmp_path)
    assert (tmp_path/'figures/curve.pdf').exists()
