import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from basil_core import fedsharp_clean_controls as study
from scripts import export_fedsharp_clean_controls as exporter
from tests.test_fedsharp_clean_controls import opts


@pytest.fixture
def completed(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir()
    views=study.selected(opts());source=study.source()
    plan=dict(revision=study.REVISION,configs=views,source=source,
              families=['sharp'],environments=['clean'],data={'mnist':{},'cifar10':{}},
              partitions={v['runId']:{} for v in views})
    (root/'plan.json').write_text(json.dumps(plan))
    (root/'campaign.json').write_text(json.dumps(dict(status='COMPLETED')))
    for name in source:
        path=root/'source_snapshots'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((exporter.ROOT/name).read_bytes())
    for view in views:
        folder=root/view['runId'];folder.mkdir()
        row=dict(status='COMPLETED',config=view,source=source,methodRevision=study.REVISION,
                 completedActivations=50000,optimizerSteps=250000,data={},partition={})
        (folder/'run.json').write_text(json.dumps(row))
        np.savez_compressed(folder/'metrics.npz',roundNodeAccuracy=np.zeros((500,100)))
        (folder/'mechanisms.json').write_text('{}')
        (folder/'checkpoint.npz').write_bytes(b'private checkpoint fixture')
        (root/(view['runId']+'.log')).write_text('private log fixture')
    def verify(command,**kwargs):
        assert command[1:4]==[str(exporter.ROOT/'scripts/run_fedsharp_clean_controls.py'),
                              '--role','verify']
        assert kwargs['check'] is True
        receipt=dict(status='ALL_COMPARISON_MODELS_VERIFIED',source=source,officialTestUsed=False,
                     verified=[dict(runId=v['runId']) for v in views])
        (root/'independent_validation.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(exporter.subprocess,'run',verify)
    return root,tmp_path/'export'


def test_verified_export_has_complete_manifest_and_excludes_runtime(completed):
    root,dest=completed
    before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=exporter.export(root,dest)
    assert result['configurations']==12 and result['freshInferenceVerified']
    manifest=json.loads((dest/'SHA256.json').read_text())
    assert set(manifest)=={str(p.relative_to(dest)) for p in dest.rglob('*')
                           if p.is_file() and p.name!='SHA256.json'}
    assert all(exporter.sha(dest/n)==h for n,h in manifest.items())
    assert not any(Path(n).name.startswith('checkpoint') or n.endswith('.log') for n in manifest)
    assert all((root/n).read_bytes()==value for n,value in before.items())
    assert exporter.export(root,dest)==result


def test_incomplete_run_refused_before_inference(completed,monkeypatch):
    root,dest=completed
    path=next(root.glob('*/run.json'));row=json.loads(path.read_text());row['status']='FAILED'
    path.write_text(json.dumps(row))
    monkeypatch.setattr(exporter.subprocess,'run',lambda *a,**k:pytest.fail('must not infer'))
    with pytest.raises(ValueError,match='Incomplete'):exporter.export(root,dest)
    assert not dest.exists()


def test_inference_failure_cannot_publish(completed,monkeypatch):
    root,dest=completed
    def fail(*args,**kwargs):raise subprocess.CalledProcessError(1,args[0])
    monkeypatch.setattr(exporter.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):exporter.export(root,dest)
    assert not dest.exists()


@pytest.mark.parametrize('change',['receipt','metrics','checkpoint'])
def test_verification_must_cover_same_stable_inputs(completed,monkeypatch,change):
    root,dest=completed;verify=exporter.subprocess.run
    def altered(*args,**kwargs):
        verify(*args,**kwargs)
        if change=='receipt':
            path=root/'independent_validation.json';row=json.loads(path.read_text())
            row['verified'].pop();path.write_text(json.dumps(row))
        else:
            path=next(root.glob('*/'+('metrics.npz' if change=='metrics' else 'checkpoint.npz')))
            path.write_bytes(path.read_bytes()+b'changed')
    monkeypatch.setattr(exporter.subprocess,'run',altered)
    with pytest.raises(ValueError):exporter.export(root,dest)
    assert not dest.exists()


def test_unexpected_export_files_are_preserved_and_refused(completed):
    root,dest=completed;dest.mkdir();(dest/'checkpoint.npz').write_bytes(b'preserve')
    with pytest.raises(ValueError,match='Unexpected'):exporter.export(root,dest)
    assert (dest/'checkpoint.npz').read_bytes()==b'preserve'
