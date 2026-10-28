import json
import pytest
from scripts.export_cart_campaign import export,copy_matching


@pytest.fixture
def campaign(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir();dest=tmp_path/'published'
    cfg=dict(runId='safe-run');(root/'plan.json').write_text(json.dumps(dict(configs=[cfg])))
    run=root/'safe-run';run.mkdir();(run/'run.json').write_text('{"status":"COMPLETED"}')
    (run/'metrics.npz').write_bytes(b'metrics');(run/'checkpoint.npz').write_bytes(b'large private checkpoint')
    summary=dict(pendingGroups=[],unpairedCartGroups=[],independentInferencePerformed=False)
    monkeypatch.setattr('scripts.report_cart_v2_completed.summarize',lambda *_:summary)
    return root,dest,summary


def test_resumes_identical_partial_export_and_excludes_checkpoints(campaign):
    root,dest,_=campaign;dest.mkdir()
    first=export(root,dest);second=export(root,dest)
    assert first==second and first['configs']==1
    assert not list(dest.rglob('checkpoint*'))
    assert json.loads((dest/'SHA256.json').read_text())['safe-run/metrics.npz']


def test_conflicting_file_is_preserved(campaign):
    root,dest,_=campaign;dest.mkdir();(dest/'summary.json').write_text('important old result')
    with pytest.raises(FileExistsError,match='Conflicting'):export(root,dest)
    assert (dest/'summary.json').read_text()=='important old result'


def test_incomplete_groups_do_not_create_destination(campaign):
    root,dest,summary=campaign;summary['pendingGroups']=[{}]
    with pytest.raises(ValueError,match='completed'):export(root,dest)
    assert not dest.exists()


def test_campaign_destination_is_rejected(campaign):
    root,_,_=campaign
    for dest in (root,root/'figures',root.parent):
        with pytest.raises(ValueError,match='outside'):export(root,dest)


def test_unexpected_files_are_not_added_to_manifest(campaign):
    root,dest,_=campaign;dest.mkdir();(dest/'checkpoint.npz').write_bytes(b'preserve')
    with pytest.raises(ValueError,match='Unexpected'):export(root,dest)
    assert not (dest/'SHA256.json').exists()
