import json
from pathlib import Path
import pytest
from tests.test_cart_revision_reporting import revision


def test_revision_export_is_repeatable_and_conflicts_are_preserved(revision,monkeypatch):
    from scripts.export_cart_revision import export
    from scripts.export_cart_campaign import sha
    _,root,_,plan=revision
    snapshot=root/'source_snapshots/fixture';snapshot.parent.mkdir();snapshot.write_text('producer source')
    plan['source']={'fixture':sha(snapshot)};(root/'plan.json').write_text(json.dumps(plan))
    for cfg in plan['configs']:
        p=root/cfg['runId']/'run.json';row=json.loads(p.read_text());row['source']=plan['source'];p.write_text(json.dumps(row))
    figures=root/'figures';figures.mkdir();(figures/'summary.json').write_text('{}')
    (figures/'artifacts_manifest.json').write_text(json.dumps(dict(revision=plan['revision'],source=plan['source'],outputSha256={'summary.json':sha(figures/'summary.json')})))
    dest=root.parent/'export';assert export(root,dest)==(6,16)
    before=(dest/'SHA256.json').read_bytes();export(root,dest)
    assert before==(dest/'SHA256.json').read_bytes()
    target=dest/plan['configs'][0]['runId']/'run.json';target.write_text('preserve conflict')
    with pytest.raises(FileExistsError):export(root,dest)
    assert target.read_text()=='preserve conflict'
