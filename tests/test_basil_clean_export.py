import json

import pytest

from basil_core import basil_plain_clean as study
from scripts import export_basil_plain_clean as exporter
from tests.test_clean_control_export import completed
from tests.test_plain_clean_export import recorded,save
from tests.test_basil_plain_clean import opts


@pytest.mark.parametrize('completed',['basil'],indirect=True)
def test_basil_export_uses_correct_study_and_fresh_verifier(completed,monkeypatch):
    root,dest=completed;seen=[]
    def routing(root,view,source,*,study_adapter):
        assert study_adapter is study
        cfg=study_adapter.effective_config(view)
        assert cfg['method']=='basil_plain_clean_ring' and not any(cfg['switches'].values())
        seen.append(view['runId'])
        return dict(runId=view['runId'],activationsChecked=50000,optimizerStepsChecked=250000)
    monkeypatch.setattr(exporter.shared,'check_routing',routing)
    result=exporter.export(root,dest)
    assert len(seen)==12 and result['configurations']==12 and result['freshInferenceVerified'] and result['plainRingRoutingVerified']
    manifest=json.loads((dest/'SHA256.json').read_text())
    assert {'plain_ring_verification.json','independent_validation.json'}<=set(manifest)
    audit=json.loads((dest/'plain_ring_verification.json').read_text())
    assert audit['algorithm']=='BASIL' and len(audit['verified'])==12
    assert all(exporter.shared.common.sha(dest/n)==h for n,h in manifest.items())
    assert not any(n.endswith('.log') or n.endswith('checkpoint.npz') for n in manifest)
    assert exporter.export(root,dest)==result


@pytest.mark.parametrize('completed',['plain'],indirect=True)
def test_fedsharp_results_cannot_be_exported_as_basil(completed):
    root,dest=completed
    with pytest.raises(ValueError,match='BASIL clean ring study'):exporter.export(root,dest)
    assert not dest.exists()


def test_basil_recorded_chain_audit_uses_its_own_effective_contract(recorded):
    root,_,_,meta=recorded
    args=opts(nodes=4,datasets=['mnist'],partitions=['iid'],seeds=[2025]);args.rounds=2
    view=study.selected(args)[0];source=study.source();cfg=study.effective_config(view)
    folder=root/view['runId'];folder.mkdir()
    contract=dict(config=cfg,source=source);meta['contract']=contract
    row=dict(status='COMPLETED',config=view,source=source,contract=contract,initialHash='initial')
    (folder/'run.json').write_text(json.dumps(row));save(root,view,meta)
    result=exporter.shared.check_routing(root,view,source,study_adapter=study)
    assert result['activationsChecked']==8 and result['optimizerStepsChecked']==40
    meta['telemetry'][1]['selection']['ranking']=[dict(sender=0,loss=1.)]
    save(root,view,meta)
    with pytest.raises(ValueError,match='ranking'):
        exporter.shared.check_routing(root,view,source,study_adapter=study)
