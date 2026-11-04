import copy
import json
from pathlib import Path

import numpy as np
import pytest

from basil_core import fedsharp_plain_clean as study
from scripts import export_fedsharp_plain_clean as exporter
from tests.test_clean_control_export import completed
from tests.test_fedsharp_plain_clean import opts


@pytest.fixture
def recorded(tmp_path):
    args=opts(nodes=4,datasets=['mnist'],partitions=['iid'],seeds=[2025]);args.rounds=2
    view=study.selected(args)[0];cfg=study.effective_config(view);source=study.source()
    folder=tmp_path/view['runId'];folder.mkdir()
    contract=dict(config=cfg,source=source)
    row=dict(status='COMPLETED',config=view,source=source,contract=contract,initialHash='initial')
    (folder/'run.json').write_text(json.dumps(row))
    records=[];previous='initial'
    for cursor in range(8):
        r,node=divmod(cursor,4);trained=f'trained-{cursor}'
        records.append(dict(round=r,node=node,activation=cursor+1,inputHash=previous,
            trainedHash=trained,outboundHash=trained,attackActive=False,configuredByzantine=False,
            selectedConfiguredAttackers=0,cart=None,
            selection=dict(selectedSenders=[] if cursor==0 else [(node-1)%4],selectedCount=1,targetCount=1,
                availableCopies=0 if cursor==0 else 1,bootstrap=cursor==0,screenApplied=False,
                screenAllFailFallback=False,screenGuardFallback=False,ranking=[],
                transferRule='immediate_predecessor',senderRound=None if cursor==0 else r if node else r-1),
            training=dict(optimizerSteps=5,proximalMu=0.,rlaCoefficient=0.,steps=[
                dict(proximalPenalty=0.,gradientSquared=0.,ce=1.,objective=1.) for _ in range(5)])))
        previous=trained
    meta=dict(contract=contract,cursor=8,telemetry=records)
    return tmp_path,view,source,meta


def save(root,view,meta):
    np.savez(root/view['runId']/'checkpoint.npz',metadata=np.frombuffer(json.dumps(meta).encode(),np.uint8))


def test_recorded_chain_has_one_bootstrap_and_wraps_to_immediate_predecessor(recorded):
    root,view,source,meta=recorded;save(root,view,meta)
    before=(root/view['runId']/'checkpoint.npz').read_bytes()
    result=exporter.check_routing(root,view,source)
    assert result['activationsChecked']==8 and result['optimizerStepsChecked']==40
    assert result['predecessorModelChainVerified'] and result['ordinarySGDVerified']
    assert (root/view['runId']/'checkpoint.npz').read_bytes()==before


@pytest.mark.parametrize('change',['input_hash','sender','screen','mu','step_budget','objective','attack'])
def test_bad_recorded_ring_or_training_policy_refuses_export(recorded,change):
    root,view,source,meta=recorded;record=meta['telemetry'][4]
    if change=='input_hash':record['inputHash']='not the preceding trained model'
    elif change=='sender':record['selection']['selectedSenders']=[0]
    elif change=='screen':record['selection']['screenApplied']=True
    elif change=='mu':record['training']['proximalMu']=1.
    elif change=='step_budget':record['training']['optimizerSteps']=4
    elif change=='objective':record['training']['steps'][1]['objective']=1.01
    else:record['attackActive']=True
    save(root,view,meta)
    with pytest.raises(ValueError):exporter.check_routing(root,view,source)


@pytest.mark.parametrize('completed',['plain'],indirect=True)
def test_plain_export_uses_correct_inference_adapter_and_publishes_both_receipts(completed,monkeypatch):
    root,dest=completed
    monkeypatch.setattr(exporter,'check_routing',lambda root,v,source:dict(runId=v['runId'],activationsChecked=50000))
    result=exporter.export(root,dest)
    assert result['configurations']==12 and result['freshInferenceVerified'] and result['plainRingRoutingVerified']
    hashes=json.loads((dest/'SHA256.json').read_text())
    assert {'plain_ring_verification.json','independent_validation.json'}<=set(hashes)
    assert all(exporter.common.sha(dest/n)==h for n,h in hashes.items())
    assert not any(Path(n).name.startswith('checkpoint') or n.endswith('.log') for n in hashes)
    assert exporter.export(root,dest)==result


def test_previous_defense_enabled_clean_export_is_not_plain_ring(completed):
    root,dest=completed
    with pytest.raises(ValueError,match='plain clean ring study'):exporter.export(root,dest)
    assert not dest.exists()
