import copy
import json

import pytest

from scripts.audit_sharp_selection import analyze,audit,text_report


def history(*,attacker=0,fallback=False):
    cfg=dict(nodes=4,window=2,attackStart=1,dataset='mnist',partition='iid',seed=2025,
        condition='attack_defense',sigma=0.,runId='fixture')
    records=[];visited=set()
    for index in range(8):
        r,node=divmod(index,4);available=[(node-d)%4 for d in (1,2) if (node-d)%4 in visited]
        selected=available[:1]
        selection=dict(selectedSenders=selected,selectedCount=1,availableCopies=len(available),
            screenApplied=False,screenAllFailFallback=False,ranking=[])
        if index==5:
            selection.update(screenApplied=True,peakRatios=[50.,50.] if fallback else [1.,50.],
                peakThreshold=10.,screenAllFailFallback=fallback,
                selectedSenders=[3] if fallback else [0],ranking=[{'sender':0,'loss':2.3},{'sender':3,'loss':.1}] if fallback else [{'sender':0,'loss':2.3}])
        records.append(dict(activation=index+1,round=r,node=node,attackActive=node==attacker and r>=1,
            configuredByzantine=node==attacker,selection=selection))
        visited.add(node)
    metrics=[dict(round=0,meanAccuracy=.9),dict(round=1,meanAccuracy=.1)]
    return cfg,records,metrics


def test_audit_distinguishes_direct_attack_and_inherited_exposure():
    cfg,records,metrics=history();before=copy.deepcopy(records)
    result=analyze(cfg,records,metrics)
    assert result[0]['selectedDirectAttackFraction']==0
    assert result[1]['selectedDirectlyAttackedCopies']==1
    assert result[1]['inputsWithAttackAncestry']==3
    assert result[1]['screenedUnattackedAvailable']==1 and result[1]['unattackedRetainedFraction']==0
    assert result[1]['screenedAttackedAvailable']==1 and result[1]['attackedRetainedFraction']==1
    first=result[1]['firstDirectAdmission']
    assert first['activation']==6 and first['retainedUnattacked']==0 and first['retainedAttacked']==1
    assert records==before


def test_all_fail_fallback_is_not_misreported_as_raw_screen_success():
    cfg,records,metrics=history(fallback=True)
    row=analyze(cfg,records,metrics)[1]
    assert row['allFailFallbacks']==1 and row['unattackedRawPassFraction']==0
    assert row['attackedRawPassFraction']==0 and row['unattackedRetainedFraction']==1
    assert row['attackedRetainedFraction']==1


def test_before_outbound_attack_sender_labels_do_not_count_as_corrupted_versions():
    cfg,records,metrics=history(attacker=3,fallback=True)
    rows=analyze(cfg,records,metrics)
    assert rows[0]['selectedDirectlyAttackedCopies']==rows[1]['selectedDirectlyAttackedCopies']==0
    assert rows[1]['inputsWithAttackAncestry']==0


def test_invalid_sender_history_is_rejected():
    cfg,records,metrics=history();records[0]['selection']['selectedSenders']=[3]
    with pytest.raises(ValueError,match='historical sender'):analyze(cfg,records,metrics)


def test_checkpoint_audit_is_read_only_and_agrees_with_raw_history(tmp_path):
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    cfg,records,metrics=history();source={'test':'same'}
    (tmp_path/'plan.json').write_text(json.dumps(dict(configs=[cfg],source=source)))
    folder=tmp_path/'fixture';folder.mkdir()
    (folder/'run.json').write_text(json.dumps(dict(status='RUNNING',completedActivations=9)))
    contract=dict(config=cfg,source=source,data={'TRAIN':'fixture'},partition={'partitionHash':'fixture'})
    metadata=dict(contract=contract,cursor=8,telemetry=records,roundMetrics=metrics)
    np.savez(folder/'checkpoint.npz',metadata=np.frombuffer(json.dumps(metadata).encode(),np.uint8))
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    reports=audit(tmp_path)
    assert reports[0]['rounds']==analyze(cfg,records,metrics)
    assert reports[0]['checkpointActivations']==8 and reports[0]['liveReportedActivations']==9
    assert 'RUNNING' in text_report(reports) and 'SelectedAttack%' in text_report(reports)
    assert {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before


def test_v1_counterexample_sparse_benign_drift_can_lose_to_dense_hidden_attack():
    # Documents a limitation of the frozen v1 rule, not a desired future defense.
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    from basil_core.sharp_math import peak_screen
    from basil_core.attacks import applyAttack
    base=np.full(1024,.1,np.float32)
    benign=np.repeat(base[None],66,axis=0)
    benign[np.arange(66),np.arange(66)]+=.001
    _,before=peak_screen(benign,full_window=True)
    assert before['screenAllFailFallback']  # All copies are benign here.
    mixed=benign.copy()
    mixed[-1]=applyAttack([base],'hidden',rng=np.random.default_rng(8))[0]
    retained,after=peak_screen(mixed,full_window=True)
    assert retained.tolist()==[65] and not after['screenAllFailFallback']


@pytest.mark.parametrize('method',['sharp_guarded','sharp_cart_v2'])
def test_versioned_survivor_guard_and_cart_activity_are_auditable(method):
    from scripts.audit_sharp_selection import phases
    cfg,records,metrics=history();cfg['method']=method
    selection=records[5]['selection']
    selection.update(peakRatios=[50.,1.],screenGuardFallback=True)
    records[5].update(cart=dict(referenceUsed=True,classGates=[.2,0.],scale=.5,referenceAdmitted=True),
        training=dict(localTrainingLoss=1.2,rlaCoefficient=.003))
    rows=analyze(cfg,records,metrics);last=rows[-1]
    assert last['cartAcceptedCorrections']==1 and last['cartGateActivations']==1
    assert last['localTrainCEMean']==1.2 and last['rlaCoefficientsMean']==.003
    result=phases(cfg,rows)
    assert result['lastBeforeAttack']['meanAccuracy']==.9
    assert result['firstAfterAttack']['meanAccuracy']==.1
    assert result['attackStartsDisplayRound']==2 and result['cartAcceptedCorrections']==1
