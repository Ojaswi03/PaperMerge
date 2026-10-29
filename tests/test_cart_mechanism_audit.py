import numpy as np
import pytest
from basil_core.cart_mechanism_audit import metrics_summary,partition_summary,telemetry_summary


def test_metric_and_partition_audit_detects_actual_heterogeneity_and_plateau():
    a=np.full((100,4),.23);a[:20]=.2
    m=metrics_summary(a,np.repeat(a[...,None],10,axis=2))
    assert m['gainAfterRound30']==0 and m['final']==pytest.approx(.23)
    base=dict(partitionHash='x',holdoutHash='same')
    iid=partition_summary(dict(base,classCounts=[[5,5],[5,5]]))
    skew=partition_summary(dict(base,classCounts=[[9,1],[1,9]]))
    assert iid['meanNormalizedClassEntropy']==1 and skew['meanNormalizedClassEntropy']<.5
    with pytest.raises(ValueError):metrics_summary([[float('nan')]])


def test_round_summary_distinguishes_no_check_from_rejected_update():
    records=[]
    for i in range(4):
        records.append(dict(round=i//2,node=i%2,activation=i+1,
            selection=dict(selectedCount=1,continuity=dict(actualRoute='own' if i%2 else 'neighbor',qualifiedClasses=[0,1])),
            training=dict(rlaCoefficient=.1,localTrainingLoss=2.),
            cart=dict(qualityCommit=dict(applied=i>0,scale=0. if i==1 else 1.))))
    rows=telemetry_summary(records,2)
    assert rows[0]['commitChecks']==1 and rows[0]['zeroCommitFractionAmongChecks']==1
    assert rows[1]['commitChecks']==2 and rows[1]['zeroCommitFractionAmongChecks']==0
    assert rows[0]['ownRouteFraction']==.5
    records[0]['activation']=2
    with pytest.raises(ValueError,match='unordered'):telemetry_summary(records,2)
