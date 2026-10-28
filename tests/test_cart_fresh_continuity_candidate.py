import numpy as np
from basil_core.cart_fresh_continuity_candidate import selector
from tests.test_cart_continuity_candidate import Worker


def setup():
    cfg=dict(nodes=4,window=2,seed=2025,sigma=0.,cartProbePerClass=2)
    state=dict(nodeParams=np.ones((4,2),np.float32),senderRounds=np.zeros(4,dtype=int),
        outbound=np.asarray([[3.,3.],[1.,1.],[1.,1.],[1.,1.]],np.float32),cursor=5)
    y=np.asarray([0]*4+[1]*4);x=y[:,None]
    evidence=dict(selectedSenders=[0,3],selectedCount=2,screenApplied=True,
        peakRatios=[0.,0.],peakThreshold=1.)
    return cfg,state,x,y,evidence


def test_latest_hop_must_pass_class_gate_and_is_recorded_as_single_copy():
    cfg,state,x,y,evidence=setup();worker=Worker()
    value,row=selector(lambda *_:(np.ones(2),evidence),'class_noise')(cfg,state,worker,x,y,1,1,None)
    np.testing.assert_array_equal(value,state['outbound'][0])
    assert row['continuity']['actualRoute']=='fresh' and row['selectedCount']==1 and row['selectedSenders']==[0]
    assert row['continuity']['empiricalInitializationBound']
    assert evidence['selectedCount']==2


def test_failed_peak_screen_cannot_be_bypassed_by_fresh_hop():
    cfg,state,x,y,evidence=setup();worker=Worker();evidence['peakRatios'][0]=2.
    value,row=selector(lambda *_:(np.ones(2),evidence),'class_noise')(cfg,state,worker,x,y,1,1,None)
    assert row['continuity']['actualRoute']=='own' and len(row['continuity']['candidateScores'])==1


def test_fresh_macro_ablation_accepts_tradeoff_rejected_by_class_gate():
    cfg,state,x,y,evidence=setup();worker=Worker();state['outbound'][0]=[3.,.8]
    old=lambda *_:(np.ones(2),evidence)
    _,strict=selector(old,'class_noise')(cfg,state,worker,x,y,1,1,None)
    _,macro=selector(old,'macro')(cfg,state,worker,x,y,1,1,None)
    assert strict['continuity']['actualRoute']=='own' and macro['continuity']['actualRoute']=='fresh'
