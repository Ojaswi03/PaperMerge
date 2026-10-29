import numpy as np
from basil_core.cart_peer_candidates import eligible_senders
from basil_core.cart_prior_neutral_routing import selector


def fixture():
    cfg=dict(nodes=10,window=6,seed=2025,cartProbePerClass=16,sigma=0.,calibrationDraws=19)
    state=dict(cursor=10,senderRounds=np.zeros(10,np.int32),nodeParams=np.arange(10,dtype=np.float32)[:,None],
        outbound=np.arange(10,dtype=np.float32)[:,None])
    base=dict(screenApplied=True,peakRatios=[2.]*6,peakThreshold=1.,screenGuardFallback=True,
        screenAllFailFallback=True,selectedSenders=[9],selectedCount=1,availableCopies=6)
    return cfg,state,base


def test_all_failed_screen_keeps_historical_control_and_explicitly_restores_guard_window():
    cfg,state,base=fixture()
    old,old_record=eligible_senders(cfg,state,0,base)
    new,new_record=eligible_senders(cfg,state,0,base,guard_fallback=True)
    assert old==[] and not old_record['peerFallbackApplied']
    assert new==[9,8,7,6,5,4] and new_record['peerFallbackApplied']
    assert new_record['screenPassingPeerCount']==0


def test_partial_survivor_guard_and_normal_screen_have_distinct_eligibility():
    cfg,state,base=fixture();base['peakRatios'][0]=0.;base['screenAllFailFallback']=False
    assert eligible_senders(cfg,state,0,base)[0]==[9]
    assert len(eligible_senders(cfg,state,0,base,guard_fallback=True)[0])==6
    base['screenGuardFallback']=False
    assert eligible_senders(cfg,state,0,base,guard_fallback=True)[0]==[9]
    base['screenApplied']=False
    assert len(eligible_senders(cfg,state,0,base)[0])==6


def test_fallback_arm_restores_coherent_candidates_without_bypassing_private_gate():
    cfg,state,base=fixture();y=np.tile(np.arange(3),20);x=y[:,None].astype(np.float32)
    def pooled(*args):return np.array([5.],np.float32),dict(base)
    class Worker:
        def outputs(self,flat,images):return np.eye(3)[images[:,0].astype(int)]
    for policy,count in [(False,0),(True,6)]:
        worker=Worker();_,row=selector(pooled,'class',guard_fallback=policy)(cfg,state,worker,x,y,0,1,np.arange(20))
        c=row['continuity']
        assert sum(v['kind']=='neighbor' for v in c['candidates'])==count
        assert c['qualityScoreBoundApplied'] and c['peerFallbackApplied']==policy
        assert c['selectedScore']>=c['carrierScore']-.01
