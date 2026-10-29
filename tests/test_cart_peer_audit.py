import numpy as np
import pytest
from basil_core.cart_peer_audit import decompose,cohort,summarize
from basil_core.cart_quality_routing_candidate import utility


def test_score_decomposition_matches_actual_cart_and_exposes_penalty():
    f=np.array([.5,.1]);k=np.array([.4,.2]);t=np.array([.6,.2]);q=np.array([True,False]);a=np.array([.05,0])
    r=decompose(f,k,t,q,a)
    assert r['score']==pytest.approx(utility(f,k,t,q,a))
    assert r['macro']==pytest.approx(.3) and r['penalty']==pytest.approx(.075)
    with pytest.raises(ValueError):decompose(f,k,t,q,[-.1,0])


def test_full_window_distinguishes_screening_shortlist_and_link_damage():
    own=dict(score=.3,macro=.3)
    def peer(sender,near,passed,received,pre):
        return dict(sender=sender,nearEight=near,screenPassed=passed,
            received=dict(score=received,macro=received,penalty=0),preLink=dict(score=pre,macro=pre,penalty=0))
    peers=[peer(1,True,True,.1,.2),peer(2,False,True,.35,.4),peer(3,False,False,.5,.6)]
    groups={mode:{name:cohort(peers,own,mode,screened=screened,near=near)
        for name,screened,near in [('nearEight',True,True),('screenedWindow',True,False),('wholeWindow',False,False)]}
        for mode in ('received','preLink')}
    assert groups['received']['nearEight']['bestScoreMargin']==pytest.approx(-.2)
    assert groups['received']['screenedWindow']['bestSender']==2
    assert groups['received']['wholeWindow']['bestSender']==3
    r=summarize([dict(cohorts=groups)])
    assert r['received_nearEight']['peerBeatsOwned']==0
    assert r['received_screenedWindow']['peerBeatsOwned']==1
    assert r['preLink_screenedWindow']['scoreMarginQuantiles'][2]==pytest.approx(.1)


@pytest.mark.parametrize('budget,expected',[(8,8),(66,65)])
def test_wider_candidate_budget_keeps_the_same_screen_and_logs_score_components(budget,expected):
    from basil_core.cart_prior_neutral_routing import selector
    y=np.tile(np.arange(10),4);x=y[:,None].astype(np.float32)
    cfg=dict(seed=2025,nodes=100,window=66,cartProbePerClass=16,sigma=0.,calibrationDraws=19)
    state=dict(cursor=100,nodeParams=np.arange(100,dtype=np.float32)[:,None],
        outbound=np.arange(100,dtype=np.float32)[:,None],senderRounds=np.zeros(100,dtype=np.int32))
    ratios=[2.]+[0.]*65
    def pooled(*args):
        return np.array([99.],np.float32),dict(selectedSenders=[99],selectedCount=1,availableCopies=66,
            screenApplied=True,peakRatios=ratios,peakThreshold=1.)
    class Worker:
        def outputs(self,flat,images):return np.eye(10)[images[:,0].astype(int)]
    _,row=selector(pooled,'class',peer_budget=budget)(cfg,state,Worker(),x,y,0,1,np.arange(20))
    candidates=row['continuity']['candidates'];peers=[c for c in candidates if c['kind']=='neighbor']
    assert len(peers)==expected and all(c['senders']!=[99] for c in peers)
    assert row['continuity']['nearPeerBudget']==budget
    assert all(c['score']==pytest.approx(c['macroScore']-c['classPenalty']) for c in candidates)
    with pytest.raises(ValueError):selector(pooled,'class',peer_budget=0)
