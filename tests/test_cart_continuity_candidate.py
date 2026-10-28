from types import SimpleNamespace
import numpy as np
import pytest
from basil_core.cart_continuity_candidate import eligible,route,selector


def test_class_constraint_rejects_macro_improvement_that_harms_one_class():
    assert not eligible([.4,1.2],[1.,1.],[.4,1.2],[1.,1.],[0.,0.],[0.,0.])
    assert eligible([.4,1.2],[1.,1.],[.4,1.2],[1.,1.],[0.,.21],[0.,.21])
    assert not eligible([.4,1.2],[1.,1.],[1.1,1.2],[1.,1.],[1.,1.],[1.,1.])


@pytest.mark.parametrize('bad',[[np.nan,1.],[-1.]])
def test_invalid_calibration_rejected(bad):
    with pytest.raises(ValueError):eligible([1.,1.],[2.,2.],[1.,1.],[2.,2.],bad,[0.,0.])


class Worker:
    cfg=dict(seed=2025,cartProbePerClass=2,sigma=0.,calibrationDraws=19)
    def outputs(self,flat,x):
        labels=x[:,0].astype(int);margins=np.asarray(flat)[labels]
        out=np.zeros((len(x),2));out[np.arange(len(x)),labels]=margins
        return out


def test_full_model_routing_and_probe_bound():
    worker=Worker();y=np.asarray([0]*4+[1]*4);x=y[:,None]
    own=np.asarray([1.,1.]);peer=np.asarray([3.,.8])
    out,row=route(worker,peer,own,x,y,0,100)
    np.testing.assert_array_equal(out,own)
    assert row['empiricalInitializationBound'] and row['fitCheckDisjoint']
    out,row=route(worker,peer,own,x,y,0,100,mode='macro')
    np.testing.assert_array_equal(out,peer)
    assert not row['empiricalInitializationBound']
    improved=np.asarray([3.,3.]);out,row=route(worker,improved,own,x,y,0,100)
    np.testing.assert_array_equal(out,improved)
    assert row['calibrationDraws']==0 and row['route']=='peer'


def test_bootstrap_and_unsupported_classes_do_not_claim_bound():
    worker=Worker();y=np.asarray([0,1]);x=y[:,None];peer=np.asarray([2.,2.]);own=np.asarray([1.,1.])
    out,row=route(worker,peer,None,x,y,0,0)
    assert row['reason']=='bootstrap' and not row['empiricalInitializationBound']
    out,row=route(worker,peer,own,x,y,0,100)
    assert row['reason']=='no class supported on both probes' and not row['empiricalInitializationBound']


def test_peer_anchor_preserved_separately_from_carrier_initialization():
    worker=Worker();peer=np.asarray([2.,2.]);own=np.asarray([1.,1.])
    state=dict(nodeParams=[own],senderRounds=[0],cursor=1)
    old=dict(selectedSenders=[7],selectedCount=1)
    value,row=selector(lambda *_:(peer,old),'own')({},state,worker,None,None,0,1,None)
    np.testing.assert_array_equal(value,own);np.testing.assert_array_equal(worker.continuity_anchor,peer)
    assert 'continuity' not in old and row['selectedSenders']==[7]
    assert row['continuity']['route']=='own'


def test_screen_preserves_declared_steps_and_is_explicitly_an_intervention():
    from scripts.probe_cart_continuity import configuration,CASES
    from basil_core.sharp_cart_v2_protocol import validate_config
    args=SimpleNamespace(partition='iid',seed=2025,nodes=100,rounds=5,attack_start=20)
    cases=[configuration(args,i) for i in range(len(CASES))]
    for case in cases:
        validate_config(case['config'])
        assert case['config']['localSteps']==5 and case['diagnosticOnly']
    assert {c['continuityMode'] for c in cases}=={None,'own','macro','class_noise'}


def test_training_keeps_the_same_sgd_when_initialization_equals_peer_anchor():
    from basil_core.sharp_worker import SharpWorker
    from basil_core.cart_continuity_candidate import ContinuityWorker
    from basil_core.sharp_study import configuration
    cfg=configuration('mnist','iid',2025,'attack_noise_defense_rla_cart',.01,nodes=4,local_steps=2)
    original=SharpWorker(cfg);candidate=ContinuityWorker(cfg)
    initial=original.export();candidate.continuity_anchor=initial.copy()
    x=np.random.default_rng(91).uniform(size=(8,28,28,1)).astype(np.float32)
    y=np.arange(8,dtype=np.int32)
    expected,old=original.train(initial,x,y,0,0,.001,np.arange(8))
    actual,new=candidate.train(initial,x,y,0,0,.001,np.arange(8))
    np.testing.assert_array_equal(actual,expected)
    assert {k:new[k] for k in old}==old
    # A different carrier is only an initialization change; the proximal anchor
    # remains the received peer, rather than silently following the carrier.
    shifted=initial+np.float32(.001)
    _,row=candidate.train(shifted,x,y,0,0,.001,np.arange(8))
    assert row['steps'][0]['proximalPenalty']>0
