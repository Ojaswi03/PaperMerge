import numpy as np
import pytest
from basil_core.cart_admitted_continuity_candidate import AdmittedWorker


@pytest.mark.parametrize('route,expected',[('own',0.),('fresh',.012),('pool',.012),('bootstrap',0.)])
def test_admitted_anchor_and_effective_noise_objective_match_explicit_sgd(route,expected):
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_study import configuration
    cfg=configuration('mnist','iid',2025,'attack_noise_defense_rla_cart',.01,nodes=4,local_steps=2)
    old=SharpWorker(cfg);new=AdmittedWorker(cfg)
    flat=old.export();new.admitted_route=route
    if route=='bootstrap':new.admitted_has_channel=False
    new.continuity_anchor=flat+np.float32(.3) # Rejected peer must be discarded.
    x=np.random.default_rng(94).uniform(size=(8,28,28,1)).astype(np.float32);y=np.arange(8,dtype=np.int32)
    correct,correct_record=old.train(flat,x,y,0,0,expected,np.arange(8))
    actual,row=new.train(flat,x,y,0,0,.012,np.arange(8))
    np.testing.assert_array_equal(actual,correct)
    assert row['rlaCoefficient']==expected and row['steps']==correct_record['steps']
    assert row['steps'][0]['proximalPenalty']==0.
    assert row['linkRlaSuppressedForLocalCarrier']==(route=='own')
