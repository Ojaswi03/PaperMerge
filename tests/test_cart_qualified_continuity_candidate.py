import numpy as np
from basil_core.cart_qualified_continuity_candidate import qualified_classes,selector
from tests.test_cart_fresh_continuity_candidate import setup
from tests.test_cart_continuity_candidate import Worker


def test_uniform_model_is_not_knowledge_to_protect():
    y=np.repeat(np.arange(10),16);z=np.zeros((len(y),10))
    assert qualified_classes(z,y,z,y,10)==[]
    z[np.arange(len(y)),y]=2.
    assert qualified_classes(z,y,z,y,10)==list(range(10))
    assert qualified_classes(z[:7],y[:7],z,y,10)==[]


def test_qualification_requires_both_probes():
    y=np.repeat(np.arange(2),16);z=3*np.eye(2)[y]
    assert qualified_classes(z,y,np.zeros_like(z),y,2)==[]


def test_bootstrap_allows_fresh_learning_without_a_macro_veto():
    cfg,state,x,y,evidence=setup();cfg['classes']=2;worker=Worker()
    value,row=selector(lambda *_:(np.ones(2),evidence),'qualified')(cfg,state,worker,x,y,1,1,None)
    np.testing.assert_array_equal(value,state['outbound'][0])
    assert row['continuity']['learnedClasses']==[] and row['continuity']['actualRoute']=='fresh'


def test_learned_class_damage_restores_local_carrier():
    cfg,state,_,_,evidence=setup();cfg['classes']=2;worker=Worker();worker.cfg=dict(worker.cfg,cartProbePerClass=16)
    cfg['cartProbePerClass']=16
    state['nodeParams'][1]=[3.,3.];state['outbound'][0]=[.1,.1]
    y=np.repeat(np.arange(2),32);x=y[:,None]
    out,row=selector(lambda *_:(np.ones(2),evidence),'qualified')(cfg,state,worker,x,y,1,1,None)
    np.testing.assert_array_equal(out,state['nodeParams'][1])
    assert row['continuity']['qualifiedInitializationBound'] and row['selectedSenders']==[]


def test_passed_fresh_route_skips_unused_pool_loss_ranking():
    cfg,state,x,y,evidence=setup();cfg['classes']=2;worker=Worker()
    def unused(*_):raise AssertionError('Unused peer pool should not be scored.')
    _,row=selector(unused,'always')(cfg,state,worker,x,y,1,1,None)
    assert row['poolRankingSkipped'] and row['selectedCount']==1
