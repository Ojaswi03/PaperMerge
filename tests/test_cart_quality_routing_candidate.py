import numpy as np
from basil_core.cart_quality_routing_candidate import class_f1,utility


def test_class_f1_detects_constant_class_bias_despite_perfect_recall():
    y=np.repeat(np.arange(10),16);z=np.zeros((len(y),10));z[:,0]=4.
    f1=class_f1(z,y,np.arange(10))
    assert f1.mean()<.02 and np.mean(z[y==0].argmax(axis=1)==0)==1.
    np.testing.assert_array_equal(class_f1(np.eye(10)[y],y,np.arange(10)),np.ones(10))


def test_class_retention_penalizes_losing_a_known_class_without_freezing_other_progress():
    target=np.asarray([.7,.1]);qualified=np.asarray([True,False]);allowance=np.zeros(2)
    degraded=utility(np.asarray([.4,.4]),np.asarray([.4,.4]),target,qualified,allowance)
    preserved=utility(np.asarray([.7,.1]),np.asarray([.7,.1]),target,qualified,allowance)
    improved=utility(np.asarray([.6,.8]),np.asarray([.6,.8]),target,qualified,allowance)
    assert degraded<preserved<improved
    assert utility([.4,.4],[.4,.4],target,qualified,allowance,penalty=False)==.4


def test_router_can_skip_a_harmful_predecessor_without_attack_labels():
    from basil_core.cart_quality_routing_candidate import selector
    from tests.test_cart_fresh_continuity_candidate import setup
    from tests.test_cart_continuity_candidate import Worker
    cfg,state,x,y,screen=setup();cfg['classes']=2;worker=Worker();state['outbound'][0]=[-3.,-3.]
    screen['availableCopies']=2
    selected,row=selector(lambda *_:(np.asarray([-2.,-2.]),screen),'class')(cfg,state,worker,x,y,1,1,None)
    assert row['selectedSenders']==[3] and row['selectedCount']==1
    np.testing.assert_array_equal(selected,state['outbound'][3])
    assert row['continuity']['selectedScore']>=row['continuity']['carrierScore']-.01
    assert worker.admitted_has_channel


def test_low_quality_peers_do_not_replace_a_discriminative_local_model():
    from basil_core.cart_quality_routing_candidate import selector
    from tests.test_cart_fresh_continuity_candidate import setup
    from tests.test_cart_continuity_candidate import Worker
    cfg,state,x,y,screen=setup();cfg['classes']=2;worker=Worker();state['outbound'][:]=[-3.,-3.];screen['availableCopies']=2
    value,row=selector(lambda *_:(np.asarray([-2.,-2.]),screen),'class')(cfg,state,worker,x,y,1,1,None)
    assert row['continuity']['actualRoute']=='own' and not worker.admitted_has_channel
    np.testing.assert_array_equal(value,state['nodeParams'][1])
