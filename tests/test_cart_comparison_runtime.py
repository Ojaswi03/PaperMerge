"""Numerical protocol checks, run on CPU without reading benchmark TEST data."""
import copy
import os
import numpy as np
import pytest
from tests.test_acds_comparison import arguments


@pytest.fixture(scope='module',autouse=True)
def cpu():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


def case(arm='cart',environment='joint'):
    from basil_core.cart_comparison_study import selected
    return selected(arguments(arms=[arm],environments=[environment],datasets=['mnist'],partitions=['iid'],
        seeds=[2025],nodes=4,rounds=2,local_steps=1,attack_start=1))[1][0]


def data():
    x=np.random.default_rng(918).uniform(size=(80,28,28,1)).astype(np.float32)
    y=np.tile(np.arange(10,dtype=np.int32),8)
    return x,y,[np.arange(i*20,(i+1)*20) for i in range(4)]


def test_full_cart_adapter_reproduces_original_revision_exactly():
    from basil_core.cart_comparison_study import configure_worker,intervention
    from basil_core.cart_revision_study import configure_worker as original_worker
    import basil_core.sharp_cart_v2_protocol as protocol
    import basil_core.sharp_cart_v2 as repair
    from basil_core.cart_prior_neutral_commit import transport
    view=case();cfg=view['comparisonCase']['protocolConfig'];x,y,chunks=data()
    old,wrap=original_worker(cfg,'CPU');new=configure_worker(view,'CPU')
    np.testing.assert_array_equal(old.export(),new.export())
    original_select,original_repair=protocol.select_input,repair.transport
    try:
        protocol.select_input=wrap(original_select);repair.transport=transport(original_repair)
        expected=protocol.run(cfg,old,x,y,chunks,x[:20],y[:20])
    finally:protocol.select_input=original_select;repair.transport=original_repair
    with intervention(view,new) as effective:actual=protocol.run(effective,new,x,y,chunks,x[:20],y[:20])
    np.testing.assert_array_equal(expected['nodeParams'],actual['nodeParams'])
    np.testing.assert_array_equal(expected['outbound'],actual['outbound'])
    assert expected['roundMetrics']==actual['roundMetrics']


@pytest.mark.parametrize('arm',['macro','balanced_commit','prior_cart'])
def test_comparison_interruption_recovery_preserves_full_model_and_metrics(tmp_path,arm):
    from basil_core.cart_comparison_study import configure_worker,intervention
    from basil_core.sharp_protocol import save_state,load_state
    import basil_core.sharp_cart_v2_protocol as protocol
    view=case(arm);x,y,chunks=data();complete=configure_worker(view,'CPU')
    with intervention(view,complete) as cfg:expected=protocol.run(cfg,complete,x,y,chunks,x[:20],y[:20])
    interrupted=configure_worker(view,'CPU')
    with intervention(view,interrupted) as cfg:
        partial=protocol.run(cfg,interrupted,x,y,chunks,x[:20],y[:20],max_activations=3)
    contract=dict(config=cfg,fixture='resume comparison')
    path=tmp_path/'checkpoint.npz';save_state(path,partial,contract)
    resumed=configure_worker(view,'CPU');state=load_state(path,contract,resumed.dimension)
    with intervention(view,resumed) as cfg:actual=protocol.run(cfg,resumed,x,y,chunks,x[:20],y[:20],state=state)
    np.testing.assert_array_equal(expected['nodeParams'],actual['nodeParams'])
    np.testing.assert_array_equal(expected['outbound'],actual['outbound'])
    assert expected['roundMetrics']==actual['roundMetrics']
    assert [(r['activation'],r['trainedHash']) for r in expected['telemetry']]==[(r['activation'],r['trainedHash']) for r in actual['telemetry']]


def test_prior_adjustment_is_training_only_and_preserves_the_sgd_budget():
    from basil_core.cart_comparison_study import configure_worker
    from basil_core.cart_local_prior import log_prior
    view=case('prior_cart');worker=configure_worker(view,'CPU')
    x,_,_=data();y=np.array([0]*70+[1]*10,np.int32);flat=worker.export()
    before=worker.outputs(flat,x)
    worker.admitted_route='own';worker.admitted_has_channel=False
    trained,record=worker.train(flat,x,y,0,0,.001,np.arange(20))
    assert record['optimizerSteps']==view['localSteps'] and record['rlaCoefficient']==0
    np.testing.assert_array_equal(record['trainingLogPrior'],log_prior(y,10).astype(np.float64))
    np.testing.assert_array_equal(before,worker.outputs(flat,x))
    assert not np.array_equal(trained,flat)


def test_local_only_does_not_benefit_from_or_consume_peer_transmissions():
    from basil_core.cart_comparison_study import configure_worker,intervention
    import basil_core.sharp_cart_v2_protocol as protocol
    x,y,chunks=data();results=[]
    for env in ('clean','joint'):
        view=case('local_only',env);worker=configure_worker(view,'CPU')
        with intervention(view,worker) as cfg:result=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20])
        assert all(not r['selection']['selectedSenders'] for r in result['telemetry'])
        assert all(r['training']['rlaCoefficient']==0 for r in result['telemetry'])
        results.append(result)
    np.testing.assert_array_equal(results[0]['nodeParams'],results[1]['nodeParams'])
    # Honest-only means average different identity subsets in the two
    # environments, even though every trained client model is identical.
    for clean,attacked in zip(results[0]['roundMetrics'],results[1]['roundMetrics']):
        assert {k:v for k,v in clean.items() if k!='honestMeanAccuracy'}=={
            k:v for k,v in attacked.items() if k!='honestMeanAccuracy'}
