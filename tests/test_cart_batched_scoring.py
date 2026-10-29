import numpy as np
import pytest


@pytest.mark.parametrize('dataset',['mnist','cifar10'])
def test_sequential_logits_batch_has_identical_losses_and_guarded_selection(dataset):
    from basil_core.cart_batched_scoring import BatchedAdmittedWorker,select_input
    from basil_core.sharp_guarded_protocol import select_input as original,guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration
    from basil_core.sharp_study import configuration as base
    from basil_core.sharp_protocol import initial_state
    cfg=configuration(guarded_configuration(base(dataset,'iid',2025,'attack_noise_defense_rla_cart',.01,nodes=4)))
    worker=BatchedAdmittedWorker(cfg);initial=worker.export();rng=np.random.default_rng(17)
    shape=(28,28,1) if dataset=='mnist' else (32,32,3)
    x=rng.uniform(size=(17,*shape)).astype(np.float32);y=np.arange(17,dtype=np.int32)%10
    flats=initial[None]+rng.normal(0,.001,size=(6,worker.dimension)).astype(np.float32)
    expected=np.asarray([worker.probe(flat,x,y) for flat in flats]);actual=worker.score_many(flats,x,y)
    np.testing.assert_array_equal(actual,expected)
    state=initial_state(cfg,initial);state['outbound'][:]=flats[:4];state['senderRounds'][:]=0
    a,ar=original(cfg,state,worker,x,y,3,1,np.arange(len(y)))
    b,br=select_input(cfg,state,worker,x,y,3,1,np.arange(len(y)))
    np.testing.assert_array_equal(a,b);assert ar==br
