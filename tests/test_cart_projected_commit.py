"""Measured class repair preserves more of an activation than scalar shrinkage."""
import pytest


def test_projected_repair_checks_true_risk_and_dominates_scalar_distance():
    import numpy as np
    from basil_core.cart_projected_commit import ProjectedCommitWorker
    from basil_core.cart_feasible_memory import within_envelope
    from tests.test_factorial_worker import small_model
    worker=ProjectedCommitWorker(small_model())
    y=np.repeat(np.arange(3,dtype=np.int32),16);x=np.zeros((48,32,32,3),np.float32)
    for i,c in enumerate(y):x[i,:,:,c]=1.
    worker.prepare(0,x,y)
    anchor=[np.zeros((3,10),np.float32),np.zeros(10,np.float32)]
    for c in range(3):anchor[0][c,c]=2.
    worker.load(anchor);bounds=worker.measure_current().tolist()
    assert bounds==[0.,0.,0.]
    proposal=[v.copy() for v in anchor];proposal[0][0,3]+=4.;proposal[0][1,1]+=1.;proposal[0][2,2]+=1.
    final,evidence=worker.repair_proposal(anchor,proposal,bounds)
    assert within_envelope(worker.measure_params(final),bounds)
    p=evidence['projection']
    assert p['unsafeClasses']==[0]
    assert {r['endpoint'] for r in p['gradientRows']}=={'reference','proposal'}
    assert p['selectedSquaredDistance']<=p['scalarSquaredDistance']
    assert p['selectedSquaredDistance']<p['scalarSquaredDistance']-1e-6
    assert 0<=p['proposalFidelity']<=1
    assert p['family']=='projected'
    unchanged,info=worker.repair_proposal(anchor,anchor,bounds)
    for a,b in zip(anchor,unchanged):np.testing.assert_array_equal(a,b)
    assert info['projection']['status']=='not_needed'


def test_full_local_training_uses_the_projection_hook_and_graph_optimizer():
    import numpy as np
    from basil_core.cart_coherent_study import configuration,make_worker
    from basil_core.cart_feasible_memory import within_envelope
    from tests.test_factorial_worker import small_model
    cfg=configuration('dirichlet',2025,.01,'coherent_projected_error',stage='smoke',device='CPU',
        training_objective='local_prior',compiled_optimizer=True)
    worker=make_worker(small_model(),cfg)
    y=np.repeat(np.arange(3,dtype=np.int32),16);x=np.zeros((48,32,32,3),np.float32)
    for i,c in enumerate(y):x[i,:,:,c]=1.
    reference=[np.zeros((3,10),np.float32),np.zeros(10,np.float32)]
    for c in range(3):reference[0][c,c]=2.
    identity=worker.prepare(0,x,y);worker.load(reference)
    worker.admit(0,reference,worker.measure_current(),identity,0,.01)
    incoming=[v.copy() for v in reference];incoming[0][0,3]+=4.
    final,record=worker.train(incoming,x,y,seed=2025,node_id=0,round_id=1,epochs=5,batch_size=128,
        learning_rate=.01,ebm_mode='gradient_norm_objective',semantics='paper_absolute_gaussian',sigma=.01)
    memory=record['cartMemory']
    assert record['steps']==5 and memory['guardEnabled']
    assert 'projection' in memory['outputBridge']
    assert memory['outputBridge']['projection']['family'] in ('scalar','projected')
    assert within_envelope(worker.measure_params(final),memory['activeBounds'])
