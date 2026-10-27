"""Cold single-label specialization is not a qualified retention anchor."""
import pytest


def test_above_chance_breadth_requires_multiple_supported_classes():
    from basil_core.cart_reference_qualification import qualified_error_reference
    assert not qualified_error_reference([0.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
    assert not qualified_error_reference([0.])
    assert not qualified_error_reference([.9,.9,.9,.9])
    assert not qualified_error_reference([0.,.2,1.,1.,1.])
    assert qualified_error_reference([0.,.2,.8,1.,1.])
    assert not qualified_error_reference([.8,.8,1.,1.])
    assert qualified_error_reference([.7,.8,1.,1.])
    assert qualified_error_reference([0.,0.])
    for invalid in ([],[1.01],[-.01],[float('nan')],[[.2,.3]]):
        with pytest.raises(ValueError):qualified_error_reference(invalid)


def test_new_breadth_does_not_admit_worse_balanced_risk():
    from basil_core.cart_boundary_worker import BoundaryCartWorker
    worker=object.__new__(BoundaryCartWorker)
    worker.require_breadth=True;worker.risk_metric='classification_error'
    worker.mean_probe_ce=lambda:0.5
    before=dict(referenceMeanCE=1.,envelope=dict(referenceLosses=[0.]*4+[1.]*6,bounds=[1.]*10))
    assert not worker.admission(before,[.6]*5+[1.]*5)
    weak=dict(referenceMeanCE=2.,envelope=dict(referenceLosses=[0.]+[1.]*9,bounds=[1.]*10))
    assert worker.admission(weak,[.1,.1]+[1.]*8)


def test_cold_reference_does_not_block_learning_then_broad_reference_protects():
    import numpy as np
    from basil_core.cart_boundary_worker import BoundaryCartWorker
    from basil_core.cart_feasible_memory import within_envelope
    from basil_core.research_protocol import params_hash
    from tests.test_factorial_worker import small_model
    worker=BoundaryCartWorker(small_model(),restore_input=False,risk_metric='classification_error',require_breadth=True)
    y=np.repeat(np.arange(3,dtype=np.int32),20)
    x=np.zeros((60,32,32,3),np.float32)
    for i,c in enumerate(y):x[i,:,:,c]=1.
    identity=worker.prepare(0,x,y)
    cold=[np.zeros((3,10),np.float32),np.array([3.]+[0.]*9,np.float32)]
    worker.load(cold);worker.admit(0,cold,worker.measure_current(),identity,0,.01)
    assert worker.memory['nodes']['0']['referenceQualified'] is False
    broad=[np.zeros((3,10),np.float32),np.zeros(10,np.float32)]
    for c in range(3):broad[0][c,c]=4.
    options=dict(seed=2025,node_id=0,epochs=5,batch_size=128,learning_rate=.01,
        ebm_mode='gradient_norm_objective',semantics='paper_absolute_gaussian',sigma=.01)
    learned,first=worker.train(broad,x,y,round_id=1,**options)
    assert first['cartMemory']['guardEnabled'] is False
    assert first['cartMemory']['referenceQualified'] is True
    assert first['cartMemory']['inputReferenceHash']==params_hash(broad)
    damaged=[v.copy() for v in learned];damaged[-1][9]+=30.
    _,second=worker.train(damaged,x,y,round_id=2,**options)
    memory=second['cartMemory']
    assert memory['guardEnabled'] and memory['referenceWasQualified']
    assert within_envelope(memory['outputBridge']['classLosses'],memory['activeBounds'])
    worker.restore_local_memory(worker.export_local_memory())
