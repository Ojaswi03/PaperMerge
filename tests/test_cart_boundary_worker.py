"""The activation transaction may explore but only commits feasible outputs."""
import pytest


@pytest.mark.parametrize('restore_input,risk_metric',[(True,'cross_entropy'),(False,'cross_entropy'),(False,'classification_error')])
def test_boundary_commits_are_measured_and_do_not_claim_step_constraints(restore_input,risk_metric):
    import numpy as np
    from basil_core.cart_boundary_worker import BoundaryCartWorker
    from basil_core.cart_feasible_memory import within_envelope
    from basil_core.research_protocol import params_hash
    from tests.test_factorial_worker import small_model
    worker=BoundaryCartWorker(small_model(),restore_input=restore_input,risk_metric=risk_metric)
    x=np.random.default_rng(91).uniform(0,1,(23,32,32,3)).astype(np.float32)
    y=np.arange(23,dtype=np.int32)%3
    options=dict(seed=2025,node_id=0,epochs=5,batch_size=128,learning_rate=.1,
        ebm_mode='gradient_norm_objective',semantics='paper_absolute_gaussian',sigma=.01)
    first,initial=worker.train(worker.export(),x,y,round_id=0,**options)
    assert not initial['cartMemory']['guardEnabled']
    damaged=[v.copy() for v in first];damaged[-1][0]+=30.
    final,row=worker.train(damaged,x,y,round_id=1,**options)
    evidence=row['cartMemory']
    assert row['steps']==5 and evidence['guardMode']=='activation_boundary'
    assert evidence['stepGuards']==[]
    assert evidence['riskMetric']==risk_metric
    assert evidence['inputRestorationEnabled']==restore_input
    if restore_input:assert evidence['inputBridge']['scale']<1.
    else:assert evidence['inputReferenceHash']==params_hash(damaged)
    assert within_envelope(evidence['outputBridge']['classLosses'],evidence['activeBounds'])
    assert params_hash(final)==evidence['committedOutputHash']
    np.testing.assert_allclose(worker.measure_params(final),evidence['outputBridge']['classLosses'],rtol=0,atol=1e-7)
    for step in row['optimizerTelemetry']:
        assert step['ebmCoefficient']==pytest.approx(.0001)
        assert 'cartAcceptedStepScale' not in step
    state=worker.export_local_memory()
    worker.restore_local_memory(state)
    with pytest.raises(InterruptedError):
        worker.train(final,x,y,round_id=2,should_stop=lambda:True,**options)
    assert worker.memory['nodes']['0']['referenceHash']==state['nodes']['0']['referenceHash']
