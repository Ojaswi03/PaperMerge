"""Full ring/checkpoint replay with the new memory schema and real CPU SGD."""
import copy
import pytest


@pytest.mark.parametrize('variant,objective',[('none','raw_ce'),('budgeted','raw_ce'),('coherent','raw_ce'),
    ('none','local_prior'),('coherent','local_prior'),('coherent_refresh','local_prior'),
    ('coherent_commit_ce','local_prior'),('coherent_commit_error','local_prior'),('coherent_qualified_error','local_prior'),
    ('coherent_projected_error','local_prior'),('coherent_head_transport','local_prior')])
def test_coherent_pilot_exact_resume_and_actual_partition_identity(monkeypatch,tmp_path,variant,objective):
    import numpy as np
    import basil_core.research_protocol as protocol
    from basil_core.cart_coherent_study import configuration,make_worker
    from basil_core.protocol_checkpoint import save_checkpoint,load_checkpoint,TelemetryStore
    from scripts.run_cart_coherent_pilot import partition_identity
    from tests.test_factorial_worker import small_model
    monkeypatch.setattr(protocol,'build_model',lambda *_:small_model())
    x=np.random.default_rng(72).uniform(0,1,(640,32,32,3)).astype(np.float32)
    y=np.repeat(np.arange(10,dtype=np.int32),64)
    initial=protocol.params_hash(small_model().get_weights())
    partition=partition_identity(y,'dirichlet',2025,'smoke')
    cfg=configuration('dirichlet',2025,.01,variant,stage='smoke',device='CPU',
        partition_hash=partition,initial_hash=initial,training_objective=objective,
        compiled_optimizer=variant in ('coherent_projected_error','coherent_head_transport'))
    def run(config=cfg,**kwargs):
        return protocol.run_research_protocol(config,x,y,x[:20],y[:20],
            worker_factory=lambda m:make_worker(m,config),**kwargs)
    states=[]
    def checkpoint(state):states[:]=[copy.deepcopy(state)]
    direct=run()
    with pytest.raises(InterruptedError):
        run(checkpoint_callback=checkpoint,should_stop=lambda:bool(states and states[0]['next_activation']>=2))
    path=tmp_path/'checkpoint.npz'
    save_checkpoint(path,cfg,states[0],partition_hash=partition,initial_hash=initial,telemetry_store=TelemetryStore(path))
    restored=load_checkpoint(path,cfg,partition_hash=partition,initial_hash=initial)
    resumed=run(resume_state=restored,checkpoint_callback=checkpoint)
    for name in ('roundNodeAccuracy','selectedSources','selectedSnapshotRounds'):
        np.testing.assert_array_equal(direct[name],resumed[name])
    for a,b in zip(direct['finalNodeParams'],resumed['finalNodeParams']):
        for first,second in zip(a,b):np.testing.assert_array_equal(first,second)
    assert direct['telemetry']==resumed['telemetry']
    if variant in ('coherent_qualified_error','coherent_projected_error','coherent_head_transport'):
        from scripts.run_cart_memory_smoke import save_json
        from scripts.report_factorial_study import verify_checkpoint
        from basil_core.protocol_checkpoint import checkpoint_implementation
        save_checkpoint(path,cfg,states[0],partition_hash=partition,initial_hash=initial,telemetry_store=TelemetryStore(path))
        save_json(tmp_path/'partition.json',dict(hash=partition,initialModelHash=initial,
            counts=[len(c) for c in resumed['partitionIndices']]))
        verify_checkpoint(tmp_path,cfg,checkpoint_implementation(cfg),resumed['roundNodeAccuracy'])
    if variant in ('coherent','coherent_refresh','coherent_commit_ce','coherent_commit_error','coherent_qualified_error','coherent_projected_error'):
        from basil_core.cart_feasible_memory import within_envelope
        assert len(restored['cart_memory']['nodes'])==2
        guarded=[r['cartMemory'] for r in resumed['telemetry'] if r['cartMemory']['guardEnabled']]
        if variant not in ('coherent_qualified_error','coherent_projected_error'):assert len(guarded)==10
        if variant in ('coherent_commit_ce','coherent_commit_error','coherent_qualified_error','coherent_projected_error'):
            assert all(not r['stepGuards'] and within_envelope(r['outputBridge']['classLosses'],r['activeBounds']) for r in guarded)
        else:assert all(within_envelope(s['classLosses'],r['activeBounds']) for r in guarded for s in r['stepGuards'])
    wrong=configuration('dirichlet',2025,.01,variant,stage='smoke',device='CPU',partition_hash='a'*64,initial_hash=initial)
    with pytest.raises(ValueError,match='partition'):run(config=wrong)


def test_declared_prior_has_the_exact_nested_gradient_and_raw_inference():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_coherent_study import configuration,make_worker
    from basil_core.dirichlet_calibration import calibration_offsets,calibrated_anchor_gradients
    from basil_core.research_audit import augment_batch
    from tests.test_factorial_worker import small_model
    x=np.random.default_rng(74).uniform(0,1,(9,32,32,3)).astype(np.float32)
    y=np.array([0,0,0,0,0,0,1,1,2],np.int32)
    cfg=configuration('dirichlet',2025,.01,'none',stage='smoke',device='CPU',training_objective='local_prior')
    worker=make_worker(small_model(),cfg)
    worker.training_offsets.assign(calibration_offsets(np.bincount(y,minlength=10),method='prior_adjustment'))
    key=tf.constant([15,19])
    gradients,values=worker._batch_gradients(x,y,key,'gradient_norm_objective','paper_absolute_gaussian',tf.constant(.01),tf.constant(0.))
    expected,metrics=calibrated_anchor_gradients(worker.model,augment_batch(x,key),y,worker.export(),worker.training_offsets,mu=0.,coefficient=.0001)
    for actual,reference,variable in zip(gradients,expected,worker.model.trainable_variables):
        if len(variable.shape)>1:reference=reference+.0005*variable
        np.testing.assert_allclose(actual.numpy(),reference.numpy(),rtol=2e-5,atol=1e-7)
    assert float(values[0])==pytest.approx(float(metrics[0]),rel=1e-6)
    np.testing.assert_array_equal(worker.predict(x).numpy(),worker.model(x,training=False).numpy())


def test_coherent_default_plan_has_no_tensorflow_or_training_side_effects():
    import json
    import subprocess
    import sys
    result=subprocess.run([sys.executable,'-c',"import runpy,sys;sys.argv=['run_cart_coherent_pilot.py'];runpy.run_module('scripts.run_cart_coherent_pilot',run_name='__main__');assert 'tensorflow' not in sys.modules"],
        capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['runs']==3 and not plan['trainingStarted'] and not plan['noveltyEstablished']
