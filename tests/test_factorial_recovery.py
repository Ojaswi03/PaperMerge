"""Real end-to-end CPU updates, attacks, transport and CART exact recovery."""
import copy


def test_factorial_cart_checkpoint_resume_matches_uninterrupted(monkeypatch,tmp_path):
    import numpy as np
    import tensorflow as tf
    import basil_core.research_protocol as protocol
    from basil_core.factorial_study import configuration,CONDITIONS
    from basil_core.factorial_worker import make_worker
    from basil_core.protocol_checkpoint import save_checkpoint,load_checkpoint,TelemetryStore
    from tests.test_factorial_worker import small_model
    cfg=configuration(CONDITIONS[-1],'dirichlet',2025,.01,stage='smoke',device='CPU')
    monkeypatch.setattr(protocol,'build_model',lambda *_:small_model())
    x=np.random.default_rng(74).uniform(0,1,(640,32,32,3)).astype(np.float32)
    y=np.repeat(np.arange(10,dtype=np.int32),64)
    vx,vy=x[:20],y[:20]
    initial=protocol.params_hash(small_model().get_weights())
    def run(**kw):
        return protocol.run_research_protocol(cfg,x,y,vx,vy,
            worker_factory=lambda model:make_worker(model,cfg),**kw)
    uninterrupted=run()
    saved=[]
    def checkpoint(current):saved[:]=[copy.deepcopy(current)]
    try:run(checkpoint_callback=checkpoint,should_stop=lambda:bool(saved and saved[0]['next_activation']>=2))
    except InterruptedError:pass
    assert saved[0]['next_activation']==2
    path=tmp_path/'checkpoint.npz'
    save_checkpoint(path,cfg,saved[0],partition_hash='a'*64,initial_hash=initial,telemetry_store=TelemetryStore(path))
    restored=load_checkpoint(path,cfg,partition_hash='a'*64,initial_hash=initial)
    assert restored['cart_memory']==saved[0]['cart_memory']
    resumed=run(resume_state=restored,checkpoint_callback=checkpoint)
    for name in ('roundNodeAccuracy','selectedSources','selectedSnapshotRounds'):
        np.testing.assert_array_equal(uninterrupted[name],resumed[name])
    for a,b in zip(uninterrupted['finalNodeParams'],resumed['finalNodeParams']):
        for first,second in zip(a,b):np.testing.assert_array_equal(first,second)
    assert uninterrupted['telemetry']==resumed['telemetry']
    assert sum(r['attackActive'] for r in resumed['telemetry'])==4
    assert all(r['channelNoiseActive'] for r in resumed['telemetry'])
    assert all(r['ebmCoefficient']>0 for r in resumed['telemetry'])
    assert all(len(r['candidates'])==5 for r in resumed['telemetry'])
    assert all(r['cartMemory']['beta']==1. for r in resumed['telemetry'])
    from scripts.run_cart_memory_smoke import save_json
    from scripts.report_factorial_study import verify_checkpoint
    from basil_core.protocol_checkpoint import checkpoint_implementation
    save_checkpoint(path,cfg,saved[0],partition_hash='a'*64,initial_hash=initial,telemetry_store=TelemetryStore(path))
    save_json(tmp_path/'partition.json',dict(hash='a'*64,initialModelHash=initial,
        counts=[len(c) for c in resumed['partitionIndices']]))
    verify_checkpoint(tmp_path,cfg,checkpoint_implementation(cfg),resumed['roundNodeAccuracy'])
