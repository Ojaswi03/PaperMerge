"""Display-independent contracts for the paired IID study."""
import numpy as np
import pytest
import tensorflow as tf

from basil_core.iid_study import (
    IidWorker, paired_configs, partition_audit, preflight_config, verify_pair, iid_output,
)
from basil_core.research_protocol import (
    IID_PROTOCOL, PROTOCOL, build_model, iid_partition_indices, keyed_rng,
    gradient_norm_objective_gradients, validate_config, resolve_attackers,
    initialize_memories, broadcast, Snapshot, select_snapshot, apply_channel_noise,
)
from basil_core.protocol_compatibility import normalize_config, is_research_protocol
from basil_core.trainer import lossFn


def test_existing_iid_split_coverage_histograms_and_stream_equivalence():
    labels = np.repeat(np.arange(10),5000)
    audit,indices = partition_audit(labels,np.repeat(np.arange(10),1000))
    assert audit["assignedSamples"]==audit["uniqueSamples"]==50000
    assert all(row["samples"]==5000 and all(350<n<650 for n in row["classCounts"]) for row in audit["nodes"])
    old = np.array_split(keyed_rng(2025,"iid_partition").permutation(50000),10)
    assert all(np.array_equal(a,b) for a,b in zip(old,indices))
    assert all(np.array_equal(a,b) for a,b in zip(indices,iid_partition_indices(labels,2025)))
    assert audit["testClassCounts"]==[1000]*10


def test_two_configs_and_unresolved_sigma_gate():
    clean,noisy = paired_configs()
    validate_config(clean)
    with pytest.raises(ValueError,match="sigma_e"): validate_config(noisy)
    assert clean["experimentProtocol"]==noisy["experimentProtocol"]==IID_PROTOCOL
    assert normalize_config(clean)["experimentProtocol"]==IID_PROTOCOL
    assert is_research_protocol(clean["experimentProtocol"]) and PROTOCOL!=IID_PROTOCOL
    assert clean["resolvedAttackerIds"]==noisy["resolvedAttackerIds"]==list(resolve_attackers(2025))
    clean,noisy = paired_configs(.01)
    validate_config(noisy)
    assert clean["localObjective"]=="cross_entropy" and not clean["useChannelNoise"] and clean["ebmMode"]=="none"
    assert noisy["localObjective"]=="source_ebm" and noisy["useChannelNoise"]
    assert len(verify_pair(clean,noisy))<=8
    noisy["batchSize"]=256
    with pytest.raises(ValueError,match="Unpaired"): verify_pair(clean,noisy)


@pytest.mark.parametrize("field,value",[("snapshotAnchorMu",.01),("monteCarloNoiseSamples",2),
    ("gradientClipNorm",1.),("momentum",.1),("weightDecayCoefficient",.01),("localEpochs",3)])
def test_protocol_rejects_unapproved_changes(field,value):
    config = paired_configs(.01)[1]; config[field]=value
    with pytest.raises(ValueError): validate_config(config)


def test_preflights_preserve_epochs_and_separate_smoke_attack():
    config = paired_configs(.01)[1]
    full = preflight_config(config)
    smoke = preflight_config(config,smoke=True)
    validate_config(full); validate_config(smoke)
    assert full["attackStartRound"]==20 and not full["researchValid"]
    assert not any(k.startswith("diagnostic") for k in full)
    assert smoke["attackStartRound"]==1 and smoke["localEpochs"]==full["localEpochs"]==5


def test_memory_rollover_and_receiver_only_selection():
    memories = initialize_memories([np.array([0.],np.float32)])
    for r in range(11):
        for sender in range(10): broadcast(memories,Snapshot(sender,r,[np.array([sender],np.float32)]))
    assert [(s.sender_id,s.round_id) for s in memories[0].candidates()]==[(i,10) for i in range(5,10)]
    broadcast(memories,Snapshot(0,11,[np.array([0.],np.float32)]))
    assert [(s.sender_id,s.round_id) for s in memories[1].candidates()]==[(i,10) for i in range(6,10)]+[(0,11)]
    candidates = memories[0].candidates()
    for s in candidates: s.sender_loss=-999 if s.sender_id==5 else 999
    evaluated=[]
    def receiver_only(params): evaluated.append(params); return 1.,.5
    selected,_ = select_snapshot(0,candidates,receiver_only)
    assert selected.sender_id==9 and len(evaluated)==5


def test_full_epoch_batch_contract():
    sizes = [len(np.arange(5000)[s:s+512]) for s in range(0,5000,512)]
    assert sizes==[512]*9+[392] and sum(sizes)==5000 and len(sizes)*5==50


def test_absolute_noise_is_not_model_relative():
    a=[np.ones(100000,dtype=np.float32)]; b=[a[0]*100]
    na,sa=apply_channel_noise(a,semantics="paper_absolute_gaussian",sigma=.01,rng=keyed_rng(2025,"channel_noise",0,0,1))
    nb,sb=apply_channel_noise(b,semantics="paper_absolute_gaussian",sigma=.01,rng=keyed_rng(2025,"channel_noise",0,0,1))
    assert sa["coordinateSigma"]==sb["coordinateSigma"]==.01
    assert abs(np.std(na[0]-a[0])-.01)<.0001
    assert sa["noiseHash"]==sb["noiseHash"]
    _,different=apply_channel_noise(a,semantics="paper_absolute_gaussian",sigma=.01,rng=keyed_rng(2025,"channel_noise",0,0,2))
    assert different["noiseHash"]!=sa["noiseHash"]


def tiny_model(shape=(2,)):
    return tf.keras.Sequential([tf.keras.layers.Input(shape=shape),tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=13))])


def test_ebm_matches_hessian_vector_product_and_objective():
    model=tiny_model(); x=tf.constant([[.3,-.2],[.1,.6]]); y=tf.constant([0,1]); c=.02**2
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce=lossFn(y,model(x))
        g=inner.gradient(ce,model.trainable_variables)
        norm=tf.add_n([tf.reduce_sum(v*v) for v in g])
    correction=outer.gradient(norm,model.trainable_variables)
    robust,ce2,norm2,_,_=gradient_norm_objective_gradients(model,x,y,c)
    for expected,base,actual in zip(correction,g,robust):
        np.testing.assert_allclose(actual,base+c*expected,rtol=2e-6,atol=2e-7)
    assert float(ce2+c*norm2)==pytest.approx(float(ce+c*norm),rel=1e-6)
    # The correction is not a scalar learning-rate multiplier.
    assert any(not np.allclose(r,(1+c)*base,rtol=1e-6,atol=1e-7) for r,base in zip(robust,g))
    # Independent finite-difference check of CE + c||grad CE||².
    variable=model.trainable_variables[0]; original=variable.numpy(); delta=.001
    def objective():
        _,ce,norm,_,_=gradient_norm_objective_gradients(model,x,y,c)
        return float(ce+c*norm)
    changed=original.copy(); changed[0,0]+=delta; variable.assign(changed); plus=objective()
    changed=original.copy(); changed[0,0]-=delta; variable.assign(changed); minus=objective()
    variable.assign(original)
    assert float(robust[0][0,0])==pytest.approx((plus-minus)/(2*delta),rel=.005,abs=2e-4)


def test_no_lambda_loss_gradients_updates_and_strict_load():
    model=tiny_model((32,32,3)); worker=IidWorker(model)
    params=worker.export(); x=tf.ones([2,32,32,3])*.2; y=tf.constant([0,1]); key=tf.constant([1,2])
    outcomes=[]
    for legacy in (1.,999999.):
        worker.load(params)
        grads,values=worker._gradients(x,y,key,"gradient_norm_objective","paper_absolute_gaussian",tf.constant(.01),tf.constant(legacy))
        optimizer=tf.keras.optimizers.SGD(.05,momentum=0.)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        outcomes.append(([g.numpy() for g in grads],[float(v) for v in values],worker.export()))
    for a,b in zip(outcomes[0][0]+outcomes[0][2],outcomes[1][0]+outcomes[1][2]): np.testing.assert_array_equal(a,b)
    assert outcomes[0][1]==outcomes[1][1]
    worker.load(params)
    for a,b in zip(worker.export(),params): np.testing.assert_array_equal(a,b)
    final,record=worker.train(params,x.numpy(),y.numpy(),seed=2025,round_id=0,node_id=0,
        epochs=5,batch_size=512,learning_rate=.05,ebm_mode="none",semantics="paper_absolute_gaussian")
    assert record["steps"]==5 and all(e["samplesSeen"]==2 for e in record["epochs"])
    assert all(s["ebmCoefficient"]==s["ebmPenalty"]==s["ebmCorrectionNorm"]==0 for s in record["optimizerTelemetry"])
    assert int(worker.optimizer.iterations)==5 and float(worker.optimizer.momentum)==0
    assert worker.optimizer.weight_decay is None and worker.optimizer.clipnorm is None


def test_nonfinite_parameters_are_rejected():
    worker=IidWorker(tiny_model((32,32,3)))
    worker.model.trainable_variables[0].assign(tf.fill(worker.model.trainable_variables[0].shape,float("nan")))
    with pytest.raises(tf.errors.InvalidArgumentError,match="Non-finite"):
        worker._gradients(tf.zeros([1,32,32,3]),tf.constant([0]),tf.constant([1,2]),
            "none","paper_absolute_gaussian",tf.constant(0.),tf.constant(0.))


def test_output_routing_rejects_non_iid_and_historical_paths():
    assert iid_output("newResults/IID/comparison").parts[-1]=="comparison"
    assert iid_output("newPlots/IID/comparison",plots=True).parts[-1]=="comparison"
    for path in ("newResults/nonIID/test","experiments/results4/test","plots4/test","../escape"):
        with pytest.raises(ValueError): iid_output(path)


def test_iid_engine_receiver_batch_handoff_attack_and_lambda_pairing(monkeypatch):
    import basil_core.research_protocol as protocol
    inputs=[]; receiver_batches=[]; lambdas=[]; sent=[]
    class Worker:
        def __init__(self,model): self.params=[np.array([0.],np.float32)]
        def export(self): return protocol.copy_params(self.params)
        def probe(self,params,x,y):
            receiver_batches.append((x.copy(),y.copy()))
            return float(params[0][0]**2),.1
        def train(self,params,x,y,**kw):
            inputs.append(protocol.params_hash(params)); lambdas.append(kw["legacy_lambda"])
            self.params=[params[0]+1]
            ev=kw["evaluation"]()
            coefficient=kw["sigma"]**2 if kw["ebm_mode"]=="gradient_norm_objective" else 0.
            return self.export(),dict(loss=1.,accuracy=.1,steps=5,beforeTraining=ev,
                epochs=[dict(epoch=i+1,localTrainingLoss=1.,**ev) for i in range(5)],
                optimizerTelemetry=[dict(ebmCoefficient=coefficient,effectiveCoordinateSigma=kw["sigma"])],
                baseGradientNorm=1.,robustGradientNorm=1.)
    monkeypatch.setattr(protocol,"build_model",lambda *a:None)
    monkeypatch.setattr(protocol,"evaluate_params",lambda *a:(.1,np.full(10,.1)))
    def attack(params,*a,**kw): return [params[0]+1000]
    monkeypatch.setattr(protocol,"applyAttack",attack)
    def channel(params,**kw):
        sent.append(params[0].copy())
        return [params[0]+.25],dict(coordinateSigma=kw["sigma"],noiseL2Norm=.25,relativeNoiseL2=.01)
    monkeypatch.setattr(protocol,"apply_channel_noise",channel)
    config=preflight_config(paired_configs(.01)[1],rounds=21)
    config.update(diagnosticSamplesPerClass=1,diagnosticTestSamplesPerClass=1,legacyEbmLambda=999999.)
    x=np.arange(10,dtype=np.float32).reshape(10,1); y=np.arange(10,dtype=np.int32)
    result=protocol.run_research_protocol(config,x,y,x,y,worker_factory=Worker)
    records=result["telemetry"]
    assert inputs==[r["inputHash"] for r in records] and set(lambdas)=={0.}
    for offset in range(0,len(receiver_batches),5):
        for actual in receiver_batches[offset+1:offset+5]:
            for a,b in zip(actual,receiver_batches[offset]): np.testing.assert_array_equal(a,b)
    assert records[110]["memorySenderIds"]==[5,6,7,8,9]
    assert records[110]["memorySnapshotRounds"]==[10]*5
    assert records[111]["memorySnapshotRounds"]==[10,10,10,10,11]
    for index,r in enumerate(records):
        active=r["nodeId"] in config["resolvedAttackerIds"] and r["round"]>=20
        assert r["attackActive"]==active
        honest=[sent[index*5]-(1000 if active else 0)]
        assert protocol.params_hash(honest)==r["trainedOutputHash"]
        assert r["channelNoiseActive"] and r["legacyLambdaIgnored"]


def test_iid_config_roundtrip_keeps_identity_and_unknown_fields(tmp_path):
    from gui.services.config_service import ConfigService
    from gui.state.experiment_state import from_persisted,to_persisted
    config=paired_configs(.01)[1]; config["additionalMetadata"]={"retained":7}
    service=ConfigService(); path=tmp_path/"iid.json"
    service.save(path,from_persisted(config)); loaded=to_persisted(service.load(path))
    assert loaded["experimentProtocol"]==IID_PROTOCOL and loaded["additionalMetadata"]==config["additionalMetadata"]
    validate_config(loaded)


def test_actual_paper_cnn_full_ebm_has_finite_correction_and_no_lambda_effect():
    worker=IidWorker(build_model("basil_paper_cnn",2025))
    params=worker.export()
    assert sum(p.size for p in params)==117706
    x=tf.constant(keyed_rng(2025,"objective_contract_fixture").normal(0,.2,(2,32,32,3)),tf.float32)
    y=tf.constant([0,1]); key=tf.constant([1,2]); outputs=[]
    for legacy in (1.,999999.):
        worker.load(params)
        gradients,values=worker._gradients(x,y,key,"gradient_norm_objective",
            "paper_absolute_gaussian",tf.constant(.01),tf.constant(legacy))
        assert all(np.isfinite(g.numpy()).all() for g in gradients)
        assert all(np.isfinite(float(v)) for v in values)
        assert float(values[3])==pytest.approx(.01**2,rel=1e-6)
        assert float(values[6])>0
        outputs.append(([g.numpy() for g in gradients],[float(v) for v in values]))
    for a,b in zip(outputs[0][0],outputs[1][0]): np.testing.assert_array_equal(a,b)
    assert outputs[0][1]==outputs[1][1]


def test_authorized_fifty_round_iid_pair_preserves_protocol():
    clean,noisy=paired_configs(.010,rounds=50)
    validate_config(clean);validate_config(noisy);verify_pair(clean,noisy)
    assert clean["researchValid"] and noisy["researchValid"]
    assert clean["roundEvaluationMetrics"] and noisy["roundEvaluationMetrics"]
    assert clean["nRounds"]==noisy["nRounds"]==50
    assert noisy["channelNoiseSigmaAbsolute"]**2==.0001
    assert clean["localEpochs"]==noisy["localEpochs"]==5


def test_evaluation_ce_is_measured_without_parameter_mutation():
    from basil_core.research_protocol import evaluate_params,params_hash
    worker=IidWorker(tiny_model((32,32,3))); params=worker.export()
    x=np.zeros((10,32,32,3),np.float32);y=np.arange(10,dtype=np.int32)
    accuracy,per_class,ce=evaluate_params(worker,params,x,y,batch_size=4,details=True)
    assert accuracy==pytest.approx(.1) and ce==pytest.approx(np.log(10),rel=1e-6)
    assert len(per_class)==10 and params_hash(worker.export())==params_hash(params)


def test_pre_post_attack_analysis_uses_distinct_periods():
    from reporting.iid_study_plots import attack_analysis
    curve=np.concatenate([np.full(20,.4),np.linspace(.3,.5,30)])
    report=attack_analysis(curve)
    assert report["preAttackMean"]==pytest.approx(.4)
    assert report["round19"]==.4 and report["round20"]==.3
    assert report["maxDropFromRound19"]==pytest.approx(.1)
    assert report["final"]==.5 and report["postAttackBest"]==.5
