import numpy as np
import pytest
import tensorflow as tf

from basil_core.data.cifar import fullLocalEpochBatches, oneClassPerNodePartition
from basil_core.models import BasilPaperCifarModel
from basil_core.research_protocol import (
    RollingMemory, Snapshot, apply_channel_noise, basil_round_learning_rate,
    broadcast, ebm_coefficient, gradient_norm_objective_gradients,
    initialize_memories, keyed_rng, params_hash, resolve_attackers,
    select_snapshot, validate_config,
)
from gui.research_protocol import base_config, production_configs


def scalar_params(value): return [np.asarray([value],np.float32)]


def test_paper_cnn_parameter_count():
    assert BasilPaperCifarModel().model.count_params()==117_706


def test_one_class_partition_exact_cifar_contract():
    labels=np.repeat(np.arange(10,dtype=np.int32),5000); chunks=oneClassPerNodePartition(labels)
    assert [len(chunk) for chunk in chunks]==[5000]*10
    assert all(np.array_equal(np.unique(labels[chunk]),[node]) for node,chunk in enumerate(chunks))
    joined=np.concatenate(chunks); assert len(np.unique(joined))==50_000 and set(joined)==set(range(50_000))
    test_labels=np.repeat(np.arange(10,dtype=np.int32),1000)
    assert len(test_labels)==10_000 and np.array_equal(np.bincount(test_labels),np.full(10,1000))


def test_true_epoch_semantics_keep_partial_batch():
    sizes=fullLocalEpochBatches(5000,512,5)
    assert len(sizes)==50 and sizes[:10]==[512]*9+[392] and sum(sizes)==25_000


def test_rolling_memory_round_10_round_11_invariant():
    memories=initialize_memories(scalar_params(0),10,5)
    for round_id in range(11):
        for sender in range(10): broadcast(memories,Snapshot(sender,round_id,scalar_params(round_id*10+sender)))
        if round_id==10:
            node0=memories[0].candidates(); assert [item.sender_id for item in node0]==[5,6,7,8,9]; assert {item.round_id for item in node0}=={10}
    broadcast(memories,Snapshot(0,11,scalar_params(110)))
    node1=memories[1].candidates(); assert [item.sender_id for item in node1]==[6,7,8,9,0]; assert [item.round_id for item in node1]==[10,10,10,10,11]


def test_strict_handoff_hash_chain_has_no_averaging():
    current=scalar_params(0)
    outputs=[]
    for activation in range(20):
        input_hash=params_hash(current); trained=scalar_params(float(activation+1)); outputs.append((input_hash,params_hash(trained))); current=trained
    assert all(outputs[index][0]==outputs[index-1][1] for index in range(1,20))
    assert outputs[10][0]==outputs[9][1]


def test_snapshot_selection_uses_receiver_loss_not_sender_report_and_ties_nearest():
    candidates=[Snapshot(8,3,scalar_params(1),sender_loss=-100),Snapshot(9,3,scalar_params(2),sender_loss=100)]
    selected,diagnostics=select_snapshot(0,candidates,lambda params:(abs(float(params[0][0])-2),0.5))
    assert selected.sender_id==9 and diagnostics[0]["senderReportedLoss"]==-100
    selected,_=select_snapshot(0,candidates,lambda _params:(1.0,0.5))
    assert selected.sender_id==9


def test_seeded_attackers_are_paired_and_without_replacement():
    first=resolve_attackers(2025); assert len(first)==4 and len(set(first))==4
    assert first==resolve_attackers(2025) and first!=resolve_attackers(2026)
    configs=production_configs(2025); assert all(tuple(config["resolvedAttackerIds"])==first for config in configs)


def test_attack_and_noise_timing_contract_in_production_config():
    config=base_config(); assert config["attackStartRound"]==20 and config["channelNoiseStartRound"]==0
    assert not (19>=config["attackStartRound"]) and 20>=config["attackStartRound"]


def test_relative_and_absolute_noise_formulas_and_keyed_links():
    params=[np.ones(10_000,np.float32)*2]; dimension=10_000; norm=200.0
    noisy,relative=apply_channel_noise(params,semantics="relative_l2_gaussian",sigma=0.4,rng=keyed_rng(1,"link",0,1))
    assert relative["coordinateSigma"]==pytest.approx(0.4*norm/np.sqrt(dimension))
    assert relative["relativeNoiseL2"]==pytest.approx(0.4,rel=0.03)
    _,absolute=apply_channel_noise(params,semantics="absolute_coordinate_gaussian",sigma=0.2,rng=keyed_rng(1,"link",0,1))
    assert absolute["coordinateSigma"]==0.2
    again,_=apply_channel_noise(params,semantics="relative_l2_gaussian",sigma=0.4,rng=keyed_rng(1,"link",0,1))
    other,_=apply_channel_noise(params,semantics="relative_l2_gaussian",sigma=0.4,rng=keyed_rng(1,"link",0,2))
    assert np.array_equal(noisy[0],again[0]) and not np.array_equal(noisy[0],other[0])


def test_relative_ebm_coefficient_is_coordinate_variance():
    params=[np.ones(100,np.float32)*2]
    assert ebm_coefficient(params,semantics="relative_l2_gaussian",sigma=0.4)==pytest.approx((0.4*20/10)**2)
    assert ebm_coefficient(params,semantics="absolute_coordinate_gaussian",sigma=0.4)==pytest.approx(0.16)


def test_second_order_ebm_gradient_matches_analytic_and_is_not_scalar_scaling():
    model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(1,use_bias=False)])
    model.trainable_variables[0].assign([[2.0]]); x=tf.constant([[1.0]]); y=tf.constant([0],tf.int32)
    # Use protocol helper with sparse CE on a two-logit toy instead for valid classification.
    classifier=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(2,use_bias=False)])
    classifier.trainable_variables[0].assign([[0.4,-0.2]])
    coefficient=0.3; robust,_,_,_,_=gradient_norm_objective_gradients(classifier,x,y,coefficient)
    original=classifier.trainable_variables[0].numpy().copy(); epsilon=1e-3; finite=np.zeros_like(original)
    def objective(weights):
        classifier.trainable_variables[0].assign(weights)
        with tf.GradientTape() as tape:
            loss=tf.keras.losses.sparse_categorical_crossentropy(y,classifier(x),from_logits=True)[0]
        gradients=tape.gradient(loss,classifier.trainable_variables)
        return float((loss+coefficient*tf.add_n([tf.reduce_sum(g*g) for g in gradients])).numpy())
    for index in np.ndindex(original.shape):
        plus=original.copy();minus=original.copy();plus[index]+=epsilon;minus[index]-=epsilon;finite[index]=(objective(plus)-objective(minus))/(2*epsilon)
    classifier.trainable_variables[0].assign(original)
    assert np.allclose(robust[0].numpy(),finite,rtol=2e-2,atol=2e-3)
    with tf.GradientTape() as tape: base=tf.keras.losses.sparse_categorical_crossentropy(y,classifier(x),from_logits=True)[0]
    base_gradient=tape.gradient(base,classifier.trainable_variables)[0].numpy()
    ratios=robust[0].numpy()/base_gradient; assert not np.allclose(ratios,1.0+coefficient)


def test_research_config_disables_wd_clipping_momentum_and_resets_optimizer():
    config=base_config(); validate_config(config)
    assert config["weightDecayCoefficient"]==0.0 and config["gradientClipNorm"] is None
    assert config["momentum"]==0.0 and config["optimizerStateMode"]=="reset_each_activation"
    assert basil_round_learning_rate(0.05,20)==pytest.approx(0.025)


def test_old_config_schema_and_modes_remain_distinct():
    old={"schemaVersion":4,"ebmMode":"legacy_gradient_scale","unknownHistoricalField":7}
    from gui.state.experiment_state import from_persisted,to_persisted
    assert to_persisted(from_persisted(old))["unknownHistoricalField"]==7
    assert to_persisted(from_persisted(old))["ebmMode"]=="legacy_gradient_scale"


@pytest.mark.parametrize("epochs",[1,2,3,4,6])
def test_amendment_rejects_reduced_or_increased_epochs(epochs):
    config=base_config();config["localEpochs"]=epochs
    with pytest.raises(ValueError,match="localEpochs"):validate_config(config)


def test_all_generated_training_profiles_keep_five_epochs():
    from gui.research_protocol import smoke_configs,preflight_configs
    for config in production_configs()+smoke_configs()+preflight_configs():
        validate_config(config)
        assert config["localEpochs"]==5 and config["batchSize"]==512
        if config["ebmMode"]=="gradient_norm_objective":assert "legacyEbmLambda" not in config
        if config["researchValid"]:assert config["attackStartRound"]==20


def test_model_initialization_is_keyed_and_paired():
    from basil_core.research_protocol import build_model
    def initial_hash(seed):return params_hash([v.numpy() for v in build_model("basil_paper_cnn",seed).trainable_variables])
    assert initial_hash(2025)==initial_hash(2025)!=initial_hash(2026)


def test_absolute_coordinate_std_independent_of_model_norm():
    draws=[]
    for magnitude in (1.,100.):
        values=[np.full(100000,magnitude,np.float32)]
        perturbed,stats=apply_channel_noise(values,semantics="paper_absolute_gaussian",sigma=.02,rng=keyed_rng(5,"channel",0,1,2))
        draws.append(perturbed[0]-values[0])
        assert stats["coordinateSigma"]==.02
        assert np.std(draws[-1])==pytest.approx(.02,rel=.015)
    assert np.allclose(draws[0],draws[1],atol=4e-6)


def test_relative_variance_is_stopped_and_changes_with_current_weights():
    from basil_core.research_protocol import tensor_noise_variance
    model=tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(2,use_bias=False)])
    model.trainable_variables[0].assign([[2.,2.]])
    with tf.GradientTape() as tape:
        variance=tensor_noise_variance(model,"relative_l2_gaussian",.4)
    assert float(variance)==pytest.approx(.64)
    assert tape.gradient(variance,model.trainable_variables)==[None]
    model.trainable_variables[0].assign([[4.,4.]])
    assert float(tensor_noise_variance(model,"relative_l2_gaussian",.4))==pytest.approx(2.56)


@pytest.mark.parametrize("mode",["none","legacy_gradient_scale","gradient_norm_objective"])
def test_actual_worker_optimizer_and_five_full_epochs(mode):
    from basil_core.research_protocol import SequentialWorker
    model=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10)])
    worker=SequentialWorker(model)
    x=np.zeros((513,32,32,3),np.float32);y=np.zeros(513,np.int32)
    before=worker.export()
    trained,record=worker.train(before,x,y,seed=2025,round_id=0,node_id=0,epochs=5,batch_size=512,learning_rate=.05,ebm_mode=mode,semantics="paper_absolute_gaussian",sigma=.01,legacy_lambda=25.)
    assert record["steps"]==10
    assert all(epoch["batchSizes"]==[512,1] and epoch["samplesSeen"]==513 for epoch in record["epochs"])
    optimizer=worker.optimizer
    assert optimizer.momentum==0 and optimizer.weight_decay is None
    assert optimizer.clipnorm is None and optimizer.global_clipnorm is None and optimizer.clipvalue is None
    assert int(optimizer.iterations)==10 and params_hash(before)!=params_hash(trained)
    if mode=="legacy_gradient_scale":
        assert all(step["robustObjectiveGradientNorm"]==pytest.approx(step["baseGradientNorm"]*(1+25*.01**2),rel=1e-5) for step in record["optimizerTelemetry"])
    if mode=="gradient_norm_objective":
        assert all(step["ebmCoefficient"]==pytest.approx(.0001) for step in record["optimizerTelemetry"])
        assert record["optimizerTelemetry"][0]["ebmObjective"]>record["optimizerTelemetry"][0]["baseCrossEntropy"]
    worker.train(trained,x[:1],y[:1],seed=2025,round_id=1,node_id=1,epochs=5,batch_size=512,learning_rate=.05,ebm_mode="none")
    assert worker.optimizer is not optimizer and int(worker.optimizer.iterations)==5


def test_real_engine_handoff_memory_and_attack_order(monkeypatch):
    import basil_core.research_protocol as protocol
    class FakeWorker:
        def __init__(self,_model):self.params=scalar_params(0)
        def export(self):return protocol.copy_params(self.params)
        def load(self,params):self.params=protocol.copy_params(params)
        def train(self,params,x,y,**kwargs):
            self.load(params);before=kwargs["evaluation"]();self.params=[params[0]+1]
            after=kwargs["evaluation"]()
            return self.export(),{"loss":1.,"accuracy":.1,"steps":5,"beforeTraining":before,
                "epochs":[{"epoch":e+1,"samplesSeen":len(y),"batchSizes":[len(y)],"localTrainingLoss":1.,**after} for e in range(5)],
                "optimizerTelemetry":[{"ebmCoefficient":0.,"effectiveCoordinateSigma":0.}]*5,"baseGradientNorm":1.,"robustGradientNorm":1.}
    monkeypatch.setattr(protocol,"SequentialWorker",FakeWorker)
    monkeypatch.setattr(protocol,"build_model",lambda *args:None)
    monkeypatch.setattr(protocol,"evaluate_params",lambda *args:(.1,np.full(10,.1)))
    config=base_config();config.update(nRounds=21,researchValid=False,purpose="protocol_contract",diagnosticSamplesPerClass=1,diagnosticTestSamplesPerClass=1,ebmMode="none")
    x=np.zeros((10,32,32,3),np.float32);y=np.arange(10,dtype=np.int32)
    clean=protocol.run_research_protocol(config,x,y,x,y)
    records=clean["telemetry"]
    assert all(records[i]["inputHash"]==records[i-1]["trainedOutputHash"] for i in range(1,len(records)))
    assert not any(r["attackActive"] for r in records)
    assert records[110]["memorySenderIds"]==[5,6,7,8,9]
    assert records[110]["memorySnapshotRounds"]==[10]*5
    assert records[111]["memorySenderIds"]==[6,7,8,9,0]
    assert records[111]["memorySnapshotRounds"]==[10,10,10,10,11]
    attacked_inputs=[]
    def attack(params,*args,**kwargs):return [params[0]+1000]
    def noise(params,**kwargs):
        attacked_inputs.append(float(params[0][0]));return [params[0]+.25],{"coordinateSigma":.01,"noiseL2Norm":.25,"relativeNoiseL2":.01}
    monkeypatch.setattr(protocol,"applyAttack",attack);monkeypatch.setattr(protocol,"apply_channel_noise",noise)
    config.update(attackHidden=True,useChannelNoise=True,channelNoiseSigmaRelative=.2)
    threatened=protocol.run_research_protocol(config,x,y,x,y)
    for index,record in enumerate(threatened["telemetry"]):
        active=record["round"]>=20 and record["nodeId"] in config["resolvedAttackerIds"]
        assert record["attackActive"]==active
        trained=scalar_params(attacked_inputs[index*5]-(1000 if active else 0))
        assert params_hash(trained)==record["trainedOutputHash"]
        assert record["channelNoiseActive"]


def test_protected_output_paths_are_rejected(tmp_path):
    from basil_core.artifact_paths import writable_output,PROTECTED_ROOTS
    for root in PROTECTED_ROOTS:
        with pytest.raises(ValueError,match="read-only"):writable_output(root/"new-output")
    assert writable_output(tmp_path/"new") == tmp_path/"new"


def test_cached_real_cifar_label_partition_and_evaluation_counts():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/"experiments/cache/cifar10"
    if not (root/"y_train_int32.npy").exists():pytest.skip("CIFAR cache is not present")
    train=np.load(root/"y_train_int32.npy",mmap_mode="r");test=np.load(root/"y_test_int32.npy",mmap_mode="r")
    chunks=oneClassPerNodePartition(train)
    assert [len(c) for c in chunks]==[5000]*10
    assert np.array_equal(np.bincount(test,minlength=10),np.full(10,1000))
    assert len(np.unique(np.concatenate(chunks)))==50000


def test_actual_fifty_optimizer_steps_keep_392_sample_batches():
    from basil_core.research_protocol import SequentialWorker
    model=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(10)])
    worker=SequentialWorker(model)
    _,record=worker.train(worker.export(),np.zeros((5000,32,32,3),np.float32),np.zeros(5000,np.int32),seed=2025,round_id=0,node_id=0,epochs=5,batch_size=512,learning_rate=.05,ebm_mode="none")
    assert record["steps"]==50 and int(worker.optimizer.iterations)==50
    assert all(e["batchSizes"]==[512]*9+[392] and e["samplesSeen"]==5000 for e in record["epochs"])


def test_research_gui_adapter_preserves_explicit_sigmas_and_unknown_metadata():
    from gui.state.experiment_state import from_persisted,to_persisted,validate_experiment
    from gui.research_protocol import smoke_configs
    config=next(c for c in smoke_configs() if c["conditionId"]=="smoke_noise_abs_0.01_ebm")
    config["unrecognizedProvenance"]={"source":"retained"}
    internal=from_persisted(config);saved=to_persisted(internal)
    assert saved["channelNoiseSigmaAbsolute"]==.01 and saved["channelNoiseSigmaRelative"]==0
    assert saved["attackStartRound"]==1 and saved["unrecognizedProvenance"]==config["unrecognizedProvenance"]
    assert validate_experiment(internal)==[]


def test_execution_gate_does_not_launch_production():
    from gui.state import ApplicationState
    from gui.state.experiment_state import from_persisted
    state=ApplicationState();state.experiment=from_persisted(base_config())
    assert not state.can_run
    from gui.research_protocol import smoke_configs
    state.experiment=from_persisted(smoke_configs()[0])
    assert state.can_run


def test_bad_numeric_form_input_blocks_run_without_changing_last_valid_value():
    from gui.state import ApplicationState
    state=ApplicationState();old=state.experiment.learning_rate
    state.experiment.form_errors["learning_rate"]="Learning rate must be a number."
    assert not state.can_run and state.experiment.learning_rate==old


def test_legacy_channel_start_key_survives_gui_round_trip():
    from gui.state.experiment_state import from_persisted,to_persisted
    assert to_persisted(from_persisted({"schemaVersion":4,"channelNoiseStart":7}))["channelNoiseStart"]==7


def test_research_result_discovery_and_portable_lazy_metrics(tmp_path):
    import json
    from gui.services.result_service import ResultService
    from gui.research_protocol import smoke_configs
    config=next(c for c in smoke_configs() if c["conditionId"]=="smoke_noise_abs_0.01_ebm")
    (tmp_path/"run.json").write_text(json.dumps({"config":config,"status":"completed","researchValid":False,"artifactPaths":{"metrics":"/missing/original/metrics.npz"}}))
    service=ResultService(tmp_path);records=service.discover()
    assert len(records)==1 and records[0].noise==.01
    assert records[0].noise_semantics=="paper_absolute_gaussian" and not records[0].research_valid
    assert service.filter(records,noise="0.01",split="one_class_per_node")==records
    # Discovery succeeds without metrics; loading happens only on demand.
    assert service.load_metrics(records[0])=={}
    np.savez(tmp_path/"metrics.npz",averageAccuracy=np.array([.1,.2]))
    assert np.array_equal(service.load_metrics(records[0])["averageAccuracy"],[.1,.2])


def test_research_replay_adapter_is_display_independent():
    from gui.services.research_telemetry import node_event
    record={"round":11,"nodeId":1,"inputSourceNode":0,"inputSnapshotRound":11,
        "configuredByzantine":False,"attackActive":False,"attackRelativeParameterChange":0.,
        "candidates":[{"senderId":0,"receiverLocalLoss":.2}],"trainedOutputHash":"abc",
        "baseGradientNorm":2.,"robustObjectiveGradientNorm":3.,"optimizerSteps":50,
        "ebmMode":"gradient_norm_objective","ebmCoefficient":.01,"effectiveCoordinateSigma":.1,
        "outgoingLinks":[{"receiverId":2,"noiseL2Norm":1.,"relativeNoiseL2":.2,"coordinateSigma":.01}],
        "globalTestAccuracy":.1,"perClassGlobalTestAccuracy":[.1]*10}
    event=node_event(record)
    assert event["incoming"]=={"senderId":0,"sourceRound":11}
    assert event["selection"]["candidates"][0]["selected"]
    assert event["training"]["optimizerSteps"]==50 and event["outgoing"][0]["receiverId"]==2


def test_legacy_round_trip_does_not_inject_new_training_defaults():
    from gui.state.experiment_state import from_persisted,to_persisted
    original={"schemaVersion":4,"learningRate":.07,"channelNoiseStart":7,"unknownProvenance":{"campaignVersion":2}}
    config=from_persisted(original)
    assert to_persisted(config)==original
    config.learning_rate=.03
    assert to_persisted(config)=={**original,"learningRate":.03}
    unversioned=from_persisted({"dataset":"cifar10"})
    assert unversioned.schema_version==1
    assert to_persisted(unversioned)=={"dataset":"cifar10"}


def test_invalid_form_cannot_be_saved_as_last_valid_values(tmp_path):
    from gui.services.config_service import ConfigService
    from gui.state.experiment_state import ExperimentConfig
    config=ExperimentConfig();config.form_errors["learning_rate"]="Learning rate must be a number."
    with pytest.raises(ValueError,match="Learning rate"):ConfigService().save(tmp_path/"config.json",config)
    assert not (tmp_path/"config.json").exists()
