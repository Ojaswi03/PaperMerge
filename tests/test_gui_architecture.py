import json
from pathlib import Path
import pytest

from gui.services import ConfigService, ExecutionEvent, ExecutionService, PlotService, QueueService, ResultService
from gui.state import ApplicationState, ExecutionState, QueueState, QueueStatus, Workspace
from gui.state.experiment_state import ExperimentConfig, from_persisted, to_persisted, validate_experiment


def valid_config(**changes):
    config=ExperimentConfig(experiment_name="Test",attacks={name:False for name in ("Gaussian","SignFlip","Hidden","ModelPoison","Scaling","Alie","Ipm","NoiseAmp")})
    for key,value in changes.items(): setattr(config,key,value)
    return config


def test_config_round_trip_preserves_camel_case_and_unknown_fields(tmp_path):
    raw={"schemaVersion":4,"experimentName":"Round trip","dataset":"cifar10","nRounds":12,"customResearchMetadata":{"keep":True}}
    service=ConfigService(); source=tmp_path/"source.json"; target=tmp_path/"target.json"; source.write_text(json.dumps(raw))
    model=service.load(source); service.save(target,model); saved=json.loads(target.read_text())
    assert saved["experimentName"]=="Round trip" and saved["nRounds"]==12
    assert saved["customResearchMetadata"]=={"keep":True}
    assert "experiment_name" not in saved


def test_schema_adapter_rejects_unsupported_version():
    with pytest.raises(ValueError,match="Unsupported schemaVersion"): from_persisted({"schemaVersion":99})


@pytest.mark.parametrize("change,field",[(dict(rounds=0),"rounds"),(dict(learning_rate=-1),"learning_rate"),(dict(batch_size=0),"batch_size")])
def test_numeric_validation(change,field): assert field in {issue.field for issue in validate_experiment(valid_config(**change))}


def test_conditional_and_attack_validation():
    config=valid_config(noise_mitigation="ebm",use_channel_noise=False,snapshot_selection=True,attacker_ids="")
    fields={issue.field for issue in validate_experiment(config)}
    assert {"noise_mitigation","snapshot_selection"} <= fields
    config.attacks["Hidden"]=True
    assert "attacker_ids" in {issue.field for issue in validate_experiment(config)}


def test_queue_transitions_reorder_and_retry():
    state=QueueState(); first=state.add(valid_config(experiment_name="A")); second=state.add(valid_config(experiment_name="B"))
    state.move(second.entry_id,-1); assert state.entries[0] is second
    state.transition(first.entry_id,QueueStatus.RUNNING); state.transition(first.entry_id,QueueStatus.FAILED); state.retry(first.entry_id)
    assert first.status is QueueStatus.PENDING
    with pytest.raises(ValueError): state.transition(first.entry_id,QueueStatus.COMPLETED)


def test_queue_persistence_accepts_legacy_and_enriched_format(tmp_path):
    path=tmp_path/"queue.json"; path.write_text(json.dumps([to_persisted(valid_config())])); service=QueueService(path); state=service.load(); assert len(state.entries)==1
    service.save(state); assert json.loads(path.read_text())[0]["experimentName"]=="Test"


def test_execution_state_and_event_dispatch():
    state=ApplicationState(experiment=valid_config()); launched=[]
    service=ExecutionService(state,launcher=lambda config: launched.append(config) or object()); service.start(to_persisted(state.experiment)); assert state.execution is ExecutionState.RUNNING
    service.post(ExecutionEvent("progress_updated",{"round":4,"progress":.4})); service.post(ExecutionEvent("accuracy_updated",{"average":.8,"worst":.6})); assert service.dispatch_pending()==2
    assert state.current_round==4 and state.average_accuracy==.8
    service.post(ExecutionEvent("execution_completed")); service.dispatch_pending(); assert state.execution is ExecutionState.COMPLETED
    with pytest.raises(ValueError): state.transition(ExecutionState.STOPPING)


def test_navigation_has_exactly_five_workspaces():
    state=ApplicationState()
    for workspace in Workspace: state.navigate(workspace); assert state.workspace is workspace
    assert {item.value for item in Workspace}=={"dashboard","builder","queue","results","network"}


def test_result_discovery_is_lazy_filters_and_loads_metrics(tmp_path):
    import numpy as np
    run_dir=tmp_path/"run"; run_dir.mkdir(); np.savez(run_dir/"metrics.npz",avg_history=[.1,.2],worst_history=[.05,.1])
    (run_dir/"run.json").write_text(json.dumps({"status":"completed","finalAverageAccuracy":.2,"config":{"experimentName":"Alpha","dataset":"cifar10","approach":"basil","split":"IID"}}))
    malformed=tmp_path/"bad"; malformed.mkdir(); (malformed/"run.json").write_text("{")
    service=ResultService(tmp_path); assert service._records is None
    records=service.discover(); assert len(records)==2 and any(r.status=="malformed" for r in records)
    selected=service.filter(records,search="alpha",dataset="cifar10"); assert len(selected)==1
    assert list(service.load_metrics(selected[0])["avg_history"])==[.1,.2]


def test_plot_service_rejects_protected_output(tmp_path):
    protected=tmp_path/"plots4"; protected.mkdir(); service=PlotService(protected)
    with pytest.raises(ValueError,match="Protected historical plots"):
        service.generate(lambda **_kwargs: None,output_dir=protected/"nested")


def test_network_event_dispatch_remains_widget_independent():
    state=ApplicationState(experiment=valid_config()); service=ExecutionService(state)
    service.post(ExecutionEvent("network_updated",{"lane":0,"payload":{"nodeId":1}}))
    assert service.dispatch_pending()==1
    assert state.network_events==[("network_updated",{"lane":0,"payload":{"nodeId":1}})]
