import json
from pathlib import Path

import pytest

from basil_core.protocol_compatibility import (
    LEGACY_PLOT_ROOTS, LEGACY_PROTOCOL_ALIASES, LEGACY_PURPOSE_ALIASES,
    LEGACY_RESULT_ROOTS, PROTOCOL, is_research_protocol, normalize_config,
)
from basil_core.artifact_paths import PROJECT_ROOT, writable_output
from basil_core.research_protocol import validate_config
from gui.research_protocol import CONFIG_ROOT, base_config, smoke_configs, write_config_library
from gui.services.config_service import ConfigService
from gui.services.queue_service import QueueService
from gui.state.application_state import ApplicationState
from gui.state.experiment_state import from_persisted, to_persisted, validate_experiment


@pytest.mark.parametrize("alias", LEGACY_PROTOCOL_ALIASES)
def test_saved_alias_load_save_and_validation(tmp_path, alias):
    old=smoke_configs()[0]
    old.update(experimentProtocol=alias, purpose=next(iter(LEGACY_PURPOSE_ALIASES)), attackStartRound=20,
               experimentName=alias.split("_")[0].title()+" | saved condition",
               unknownCompatibleField={"provenance": [1, 2, 3]})
    original=json.loads(json.dumps(old))
    model=from_persisted(old)
    assert not validate_experiment(model)
    validate_config(old)
    saved=to_persisted(model)
    for key in old.keys() - {"experimentProtocol", "purpose", "experimentName"}:
        assert saved[key] == old[key]
    assert old == original
    assert saved["experimentProtocol"] == PROTOCOL
    assert saved["purpose"] == "research_protocol_production"
    assert saved["experimentName"].startswith("Sequential BASIL")
    service=ConfigService()
    target=tmp_path/"saved.json"
    service.save(target, model)
    assert to_persisted(service.load(target)) == saved
    assert not any(alias in target.read_text() for alias in LEGACY_PROTOCOL_ALIASES)


def test_protocol_field_alias_does_not_invent_scientific_defaults():
    alias=next(iter(LEGACY_PROTOCOL_ALIASES))
    migrated=normalize_config({"protocol":alias,"unknown":17})
    assert migrated == {"protocol":PROTOCOL,"experimentProtocol":PROTOCOL,"unknown":17}
    assert not is_research_protocol("unrecognized_protocol")


def test_queue_alias_migration_is_read_only_until_explicit_save(tmp_path):
    data=smoke_configs()[0]
    data["experimentProtocol"]=next(iter(LEGACY_PROTOCOL_ALIASES))
    path=tmp_path/"queue.json"
    path.write_text(json.dumps([data]))
    before=path.read_bytes()
    service=QueueService(path)
    queue=service.load()
    assert path.read_bytes() == before
    assert to_persisted(queue.entries[0].config)["experimentProtocol"] == PROTOCOL
    service.save(queue)
    assert json.loads(path.read_text())[0]["experimentProtocol"] == PROTOCOL


def test_production_gate_recognizes_deprecated_identifier():
    state=ApplicationState(experiment=from_persisted(base_config()))
    state.experiment.extra["experimentProtocol"]=next(iter(LEGACY_PROTOCOL_ALIASES))
    assert not state.can_run


@pytest.mark.parametrize("root", (*LEGACY_RESULT_ROOTS, *LEGACY_PLOT_ROOTS))
def test_historical_namespace_is_read_only(root):
    with pytest.raises(ValueError, match="read-only"):
        writable_output(PROJECT_ROOT/root/"new_output.json")


def test_new_config_library_uses_canonical_scientific_names(tmp_path):
    assert CONFIG_ROOT.name == "sequential_basil"
    assert write_config_library(tmp_path) == (47,47)
    configs=list(tmp_path.glob("*/*.json"))
    assert len(configs) == 97
    for path in configs:
        config=json.loads(path.read_text())
        validate_config(config)
        assert config["experimentProtocol"] == PROTOCOL
        assert config["runId"].startswith("basil-")
        assert not any(alias in path.read_text() for alias in LEGACY_PROTOCOL_ALIASES)


def test_alias_worker_dispatch_uses_canonical_script_and_config(monkeypatch):
    from gui.services.execution_service import IsolatedWorkerHandle
    import gui.worker_pool
    captured={}
    class Pool:
        def __init__(self, **kwargs): captured.update(kwargs)
        def launch(self, config): captured["config"]=config
    monkeypatch.setattr(gui.worker_pool, "CampaignWorkerPool", Pool)
    data=smoke_configs()[0]
    data["experimentProtocol"]=next(iter(LEGACY_PROTOCOL_ALIASES))
    IsolatedWorkerHandle(data, lambda _event: None)
    assert captured["worker_script"] == Path("scripts/run_research_protocol.py")
    assert captured["work_dir"] == Path("experiments/research_protocol_results/workers")
    assert captured["config"]["experimentProtocol"] == PROTOCOL
