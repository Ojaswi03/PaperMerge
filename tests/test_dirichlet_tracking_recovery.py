"""Real tiny checkpoint decoding for tracking retries; no SGD or cloud access."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def saved(monkeypatch, tmp_path):
    import numpy as np
    import basil_core.dirichlet_horizon_study as study
    import basil_core.protocol_checkpoint as checkpoint
    from basil_core.research_protocol import Snapshot, broadcast, initialize_memories

    cfg=study.configurations('a'*64,'b'*64,'c'*64,500)[0]
    fingerprints={'synthetic_fixture.py':'d'*64}
    monkeypatch.setattr(study,'validate_evidence',lambda _:None)
    monkeypatch.setattr(study,'execution_fingerprints',lambda _:dict(fingerprints))
    monkeypatch.setattr(checkpoint,'checkpoint_implementation',lambda _:dict(fingerprints))
    params=[np.ones(3,np.float32)]
    memories=initialize_memories(params)
    for cursor in range(123):
        round_id,node=divmod(cursor,10)
        broadcast(memories,Snapshot(node,round_id,params))
    state=dict(next_activation=123,completed_rounds=12,
        node_params=[params for _ in range(10)],memories=memories,
        telemetry=[dict(round=i//10,nodeId=i%10) for i in range(123)])
    for name in checkpoint.STATE_ARRAYS:
        shape=(500,10,10) if name in ('activation_per_class','round_node_per_class') else (500,10)
        state[name]=np.zeros(shape,np.float32)
    manifest=dict(status='tracking_unavailable',config=cfg,
        completedActivations=123,completedRounds=12,implementationSha256=dict(fingerprints))
    def write_manifest():
        (tmp_path/'run.json').write_text(json.dumps(manifest))
    write_manifest()
    checkpoint.save_checkpoint(tmp_path/'checkpoint.npz',cfg,state,
        partition_hash=cfg['partitionHash'],initial_hash=cfg['initialModelHash'])
    return study,cfg,manifest,write_manifest,tmp_path,fingerprints


def test_legacy_nonzero_tracking_failure_uses_verified_checkpoint(saved):
    study,cfg,_,_,directory,_=saved
    before={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
    plan=study.recovery_plan(directory,cfg)
    assert plan['mode']=='checkpoint' and plan['startActivation']==123
    assert plan['completedRounds']==12
    assert before=={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}


def test_legacy_tracking_failure_falls_back_to_previous_verified_checkpoint(saved):
    study,cfg,_,_,directory,_=saved
    path=directory/'checkpoint.npz'
    path.rename(directory/'checkpoint.previous.npz')
    path.write_bytes(b'corrupt current checkpoint')
    plan=study.recovery_plan(directory,cfg)
    assert Path(plan['checkpoint']).name=='checkpoint.previous.npz'
    assert plan['startActivation']==123


@pytest.mark.parametrize('missing',[False,True])
def test_nonzero_tracking_failure_cannot_fall_back_to_fresh_start(saved,missing):
    study,cfg,_,_,directory,_=saved
    path=directory/'checkpoint.npz'
    if missing:path.unlink()
    else:path.write_bytes(b'corrupt checkpoint')
    with pytest.raises(ValueError,match='No verified'):
        study.recovery_plan(directory,cfg)


def test_changed_source_still_blocks_tracking_retry(saved):
    study,cfg,_,_,directory,fingerprints=saved
    fingerprints['synthetic_fixture.py']='e'*64
    with pytest.raises(ValueError,match='inconsistent scientific state'):
        study.recovery_plan(directory,cfg)


def test_interrupted_tracking_retry_still_checks_checkpoint_source(saved):
    study,cfg,manifest,write,directory,fingerprints=saved
    manifest['status']='interrupted';write()
    fingerprints['synthetic_fixture.py']='e'*64
    with pytest.raises(ValueError,match='No verified'):
        study.recovery_plan(directory,cfg)


@pytest.mark.parametrize('field,value',[
    ('completedActivations',-1),('completedActivations',True),('completedActivations',5001),
    ('completedRounds',13),('completedRounds',None),('config',{}),
])
def test_legacy_tracking_retry_rejects_inconsistent_manifest(saved,field,value):
    study,cfg,manifest,write,directory,_=saved
    manifest[field]=value;write()
    with pytest.raises(ValueError,match='inconsistent scientific state'):
        study.recovery_plan(directory,cfg)


@pytest.mark.parametrize('orphan',[None,'checkpoint.previous.npz','metrics.npz',
                                  'activation_telemetry.json.gz','checkpoint_telemetry/record.json.gz'])
def test_zero_work_startup_retry_requires_absence_of_committed_evidence(saved,orphan):
    study,cfg,manifest,write,directory,_=saved
    manifest.update(completedActivations=0,completedRounds=0);write()
    path=directory/'checkpoint.npz'
    if orphan=='checkpoint.previous.npz':
        # Even stale zero counters must not discard a real nonzero checkpoint.
        path.rename(directory/orphan)
        plan=study.recovery_plan(directory,cfg)
        assert plan['mode']=='checkpoint' and plan['startActivation']==123
        return
    path.unlink()
    if orphan:
        target=directory/orphan;target.parent.mkdir(exist_ok=True)
        target.write_bytes(b'orphan evidence')
        with pytest.raises(ValueError,match='without a checkpoint'):
            study.recovery_plan(directory,cfg)
    else:
        assert study.recovery_plan(directory,cfg)==dict(mode='startup_retry',
            savedActivations=0,startActivation=0,completedRounds=0)


def test_tracking_failure_does_not_bypass_prerequisite_approval(saved,monkeypatch):
    study,cfg,_,_,directory,_=saved
    def blocked(_):raise ValueError('Prerequisite approval is stale')
    monkeypatch.setattr(study,'validate_evidence',blocked)
    with pytest.raises(ValueError,match='approval is stale'):
        study.recovery_plan(directory,cfg)
