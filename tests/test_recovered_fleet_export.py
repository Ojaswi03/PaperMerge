import hashlib
from pathlib import Path
import pytest
from basil_core.continuation_campaign import atomic
from basil_core.round_continuation import sha
from scripts import export_recovered_fedsharp_fleet as exporter


def setup(tmp_path,monkeypatch,bad=False):
    root=tmp_path/'local';run='run0';folder=root/run;folder.mkdir(parents=True)
    for name in ('run.json','metrics.npz','checkpoint.npz'):(folder/name).write_bytes(name.encode())
    payload=b'recovery source';digest=hashlib.sha256(payload).hexdigest()
    source=root/'recovery_source_snapshots'/digest/'scripts/recover_fedsharp_outbound.py'
    source.parent.mkdir(parents=True);source.write_bytes(payload)
    receipt=dict(status='RECOVERED',runId=run,source={},recipe='deterministic_outbound_from_trained_models',
        optimizerUpdatesAdded=0,outboundHashesMatched=3 if bad else 100,trainedModelHashesMatched=100,
        repairedCheckpointSha256=sha(folder/'checkpoint.npz'),metricsSha256=sha(folder/'metrics.npz'),
        runRecordSha256=sha(folder/'run.json'),repairScriptSha256=digest)
    atomic(folder/'checkpoint_recovery.json',receipt)
    monkeypatch.setattr(exporter,'host_name',lambda:'hal9k')
    monkeypatch.setattr(exporter.fleet_queue,'state',lambda q:dict(runs={run:dict(owner='hal9k',state='DONE',location=dict(output=str(root)))}))
    def verified_export(queue,destination,verify=False):
        assert verify
        destination.mkdir();atomic(destination/'plan.json',dict(source={},configs=[dict(runId=run,nodes=100)]))
        (destination/run).mkdir()
        for name in ('run.json','metrics.npz'):(destination/run/name).write_bytes((folder/name).read_bytes())
        atomic(destination/'independent_validation.json',dict(verified=[dict(runId=run)]))
        return 1
    monkeypatch.setattr(exporter,'export',verified_export)
    return tmp_path/'queue',tmp_path/'export'


def test_export_requires_fresh_verification_and_includes_repair_evidence(tmp_path,monkeypatch):
    queue,destination=setup(tmp_path,monkeypatch)
    result=exporter.recovered_export(queue,destination)
    assert result==dict(configurations=1,recoveryReceipts=1,freshInferenceVerified=True)
    assert (destination/'run0/checkpoint_recovery.json').exists()
    assert not list(destination.rglob('checkpoint.npz'))
    from basil_core.cart_campaign_io import read_json
    assert all(sha(destination/name)==digest for name,digest in read_json(destination/'SHA256.json').items())


def test_hash_provenance_mismatch_never_publishes_destination(tmp_path,monkeypatch):
    queue,destination=setup(tmp_path,monkeypatch,bad=True)
    with pytest.raises(ValueError,match='provenance differs'):exporter.recovered_export(queue,destination)
    assert not destination.exists() and not list(tmp_path.glob('export.recovery-*'))
