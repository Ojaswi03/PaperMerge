import json
from pathlib import Path

import numpy as np

from basil_core.round_continuation import sha
from scripts import diagnose_fedsharp_500r as diagnostic


def test_checkpoint_probe_preserves_bytes_and_requires_original_contract(tmp_path):
    path=tmp_path/'checkpoint.npz';contract=dict(source='original')
    raw=json.dumps(dict(contract=contract,cursor=10000)).encode()
    np.savez(path,metadata=np.frombuffer(raw,np.uint8))
    before=path.read_bytes()
    result=diagnostic.checkpoint(path,sha(path),contract,10000)
    assert result['matchesFrozenSeed'] and result['metadataReadable']
    assert result['originalContractMatches'] and result['boundaryMatches'] and result['stableChecksum']
    assert path.read_bytes()==before
    assert not diagnostic.checkpoint(path,sha(path),dict(source='different'),10000)['originalContractMatches']


def test_checksum_match_does_not_hide_invalid_metadata(tmp_path):
    path=tmp_path/'checkpoint.npz';np.savez(path,metadata=np.array([0xe1],np.uint8))
    result=diagnostic.checkpoint(path,sha(path),{},10000)
    assert result['matchesFrozenSeed'] and result['error'].startswith('UnicodeDecodeError:')


def test_missing_or_changed_copy_is_reported_without_repair(tmp_path):
    path=tmp_path/'checkpoint.npz'
    assert 'FileNotFoundError' in diagnostic.checkpoint(path,'0'*64,{},10000)['error']
    path.write_bytes(b'changed')
    result=diagnostic.checkpoint(path,'0'*64,{},10000)
    assert not result['matchesFrozenSeed'] and path.read_bytes()==b'changed'


def test_dataset_probe_reports_intermittent_failure_without_relaxing_checks(monkeypatch):
    from basil_core import sharp_data,thesis_raw_data
    original=sharp_data.digest;calls=[]
    def load(name):
        calls.append(name);sharp_data.digest(np.array([len(calls)],np.int32))
        if len(calls)==1:raise ValueError('fingerprint mismatch')
        return np.zeros(1),np.zeros(1)
    monkeypatch.setattr(thesis_raw_data,'training_data',load)
    result=diagnostic.dataset_reads('mnist')
    assert [r['verified'] for r in result['reads']]==[False,True]
    assert result['reads'][0]['observedArrayHashes']!=result['reads'][1]['observedArrayHashes']
    assert sharp_data.digest is original


def test_diagnosis_reads_only_local_failed_runs(tmp_path,monkeypatch):
    from basil_core import basil_sharp_thesis as study
    queue=tmp_path/'queue';output=tmp_path/'output';host='hal9k'
    (queue/'host_bindings'/host).mkdir(parents=True);output.mkdir()
    (queue/'host_bindings'/host/'owner.json').write_text(json.dumps(dict(output=str(output))))
    source={'frozen':'hash'};monkeypatch.setattr(study,'source',lambda:source)
    monkeypatch.setattr(diagnostic.socket,'gethostname',lambda:'hal9k.example.edu')
    plan=dict(source=source,configs=[dict(runId='failed'),dict(runId='healthy')],continuations={'failed':{}})
    (output/'plan.json').write_text(json.dumps(plan))
    for name,status in [('failed','FAILED'),('healthy','RUNNING')]:
        folder=output/name;folder.mkdir()
        (folder/'run.json').write_text(json.dumps(dict(status=status,error='reason',completedActivations=10000)))
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=diagnostic.diagnose(queue)
    assert result['producerUnchanged'] and not result['trainingModified']
    assert [r['runId'] for r in result['failures']]==['failed']
    assert all(p.read_bytes()==raw for p,raw in before.items())
