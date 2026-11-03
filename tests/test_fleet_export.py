import json
from pathlib import Path
import socket

import pytest

from basil_core import fleet_queue as fleet
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic
from basil_core.round_continuation import sha
from scripts import export_fedsharp_fleet as exporter
from tests.test_fleet_queue import setup_queue,join


@pytest.mark.parametrize('optimized',[False,True])
def test_final_export_collects_local_original_and_transferred_results_without_checkpoints(tmp_path,monkeypatch,optimized):
    from reporting import round_continuation
    queue,master=setup_queue(tmp_path,count=2)
    master['revision']='revision'
    for item in master['items']:item.update(sourceRunId='parent-'+item['config']['runId'],seedSha256='a'*64)
    atomic(queue/'continuation_plan.json',master)
    original,output=join(queue,'hal9k');join(queue,'themachine');output.mkdir()
    atomic(original/'plan.json',dict(host=socket.gethostname(),source={},configs=[master['items'][0]['config']],
        continuationPlanSha256=sha(queue/'continuation_plan.json')))
    atomic(output/'plan.json',dict(host=socket.gethostname(),source={},configs=[i['config'] for i in master['items']],
        continuationPlanSha256=sha(queue/'continuation_plan.json')))
    for _ in range(2):
        claim=fleet.claim(queue,'hal9k',0);run=claim['runId'];root=Path(claim['location']['output'])
        atomic(root/run/'run.json',dict(status='COMPLETED',runtime=dict(profile=dict(gpu=0))))
        if optimized:
            import hashlib
            payload=b'optimization source fixture';digest=hashlib.sha256(payload).hexdigest();name='scripts/optimized_fedsharp_worker.py'
            source=root/'optimization_snapshots'/digest/name;source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(payload)
            adapter=dict(revision='fixture',source={name:digest},features=['cached_checkpoint_json'],
                zeroCoefficientQualification=dict(enabled=False),runtime=dict(profile=dict(output_root=str(root))))
            row=read_json(root/run/'run.json');row['runtimeAdapter']=dict(executionOptimization=adapter)
            atomic(root/run/'run.json',row);atomic(root/run/'optimization.json',adapter)
        for name in ('metrics.npz','mechanisms.json','continuation.json','checkpoint.npz'):
            (root/run/name).write_bytes(name.encode())
        fleet.update(queue,run,'hal9k',claim['claim'],'DONE')
    monkeypatch.setattr(exporter,'host_name',lambda:'hal9k')
    monkeypatch.setattr(exporter,'source_record',lambda item:dict(data={'same':True},partition=dict(seed=item['config']['seed'])))
    checked=[]
    def generate(root,destination):
        plan=read_json(root/'plan.json');checked.append(plan)
        assert plan['configs']==[item['config'] for item in master['items']]
        assert (root/'run0/run.json').read_bytes()==(original/'run0/run.json').read_bytes()
        assert (root/'run1/run.json').read_bytes()==(output/'run1/run.json').read_bytes()
    monkeypatch.setattr(round_continuation,'generate',generate)
    destination=tmp_path/'export'
    assert exporter.export(queue,destination)==2
    assert checked and not list(destination.rglob('checkpoint.npz'))
    if optimized:
        assert list((destination/'optimization_snapshots').rglob('optimized_fedsharp_worker.py'))
        assert (destination/'run0/optimization.json').exists()
    provenance=read_json(destination/'execution_provenance.json')
    assert not provenance['run0']['transferred'] and provenance['run1']['transferred']
    assert 'pid' not in json.dumps(provenance) and 'output' not in json.dumps(provenance)
    assert all(sha(destination/name)==digest for name,digest in read_json(destination/'SHA256.json').items())


def test_unfinished_fleet_cannot_be_exported_as_completed(tmp_path):
    queue,master=setup_queue(tmp_path,count=2);join(queue,'hal9k')
    destination=tmp_path/'export'
    with pytest.raises(ValueError,match='complete successfully'):exporter.export(queue,destination)
    assert not destination.exists()
