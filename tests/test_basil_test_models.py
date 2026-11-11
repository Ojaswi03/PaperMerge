import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from scripts import fetch_basil_test_models as fetch
from scripts import test_fedsharp_image as frozen


def test_all_four_methods_select_eight_cases_and_acds_keeps_its_extra_data_identity(tmp_path):
    cases=frozen.select_local_runs(tmp_path,['mnist','cifar10'],'hidden',2025,
        ('basil','basil_acds','fedsharp','fedsharp_plus'))
    assert len(cases)==8
    assert [c['method'] for c in cases]==['BASIL','BASIL + ACDS','FedSharp','FedSharp+']*2
    for c in cases:
        c['checkpoint'].parent.mkdir();c['checkpoint'].write_bytes(b'preflight-only fixture')
    got=frozen.preflight(cases)
    for ds in ('mnist','cifar10'):
        selected=[c for c in got if c['config']['dataset']==ds]
        assert all(c['view']['partition']=='dirichlet' and c['config']['rounds']==500 for c in selected)
        assert 'acds' in next(c for c in selected if c['method']=='BASIL + ACDS')['row']['partition']
        assert 'acds' not in next(c for c in selected if c['method']=='BASIL')['row']['partition']


def fixture(tmp_path):
    run='completed';root=tmp_path/'original';folder=root/run;folder.mkdir(parents=True)
    cfg=dict(nodes=100,rounds=500);contract=dict(config=cfg)
    params=np.arange(1000,dtype=np.float32).reshape(100,10)/1000
    hashes=[__import__('hashlib').sha256(p.tobytes()).hexdigest() for p in params]
    meta=dict(contract=contract,cursor=50000,roundMetrics=[dict(nodeAccuracy=[.25]*100) for _ in range(500)],
        telemetry=[dict(node=i%100,trainedHash=hashes[i%100]) for i in range(50000)])
    np.savez(folder/'checkpoint.npz',nodeParams=params,metadata=np.frombuffer(json.dumps(meta).encode(),np.uint8))
    (folder/'run.json').write_text(json.dumps(dict(status='COMPLETED',contract=contract)))
    np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((500,100),.25))
    selected=[dict(view=dict(runId=run),output=root)]
    return root,run,params,selected


def test_remote_export_uses_stdlib_and_preserves_checkpoint_then_compact_loader_checks_every_hash(tmp_path):
    root,run,params,selected=fixture(tmp_path)
    before=frozen.sha(root/run/'checkpoint.npz');archive=tmp_path/'bundle.zip'
    with archive.open('wb') as stream:
        subprocess.run([sys.executable,'-c',fetch.REMOTE,str(root),json.dumps([run])],stdout=stream,check=True)
    assert frozen.sha(root/run/'checkpoint.npz')==before
    assert archive.stat().st_size<(root/run/'checkpoint.npz').stat().st_size/10
    destination=tmp_path/'models';fetch.unpack(archive,selected,destination)
    checkpoint=destination/run/'model.npz'
    item=dict(config=dict(nodes=100,rounds=500),row=dict(contract=dict(config=dict(nodes=100,rounds=500))),
        checkpoint=checkpoint,output=root,view=dict(runId=run))
    got,accuracy=frozen.load_final_parameters(item,10)
    np.testing.assert_array_equal(got,params);assert accuracy.shape==(500,100)
    assert set(p.name for p in (root/run).iterdir())=={'checkpoint.npz','run.json','metrics.npz'}
    with pytest.raises(FileExistsError):fetch.unpack(archive,selected,destination)
    with pytest.raises(ValueError):frozen.load_final_parameters(item,11)


def test_compact_transfer_rejects_unrequested_archive_entries(tmp_path):
    root,run,_,selected=fixture(tmp_path);archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside',b'wrong')
    with pytest.raises(ValueError,match='Unexpected downloaded'):fetch.unpack(archive,selected,tmp_path/'models')
    assert not (tmp_path/'outside').exists()
