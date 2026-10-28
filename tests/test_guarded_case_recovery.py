import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace


def test_reconciliation_preserves_completed_weights_and_metrics(tmp_path,monkeypatch):
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    import scripts.recover_guarded_sharp_case as module
    import basil_core.sharp_guarded_protocol as guarded
    import basil_core.sharp_protocol as protocol
    import basil_core.sharp_worker as workers
    import basil_core.sharp_data as data
    from basil_core.sharp_study import configuration
    cfg=guarded.guarded_configuration(configuration('mnist','iid',2025,'attack_noise_defense_rla_cart',.01,nodes=4,rounds=2))
    folder=tmp_path/cfg['runId'];folder.mkdir()
    x=np.zeros((20,1),np.float32);y=np.arange(20,dtype=np.int32)%10
    audit={'partitionHash':'fixture'};source={'fixture':'same'}
    identity=dict(imagesSha256=data.digest(x),labelsSha256=data.digest(y),samples=len(y),officialTestLoaded=False)
    runtime=dict(device='/CPU:0',tensorflow='fixture',numpy=np.__version__,host=socket.gethostname())
    manifest=dict(config=cfg,source=source,data=identity,partition=audit,runtime=runtime,
        initialHash=protocol.vector_hash(np.zeros(3,np.float32)),status='FAILED',completedActivations=8,
        error='CART report requires the paired non-CART condition.')
    (folder/'run.json').write_text(json.dumps(manifest))
    (tmp_path/'plan.json').write_text(json.dumps(dict(configs=[cfg],source=source,
        data={'mnist':identity},partitions={cfg['runId']:audit},executionProgram={})))
    (tmp_path/'campaign.json').write_text(json.dumps(dict(status='FAILED')))
    checkpoint=folder/'checkpoint.npz';checkpoint.write_bytes(b'unchanged-completed-checkpoint')
    records=[dict(round=r,meanAccuracy=.75,nodeAccuracy=[.75]*4,perClass=[[.75]*10]*4) for r in range(2)]
    state=dict(cursor=8,roundMetrics=records,telemetry=[{}]*8,elapsedSeconds=12.)
    np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((2,4),.75),roundNodePerClassAccuracy=np.full((2,4,10),.75))
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint,folder/'metrics.npz')}
    monkeypatch.setattr(module,'active_writer',lambda _:None)
    monkeypatch.setattr(guarded,'check_source',lambda _:None)
    monkeypatch.setattr(data,'training_data',lambda _:(x,y))
    monkeypatch.setattr(data,'prepare_partition',lambda *_:([[]]*4,np.arange(4),audit))
    monkeypatch.setattr(protocol,'load_state',lambda *_:state)
    def never_run(*_,**__):raise AssertionError('Completed training must not be repeated.')
    monkeypatch.setattr(guarded,'run',never_run)
    class Worker:
        dimension=3
        def __init__(self,*_):pass
        def export(self):return np.zeros(3,np.float32)
    monkeypatch.setattr(workers,'SharpWorker',Worker)
    config=SimpleNamespace(set_visible_devices=lambda *_:None,
        threading=SimpleNamespace(set_intra_op_parallelism_threads=lambda _:None,set_inter_op_parallelism_threads=lambda _:None),
        experimental=SimpleNamespace(enable_op_determinism=lambda:None))
    monkeypatch.setitem(sys.modules,'tensorflow',SimpleNamespace(config=config,__version__='fixture'))
    module.recover(tmp_path)
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint,folder/'metrics.npz')}==before
    result=json.loads((folder/'run.json').read_text())
    assert result['status']=='COMPLETED'
    assert result['continuations'][0]['previousError']==manifest['error']
    assert list((tmp_path/'recovery_records').glob('*-run.json'))
    assert json.loads((tmp_path/'campaign.json').read_text())['status']=='COMPLETED'


def test_recovery_detects_an_existing_writer(tmp_path):
    from scripts.recover_guarded_sharp_case import active_writer
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)',
        'run_guarded_sharp_case.py','--output',str(tmp_path)])
    try:assert active_writer(tmp_path)==process.pid
    finally:process.terminate();process.wait()
