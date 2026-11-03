import copy
import json
import os
import zipfile

import numpy as np
import pytest

from scripts import recover_fedsharp_outbound as recovery
from basil_core import basil_sharp_thesis as study
from basil_core import live_gpu_scheduler as live
from basil_core.continuation_campaign import atomic
from basil_core.round_continuation import sha
from tests.test_checkpoint_integrity_scan import corrupt


def fixture(tmp_path,monkeypatch,attack='hidden',bad_hash=False):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()
    from tests.test_round_continuation import parent_fixture
    from basil_core.sharp_protocol import save_state
    import basil_core.sharp_cart_v2_protocol as protocol
    f=parent_fixture(tmp_path,monkeypatch,'basil_acds',attack)
    worker=study.configure_worker(f.new,'CPU')
    with study.intervention(f.new,worker) as cfg:
        state=protocol.run(cfg,worker,f.x,f.y,f.chunks,f.x[f.held],f.y[f.held],state=copy.deepcopy(f.state))
    if bad_hash:state['telemetry'][-1]['outboundHash']='0'*64
    contract=dict(f.contract,config=cfg,comparisonCase=f.new['comparisonCase'])
    root=f.output;run=f.new['runId'];folder=root/run;folder.mkdir(parents=True)
    save_state(folder/'checkpoint.npz',state,contract)
    with zipfile.ZipFile(folder/'checkpoint.npz') as z:good={name:z.read(name) for name in z.namelist()}
    corrupt(folder/'checkpoint.npz','outbound.npy')
    atomic(folder/'run.json',dict(status='COMPLETED',config=f.new,source=f.source,initialHash=f.contract['initialHash'],contract=contract))
    np.savez(folder/'metrics.npz',roundNodeAccuracy=np.asarray([r['nodeAccuracy'] for r in state['roundMetrics']]),
        roundNodePerClassAccuracy=np.asarray([r['perClass'] for r in state['roundMetrics']]),
        roundHonestMeanAccuracy=np.asarray([r['honestMeanAccuracy'] for r in state['roundMetrics']]))
    atomic(root/'plan.json',f.plan)
    queue=tmp_path/'shared';atomic(queue/'fleet/state.json',dict(runs={run:dict(owner='hal9k',state='DONE',location=dict(output=str(root),index=0))}))
    monkeypatch.setattr(recovery,'host_name',lambda:'hal9k')
    monkeypatch.setattr(live,'discover',lambda *a:{})
    return queue,folder,run,good


@pytest.mark.parametrize('attack',['hidden','gaussian'])
def test_reconstructs_exact_original_arrays_preserving_all_intact_members(tmp_path,monkeypatch,attack):
    queue,folder,run,good=fixture(tmp_path,monkeypatch,attack)
    path=folder/'checkpoint.npz';damaged=path.read_bytes();digest=sha(path)
    row=(folder/'run.json').read_bytes();metrics=(folder/'metrics.npz').read_bytes()
    receipt=recovery.recover(queue,run,digest,True)
    assert receipt['status']=='RECOVERED' and receipt['outboundHashesMatched']==4
    assert receipt['trainedModelHashesMatched']==4 and receipt['optimizerUpdatesAdded']==0
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,payload in good.items():assert z.read(name)==payload
    assert (folder/'run.json').read_bytes()==row and (folder/'metrics.npz').read_bytes()==metrics
    backup=folder.parent/'operational_adapters/outbound_recovery'/run/'original-checkpoint.npz'
    assert backup.read_bytes()==damaged
    assert sha(path)==receipt['repairedCheckpointSha256']


def test_recorded_hash_mismatch_stops_without_installing(tmp_path,monkeypatch):
    queue,folder,run,good=fixture(tmp_path,monkeypatch,bad_hash=True)
    before=(folder/'checkpoint.npz').read_bytes()
    with pytest.raises(ValueError,match='does not match the recorded hash'):
        recovery.recover(queue,run,sha(folder/'checkpoint.npz'),True)
    assert (folder/'checkpoint.npz').read_bytes()==before
    assert not (folder/'checkpoint_recovery.json').exists()


def test_dry_run_keeps_checkpoint_and_run_untouched(tmp_path,monkeypatch):
    queue,folder,run,good=fixture(tmp_path,monkeypatch)
    before=(folder/'checkpoint.npz').read_bytes()
    receipt=recovery.recover(queue,run,sha(folder/'checkpoint.npz'))
    assert receipt['status']=='PREPARED' and (folder/'checkpoint.npz').read_bytes()==before
    assert not (folder/'checkpoint_recovery.json').exists()
