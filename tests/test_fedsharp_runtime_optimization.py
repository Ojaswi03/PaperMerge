import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from basil_core import fedsharp_runtime_optimization as opt
from basil_core import sharp_protocol
from basil_core.cart_mechanism_audit import telemetry_summary


def records(count,nodes=4):
    return [dict(round=i//nodes,node=i%nodes,activation=i+1,selection=dict(selectedCount=1),
        training=dict(optimizerSteps=5,rlaCoefficient=.01,localTrainingLoss=.4)) for i in range(count)]


def test_history_bytes_match_original_json_and_replacement_invalidates():
    history=records(8);cache=opt.HistoryCache()
    assert cache.render(history)==json.dumps(history,sort_keys=True,allow_nan=False).encode()
    prefix=cache.encoded.copy();history.extend(records(12)[8:])
    assert cache.render(history)==json.dumps(history,sort_keys=True,allow_nan=False).encode()
    assert cache.encoded[:8]==prefix
    history[0]=dict(history[0],extra='replacement')
    assert cache.render(history)==json.dumps(history,sort_keys=True,allow_nan=False).encode()
    with pytest.raises(ValueError):cache.render([dict(value=float('nan'))])


def test_incremental_summary_exact_across_partial_rounds_and_restarts():
    history=[];fast=opt.RoundSummary(telemetry_summary)
    for record in records(37):
        history.append(record)
        assert fast(history,4)==telemetry_summary(history,4)
    assert fast(copy.deepcopy(history),4)==telemetry_summary(history,4)
    history[0]=dict(history[0],activation=9)
    with pytest.raises(ValueError):fast(history,4)


def test_optimized_checkpoint_is_strictly_loadable_and_keeps_old_inode(tmp_path):
    cfg=dict(nodes=4,rounds=3,classes=10);contract=dict(config=cfg)
    state=sharp_protocol.initial_state(cfg,np.arange(7,dtype=np.float32))
    state.update(cursor=5,telemetry=records(5),roundMetrics=[dict(round=0)],elapsedSeconds=1.5)
    state['senderRounds']=np.array([1,0,0,0],np.int32)
    path=tmp_path/'checkpoint.npz';sharp_protocol.save_state(path,state,contract)
    backup=tmp_path/'original.npz';os.link(path,backup);before=backup.read_bytes()
    writer=opt.CheckpointWriter(dict(revision='test'));writer(path,state,contract)
    restored=sharp_protocol.load_state(path,contract,7)
    for key in ('nodeParams','outbound','senderRounds'):np.testing.assert_array_equal(restored[key],state[key])
    assert restored['telemetry']==state['telemetry'] and restored['roundMetrics']==state['roundMetrics']
    assert backup.read_bytes()==before and path.stat().st_ino!=backup.stat().st_ino
    with np.load(path) as a:assert json.loads(a['metadata'].tobytes())['executionOptimization']==dict(revision='test')
    with pytest.raises(ValueError,match='identity differs'):
        sharp_protocol.load_state(path,dict(config=cfg,changed=True),7)


@pytest.fixture
def cpu():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


@pytest.mark.parametrize('dataset',['mnist','cifar10'])
def test_zero_coefficient_device_qualification_preserves_five_steps(cpu,dataset):
    from basil_core import basil_sharp_thesis as study
    from tests.test_basil_sharp_thesis import options
    view=next(v for v in study.selected(options(datasets=[dataset],partitions=['dirichlet'],seeds=[2025]))
        if v['comparisonCase']['addon']=='cart')
    receipt=opt.qualify_zero_step(view,study.configure_worker,'CPU')
    assert receipt['enabled'],receipt
    assert len(receipt['checks'])==4 and all(c['exact'] for c in receipt['checks'])


def test_mismatched_fast_path_is_disabled(monkeypatch):
    class Worker:
        cfg=dict(dataset='mnist',batchSize=14,classes=10)
        def export(self):return np.zeros(3,np.float32)
        def train(self,flat,*args):return flat+np.float32(.01),dict(optimizerSteps=5)
    def incorrect(worker):
        worker.train=lambda flat,*args:(flat+1.,dict(optimizerSteps=5))
        return True
    monkeypatch.setattr(opt,'install_zero_step',incorrect)
    result=opt.qualify_zero_step(dict(comparisonCase=dict(addon='cart')),lambda *a:Worker(),'CPU')
    assert not result['enabled'] and not result['checks'][0]['exact']


@pytest.mark.parametrize('arm',['basil_joint','sharp_joint','sharp_cart'])
def test_optimized_full_trajectory_and_resume_match_original(cpu,tmp_path,arm):
    from basil_core import basil_sharp_thesis as study,cart_comparison_study as comparison
    from tests.test_basil_sharp_thesis import options
    from tests.test_cart_comparison_runtime import data
    import basil_core.sharp_cart_v2_protocol as protocol
    view=next(v for v in study.selected(options(datasets=['mnist'],partitions=['dirichlet'],seeds=[2025],
        nodes=4,rounds=3,local_steps=5,attack_start=1)) if v['comparisonCase']['arm']==arm)
    x,y,chunks=data();worker=study.configure_worker(view,'CPU')
    with study.intervention(view,worker) as cfg:
        expected=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20])
    root=tmp_path/'optimized';(root/view['runId']).mkdir(parents=True)
    policy=dict(revision=opt.REVISION,source=opt.source(),transitionId='test')
    contract=dict(config=cfg);path=root/'checkpoint.npz'
    with study.installed(),opt.installed(root,view,policy):
        worker=comparison.configure_worker(view,'CPU')
        with study.intervention(view,worker) as effective:
            state=protocol.run(effective,worker,x,y,chunks,x[:20],y[:20],max_activations=5)
        sharp_protocol.save_state(path,state,contract)
    with study.installed(),opt.installed(root,view,policy):
        worker=comparison.configure_worker(view,'CPU')
        state=sharp_protocol.load_state(path,contract,worker.dimension)
        with study.intervention(view,worker) as effective:
            actual=protocol.run(effective,worker,x,y,chunks,x[:20],y[:20],state=state)
    for key in ('nodeParams','outbound','senderRounds'):np.testing.assert_array_equal(actual[key],expected[key])
    assert actual['telemetry']==expected['telemetry'] and actual['roundMetrics']==expected['roundMetrics']
    for node,row in expected['references'].items():
        np.testing.assert_array_equal(actual['references'][node]['params'],row['params'])
    assert (root/view['runId']/'optimization.json').is_file()


def test_real_worker_resume_keeps_original_contract_and_records_adapter(cpu,tmp_path,monkeypatch):
    from tests.test_round_continuation import parent_fixture
    from basil_core import basil_sharp_thesis as study,cart_comparison_study as comparison,sharp_data
    from scripts.cart_comparison_worker import run
    fixture=parent_fixture(tmp_path,monkeypatch,'sharp_cart')
    root=fixture.output;(root/fixture.new['runId']).mkdir(parents=True)
    (root/'plan.json').write_text(json.dumps(fixture.plan))
    policy=dict(revision=opt.REVISION,source=opt.source(),transitionId='cpu-worker-test')
    with study.installed(),opt.installed(root,fixture.new,policy):
        monkeypatch.setattr(sharp_data,'training_data',lambda _: (fixture.x,fixture.y))
        monkeypatch.setattr(comparison,'partitions_for',lambda *args:(fixture.chunks,fixture.held,fixture.audit))
        run(root,0,cpu=True)
    row=json.loads((root/fixture.new['runId']/'run.json').read_text())
    assert row['status']=='COMPLETED' and row['completedActivations']==16
    provenance=row['runtimeAdapter']['executionOptimization']
    assert provenance['startedCursor']==8 and provenance['source']==opt.source()
    assert json.loads((root/fixture.new['runId']/'optimization.json').read_text())==provenance
    assert row['contract']['source']==fixture.source
    with np.load(root/fixture.new['runId']/'checkpoint.npz') as archive:
        assert json.loads(archive['metadata'].tobytes())['executionOptimization']==provenance
