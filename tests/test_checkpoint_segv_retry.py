import copy
import json
import os
import pytest

from basil_core import basil_sharp_thesis as study
from basil_core.continuation_campaign import atomic
from basil_core.round_continuation import sha
from scripts import retry_fedsharp_checkpoint as recovery


@pytest.fixture
def failed(tmp_path,monkeypatch):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()
    from tests.test_round_continuation import parent_fixture
    from basil_core.sharp_protocol import save_state
    import basil_core.sharp_cart_v2_protocol as protocol
    fixture=parent_fixture(tmp_path,monkeypatch,'sharp_joint')
    worker=study.configure_worker(fixture.new,'CPU')
    with study.intervention(fixture.new,worker) as cfg:
        state=protocol.run(cfg,worker,fixture.x,fixture.y,fixture.chunks,fixture.x[fixture.held],fixture.y[fixture.held],
            state=copy.deepcopy(fixture.state),max_activations=4)
    contract=dict(fixture.contract,config=cfg,comparisonCase=fixture.new['comparisonCase'])
    root=fixture.output;run=fixture.new['runId'];folder=root/run;folder.mkdir(parents=True)
    save_state(folder/'checkpoint.npz',state,contract)
    atomic(root/'plan.json',fixture.plan)
    atomic(folder/'run.json',dict(status='FAILED',contract=contract,error='Fleet worker exited -11 without a completed result.',
        completedActivations=14,elapsedSeconds=100))
    queue=root.parent/'shared';master=dict(queueId='retry',items=[fixture.item])
    atomic(queue/'continuation_plan.json',master)
    entry=dict(owner='hal9k',state='FAILED',location=dict(output=str(root),index=0),boundGpu=1,claim='token',index=0)
    registry=dict(queueId='retry',masterSha256=sha(queue/'continuation_plan.json'),hosts={'hal9k':dict(fleet=str(root))},runs={run:entry})
    atomic(queue/'fleet/state.json',registry)
    atomic(queue/'fleet/claims'/(run+'.json'),dict(host='hal9k',token='token',location=entry['location'],gpu=1))
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    monkeypatch.setattr(recovery,'host_name',lambda:'hal9k')
    monkeypatch.setattr(recovery.live,'discover',lambda *a:{})
    return queue,root,run


def test_readonly_retry_checks_actual_models_and_prefix_without_mutation(failed):
    queue,root,run=failed
    before={p:p.read_bytes() for p in root.parent.rglob('*') if p.is_file()}
    receipt=recovery.retry(queue,run)
    assert receipt['resumeActivations']==12 and receipt['replayActivations']==2
    assert all(p.read_bytes()==value for p,value in before.items())
    assert set(before)=={p for p in root.parent.rglob('*') if p.is_file()}


def test_retry_preserves_claim_and_checkpoint_and_prevents_repeat(failed):
    queue,root,run=failed
    checkpoint=(root/run/'checkpoint.npz').read_bytes();claim=(queue/'fleet/claims'/(run+'.json')).read_bytes()
    receipt=recovery.retry(queue,run,True)
    row=json.loads((root/run/'run.json').read_text())
    assert row['completedActivations']==12 and row['status']=='INTERRUPTED' and 'error' not in row
    assert (root/run/'checkpoint.npz').read_bytes()==checkpoint
    assert (queue/'fleet/claims'/(run+'.json')).read_bytes()==claim
    assert json.loads((queue/'fleet/state.json').read_text())['runs'][run]['state']=='WAITING'
    assert (root/'operational_adapters/checkpoint_retries'/run/'checkpoint.npz').read_bytes()==checkpoint
    with pytest.raises(ValueError):recovery.retry(queue,run,True)


def test_live_target_or_wrong_failure_refused(failed,monkeypatch):
    queue,root,run=failed
    monkeypatch.setattr(recovery.live,'discover',lambda *a:{0:dict(pid=42)})
    with pytest.raises(ValueError,match='already has a worker'):recovery.retry(queue,run,True)
    monkeypatch.setattr(recovery.live,'discover',lambda *a:{})
    path=root/run/'run.json';row=json.loads(path.read_text());row['error']='unknown failure';atomic(path,row)
    with pytest.raises(ValueError,match='only for'):recovery.retry(queue,run,True)
