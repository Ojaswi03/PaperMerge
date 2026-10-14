"""Step/recovery/wire contracts; no research run or external service."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def setup():
    import numpy as np
    import tensorflow as tf
    from basil_core.replicated_training_service import TrainingReference,TrainingReplica,ReplicatedTrainingService
    from basil_core.verified_cart_runner import VerifiedCartRunner
    from basil_core.research_protocol import Snapshot
    x=np.random.default_rng(4).uniform(size=(20,32,32,3)).astype(np.float32)
    y=np.repeat(np.arange(10,dtype=np.int32),2)
    ref=TrainingReference(x,y,source_split='train',synthetic=True)
    def model():
        return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.Constant(.1))],name='fixture')
    replicas={i:TrainingReplica(i,model(),ref) for i in range(10)}
    service=ReplicatedTrainingService(replicas)
    def runner():
        r=VerifiedCartRunner(model(),ref,service,run_id='unit',ebm=True,sigma_e=.01)
        r.round=1
        for memory in r.memories:
            for s in memory.candidates():s.round_id=0 if s.sender_id>=memory.receiver_id else 1
        return r
    batch=((0,1),(7,8),{c:((2*c,2*c+1),(7,8)) for c in range(1,10)})
    return runner,service,replicas,ref,batch


def test_checkpoint_resume_reproduces_next_step_and_memory(setup,tmp_path):
    import numpy as np
    from basil_core.research_protocol import params_hash
    make,_,_,_,batch=setup
    r=make();r.select({c:(2*c,2*c+1) for c in range(10)})
    assert params_hash(r.anchor)==r.selection['hash']==params_hash(r.params)
    first=r.advance(unit_batch=batch);r.save(tmp_path/'state.json')
    second=r.advance(unit_batch=batch)
    resumed=make();resumed.restore(tmp_path/'state.json')
    assert resumed.step==1 and resumed.round==1 and resumed.node==0
    for m in resumed.memories:
        assert len(m.candidates())==5
    replay=resumed.advance(unit_batch=batch)
    assert second==replay and first['outputHash']==second['inputHash']
    for a,b in zip(r.params,resumed.params):np.testing.assert_array_equal(a,b)
    assert params_hash(resumed.anchor)==resumed.selection['hash']


def test_active_node_wrong_update_rejected_without_progress(setup):
    import numpy as np
    from basil_core.research_protocol import params_hash
    make,_,_,_,batch=setup;r=make();r.select({c:(2*c,2*c+1) for c in range(10)})
    before=params_hash(r.params)
    with pytest.raises(ValueError,match='Active-node'):
        r.advance(unit_batch=batch,proposed_params=[np.zeros_like(p) for p in r.params])
    assert r.step==0 and not r.records and params_hash(r.params)==before
    assert params_hash([w.numpy() for w in r.model.trainable_variables])==before


def test_four_poisoned_endpoints_do_not_change_step(setup):
    import numpy as np
    from basil_core.replicated_verification import ComputeResult,sign_result
    make,_,replicas,_,batch=setup
    def poisoned_outbound_candidate():
        r=make()
        # Arbitrary malicious outbound model: nearly constant class-7 logits.
        # This is a fault-injection fixture, not a replacement for Hidden attack.
        r.memories[0].snapshots[7].params[-1][7]=80.
        return r
    honest=poisoned_outbound_candidate();honest.select({c:(2*c,2*c+1) for c in range(10)})
    honest.advance(unit_batch=batch)
    for i in (0,1,5,7):
        replica=replicas[i]
        def poison(req,params,r=replica):
            good=r.compute(req,params)
            return sign_result(r._private,r.identity,req,ComputeResult.from_values(999.,
                [np.full_like(g,-100.) for g in good.arrays()]))
        replica.reply=poison
    attacked=poisoned_outbound_candidate();attacked.select({c:(2*c,2*c+1) for c in range(10)})
    assert honest.selection['sender']==attacked.selection['sender']==9
    scores={record['senderId']:record['balancedTrainingLoss'] for record in attacked.selection['records']}
    assert scores[7]>scores[9]
    attacked.advance(unit_batch=batch)
    for a,b in zip(honest.params,attacked.params):np.testing.assert_array_equal(a,b)


def test_corrupt_checkpoint_and_interrupted_write_fail_safely(setup,tmp_path,monkeypatch):
    import basil_core.verified_cart_runner as module
    make,*_=setup;r=make();path=tmp_path/'state.json';r.save(path);before=path.read_bytes()
    def interrupted(*_):raise OSError('Injected disk failure before commit')
    with monkeypatch.context() as m:
        m.setattr(module.os,'replace',interrupted)
        with pytest.raises(OSError):r.save(path)
    assert path.read_bytes()==before and not list(tmp_path.glob('.verified-step-*'))
    raw=json.loads(before);raw['state']['node']=9;path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='integrity'):make().restore(path)
    path.write_text('{')
    with pytest.raises(ValueError):make().restore(path)


def test_five_epoch_schedule_and_last_batch(setup):
    import numpy as np
    make,*_=setup;r=make()
    r.reference=SimpleNamespace(labels=np.repeat(np.arange(10,dtype=np.int32),5000))
    for epoch in range(5):
        seen=[]
        for batch in range(10):
            r.step=epoch*10+batch
            ids,key,helpers=r.next_batch()
            assert len(ids)==(392 if batch==9 else 512)
            seen.extend(ids)
            assert len(helpers)==9
        assert sorted(seen)==list(range(5000))
    r.step=50
    with pytest.raises(ValueError):r.next_batch()


def test_finish_requires_real_fifty_steps(setup):
    make,*_=setup;r=make()
    with pytest.raises(ValueError,match='fifty'):r.finish_activation()
    r.step=50
    with pytest.raises(ValueError,match='Five full'):r.finish_activation()


def test_wire_rejects_nonfinite_oversize_and_object_payloads():
    from basil_core.replica_wire import decode,unpack_arrays,MAX_FRAME
    for raw in (b'NaN',b'Infinity',b'x'*(MAX_FRAME+1),b'\x80\x04pickle'):
        with pytest.raises((ValueError,UnicodeError)):decode(raw)
    with pytest.raises(ValueError):unpack_arrays([dict(shape=[10**9],data='')])


def test_sandbox_mounts_exclude_home_credentials_and_gpu(tmp_path):
    from basil_core.isolated_replicas import sandbox_command
    fixture=tmp_path/'train.npz';fixture.touch()
    command=sandbox_command(fixture,0,'none')
    assert '--unshare-all' in command and '--clearenv' in command
    assert '/home/ojaswi/.netrc' not in command and '/dev/dxg' not in command
    assert not any(x in command for x in ('--share-net','--bind','--dev-bind'))
    assert 'CUDA_VISIBLE_DEVICES' in command


def test_pipe_deadline_kills_only_owned_worker():
    from basil_core.isolated_replicas import SandboxEndpoint
    endpoint=SandboxEndpoint.__new__(SandboxEndpoint)
    endpoint.process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
    process=endpoint.process
    try:
        with pytest.raises(TimeoutError):endpoint._read(.02)
    finally:endpoint.close()
    assert process.poll() is not None


def test_new_modules_remain_lazy():
    code=('import basil_core.replica_wire,basil_core.sandbox_replica,basil_core.isolated_replicas,'
          'basil_core.verified_cart_runner;import sys;'
          'assert not ({"tensorflow","numpy","cryptography","wandb"}&sys.modules.keys())')
    subprocess.run([sys.executable,'-c',code],check=True)
