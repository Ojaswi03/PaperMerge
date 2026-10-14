"""Recovery/resource contracts, without host cgroup changes in unit tests."""
import json
from types import SimpleNamespace

import pytest

from test_isolated_cart_runner import setup


@pytest.mark.parametrize('failure',['timeout','exit','refused','invalid_message','incorrect_update'])
def test_explicit_recovery_equals_honest_step(setup,tmp_path,failure):
    import numpy as np
    from basil_core.verified_cart_runner import VerifiedCartRunner
    make,service,_,ref,batch=setup
    honest=make();honest.select({c:(2*c,2*c+1) for c in range(10)})
    recovered=VerifiedCartRunner(make().model,ref,service,run_id='unit',ebm=True,sigma_e=.01,
        trainer_failure_policy='verified_coordinator_recompute')
    recovered.round=1
    for memory in recovered.memories:
        for s in memory.candidates():s.round_id=0 if s.sender_id>=memory.receiver_id else 1
    recovered.select({c:(2*c,2*c+1) for c in range(10)})
    normal=honest.advance(unit_batch=batch)
    kwargs=dict(proposed_params=[np.zeros_like(p) for p in recovered.params]) if failure=='incorrect_update' else dict(proposal_failure=failure)
    receipt=recovered.advance(unit_batch=batch,**kwargs)
    assert receipt['outputHash']==normal['outputHash'] and receipt['objective']==normal['objective']
    assert receipt['recoveredByCoordinator'] and receipt['trainerFailure']==failure
    assert recovered.step==1 and receipt['samples']==2
    recovered.save(tmp_path/'checkpoint.json')
    assert json.loads((tmp_path/'checkpoint.json').read_text())['state']['records'][0]['recoveredByCoordinator']


def test_recovery_does_not_bypass_missing_helper_quorum(setup):
    from basil_core.verified_cart_runner import VerifiedCartRunner
    make,service,replicas,ref,batch=setup
    r=VerifiedCartRunner(make().model,ref,service,run_id='unit',trainer_failure_policy='verified_coordinator_recompute')
    r.round=1;r.anchor=[p.copy() for p in r.params]
    for identity in (0,2,3,4,5):
        def unavailable(*_):raise TimeoutError('Injected unavailable response')
        replicas[identity].reply=unavailable
    with pytest.raises(ValueError,match='certificate'):
        r.advance(unit_batch=batch,proposal_failure='refused')
    assert r.step==0 and not r.records


def test_checkpoint_disk_guard_preserves_previous(setup,tmp_path,monkeypatch):
    import basil_core.verified_cart_runner as module
    make,*_=setup;r=make();path=tmp_path/'checkpoint.json';r.save(path);old=path.read_bytes()
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _:SimpleNamespace(free=128))
    with pytest.raises(OSError,match='headroom'):r.save(path)
    assert path.read_bytes()==old


def test_resource_limits_are_bounded_and_unique():
    from basil_core.replica_resources import ResourceScope,MIB,POOL_MEMORY_CEILING_BYTES
    a,b=ResourceScope.create(),ResourceScope.create()
    assert a.unit!=b.unit and POOL_MEMORY_CEILING_BYTES==20*1024**3
    command=a.command(['/usr/bin/true'])
    for flag in ('--property=MemoryMax=2147483648','--property=MemorySwapMax=0',
                 '--property=OOMPolicy=kill','--property=TasksMax=64','--property=CPUQuota=100%'):
        assert flag in command
    with pytest.raises(ValueError):ResourceScope.create(memory_bytes=4096*MIB)


def test_gpu_requires_actual_placement_not_label(setup):
    from basil_core.verified_cart_runner import VerifiedCartRunner
    make,service,_,ref,_=setup
    with pytest.raises(ValueError,match='actual model placement'):
        VerifiedCartRunner(make().model,ref,service,run_id='not-gpu',device='GPU')
