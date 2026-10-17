"""Resource contracts use synthetic inventories, never allocated GPU kernels."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from basil_core import execution_profile as profiles
from basil_core import iid_runtime as runtime


@pytest.fixture(autouse=True)
def isolated_profile(monkeypatch):
    monkeypatch.delenv(profiles.ENV,raising=False)


def inventory():
    return [dict(index=i,uuid=f'GPU-{i:08x}-abcd',name=f'Test GPU {i}',
        **{'memory.total':97887,'memory.free':2000 if i==0 else 90000,
           'memory.used':95887 if i==0 else 7887,'utilization.gpu':0,
           'temperature.gpu':40,'driver_version':'test-driver'}) for i in range(4)]


def policy(tmp_path, **changes):
    return replace(profiles.ExecutionProfile(name='hpc',gpu=1,memory_mib=8192,
        headroom_mib=2048,output_root=str(tmp_path/'runs')),**changes).validate()


def activate(monkeypatch,value):
    monkeypatch.setenv(profiles.ENV,value.environment()[profiles.ENV])


def test_legacy_profile_preserves_local_environment():
    p=profiles.current_profile()
    assert (p.name,p.gpu,p.memory_mib,p.headroom_mib,p.threads)==('local',0,6144,1024,1)
    env=profiles.worker_environment('GPU')
    assert env['CUDA_VISIBLE_DEVICES']=='0' and env['TF_NUM_INTRAOP_THREADS']=='1'
    assert profiles.worker_environment('CPU')['CUDA_VISIBLE_DEVICES']=='-1'
    assert profiles.profile_path(profiles.ROOT/'newResults/a')==profiles.ROOT/'newResults/a'
    with pytest.raises(ValueError):replace(p,gpu=1).validate()
    with pytest.raises(ValueError):replace(p,memory_mib=8192).validate()


def test_selected_physical_gpu_maps_to_one_visible_logical_gpu(tmp_path,monkeypatch):
    p=policy(tmp_path).bind(inventory());activate(monkeypatch,p)
    monkeypatch.setattr(profiles,'gpu_inventory',inventory)
    env=profiles.worker_environment('GPU',{'EXISTING':'value'})
    assert env['CUDA_VISIBLE_DEVICES']==inventory()[1]['uuid']
    assert env['EXISTING']=='value'
    assert env['TF_DETERMINISTIC_OPS']=='1' and env['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
    actual=profiles.resource_identity()
    assert actual['physicalGpuId']==1 and actual['physicalGpuUuid']==inventory()[1]['uuid']
    assert actual['logicalTensorflowDevice']=='/GPU:0'
    assert profiles.verification_path().is_relative_to(tmp_path/'runs/verification')


def test_free_memory_checks_selected_gpu_not_first(tmp_path,monkeypatch):
    monkeypatch.setattr(profiles,'gpu_inventory',inventory)
    p=policy(tmp_path).bind(inventory());activate(monkeypatch,p)
    assert runtime.free_gpu_memory()==90000
    activate(monkeypatch,replace(p,gpu=0,gpu_uuid=inventory()[0]['uuid']))
    with pytest.raises(RuntimeError,match='Physical GPU 0'):runtime.free_gpu_memory()


def test_prelaunch_resource_checks_do_not_pool_vram(tmp_path):
    p=policy(tmp_path)
    profiles.validate_resources([p],inventory())
    with pytest.raises(RuntimeError,match='Physical GPU 0'):
        profiles.validate_resources([replace(p,gpu=0)],inventory())
    with pytest.raises(ValueError,match='distinct physical GPUs'):
        profiles.validate_resources([p,p],inventory())


def test_discovery_and_missing_gpu(tmp_path,monkeypatch):
    output='1, GPU-00000001-abcd, RTX PRO 6000, 97887, 90000, 7887, 87, 42, 580.126.20\n'
    calls=[]
    def run(*args,**kwargs):
        calls.append(args[0]);return SimpleNamespace(stdout=output)
    monkeypatch.setattr(profiles.subprocess,'run',run)
    devices=profiles.gpu_inventory()
    assert devices[0]['index']==1 and devices[0]['memory.free']==90000
    assert '--format=csv,noheader,nounits' in calls[0]
    with pytest.raises(RuntimeError,match='unavailable'):policy(tmp_path,gpu=3).bind(devices)
    with pytest.raises(RuntimeError,match='UUID changed'):
        policy(tmp_path,gpu_uuid='GPU-ffffffff').bind(devices)
    monkeypatch.setattr(profiles.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='broken'))
    with pytest.raises(RuntimeError,match='format'):profiles.gpu_inventory()


@pytest.mark.parametrize('changes',[dict(memory_mib=0),dict(headroom_mib=0),dict(threads=64),
    dict(data_workers=1),dict(gpu=-1),dict(ram_mib=128),dict(gpu_uuid='GPU-;echo')])
def test_invalid_resource_policy(tmp_path,changes):
    with pytest.raises(ValueError):policy(tmp_path,**changes)


@pytest.mark.parametrize('name',['newResults','newPlots','Papers','gui','scripts','.git','environment'])
def test_hpc_cannot_route_into_protected_repository_trees(tmp_path,name):
    with pytest.raises(ValueError):policy(tmp_path,output_root=str(profiles.ROOT/name/'hpc'))


def test_output_routing_preserves_scientific_configs(tmp_path,monkeypatch):
    activate(monkeypatch,policy(tmp_path))
    for prefix in ('newResults','newPlots','gui/worker_state'):
        target=profiles.profile_path(profiles.ROOT/prefix/'condition')
        assert target==tmp_path/'runs'/prefix/'condition'
        assert profiles.profile_path(target)==target
    assert profiles.profile_path(profiles.ROOT/'Papers')==profiles.ROOT/'Papers'
    assert profiles.profile_path(profiles.ROOT/'experiments/cache')==profiles.ROOT/'experiments/cache'


def test_gpu_lock_conflict_is_per_physical_uuid(tmp_path,monkeypatch):
    import fcntl
    import os
    p=policy(tmp_path).bind(inventory());activate(monkeypatch,p)
    monkeypatch.setattr(profiles,'gpu_inventory',inventory)
    monkeypatch.setattr(runtime,'_gpu_lease',None)
    # The UUID is synthetic and cannot overlap the host's physical GPUs.
    from pathlib import Path
    path=Path('/tmp')/f'papermerge-{os.getuid()}-{p.gpu_uuid}.lock'
    with path.open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='conflicting worker'):runtime.acquire_gpu_lease()
    runtime.acquire_gpu_lease()
    assert runtime._gpu_lease
    for handle in runtime._gpu_lease:handle.close()


def test_configure_tensorflow_cap_no_cpu_fallback(tmp_path,monkeypatch):
    activate(monkeypatch,policy(tmp_path))
    calls=[]
    monkeypatch.setattr(runtime,'acquire_gpu_lease',lambda:None)
    monkeypatch.setattr(runtime,'free_gpu_memory',lambda:90000)
    tf=SimpleNamespace(config=SimpleNamespace(threading=SimpleNamespace(
        set_intra_op_parallelism_threads=lambda n:calls.append(('intra',n)),
        set_inter_op_parallelism_threads=lambda n:calls.append(('inter',n))),
        experimental=SimpleNamespace(enable_op_determinism=lambda:calls.append('determinism'),
            enable_tensor_float_32_execution=lambda b:calls.append(('tf32',b))),
        list_physical_devices=lambda _:['physical_visible_0'],
        LogicalDeviceConfiguration=lambda **kwargs:kwargs,
        set_logical_device_configuration=lambda d,c:calls.append((d,c))))
    runtime.configure_tensorflow(tf,'GPU')
    assert ('physical_visible_0',[{'memory_limit':8192}]) in calls
    assert ('tf32',False) in calls
    tf.config.list_physical_devices=lambda _:[]
    with pytest.raises(RuntimeError,match='No CPU fallback'):runtime.configure_tensorflow(tf,'GPU')


def test_recovery_resource_identity_is_strict():
    old={'device':'GPU','numericalExecution':{'resources':{'gpu':'old'}}}
    new={'device':'GPU','numericalExecution':{'resources':{'gpu':'new'}}}
    with pytest.raises(ValueError,match='same validated'):runtime.check_recovery_device(old,new)


def test_hpc_fingerprint_is_explicit_and_legacy_scope_remains(tmp_path,monkeypatch):
    from basil_core.protocol_checkpoint import implementation_fingerprint
    local=implementation_fingerprint()
    assert len(local)==6 and 'basil_core/iid_runtime.py' not in local
    activate(monkeypatch,policy(tmp_path))
    hpc=implementation_fingerprint()
    assert all(hpc[k]==v for k,v in local.items())
    assert 'basil_core/execution_profile.py' in hpc and 'scripts/hpc.py' in hpc


def test_scoped_profile_restores_controller_after_failed_worker_check(tmp_path,monkeypatch):
    first=policy(tmp_path);second=replace(first,gpu=2)
    activate(monkeypatch,first)
    with pytest.raises(ValueError,match='synthetic rejection'):
        with profiles.use_profile(second):
            assert profiles.current_profile()==second
            raise ValueError('synthetic rejection')
    assert profiles.current_profile()==first
    monkeypatch.delenv(profiles.ENV)
    with profiles.use_profile(first):assert profiles.current_profile()==first
    assert profiles.ENV not in profiles.os.environ
