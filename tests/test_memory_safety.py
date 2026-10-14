"""Memory guards never clear sessions, change tensors, or launch training."""
from types import SimpleNamespace
import pytest
from basil_core import memory_safety as safety


def test_guard_keeps_live_state_and_collects_only_dead_objects(monkeypatch):
    calls=[]
    tf=SimpleNamespace(config=SimpleNamespace(experimental=SimpleNamespace(
        get_memory_info=lambda device:dict(current=128*1024**2,peak=400*1024**2))))
    monkeypatch.setattr(safety,'host_available_mib',lambda:8000)
    monkeypatch.setattr(safety.gc,'collect',lambda *args:(calls.append(args) or 3))
    guard=safety.RuntimeMemoryGuard(tf,'GPU')
    result=guard.checkpoint()
    assert calls==[(0,)] and result['gpuCurrentMiB']==128
    assert result['maximumGpuPeakMiB']==400 and result['gpuLimitMiB']==6144
    assert result['pressureEvents']==0 and result['unreachableObjectsCollected']==3


def test_memory_pressure_collects_without_touching_models(monkeypatch):
    calls=[]
    tf=SimpleNamespace(config=SimpleNamespace(experimental=SimpleNamespace(
        get_memory_info=lambda device:dict(current=6000*1024**2,peak=6100*1024**2))))
    monkeypatch.setattr(safety,'host_available_mib',lambda:3000)
    monkeypatch.setattr(safety.gc,'collect',lambda *args:(calls.append(args) or 2))
    result=safety.RuntimeMemoryGuard(tf,'GPU').checkpoint()
    assert calls==[()] and result['pressureEvents']==1


def test_low_host_ram_blocks_before_dataset_load(monkeypatch):
    monkeypatch.setattr(safety,'host_available_mib',lambda:1000)
    with pytest.raises(RuntimeError,match='before loading CIFAR'):safety.check_host_memory()


def test_cpu_guard_never_accesses_cuda(monkeypatch):
    monkeypatch.setattr(safety,'host_available_mib',lambda:9000)
    monkeypatch.setattr(safety.gc,'collect',lambda *args:0)
    assert safety.RuntimeMemoryGuard(None,'CPU').checkpoint()['gpuLimitMiB'] is None
