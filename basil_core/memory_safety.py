"""Observation and unreachable-object cleanup; never alter live training state."""
from __future__ import annotations
import gc
from pathlib import Path

from basil_core.iid_runtime import GPU_MEMORY_LIMIT_MB


def host_available_mib():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])/1024
    except (OSError,ValueError):
        pass
    return None


def check_host_memory(minimum_mib=4096):
    available=host_available_mib()
    if available is not None and available<minimum_mib:
        raise RuntimeError(f'Only {available:.0f} MiB host RAM available; at least {minimum_mib} MiB is required before loading CIFAR. Close memory-heavy applications.')
    return available


class RuntimeMemoryGuard:
    """Release unreachable Python objects only; retain all live tensors/caches.

    Clearing Keras sessions/graphs or forcibly emptying a CUDA allocator during
    SGD is unsafe. TensorFlow reuses cached blocks inside its bounded allocator.
    We measure usage, collect dead objects, and preserve necessary model, SGD,
    BASIL memories and telemetry. No cache reset is disguised as freeing VRAM.
    """
    def __init__(self, tf, device):
        self.tf=tf
        self.device=device
        self.metrics={'gpuLimitMiB':GPU_MEMORY_LIMIT_MB if device=='GPU' else None,
            'cleanupCalls':0,'unreachableObjectsCollected':0,'maximumGpuPeakMiB':0.,
            'pressureEvents':0}

    def checkpoint(self):
        available=host_available_mib()
        gpu={}
        if self.device=='GPU':
            try:gpu=self.tf.config.experimental.get_memory_info('GPU:0')
            except (ValueError,RuntimeError):pass
        current=gpu.get('current',0)/1024**2
        peak=gpu.get('peak',0)/1024**2
        pressure=(available is not None and available<4096) or current>.85*GPU_MEMORY_LIMIT_MB
        collected=gc.collect() if pressure else gc.collect(0)
        self.metrics.update(cleanupCalls=self.metrics['cleanupCalls']+1,
            unreachableObjectsCollected=self.metrics['unreachableObjectsCollected']+collected,
            maximumGpuPeakMiB=max(self.metrics['maximumGpuPeakMiB'],peak),
            gpuCurrentMiB=current,gpuPeakMiB=peak,hostAvailableMiB=available,
            pressureEvents=self.metrics['pressureEvents']+int(pressure))
        return dict(self.metrics)
