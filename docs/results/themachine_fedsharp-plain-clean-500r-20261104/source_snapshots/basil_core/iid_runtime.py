"""Explicit serial CPU/GPU execution policy; never choose a device silently."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
from pathlib import Path
from basil_core.execution_profile import current_profile, resource_identity, verification_path, worker_environment

ROOT = Path(__file__).resolve().parents[1]
GPU_VERIFICATION = verification_path()
POLICY_REVISION = 'iid_serial_float32_v1'
GPU_MEMORY_LIMIT_MB = 6144
GPU_HEADROOM_MB = 1024
_gpu_lease = None


def gpu_worker_hash() -> str:
    return hashlib.sha256((ROOT/'basil_core/iid_gpu_worker.py').read_bytes()).hexdigest()


def execution_device(config: dict) -> str:
    device = config.get('executionDevice', 'CPU')
    if device not in {'CPU', 'GPU'}:
        raise ValueError('executionDevice must be CPU or GPU.')
    return device


def prepare_environment(device: str) -> dict[str, str]:
    execution_device({'executionDevice': device})
    values = worker_environment(device)
    os.environ.update(values)
    return values


def configure_tensorflow(tf, device: str) -> None:
    profile=current_profile()
    tf.config.threading.set_intra_op_parallelism_threads(profile.threads)
    tf.config.threading.set_inter_op_parallelism_threads(profile.threads)
    tf.config.experimental.enable_op_determinism()
    if device == 'GPU':
        acquire_gpu_lease()
        free_gpu_memory()
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError('GPU was requested but TensorFlow cannot access CUDA. No CPU fallback was started.')
        tf.config.set_logical_device_configuration(gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit())])
        # Preserve float32 arithmetic rather than using Ampere/Ada TF32 kernels.
        tf.config.experimental.enable_tensor_float_32_execution(False)


def free_gpu_memory() -> int:
    profile=current_profile()
    if profile.name=='hpc':
        from basil_core.execution_profile import gpu_inventory
        devices=gpu_inventory()
        profile.bind(devices)
        free=next(g['memory.free'] for g in devices if g['index']==profile.gpu)
        required=profile.memory_mib+profile.headroom_mib
        if free<required:
            raise RuntimeError(f'Physical GPU {profile.gpu} has {free} MiB free; at least {required} MiB is required.')
        return free
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
            check=True,capture_output=True,text=True,timeout=10)
        free=int(result.stdout.splitlines()[0].strip())
    except (OSError,ValueError,IndexError,subprocess.SubprocessError) as error:
        raise RuntimeError('Cannot verify free GPU memory; research was not started.') from error
    required=GPU_MEMORY_LIMIT_MB+GPU_HEADROOM_MB
    if free<required:
        raise RuntimeError(f'GPU has {free} MiB free; at least {required} MiB is required '
            f'for the {GPU_MEMORY_LIMIT_MB} MiB cap plus safety headroom. Close other GPU applications.')
    return free


def acquire_gpu_lease() -> None:
    global _gpu_lease
    if _gpu_lease is not None:return
    import fcntl
    profile=current_profile()
    from basil_core.execution_profile import gpu_inventory
    bound=profile.bind(gpu_inventory())
    paths=[Path('/tmp')/f'papermerge-{os.getuid()}-{bound.gpu_uuid}.lock']
    if profile.name=='local':paths.append(ROOT/'gui/worker_state/iid/gpu.lock')
    handles=[]
    try:
        for path in paths:
            path.parent.mkdir(parents=True,exist_ok=True)
            handle=path.open('a');handles.append(handle)
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as error:
        for handle in handles:handle.close()
        raise RuntimeError(f'Another PaperMerge worker holds physical GPU {profile.gpu}; refusing a conflicting worker.') from error
    _gpu_lease=handles  # OS releases these locks on normal exit or a crash.


def gpu_memory_limit():
    profile=current_profile()
    return profile.memory_mib if profile.name=='hpc' else GPU_MEMORY_LIMIT_MB


def runtime_metadata(tf, config: dict) -> dict:
    device = execution_device(config)
    gpu_name = None
    if device == 'GPU':
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError('GPU requested but no CUDA device is available. Refusing CPU fallback.')
        gpu_name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', gpus[0].name)
        logical=tf.config.get_logical_device_configuration(gpus[0])
        if not logical or logical[0].memory_limit!=gpu_memory_limit():
            raise RuntimeError(f'GPU requires the validated {gpu_memory_limit()} MiB memory cap before execution.')
        if tf.config.experimental.tensor_float_32_execution_enabled():
            raise RuntimeError('The validated GPU policy requires TF32 to be disabled.')
    numerical = dict(policyRevision=POLICY_REVISION, device=device, gpuName=gpu_name,
        tensorflowVersion=tf.__version__, tensorFloat32=False if device == 'GPU' else None,
        intraOpThreads=tf.config.threading.get_intra_op_parallelism_threads(),
        interOpThreads=tf.config.threading.get_inter_op_parallelism_threads())
    if device == 'GPU':
        build = tf.sysconfig.get_build_info()
        numerical.update(cudaVersion=build.get('cuda_version'), cudnnVersion=build.get('cudnn_version'),
            gpuWorkerSha256=gpu_worker_hash(),gpuMemoryLimitMb=gpu_memory_limit())
    if current_profile().name=='hpc':numerical['resources']=resource_identity()
    verified = False
    if device == 'GPU':
        from basil_core.protocol_checkpoint import implementation_fingerprint
        try:
            path=verification_path() if current_profile().name=='hpc' else GPU_VERIFICATION
            report = json.loads(path.read_text())
            verified = (report.get('status') == 'passed' and
                report.get('numericalExecution') == numerical and
                report.get('implementationSha256') == implementation_fingerprint())
        except (OSError, ValueError):
            pass
        if not verified:
            raise RuntimeError('GPU verification is missing or does not match this hardware/software. '
                'Run scripts/verify_iid_gpu.py before starting GPU research.')
    return dict(device=device, deviceName=gpu_name or 'CPU', tensorflowDevice=f'/{device}:0',
        gpuVerified=verified, gpuVerificationScope='synthetic batch CE/EBM, SGD, evaluation and same-device repeatability' if verified else None,
        numericalExecution=numerical, intraOpThreads=numerical['intraOpThreads'],
        interOpThreads=numerical['interOpThreads'])


def check_recovery_device(previous: dict, runtime: dict) -> None:
    if previous.get('device', 'CPU') != runtime['device']:
        raise ValueError('Cannot exactly resume CPU evidence on GPU or GPU evidence on CPU. Use a separate fresh output path.')
    if runtime['device'] == 'GPU' and previous.get('numericalExecution') != runtime['numericalExecution']:
        raise ValueError('GPU recovery requires the same validated hardware and numerical execution policy.')
