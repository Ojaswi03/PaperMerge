"""Opt-in machine resources and output routing; no scientific configuration changes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENV = 'PAPERMERGE_EXECUTION_PROFILE'


@dataclass(frozen=True)
class ExecutionProfile:
    name: str = 'local'
    gpu: int = 0
    gpu_uuid: str = ''
    memory_mib: int = 6144
    headroom_mib: int = 1024
    threads: int = 1
    ram_mib: int = 32768
    data_workers: int = 0
    output_root: str = ''

    def validate(self):
        if self.name not in ('local','hpc'):
            raise ValueError('Execution profile must be local or hpc.')
        for key,low,high in (('gpu',0,1023),('memory_mib',512,1048576),
                            ('headroom_mib',256,1048576),('threads',1,8),('ram_mib',4096,1048576)):
            value=getattr(self,key)
            if type(value) is not int or not low<=value<=high:
                raise ValueError(f'{key} must be an integer in [{low}, {high}].')
        if type(self.data_workers) is not int or self.data_workers!=0:
            raise ValueError('The deterministic research workers require data_workers=0; no loader processes are authorized.')
        if self.gpu_uuid and (not self.gpu_uuid.startswith('GPU-') or
                             any(c not in '0123456789abcdefABCDEF-GPU' for c in self.gpu_uuid)):
            raise ValueError('Invalid NVIDIA GPU UUID.')
        if self.name=='local' and (self.gpu!=0 or self.memory_mib!=6144 or
                self.headroom_mib!=1024 or self.threads!=1 or self.output_root):
            raise ValueError('The local profile preserves GPU 0, 6144 MiB, 1024 MiB headroom and one thread.')
        if self.name=='hpc':
            if not self.output_root or not Path(self.output_root).is_absolute():
                raise ValueError('HPC requires an absolute, separate output namespace.')
            target=Path(self.output_root).resolve()
            if ROOT.is_relative_to(target) or any(target==ROOT/p or target.is_relative_to(ROOT/p)
                                  for p in ('newResults','newPlots','gui','basil_core','scripts','Papers','papers','environment','.venv-hpc','.git')):
                raise ValueError('HPC output root must be separate from existing results, source and environments.')
        return self

    def environment(self):
        self.validate()
        return {ENV:json.dumps(asdict(self),sort_keys=True), 'MPLBACKEND':'Agg',
                'PYTHONUNBUFFERED':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}

    def bind(self, gpus):
        matches=[g for g in gpus if g['index']==self.gpu]
        if len(matches)!=1:
            raise RuntimeError(f'Requested physical GPU {self.gpu} is unavailable.')
        if self.gpu_uuid and self.gpu_uuid!=matches[0]['uuid']:
            raise RuntimeError('Selected GPU UUID changed; refusing device substitution.')
        return replace(self,gpu_uuid=matches[0]['uuid']).validate()


def current_profile():
    raw=os.environ.get(ENV)
    return ExecutionProfile(**json.loads(raw)).validate() if raw else ExecutionProfile()


@contextmanager
def use_profile(profile):
    """Scope read-only validation to one worker's resources in the controller."""
    previous=os.environ.get(ENV)
    os.environ[ENV]=profile.environment()[ENV]
    try:yield
    finally:
        if previous is None:os.environ.pop(ENV,None)
        else:os.environ[ENV]=previous


def profile_path(path):
    """Route only generated results/plots/receipts, leaving source and data intact."""
    value=Path(path).resolve()
    profile=current_profile()
    if profile.name=='hpc':
        for prefix in ('newResults','newPlots','gui/worker_state'):
            base=ROOT/prefix
            if value==base or value.is_relative_to(base):
                return Path(profile.output_root)/value.relative_to(ROOT)
    return value


def gpu_inventory():
    fields=('index','uuid','name','memory.total','memory.free','memory.used',
            'utilization.gpu','temperature.gpu','driver_version')
    try:
        response=subprocess.run(['nvidia-smi','--query-gpu='+','.join(fields),
            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    except (OSError,subprocess.SubprocessError) as error:
        raise RuntimeError('nvidia-smi GPU discovery failed; verify driver access and allocation.') from error
    result=[]
    for row in csv.reader(io.StringIO(response.stdout),skipinitialspace=True):
        if not row:continue
        if len(row)!=len(fields):raise RuntimeError('Unexpected nvidia-smi GPU inventory format.')
        item=dict(zip(fields,(v.strip() for v in row)))
        for key in fields:
            if key in ('uuid','name','driver_version'):continue
            try:item[key]=int(item[key])
            except ValueError:
                if key in ('index','memory.total','memory.free','memory.used'):
                    raise RuntimeError(f'nvidia-smi did not report usable {key}.')
                item[key]=None
        result.append(item)
    return result


def gpu_processes():
    try:
        output=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
        return [dict(zip(('gpu_uuid','pid','process','memory_mib'),row))
                for row in csv.reader(io.StringIO(output.stdout),skipinitialspace=True) if len(row)==4]
    except (OSError,subprocess.SubprocessError):return None


def validate_resources(policies, inventory):
    """Check each selected device independently; VRAM is never pooled."""
    seen=set()
    for policy in policies:
        policy=policy.bind(inventory)
        if policy.gpu_uuid in seen:raise ValueError('Independent workers need distinct physical GPUs.')
        seen.add(policy.gpu_uuid)
        gpu=next(g for g in inventory if g['index']==policy.gpu)
        required=policy.memory_mib+policy.headroom_mib
        if gpu['memory.free']<required:
            raise RuntimeError(f'Physical GPU {policy.gpu} has {gpu["memory.free"]} MiB free; '
                f'at least {required} MiB is required for this worker and headroom.')


def verification_path(profile=None):
    profile=profile or current_profile()
    if profile.name=='local':return ROOT/'gui/worker_state/GPU_EXECUTION_VERIFICATION.json'
    identity=hashlib.sha256(json.dumps(asdict(profile),sort_keys=True).encode()).hexdigest()[:20]
    return Path(profile.output_root)/'verification'/f'{socket.gethostname()}-{identity}.json'


def worker_environment(device, base=None, profile=None):
    profile=profile or current_profile()
    values=dict(base or {})
    values.update(PAPERMERGE_IID_DEVICE=device,
        CUDA_VISIBLE_DEVICES=((profile.gpu_uuid or str(profile.gpu)) if profile.name=='hpc' else '0') if device=='GPU' else '-1',
        TF_ENABLE_ONEDNN_OPTS='0',TF_DETERMINISTIC_OPS='1',
        TF_NUM_INTRAOP_THREADS=str(profile.threads),TF_NUM_INTEROP_THREADS=str(profile.threads),
        OMP_NUM_THREADS=str(profile.threads),OPENBLAS_NUM_THREADS=str(profile.threads),MKL_NUM_THREADS=str(profile.threads))
    if profile.name=='hpc':values.update(profile.environment())
    return values


def resource_identity(profile=None):
    profile=profile or current_profile()
    from importlib.metadata import PackageNotFoundError, version
    import platform
    software={'python':platform.python_version()}
    packages=('keras','numpy')
    if profile.name=='hpc':
        packages+=('tensorflow','scipy','h5py','ml-dtypes','protobuf')
        packages+=tuple(f'nvidia-{library}-{generation}' for generation in ('cu12','cu13')
            for library in ('cublas','cuda-cupti','cuda-nvcc','cuda-nvrtc','cuda-runtime',
                            'cudnn','cufft','curand','cusolver','cusparse','nccl','nvjitlink'))
    for package in packages:
        try:software[package]=version(package)
        except PackageNotFoundError:software[package]=None
    driver=None
    if profile.name=='hpc':
        devices=gpu_inventory();profile.bind(devices)
        driver=next(g['driver_version'] for g in devices if g['index']==profile.gpu)
    return dict(profile=profile.name,physicalGpuId=profile.gpu,physicalGpuUuid=profile.gpu_uuid,
        logicalTensorflowDevice='/GPU:0',host=socket.gethostname(),gpuMemoryLimitMb=profile.memory_mib,
        gpuHeadroomMb=profile.headroom_mib,cpuThreads=profile.threads,hostRamBudgetMiB=profile.ram_mib,
        dataLoadingWorkers=profile.data_workers,outputRoot=profile.output_root,driverVersion=driver,software=software)
