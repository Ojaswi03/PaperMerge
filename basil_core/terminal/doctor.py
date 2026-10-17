"""Non-research hardware diagnostics in isolated verification subprocesses."""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from basil_core.execution_profile import ROOT, gpu_inventory
from basil_core.terminal.state import logger, safe_text, read_json


def dependencies():
    result={}
    for package in ('tensorflow','keras','numpy','scipy','scikit-learn','h5py','matplotlib','pillow','tqdm','psutil','wandb'):
        try:result[package]=version(package)
        except PackageNotFoundError:result[package]=None
    return result


def writable(path):
    path=Path(path)
    try:
        path.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=path) as stream:stream.write(b'permission check');stream.flush()
        return True
    except OSError:return False


def diagnose(policy, state_dir, *, verify_gpu=True, objective_checks=True, runner=subprocess.run):
    from basil_core.terminal.dashboard import system_info
    data=dict(os=platform.platform(),kernel=platform.release(),python=sys.version.split()[0],
        executable=sys.executable,virtualEnvironment=sys.prefix if sys.prefix!=sys.base_prefix else None,
        cpuLogical=os.cpu_count(),dependencies=dependencies(),
        tools={t:shutil.which(t) for t in ('python3','pip3','gcc','g++','nvidia-smi','tmux','nvcc','sbatch','sinfo','clang')},
        errors=[],gpuVerified=False,researchRunsStarted=0,
        trackingAuthentication='Not read by doctor; use wandb login. Required tracking is checked by the worker before updates.')
    data['system']=system_info()
    try:
        import psutil
        data['cpuPhysical']=psutil.cpu_count(logical=False)
    except ImportError:data['cpuPhysical']=None
    data['cpuAffinityCount']=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    data['dataCache']={name:(ROOT/'experiments/cache/cifar10'/name).is_file() for name in
        ('x_train_float32.npy','y_train_int32.npy','x_test_float32.npy','y_test_int32.npy')}
    from basil_core.execution_profile import profile_path
    data['researchEvidence']={name:profile_path(ROOT/'gui/worker_state'/name).is_file() for name in
        ('dirichlet_horizon_readiness.json','dirichlet_prior_100r_validation.json',
         'dirichlet_prior_readiness.json','CART_RESOURCE_LIMITED_RECOVERY_GPU.json')}
    data['researchReady']=False  # Only the selected queue's exact gates can establish readiness.
    from basil_core.terminal.queues import tracking_availability
    data['tracking']=tracking_availability({})
    try:
        inventory=gpu_inventory();data['gpus']=inventory;policy=policy.bind(inventory)
        gpu=next(g for g in inventory if g['index']==policy.gpu)
        if gpu['memory.free']<policy.memory_mib+policy.headroom_mib:
            raise ValueError('Selected GPU lacks cap plus required safety headroom.')
    except (RuntimeError,ValueError) as error:data['errors'].append(str(error))
    data['resources']=asdict(policy)
    data['writable']={str(p):writable(p) for p in (Path(state_dir),Path(policy.output_root) if policy.output_root else ROOT/'newResults')}
    if not all(data['writable'].values()):data['errors'].append('An operational/output directory is not writable.')
    if sys.version_info[:2]!=(3,12):data['errors'].append('The reproducible deployment profile requires Python 3.12.')
    if platform.system()!='Linux':data['errors'].append('The terminal resource/locking backend requires Linux or WSL.')
    if not data['virtualEnvironment']:data['errors'].append('Activate a dedicated virtual environment before execution.')
    required=('tensorflow','keras','numpy','scipy','scikit-learn','h5py','matplotlib','pillow','tqdm')
    if policy.name=='hpc':required+=('psutil',)
    if any(data['dependencies'][p] is None for p in required):data['errors'].append('Required scientific dependencies are missing.')
    try:
        check=runner([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=60)
        data['dependencyCheck']='passed' if check.returncode==0 else safe_text(check.stdout+check.stderr)
        if check.returncode:data['errors'].append('pip check found incompatible dependencies.')
    except (OSError,subprocess.SubprocessError) as error:data['errors'].append('pip check failed: '+str(error))
    if verify_gpu and not data['errors']:
        scripts=[['scripts/verify_iid_gpu.py']]
        if objective_checks:
            scripts += [['scripts/verify_snapshot_anchor_gpu.py'],
                        ['scripts/run_dirichlet_calibration.py','--verify-runtime','--stage','noise']]
        env=dict(os.environ,**policy.environment(),PAPERMERGE_LIVE_TRACKING='disabled',WANDB_MODE='disabled')
        log=logger(Path(state_dir)/'doctor.log')
        for command in scripts:
            try:
                print('Verifying real GPU operations: '+command[0],flush=True)
                check=runner([sys.executable,*command],cwd=ROOT,env=env,capture_output=True,text=True,timeout=600)
                log.info(safe_text(check.stdout+check.stderr))
                if check.returncode:raise RuntimeError(safe_text((check.stderr or check.stdout)[-3000:]))
            except (OSError,subprocess.SubprocessError,RuntimeError) as error:
                data['errors'].append('GPU kernel/protocol verification failed: '+str(error));break
        else:
            from basil_core.execution_profile import verification_path, resource_identity
            proof=read_json(verification_path(policy),{})
            numerical=proof.get('numericalExecution',{})
            if (proof.get('status')!='passed' or not numerical.get('cudaVersion')
                    or not numerical.get('cudnnVersion')
                    or policy.name=='hpc' and numerical.get('resources')!=resource_identity(policy)):
                data['errors'].append('GPU verifier did not produce a compatible CUDA/cuDNN execution receipt.')
            else:
                data['gpuVerified']=True;data['numericalExecution']=numerical
                data['gpuMemory']=proof.get('gpuMemory')
    else:
        data['errors'].append('GPU computation was not verified in this invocation.')
    data['status']='passed' if not data['errors'] else 'blocked'
    return data
