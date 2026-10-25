"""Non-research hardware diagnostics in isolated verification subprocesses."""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from basil_core.execution_profile import ROOT, gpu_inventory
from basil_core.terminal.state import logger, safe_text, read_json

_SESSION_TIMINGS={}
_PREFLIGHT='System, GPU resources and filesystem checks'
_DEPENDENCIES='Dependency compatibility (pip check)'
_LOADER='CUDA libraries and TensorFlow import'
_KERNELS=(('IID CNN: forward, CE, EBM, SGD and repeatability',['scripts/verify_iid_gpu.py']),
          ('Frozen anchor and partial batches',['scripts/verify_snapshot_anchor_gpu.py']),
          ('TRAIN-prior CE and second-order EBM',['scripts/run_dirichlet_calibration.py','--verify-runtime','--stage','noise']))
_RECEIPT='Verify hardware/software execution receipt'


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


def library_probe(environment, runner=subprocess.run):
    """Inspect dlopen failures in a fresh process with the worker's loader path."""
    program='''
import ctypes, json, sys
def stage(message):
    print('PAPERMERGE_PROBE_STAGE=' + message, file=sys.stderr, flush=True)
libraries = {}
handles = []
for name in ('libcuda.so.1', 'libcudart.so.12', 'libcublas.so.12',
             'libcublasLt.so.12', 'libcufft.so.11', 'libcurand.so.10',
             'libcusolver.so.11', 'libcusparse.so.12', 'libcudnn.so.9'):
    stage('Loading ' + name)
    try:
        handles.append(ctypes.CDLL(name))
        libraries[name] = {'loadable': True}
    except OSError as error:
        libraries[name] = {'loadable': False, 'error': str(error)}
result = {'libraries': libraries, 'status': 'passed' if all(
    item['loadable'] for item in libraries.values()) else 'blocked'}
try:
    stage('Importing TensorFlow')
    import tensorflow as tf
    stage('Reading TensorFlow build information')
    result['tensorflowVersion'] = tf.__version__
    result['tensorflowBuild'] = tf.sysconfig.get_build_info()
except Exception as error:
    result.update(status='blocked', tensorflowError=str(error))
print(json.dumps(result))
'''
    try:
        check=runner([sys.executable,'-c',program],cwd=ROOT,env=environment,
            capture_output=True,text=True,timeout=60)
        if check.returncode:raise ValueError(safe_text((check.stderr or check.stdout)[-3000:]))
        return json.loads(check.stdout)
    except subprocess.TimeoutExpired as error:
        stderr=error.stderr or ''
        if isinstance(stderr,bytes):stderr=stderr.decode('utf-8',errors='replace')
        stages=[line.split('=',1)[1] for line in stderr.splitlines() if line.startswith('PAPERMERGE_PROBE_STAGE=')]
        last=safe_text(stages[-1]) if stages else 'child startup (no stage reported)'
        return dict(status='blocked',timedOut=True,timeoutSeconds=error.timeout,lastStage=last,
            error=f'CUDA/TensorFlow probe timed out after {error.timeout:g} seconds during {last}. '
                'Kernel checks were not started. Retry doctor on this physical GPU; '
                'a timeout alone does not establish a missing library or faulty GPU.')
    except (OSError,subprocess.SubprocessError,ValueError) as error:
        return dict(status='blocked',error=safe_text(error))


def diagnose(policy, state_dir, *, verify_gpu=True, objective_checks=True, runner=subprocess.run):
    from basil_core.terminal.progress import DiagnosticProgress
    kernels=_KERNELS if objective_checks else _KERNELS[:1]
    stages=[_PREFLIGHT,_DEPENDENCIES]
    if verify_gpu:
        if policy.name=='hpc':stages.append(_LOADER)
        stages.extend(label for label,_ in kernels)
        stages.append(_RECEIPT)
    key=(policy.name,policy.memory_mib,policy.threads,tuple(stages))
    estimates=_SESSION_TIMINGS.setdefault(key,{})
    with DiagnosticProgress(f'Physical GPU {policy.gpu} ({policy.name})',stages,estimates=estimates) as progress:
        data=_diagnose(policy,state_dir,verify_gpu=verify_gpu,kernels=kernels,runner=runner,progress=progress)
        progress.finish(data['status'])
    return data


def _diagnose(policy,state_dir,*,verify_gpu,kernels,runner,progress):
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
        check=progress.run(_DEPENDENCIES,lambda:runner([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=60))
        data['dependencyCheck']='passed' if check.returncode==0 else safe_text(check.stdout+check.stderr)
        if check.returncode:data['errors'].append('pip check found incompatible dependencies.')
    except (OSError,subprocess.SubprocessError) as error:data['errors'].append('pip check failed: '+str(error))
    if verify_gpu and not data['errors']:
        from basil_core.execution_profile import worker_environment, cuda_wheel_paths
        env=worker_environment('GPU',dict(os.environ,PAPERMERGE_LIVE_TRACKING='disabled',WANDB_MODE='disabled'),policy)
        if policy.name=='hpc':
            probe=progress.run(_LOADER,lambda:library_probe(env,runner))
            data['cudaLoader']=dict(paths=cuda_wheel_paths(),**probe)
            data['cudaLoader']['ptxas']=shutil.which('ptxas',path=env.get('PATH',''))
            if data['cudaLoader']['status']!='passed':
                failures=[name+': '+item.get('error','not loadable') for name,item in
                    data['cudaLoader'].get('libraries',{}).items() if not item['loadable']]
                failures.extend(str(data['cudaLoader'][key]) for key in ('error','tensorflowError') if key in data['cudaLoader'])
                data['errors'].append('CUDA/TensorFlow startup probe failed: '+'; '.join(failures))
    if verify_gpu and not data['errors']:
        log=logger(Path(state_dir)/'doctor.log')
        for label,command in kernels:
            try:
                check=progress.run(label,lambda:runner([sys.executable,*command],cwd=ROOT,env=env,capture_output=True,text=True,timeout=600))
                log.info(safe_text(check.stdout+check.stderr))
                if check.returncode:raise RuntimeError(safe_text((check.stderr or check.stdout)[-3000:]))
            except (OSError,subprocess.SubprocessError,RuntimeError) as error:
                data['errors'].append('GPU kernel/protocol verification failed: '+str(error));break
        else:
            progress.begin(_RECEIPT)
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
