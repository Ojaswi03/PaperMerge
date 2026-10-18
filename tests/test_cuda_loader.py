"""Private CUDA wheel discovery and loader diagnostics; no GPU computation."""
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from basil_core import execution_profile as profiles
from basil_core.terminal import doctor


def test_wheel_discovery_needs_no_tensorflow_import(tmp_path):
    root=tmp_path/'site-packages'
    library=root/'nvidia/cublas/lib';library.mkdir(parents=True)
    (library/'libcublas.so.12').touch()
    empty=root/'nvidia/unused/lib';empty.mkdir(parents=True)
    binary=root/'nvidia/cuda_nvcc/bin';binary.mkdir(parents=True)
    (binary/'ptxas').touch()
    other=root/'unrelated/lib';other.mkdir(parents=True)
    (other/'libother.so').touch()
    assert profiles.cuda_wheel_paths([root,root])==dict(
        libraryDirectories=[str(library)],binaryDirectories=[str(binary)])


def test_managed_loader_paths_precede_existing_paths_without_empty_entries(monkeypatch):
    monkeypatch.setattr(profiles,'cuda_wheel_paths',lambda:dict(
        libraryDirectories=['/private/cublas','/private/cudnn'],binaryDirectories=['/private/nvcc/bin']))
    base=dict(LD_LIBRARY_PATH='/existing/lib::/private/cublas:',PATH='/usr/bin:/private/nvcc/bin')
    result=profiles.cuda_wheel_environment(base)
    assert result['LD_LIBRARY_PATH']=='/private/cublas:/private/cudnn:/existing/lib'
    assert result['PATH']=='/private/nvcc/bin:/usr/bin'
    assert base['LD_LIBRARY_PATH'].startswith('/existing/lib')


@pytest.mark.parametrize('name,device,managed',[('local','GPU',False),('local','CPU',False),
    ('hpc','CPU',False),('hpc','GPU',True)])
def test_loader_policy_is_opt_in_for_hpc_gpu_workers(tmp_path,monkeypatch,name,device,managed):
    monkeypatch.setattr(profiles,'cuda_wheel_paths',lambda:dict(
        libraryDirectories=['/private/cuda/lib'],binaryDirectories=['/private/cuda/bin']))
    profile=profiles.ExecutionProfile(name=name,output_root=str(tmp_path/'runs') if name=='hpc' else '')
    base=dict(LD_LIBRARY_PATH='/existing/lib',PATH='/usr/bin')
    env=profiles.worker_environment(device,base,profile)
    assert env['LD_LIBRARY_PATH']==('/private/cuda/lib:/existing/lib' if managed else '/existing/lib')
    assert env['PATH']==('/private/cuda/bin:/usr/bin' if managed else '/usr/bin')
    assert env['CUDA_VISIBLE_DEVICES']==('0' if device=='GPU' else '-1')


def test_no_wheels_does_not_fabricate_paths(monkeypatch):
    monkeypatch.setattr(profiles,'cuda_wheel_paths',lambda:dict(libraryDirectories=[],binaryDirectories=[]))
    assert profiles.cuda_wheel_environment({})=={}


def test_runtime_preparation_preserves_system_tool_paths(tmp_path,monkeypatch):
    from basil_core import iid_runtime
    profile=profiles.ExecutionProfile(name='hpc',output_root=str(tmp_path/'runs'))
    monkeypatch.setattr(os,'environ',dict(profile.environment(),PATH='/usr/bin:/bin',LD_LIBRARY_PATH='/university/lib'))
    monkeypatch.setattr(profiles,'cuda_wheel_paths',lambda:dict(
        libraryDirectories=['/private/cuda/lib'],binaryDirectories=['/private/cuda/bin']))
    result=iid_runtime.prepare_environment('GPU')
    assert result['PATH']=='/private/cuda/bin:/usr/bin:/bin'
    assert result['LD_LIBRARY_PATH']=='/private/cuda/lib:/university/lib'
    assert os.environ['PATH'].endswith('/usr/bin:/bin')


@pytest.mark.skipif(shutil.which('gcc') is None,reason='C compiler needed for a tiny CPU shared-library fixture')
def test_managed_paths_fix_real_dlopen_in_a_fresh_process(tmp_path,monkeypatch):
    library=tmp_path/'private-wheel/lib';library.mkdir(parents=True)
    source=tmp_path/'probe.c';source.write_text('int papermerge_probe(void) { return 42; }\n')
    shared=library/'libpapermerge_loader_probe.so'
    subprocess.run(['gcc','-shared','-fPIC',str(source),'-o',str(shared)],check=True,capture_output=True)
    program="import ctypes; print(ctypes.CDLL('libpapermerge_loader_probe.so').papermerge_probe())"
    base=dict(os.environ,LD_LIBRARY_PATH='')
    before=subprocess.run([sys.executable,'-c',program],env=base,cwd=tmp_path,capture_output=True,text=True)
    assert before.returncode!=0 and 'cannot open shared object' in before.stderr
    monkeypatch.setattr(profiles,'cuda_wheel_paths',lambda:dict(
        libraryDirectories=[str(library)],binaryDirectories=[]))
    after=subprocess.run([sys.executable,'-c',program],env=dict(base,**profiles.cuda_wheel_environment(base)),
        cwd=tmp_path,capture_output=True,text=True,check=True)
    assert after.stdout.strip()=='42'


def test_library_probe_preserves_exact_loader_failure_and_uses_child_environment():
    calls=[]
    expected=dict(status='blocked',libraries={'libcudnn.so.9':dict(loadable=False,
        error='libcudnn.so.9: cannot open shared object file')})
    def runner(command,**kwargs):
        calls.append((command,kwargs));return SimpleNamespace(returncode=0,stdout=json.dumps(expected))
    env=dict(LD_LIBRARY_PATH='/private/cuda/lib',CUDA_VISIBLE_DEVICES='GPU-synthetic')
    result=doctor.library_probe(env,runner)
    assert result==expected
    assert calls[0][1]['env']==env and calls[0][0][1]=='-c'


def test_library_probe_crash_is_blocked():
    runner=lambda *args,**kwargs:SimpleNamespace(returncode=1,stderr='symbol lookup error: synthetic',stdout='')
    assert doctor.library_probe({},runner)==dict(status='blocked',error='symbol lookup error: synthetic')


def test_doctor_loader_failure_prevents_kernel_checks(tmp_path,monkeypatch):
    from basil_core.terminal import dashboard
    from test_hpc_profiles import inventory,policy
    selected=policy(tmp_path).bind(inventory())
    monkeypatch.setenv(profiles.ENV,selected.environment()[profiles.ENV])
    monkeypatch.setattr(doctor.sys,'prefix',str(tmp_path/'venv'))
    monkeypatch.setattr(doctor,'gpu_inventory',inventory)
    monkeypatch.setattr(dashboard,'system_info',lambda:{})
    monkeypatch.setattr(doctor,'dependencies',lambda:{name:'synthetic' for name in
        ('tensorflow','keras','numpy','scipy','scikit-learn','h5py','matplotlib','pillow','tqdm','psutil','wandb')})
    monkeypatch.setattr(doctor,'library_probe',lambda *args,**kwargs:dict(status='blocked',
        libraries={'libcudnn.so.9':dict(loadable=False,error='synthetic missing library')}))
    calls=[]
    def runner(command,**kwargs):
        calls.append(command);return SimpleNamespace(returncode=0,stdout='',stderr='')
    result=doctor.diagnose(selected,tmp_path/'state',runner=runner)
    assert result['status']=='blocked' and not result['gpuVerified'] and result['researchRunsStarted']==0
    assert any('libcudnn.so.9: synthetic missing library' in message for message in result['errors'])
    assert all(command[1:]==['-m','pip','check'] for command in calls)
