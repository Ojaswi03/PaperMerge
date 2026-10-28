import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def args(tmp_path,jobs=6):
    return SimpleNamespace(output=tmp_path,gpus='0',profile='hpc',jobs_per_gpu=jobs,
        gpu_memory_mib=8192,gpu_headroom_mib=8192,context_overhead_mib=1024,
        ram_mib_per_worker=32768,ram_headroom_mib=32768)


def inventory(free=97887):return [dict(index=0,uuid='GPU-aaaa',**{'memory.free':free})]


@pytest.mark.parametrize('jobs,required',[(4,45056),(6,63488),(8,81920)])
def test_resource_budget_counts_contexts_headroom_and_ram(tmp_path,monkeypatch,jobs,required):
    import scripts.run_cart_v2_scaled as scaled
    import basil_core.execution_profile as resources
    monkeypatch.setattr(scaled,'available_ram_mib',lambda:2_000_000)
    monkeypatch.setattr(resources,'gpu_inventory',lambda:inventory(required-1))
    with pytest.raises(RuntimeError,match=str(required)):scaled.profiles(args(tmp_path,jobs))
    monkeypatch.setattr(resources,'gpu_inventory',lambda:inventory(required))
    assert scaled.profiles(args(tmp_path,jobs))[0].headroom_mib==8192
    monkeypatch.setattr(scaled,'available_ram_mib',lambda:100)
    with pytest.raises(RuntimeError,match='Host RAM'):scaled.profiles(args(tmp_path,jobs))


def test_live_admission_counts_only_our_own_gpu_allocations(tmp_path,monkeypatch):
    import scripts.run_cart_v2_scaled as scaled
    import basil_core.execution_profile as resources
    a=args(tmp_path);monkeypatch.setattr(scaled,'available_ram_mib',lambda:2_000_000)
    monkeypatch.setattr(resources,'gpu_inventory',lambda:inventory(63488))
    p=scaled.profiles(a)[0]
    monkeypatch.setattr(resources,'gpu_inventory',lambda:inventory(10000))
    monkeypatch.setattr(resources,'gpu_processes',lambda:[dict(gpu_uuid='GPU-aaaa',pid='10',memory_mib='54000')])
    assert scaled.can_admit(p,[SimpleNamespace(pid=10)],a)
    assert not scaled.can_admit(p,[SimpleNamespace(pid=11)],a)
    monkeypatch.setattr(resources,'gpu_processes',lambda:[dict(gpu_uuid='GPU-aaaa',pid='10',memory_mib='[N/A]')])
    with pytest.raises(RuntimeError,match='telemetry'):scaled.can_admit(p,[SimpleNamespace(pid=10)],a)
    a.allow_unattributed_process_memory=True
    assert not scaled.can_admit(p,[SimpleNamespace(pid=10)],a)


def test_scaled_scheduler_preserves_training_configurations_and_100_rounds(tmp_path):
    import subprocess,sys
    command=[sys.executable,'scripts/run_cart_v2_scaled.py','--output',str(tmp_path),'--jobs-per-gpu','6',
        '--datasets','mnist','cifar10','--conditions','clean','attack_noise','attack_noise_defense',
        'attack_noise_rla','attack_noise_defense_rla','attack_noise_defense_rla_cart','--only-sigma','0.01']
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    summary=json.loads(result.stdout)
    assert summary['configs']==72 and summary['rounds']==100 and summary['nodes']==100


def test_scaled_help_remains_lazy():
    import subprocess,sys
    path=Path(__file__).resolve().parents[1]/'scripts/run_cart_v2_scaled.py'
    code=(f'import sys,runpy;sys.argv=[{str(path)!r},"--help"];'
        f'\ntry:runpy.run_path({str(path)!r},run_name="__main__")'
        '\nexcept SystemExit as e:assert e.code==0'
        '\nassert not {"tensorflow","numpy","matplotlib"}.intersection(sys.modules)')
    assert subprocess.run([sys.executable,'-I','-c',code],capture_output=True).returncode==0


@pytest.mark.parametrize('jobs',[3,4,6,8])
def test_scheduler_obeys_each_requested_slot_bound(tmp_path,monkeypatch,jobs):
    from dataclasses import asdict
    from basil_core.execution_profile import ExecutionProfile
    import scripts.run_cart_v2_scaled as scaled
    p=ExecutionProfile(name='hpc',gpu_uuid='GPU-aaaa',output_root=str(tmp_path))
    configs=[dict(runId=f'job-{i}',dataset='mnist',model='paper') for i in range(jobs+2)]
    monkeypatch.setattr(scaled,'profiles',lambda _:[p]);monkeypatch.setattr(scaled,'can_admit',lambda *_:True)
    monkeypatch.setattr(scaled.core,'selected',lambda _:configs);monkeypatch.setattr(scaled.core,'fingerprint',lambda:{})
    monkeypatch.setattr(scaled.core,'check_source',lambda _:None);monkeypatch.setattr(scaled.core,'data_contract',lambda _:({},{}))
    monkeypatch.setattr(scaled.core,'worker_environment',lambda _:{})
    monkeypatch.setattr(scaled.signal,'signal',lambda *_:None);monkeypatch.setattr(scaled.time,'sleep',lambda _:None)
    (tmp_path/'gpu_verification_0.json').write_text(json.dumps(dict(status='PASSED',source={},
        runtime={'profile':asdict(p)},checks=[dict(dataset='mnist',model='paper')])))
    state=dict(active=0,peak=0,count=0)
    class Process:
        def __init__(self,*_,**__):
            self.returncode=None;self.polls=0;state['active']+=1;state['count']+=1;self.pid=state['count']
            state['peak']=max(state['peak'],state['active'])
        def poll(self):
            self.polls+=1
            if self.polls>=2 and self.returncode is None:self.returncode=0;state['active']-=1
            return self.returncode
    monkeypatch.setattr(scaled.subprocess,'Popen',Process)
    a=args(tmp_path,jobs);a.recover=False
    scaled.execute(a)
    assert state['peak']==jobs and state['count']==jobs+2
    assert json.loads((tmp_path/'campaign.json').read_text())['status']=='COMPLETED'


def test_manager_selects_scaled_entry_without_touching_training_source(tmp_path,monkeypatch):
    import scripts.run_cart_v2_managed as managed
    seen=[]
    class FailedStartup:
        returncode=1
        def __init__(self,command,**kwargs):seen.extend(command)
        def poll(self):return 1
    monkeypatch.setattr(managed.subprocess,'Popen',FailedStartup)
    monkeypatch.setattr(managed.signal,'signal',lambda *_:None)
    a=SimpleNamespace(output=tmp_path,figures=None,ntfy_topic=None,watch=False,postprocess_only=False,
        scheduler='scaled',timing_from=[])
    assert managed.run(a,['--jobs-per-gpu','6'])==1
    assert any(str(v).endswith('run_cart_v2_scaled.py') for v in seen)
