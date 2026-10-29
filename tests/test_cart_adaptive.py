"""Scheduler resource races, checkpoint handoff and frozen placement contracts."""
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest

from basil_core.cart_worker_admission import Resident, admission, placement
from basil_core.execution_profile import ExecutionProfile
from scripts import run_cart_adaptive as runner


def profile(gpu=0, root='/tmp/cart-adaptive-test'):
    return ExecutionProfile(name='hpc',gpu=gpu,gpu_uuid=f'GPU-{gpu:08x}',
        memory_mib=8192,headroom_mib=8192,ram_mib=32768,output_root=str(root))


def inventory(free=97600):
    return [dict(index=i,uuid=profile(i).gpu_uuid,**{'memory.free':free}) for i in range(4)]


def admit(residents=(),processes=(),free=97600,ram=1500000,maximum=8):
    return admission(profile(),residents,inventory(free),processes,ram,
        context_mib=1024,ram_headroom_mib=32768,maximum=maximum)


def test_startup_workers_reserve_their_entire_future_gpu_and_host_allocations():
    residents=[Resident(i,0,0) for i in range(6)]
    result=admit(residents,free=64000,ram=230000)
    assert result['gpuRequiredMiB']==8192+7*9216
    assert result['ramRequiredMiB']==8*32768
    assert not result['admit']


def test_existing_allocations_are_not_double_counted_and_other_users_are_not_credited():
    residents=[Resident(101,0,30000),Resident(102,1,30000)]
    processes=[dict(gpu_uuid=profile().gpu_uuid,pid='101',memory_mib='8776'),
               dict(gpu_uuid=profile().gpu_uuid,pid='999',memory_mib='70000')]
    result=admit(residents,processes,free=18000,ram=80000)
    assert result['gpuRequiredMiB']==8192+(9216-8776)+9216
    assert result['ramRequiredMiB']==32768+2*(32768-30000)+32768
    assert result['admit']
    assert not admit(residents,processes,free=17000)['admit']


@pytest.mark.parametrize('processes',[None,[dict(gpu_uuid=profile().gpu_uuid,pid='101',memory_mib='[N/A]')]])
def test_missing_telemetry_pauses_admission(processes):
    assert not admit(processes=processes)['admit']


def test_headroom_and_ram_limits_and_worker_ceiling_are_independent():
    assert admit(free=17408,ram=65536)['admit']
    assert not admit(free=17407)['admit']
    assert not admit(ram=65535)['admit']
    assert not admit([Resident(1,0,1)],maximum=1)['admit']
    assert not admit([Resident(1,1,0)],ram=90000)['admit']


def test_live_memory_recovery_admits_later_without_lowering_budgets():
    assert not admit(free=10000)['admit']
    assert admit(free=98000)['admit']


def make_plan(root,n=24):
    return dict(configs=[dict(runId=f'run{i}',nodes=100,rounds=100) for i in range(n)],
        profiles=[asdict(profile(i,root)) for i in range(4)]*2,
        jobsPerGpu=2,source={'fixture':'original'},revision='original',
        resourceBudget=dict(context_overhead_mib=1024,ram_headroom_mib=32768))


@pytest.mark.parametrize('count',[1,2,3,4,5,24])
def test_original_assignment_spreads_small_cohorts_and_keeps_checkpoint_gpu_identity(tmp_path,count):
    plan=make_plan(tmp_path,count)
    before=json.dumps(plan,sort_keys=True)
    hardware,assigned=placement(plan)
    assert len(hardware)==4
    assert [p.gpu for p in assigned]==[i%4 for i in range(count)]
    assert json.dumps(plan,sort_keys=True)==before


def test_conflicting_or_heterogeneous_profiles_are_rejected(tmp_path):
    plan=make_plan(tmp_path)
    plan['profiles'][4]=dict(plan['profiles'][4],memory_mib=16384)
    with pytest.raises(ValueError,match='Conflicting'):placement(plan)
    plan=make_plan(tmp_path)
    plan['profiles']=[dict(p,ram_mib=16384) if p['gpu']==1 else p for p in plan['profiles']]
    with pytest.raises(ValueError,match='equal'):placement(plan)


def test_archive_preserves_checkpoint_and_original_provenance_bytes(tmp_path):
    for name in ('plan.json','revision_execution.json','campaign.json','supervisor.log'):
        (tmp_path/name).write_bytes(b'original\n')
    checkpoint=tmp_path/'run0/checkpoint.npz'
    checkpoint.parent.mkdir();checkpoint.write_bytes(b'checkpoint\x00')
    target=tmp_path/runner.archive(tmp_path)
    for name in ('plan.json','revision_execution.json','campaign.json','supervisor.log'):
        assert (target/name).read_bytes()==(tmp_path/name).read_bytes()==b'original\n'
    assert checkpoint.read_bytes()==b'checkpoint\x00'
    assert json.loads((target/'manifest.json').read_text())['checkpointsUnmodified']


def test_owner_validation_refuses_other_host_and_unrelated_current_process(tmp_path):
    with pytest.raises(ValueError,match='another host'):
        runner.validated_owner(tmp_path,dict(host='different',ownerPid=2))
    with pytest.raises(ValueError,match='Invalid supervisor'):
        runner.validated_owner(tmp_path,dict(host=socket.gethostname(),ownerPid=os.getpid()))
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])
    try:
        with pytest.raises(ValueError,match='recognized comparison'):
            runner.validated_owner(tmp_path,dict(host=socket.gethostname(),ownerPid=process.pid))
    finally:
        process.terminate();process.wait()


def test_graceful_handoff_waits_for_checkpoint_and_never_changes_the_plan(tmp_path,monkeypatch):
    root=tmp_path/'output';root.mkdir()
    (root/'plan.json').write_bytes(b'frozen-plan')
    script=tmp_path/'scripts/run_cart_comparison.py';script.parent.mkdir()
    script.write_text('''import fcntl,json,os,signal,socket,sys,time
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--output')+1])
with (root/'supervisor.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    def stop(*args):
        (root/'checkpoint.npz').write_bytes(b'saved-complete-activation')
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    (root/'campaign.json').write_text(json.dumps(dict(host=socket.gethostname(),ownerPid=os.getpid())))
    while True: time.sleep(.01)
''')
    process=subprocess.Popen([sys.executable,str(script),'--execute','--output',str(root)])
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    try:
        deadline=time.monotonic()+5
        while not (root/'campaign.json').exists():
            assert time.monotonic()<deadline
            time.sleep(.01)
        with pytest.raises(RuntimeError,match='--take-over'):
            runner.acquire_supervisor(root,False,5)
        assert process.poll() is None
        with runner.acquire_supervisor(root,True,5):
            assert (root/'checkpoint.npz').read_bytes()==b'saved-complete-activation'
            assert (root/'plan.json').read_bytes()==b'frozen-plan'
        process.wait(timeout=5)
    finally:
        if process.poll() is None:process.terminate();process.wait()


def test_scheduler_reaches_24_workers_without_changing_plan_or_worker_profiles(tmp_path,monkeypatch):
    plan=make_plan(tmp_path)
    runner.write_json(tmp_path/'plan.json',plan)
    original=(tmp_path/'plan.json').read_bytes()
    hardware,assigned=placement(plan)
    calls=[]
    # No CUDA allocations are reported yet. Reservations alone must stop
    # startup races, and with this server's capacity all six/GPU should fit.
    class Process:
        def __init__(self,command,**kwargs):
            self.pid=1000+len(calls);self.index=int(command[command.index('--worker')+1])
            calls.append((command,kwargs,self))
        def poll(self):
            if len(calls)<24:return None
            runner.write_json(tmp_path/f'run{self.index}/run.json',dict(status='COMPLETED'))
            return 0
    for i in range(24):(tmp_path/f'run{i}').mkdir()
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner,'gpu_inventory',inventory)
    monkeypatch.setattr(runner,'gpu_processes',lambda:[])
    monkeypatch.setattr(runner.scaled,'available_ram_mib',lambda:1500000)
    monkeypatch.setattr(runner,'residents_of',lambda active,assigned:[Resident(p.pid,assigned[i].gpu,0) for i,(p,_) in active.items()])
    monkeypatch.setattr(runner,'check_source',lambda _:None)
    monkeypatch.setattr(runner.scaled.core,'worker_environment',lambda p:dict(profile=asdict(p)))
    monkeypatch.setattr(runner.signal,'signal',lambda *_:None)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    import reporting.cart_comparison
    monkeypatch.setattr(reporting.cart_comparison,'generate',lambda *a,**k:None)
    leases={g:io.StringIO() for g in hardware}
    for lease in leases.values():lease.fileno=lambda:123
    args=SimpleNamespace(output=tmp_path,max_workers_per_gpu=8,poll_seconds=1,ntfy_topic=None)
    assert runner.schedule(args,plan,hardware,assigned,runner.sha(tmp_path/'plan.json'),leases)==0
    assert len(calls)==24
    for command,kwargs,process in calls:
        assert kwargs['env']['profile']==plan['profiles'][process.index%8]
    assert (tmp_path/'plan.json').read_bytes()==original
    result=runner.read_json(tmp_path/'campaign.json')
    assert result['completed']==24 and result['status']=='COMPLETED'


def test_admission_round_robin_fills_one_worker_per_gpu_then_queues_until_memory_frees():
    # Simulate CUDA initialization lag: no allocation telemetry yet, so every
    # startup worker must consume a reservation even though nvidia-smi is empty.
    residents=[]
    order=[]
    pending={g:6 for g in range(4)}
    for _ in range(3):
        for gpu in range(4):
            p=profile(gpu)
            result=admission(p,residents,inventory(17408),[],1500000,
                context_mib=1024,ram_headroom_mib=32768,maximum=8)
            if pending[gpu] and result['admit']:
                residents.append(Resident(100+len(order),gpu,0));order.append(gpu);pending[gpu]-=1
    assert order==[0,1,2,3] and sum(pending.values())==20
    # GPU 2's worker exits. Only its GPU may accept the next startup worker.
    residents=[r for r in residents if r.gpu!=2]
    decisions=[admission(profile(g),residents,inventory(17408),[],1500000,
        context_mib=1024,ram_headroom_mib=32768,maximum=8)['admit'] for g in range(4)]
    assert decisions==[False,False,True,False]


def test_monitor_uses_distinct_adaptive_lanes_for_simultaneous_cases(tmp_path):
    from basil_core.sharp_monitor import snapshot
    from tests.test_acds_comparison import arguments
    from basil_core.cart_comparison_study import selected
    configs=selected(arguments(arms=['acds','prior_fallback'],datasets=['mnist'],
        partitions=['iid','dirichlet'],seeds=[2025]))[1]
    plan=make_plan(tmp_path,4);plan['configs']=configs
    runner.write_json(tmp_path/'plan.json',plan)
    runner.write_json(tmp_path/'campaign.json',dict(status='RUNNING',workerFailurePolicy='continue',
        activeRunIds=[c['runId'] for c in configs],scheduler=dict(mode='adaptive',maximumPerGpu=8,
            lanesByRunId={c['runId']:f'{i}/adaptive{i}' for i,c in enumerate(configs)})))
    result=snapshot(tmp_path)
    assert len(set(e['lane'] for e in result['entries']))==4
    assert all(e['status']=='STARTING' for e in result['entries'])
