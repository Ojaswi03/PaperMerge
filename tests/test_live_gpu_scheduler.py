"""Live balancing preserves ongoing workers, contracts, leases and reservations."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from basil_core import live_gpu_scheduler as live
from basil_core.cart_worker_admission import Resident
from basil_core.execution_profile import ExecutionProfile
from basil_core.cart_campaign_io import write_json, read_json


def profile(root, gpu):
    return ExecutionProfile(name='hpc',gpu=gpu,gpu_uuid=f'GPU-{gpu:08x}',memory_mib=3072,
        headroom_mib=8192,ram_mib=8192,output_root=str(root))


def choose(tmp_path, *, residents=(), ram=90000, free=(97000,97000), processes=(),
           pending=(0,1), allowed=None, quarantine=None, leases=(0,1), maximum=24):
    hardware={g:profile(tmp_path,g) for g in (0,1)}
    inventory=[dict(index=g,uuid=hardware[g].gpu_uuid,**{'memory.free':free[g]}) for g in hardware]
    return live.choose(list(pending),[{},{}],allowed or {0:{0,1},1:{0,1}},hardware,
        list(residents),inventory,processes,ram,
        dict(context_overhead_mib=1024,ram_headroom_mib=16384),maximum,
        quarantine or {},{g:None for g in leases})


def test_idle_gpu_wins_over_original_queue_assignment(tmp_path):
    selected,_=choose(tmp_path,residents=[Resident(1,0,3000)])
    assert selected[0]==0 and selected[1].gpu==1


def test_host_memory_is_global_even_when_second_gpu_is_empty(tmp_path):
    workers=[Resident(i,0,1000) for i in range(9)]
    assert choose(tmp_path,residents=workers,ram=75000)[0] is None


def test_each_startup_worker_reserves_full_future_memory(tmp_path):
    assert choose(tmp_path,residents=[Resident(1,0,0)],ram=32000)[0] is None


def test_free_vram_cannot_override_gpu_ceiling(tmp_path):
    selected,_=choose(tmp_path,residents=[Resident(1,0,0)],maximum=1)
    assert selected[1].gpu==1


def test_no_cpu_fallback_when_gpu_memory_is_low(tmp_path):
    assert choose(tmp_path,free=(9000,9000))[0] is None


def test_unknown_gpu_telemetry_blocks_admission(tmp_path):
    assert choose(tmp_path,processes=None)[0] is None


def test_quarantined_gpu_does_not_receive_new_workers(tmp_path):
    selected,_=choose(tmp_path,quarantine={0:'OOM'})
    assert selected[1].gpu==1


def test_must_own_exclusive_lease_before_launching(tmp_path):
    selected,_=choose(tmp_path,leases=(1,))
    assert selected[1].gpu==1
    assert choose(tmp_path,leases=())[0] is None


def test_checkpointed_work_cannot_move_to_another_gpu(tmp_path):
    hardware={g:profile(tmp_path,g) for g in (0,1)}
    folder=tmp_path/'case';folder.mkdir()
    (folder/'checkpoint.npz').write_bytes(b'checkpoint unchanged')
    write_json(folder/'run.json',dict(contract=dict(runtime=dict(profile=asdict(hardware[0])))))
    assert live.eligible_profile(tmp_path,dict(runId='case'),hardware)=={0}
    assert choose(tmp_path,pending=(0,),allowed={0:{0}},leases=(1,))[0] is None
    assert (folder/'checkpoint.npz').read_bytes()==b'checkpoint unchanged'


def test_incomplete_checkpoint_contract_fails_closed(tmp_path):
    folder=tmp_path/'case';folder.mkdir();(folder/'checkpoint.npz').touch()
    with pytest.raises(ValueError,match='no recorded'):
        live.eligible_profile(tmp_path,dict(runId='case'),{0:profile(tmp_path,0)})


@pytest.mark.parametrize('text',['ResourceExhaustedError: OOM when allocating tensor',
    'CUDA_ERROR_OUT_OF_MEMORY','CUDA error: device is lost'])
def test_cuda_oom_and_device_failures_trigger_quarantine(text):
    assert live.gpu_failure(text)


def test_scientific_worker_error_does_not_quarantine_a_gpu():
    assert not live.gpu_failure('ValueError: invalid labels')


def test_process_identity_pins_original_start_time():
    item=live.identity(os.getpid())
    assert live.same_process(item)
    assert not live.same_process(dict(item,start='another process'))


def test_refused_handover_does_not_signal_any_process(monkeypatch,tmp_path):
    worker=live.identity(os.getpid())
    monkeypatch.setattr(signal,'pidfd_send_signal',lambda *a:pytest.fail('Unexpected signal'))
    with pytest.raises(ValueError,match='overlap'):
        live.replace_controller(tmp_path,worker,{0:worker})


@pytest.mark.skipif(not hasattr(os,'pidfd_open'),reason='Linux pidfd required')
def test_real_controller_replacement_leaves_training_child_alive(tmp_path):
    # An isolated fake supervisor deliberately terminates children on SIGTERM,
    # just like the production supervisor. The handover must avoid that path.
    source='''import os,signal,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
signal.signal(signal.SIGTERM,lambda *_:(child.terminate(),sys.exit(1)))
open(sys.argv[1],'w').write(str(child.pid))
while True:time.sleep(1)
'''
    record=tmp_path/'child.pid'
    parent=subprocess.Popen([sys.executable,'-c',source,str(record)])
    child=None;adopted=None
    try:
        deadline=time.monotonic()+5
        while not record.exists() and time.monotonic()<deadline:time.sleep(.01)
        child=int(record.read_text());child_identity=live.identity(child)
        adopted=live.Adopted(child_identity)
        live.replace_controller(tmp_path,live.identity(parent.pid),{0:child_identity})
        assert parent.wait(timeout=5)==-signal.SIGKILL
        assert adopted.poll() is None and live.same_process(child_identity)
    finally:
        if child:
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
        if adopted:adopted.close()
        if parent.poll() is None:parent.kill();parent.wait()


def test_worker_failure_is_set_aside_and_next_case_uses_healthy_gpu(tmp_path,monkeypatch):
    from scripts import run_cart_v2_scaled as scaled, run_cart_v2_study as core
    from basil_core import basil_sharp_thesis as study
    from reporting import basil_sharp_thesis as reporting
    hardware={g:profile(tmp_path,g) for g in (0,1)}
    assigned=[hardware[0],hardware[1],hardware[0]]
    configs=[dict(runId=f'case{i}') for i in range(3)]
    plan=dict(configs=configs,revision=study.REVISION,source={},
        resourceBudget=dict(context_overhead_mib=1024,ram_headroom_mib=16384))
    write_json(tmp_path/'plan.json',plan);digest=live.sha(tmp_path/'plan.json')
    write_json(tmp_path/'campaign.json',dict(elapsedSeconds=100))
    (tmp_path/'operational_adapters').mkdir()
    inventory=[dict(index=g,uuid=hardware[g].gpu_uuid,**{'memory.free':97000}) for g in hardware]
    notifications=[];launched=[]
    monkeypatch.setattr(live,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(live,'gpu_processes',lambda:[])
    monkeypatch.setattr(live,'residents_of',lambda active,assigned:[Resident(i,assigned[i].gpu,0) for i in active])
    monkeypatch.setattr(scaled,'available_ram_mib',lambda:90000)
    monkeypatch.setattr(core,'worker_environment',lambda p:dict(gpu=p.gpu))
    monkeypatch.setattr(study,'check_source',lambda _:None)
    monkeypatch.setattr(live.time,'sleep',lambda _:None)
    monkeypatch.setattr(live,'notify',lambda root,topic,title,message:notifications.append(title))
    monkeypatch.setattr(live.signal,'signal',lambda *_:None)
    monkeypatch.setattr(reporting,'generate',lambda *_:pytest.fail('Do not publish complete figures after failures'))
    class Process:
        def __init__(self,command,**kwargs):
            i=int(command[command.index('--worker')+1]);gpu=kwargs['env']['gpu']
            self.pid=100+i;self.index=i;launched.append((i,gpu))
            folder=tmp_path/f'case{i}';folder.mkdir()
            write_json(folder/'run.json',dict(status='FAILED' if i==0 else 'COMPLETED',
                error='ResourceExhaustedError: out of memory' if i==0 else None))
        def poll(self):return 1 if self.index==0 else 0
    monkeypatch.setattr(live.subprocess,'Popen',Process)
    monkeypatch.setattr(live,'lease_available',lambda p,leases:True)
    with (tmp_path/'lease0').open('a') as a,(tmp_path/'lease1').open('a') as b:
        result=live.schedule(SimpleNamespace(output=tmp_path,maximum=24,poll_seconds=1,ntfy_topic='topic'),
            plan,hardware,assigned,digest,{}, {0:a,1:b})
    assert result==2
    assert launched==[(0,0),(1,1),(2,1)]
    assert 'PaperMerge worker failed' in notifications and 'PaperMerge GPU 0 paused' in notifications
    campaign=read_json(tmp_path/'campaign.json')
    assert campaign['status']=='COMPLETED_WITH_FAILURES' and campaign['completed']==2 and campaign['failed']==1
    assert read_json(tmp_path/'operational_adapters'/'gpu-quarantine.json')['0']
    assert live.sha(tmp_path/'plan.json')==digest


def test_operational_source_does_not_change_frozen_training_fingerprints():
    from basil_core import basil_sharp_thesis as study
    names=study.source()
    assert 'basil_core/live_gpu_scheduler.py' not in names
    assert 'scripts/rebalance_basil_sharp_live.py' not in names


def test_gpu_disappearance_leaves_other_gpu_available(tmp_path):
    hardware={g:profile(tmp_path,g) for g in (0,1)}
    inventory=[dict(index=1,uuid=hardware[1].gpu_uuid,**{'memory.free':97000})]
    selected,_=live.choose([0],[{}],{0:{0,1}},hardware,[],inventory,[],90000,
        dict(context_overhead_mib=1024,ram_headroom_mib=16384),24,{0:'missing'},{1:None})
    assert selected[1].gpu==1


def test_controller_identity_rejects_training_worker_pid(tmp_path,monkeypatch):
    write_json(tmp_path/'campaign.json',dict(host=socket.gethostname(),ownerPid=123,status='RUNNING'))
    monkeypatch.setattr(live,'identity',lambda _:dict(pid=123,argv=[sys.executable,str(live.WORKER),
        '--output',str(tmp_path),'--execute']))
    with pytest.raises(ValueError,match='not this campaign supervisor'):
        live.controller(tmp_path)


def test_adoption_failure_resumes_original_dispatch_without_touching_workers(tmp_path,monkeypatch):
    from basil_core import basil_sharp_thesis as study
    hardware={0:profile(tmp_path,0)}
    plan=dict(configs=[],source={},revision=study.REVISION)
    owner=dict(pid=9876,start='1',argv=[])
    monkeypatch.setattr(live,'verified',lambda _: (plan,hardware,[],'hash'))
    monkeypatch.setattr(live,'controller',lambda _:owner)
    monkeypatch.setattr(live,'same_process',lambda _:True)
    descriptor=os.open('/dev/null',os.O_RDONLY)
    monkeypatch.setattr(live.os,'pidfd_open',lambda _:os.dup(descriptor))
    original_read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda path,*a,**kw:'State: T (stopped)\n'
        if str(path)=='/proc/9876/status' else original_read(path,*a,**kw))
    def refuse(*_):raise ValueError('cannot validate workers')
    monkeypatch.setattr(live,'discover',refuse)
    signals=[]
    monkeypatch.setattr(live.signal,'pidfd_send_signal',lambda fd,sig:signals.append(sig))
    try:
        with pytest.raises(ValueError,match='cannot validate'):
            live.execute(SimpleNamespace(output=tmp_path))
    finally:os.close(descriptor)
    assert signals==[signal.SIGSTOP,signal.SIGCONT]


@pytest.mark.skipif(not hasattr(os,'pidfd_open'),reason='Linux pidfd required')
def test_full_live_handover_discovers_and_adopts_real_unchanged_worker(tmp_path,monkeypatch):
    worker_script=tmp_path/'fake_training.py'
    worker_script.write_text('import time\ntime.sleep(60)\n')
    p=profile(tmp_path,0)
    plan=dict(configs=[dict(runId='case')],source={})
    write_json(tmp_path/'plan.json',plan);digest=live.sha(tmp_path/'plan.json')
    record=tmp_path/'child.pid'
    source='''import os,signal,subprocess,sys,time
child=subprocess.Popen([sys.executable,sys.argv[1],'--worker','0','--output',sys.argv[2]],env=os.environ.copy())
signal.signal(signal.SIGTERM,lambda *_:(child.terminate(),sys.exit(1)))
open(sys.argv[3],'w').write(str(child.pid))
while True:time.sleep(1)
'''
    parent=subprocess.Popen([sys.executable,'-c',source,str(worker_script),str(tmp_path),str(record)],
        env={**os.environ,'PAPERMERGE_EXECUTION_PROFILE':json.dumps(asdict(p))})
    child=None
    try:
        deadline=time.monotonic()+5
        while not record.exists() and time.monotonic()<deadline:time.sleep(.01)
        child=int(record.read_text());child_identity=live.identity(child)
        monkeypatch.setattr(live,'WORKER',worker_script)
        monkeypatch.setattr(live,'verified',lambda _: (plan,{0:p},[p],digest))
        monkeypatch.setattr(live,'controller',lambda _:live.identity(parent.pid))
        def schedule(args,actual_plan,hardware,assigned,actual_digest,active,leases):
            assert active[0][0].pid==child and active[0][0].poll() is None
            assert live.same_process(child_identity)
            assert parent.wait(timeout=5)==-signal.SIGKILL
            assert assigned==[p] and actual_digest==digest and leases=={}
            return 0
        monkeypatch.setattr(live,'schedule',schedule)
        assert live.execute(SimpleNamespace(output=tmp_path))==0
        assert live.same_process(child_identity) and live.sha(tmp_path/'plan.json')==digest
        receipt=read_json(next((tmp_path/'operational_adapters').glob('live-handover-*.json')))
        assert receipt['activeWorkersSignalled'] is False and receipt['adoptedWorkers']['0']['pid']==child
    finally:
        if child:
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
        if parent.poll() is None:parent.kill();parent.wait()
