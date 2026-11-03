import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from basil_core import live_gpu_scheduler as live
from basil_core.continuation_campaign import atomic
from scripts import diagnose_fedsharp_activity as diagnostic
from tests.test_fleet_queue import setup_queue,join


def test_kernel_confirms_inherited_lease_without_signalling_process(tmp_path):
    path=tmp_path/'gpu.lock';lease=path.open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=subprocess.Popen([sys.executable,'-c','import time; print("ready",flush=True); time.sleep(30)'],
        pass_fds=(lease.fileno(),),stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        expected=live.identity(child.pid);lease.close()
        result=diagnostic.lease_fds(expected,path)
        assert result['status']=='confirmed holder' and result['descriptors']
        assert child.poll() is None and live.same_process(expected)
        stale=dict(expected,start='wrong')
        assert diagnostic.lease_fds(stale,path)['status']=='process exited or identity changed'
    finally:
        if child.poll() is None:child.terminate()
        child.wait(timeout=5);child.stdout.close();lease.close()


def test_inspection_reads_both_reports_and_preserves_state_and_job_records(tmp_path,monkeypatch):
    queue,master=setup_queue(tmp_path,count=2)
    identity=dict(pid=123,start='original',argv=['python'],profile=dict(gpu=0))
    original,output=join(queue,'hal9k',{'run0':identity});join(queue,'themachine')
    atomic(output/'plan.json',dict(profiles=[dict(gpu=0,gpu_uuid='GPU-00000000')]))
    for host,runs in [('hal9k',['run0']),('themachine',[])]:
        atomic(queue/'fleet/reports'/f'{host}.json',dict(updatedAt=time.time(),runIds=runs,
            snapshot=dict(runs=[dict(completedActivations=20000,completedRounds=200,
                etaSeconds=100,phase='TRAINING',condition='BASIL') for _ in runs]),
            blocked={'0':dict(reason='waiting for existing GPU lease')}))
    monkeypatch.setattr(diagnostic.socket,'gethostname',lambda:'hal9k.example.edu')
    monkeypatch.setattr(diagnostic,'counters',lambda _:None)
    monkeypatch.setattr(diagnostic,'lease_fds',lambda *a:dict(status='confirmed holder',descriptors=[8]))
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=diagnostic.inspect(queue,seconds=0)
    assert not result['trainingModified'] and not result['workerSignalsSent']
    assert set(result['hosts'])=={'hal9k','themachine'}
    worker=result['hosts']['hal9k']['workers'][0]
    assert worker['pid']==123 and worker['legacyWorker'] and worker['lease']['status']=='confirmed holder'
    assert all(p.read_bytes()==value for p,value in before.items())
    assert {p for p in tmp_path.rglob('*') if p.is_file()}==set(before)
    rendered=diagnostic.render(result)
    assert 'waiting for existing GPU lease' in rendered and 'PID 123 adopted' in rendered


def test_cpu_benchmark_is_synthetic_and_never_initializes_tensorflow():
    script=Path(__file__).resolve().parents[1]/'scripts/benchmark_fedsharp_cpu.py'
    result=subprocess.run([sys.executable,str(script),'--dataset','mnist','--iterations','1'],
        capture_output=True,text=True,timeout=20,check=True)
    data=json.loads(result.stdout)
    assert data['synthetic'] and not data['trainingModified'] and not data['gpuUsed']
    assert not data['tensorflowLoaded'] and data['dimensions']==52650 and data['candidates']==66
    assert all(row['wallMedianSeconds']>0 and row['cpuMedianSeconds']>=0 for row in data['timings'].values())
