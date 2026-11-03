import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from basil_core import fleet_lease_transfer as transfer
from basil_core import live_gpu_scheduler as live
from scripts import refill_fedsharp_gpus as refill
from tests.test_live_gpu_scheduler import profile


def test_duplicate_open_description_reuses_lock_without_unlocking(tmp_path, monkeypatch):
    path=tmp_path/'gpu.lock'
    with path.open('a') as original:
        fcntl.flock(original,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=live.identity(os.getpid())
        monkeypatch.setattr(transfer,'lease_path',lambda _:path)
        monkeypatch.setattr(transfer,'duplicate_fd',lambda item,fd:os.dup(fd))
        handle,receipt=transfer.capture(profile(tmp_path,0),[identity])
        with handle:
            assert receipt['method']=='existing lease duplicated'
            original.close()
            with path.open('a') as competitor:
                with pytest.raises(BlockingIOError):fcntl.flock(competitor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Pass the SAME OFD to a real child, as the new dispatcher does.
            code='import fcntl,sys; fcntl.flock(int(sys.argv[1]),fcntl.LOCK_EX|fcntl.LOCK_NB)'
            subprocess.run([sys.executable,'-c',code,str(handle.fileno())],
                pass_fds=(handle.fileno(),),check=True,timeout=5)
        with path.open('a') as free:fcntl.flock(free,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_denied_duplication_preserves_holder_and_inode(tmp_path,monkeypatch):
    path=tmp_path/'gpu.lock'
    with path.open('a') as original:
        fcntl.flock(original,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inode=path.stat().st_ino
        monkeypatch.setattr(transfer,'lease_path',lambda _:path)
        def denied(*args):raise PermissionError(1,'denied')
        monkeypatch.setattr(transfer,'duplicate_fd',denied)
        with pytest.raises(PermissionError,match='No dispatcher or worker was signalled'):
            transfer.capture(profile(tmp_path,0),[live.identity(os.getpid())])
        assert path.stat().st_ino==inode
        with path.open('a') as other:
            with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_inherited_mapping_rejects_wrong_inode_and_missing_gpu(tmp_path,monkeypatch):
    path=tmp_path/'gpu.lock';other=tmp_path/'other.lock'
    monkeypatch.setattr(transfer,'lease_path',lambda _:path)
    profiles={0:profile(tmp_path,0)}
    with path.open('a') as valid,other.open('a') as wrong:
        fcntl.flock(valid,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='different inode'):
            transfer.inherited_handles([f'0:{wrong.fileno()}'],profiles)
        with pytest.raises(ValueError,match='every GPU'):
            transfer.inherited_handles([],profiles)
        handles=transfer.inherited_handles([f'0:{valid.fileno()}'],profiles)
        handles[0].close()
        assert not valid.closed


def setup_handoff(tmp_path,monkeypatch):
    original=tmp_path/'original';output=tmp_path/'fleet';queue=tmp_path/'queue'
    for p in (original,output,queue):p.mkdir()
    (original/'fleet_owner.json').write_text('{}')
    from dataclasses import asdict
    plan=dict(profiles=[asdict(profile(output,0)),asdict(profile(output,1))])
    monkeypatch.setattr(refill.runner,'prepare',lambda q:(original,output,{},plan,{}))
    monkeypatch.setattr(refill,'agent_identity',lambda *args:live.identity(os.getpid()))
    monkeypatch.setattr(refill.runner,'discover',lambda *args:{})
    monkeypatch.setattr(refill.runner,'discover_gates',lambda *args:{})
    return output,queue


def test_all_leases_required_before_any_dispatcher_signal(tmp_path,monkeypatch):
    output,queue=setup_handoff(tmp_path,monkeypatch)
    held=[]
    def capture(p,identities):
        if p.gpu==1:raise PermissionError('host denied')
        handle=(tmp_path/'gpu0').open('a');held.append(handle)
        return handle,dict(gpu=0)
    monkeypatch.setattr(refill,'capture',capture)
    monkeypatch.setattr(refill.signal,'pidfd_send_signal',lambda *a:pytest.fail('must not signal'))
    monkeypatch.setattr(refill.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not launch'))
    with pytest.raises(PermissionError):refill.handoff(queue,execute=True)
    assert held[0].closed
    assert not (output/'lease_transfers').exists()


def test_dry_run_never_signals_or_launches(tmp_path,monkeypatch):
    output,queue=setup_handoff(tmp_path,monkeypatch)
    monkeypatch.setattr(refill,'capture',lambda p,ids:((tmp_path/f'gpu{p.gpu}').open('a'),dict(gpu=p.gpu)))
    monkeypatch.setattr(refill.signal,'pidfd_send_signal',lambda *a:pytest.fail('must not signal'))
    monkeypatch.setattr(refill.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not launch'))
    result=refill.handoff(queue)
    assert result['eligible'] and not result['execute'] and not result['workerSignalsSent']
    assert not (output/'lease_transfers').exists()


def test_actual_pidfd_getfd_is_safe_or_explicitly_denied(tmp_path):
    with (tmp_path/'fd').open('a') as handle:
        try:fd=transfer.duplicate_fd(live.identity(os.getpid()),handle.fileno())
        except OSError as error:
            assert error.errno in (1,13,38) # sandbox/security policy or unavailable syscall
        else:
            try:assert os.fstat(fd).st_ino==os.fstat(handle.fileno()).st_ino
            finally:os.close(fd)


def test_dispatcher_handoff_preserves_live_worker_and_passes_all_handles(tmp_path,monkeypatch):
    output,queue=setup_handoff(tmp_path,monkeypatch)
    real_popen=subprocess.Popen
    progress=tmp_path/'progress'
    worker=real_popen([sys.executable,'-c',
        'import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); n=0\nwhile True:\n n+=1; p.write_text(str(n)); time.sleep(.02)',str(progress)])
    owner=real_popen([sys.executable,'-c','import time; time.sleep(60)'])
    child=None
    try:
        deadline=time.monotonic()+5
        while not progress.exists():
            assert time.monotonic()<deadline
            time.sleep(.02)
        identity=live.identity(worker.pid)
        original_count=int(progress.read_text())
        monkeypatch.setattr(refill,'agent_identity',lambda *args:live.identity(owner.pid))
        monkeypatch.setattr(refill.runner,'discover',lambda *args:{'run0':identity})
        def capture(p,ids):
            handle=(tmp_path/f'gpu{p.gpu}').open('a')
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return handle,dict(gpu=p.gpu)
        monkeypatch.setattr(refill,'capture',capture)
        # Only replace the executable body: real descriptor inheritance, real
        # dispatcher termination, real readiness and worker survival checks.
        code='''import fcntl,json,os,pathlib,sys,time
from basil_core import live_gpu_scheduler as live
args=sys.argv[1:]; token=args[args.index('--handover-id')+1]; gpus=[]
for i,arg in enumerate(args):
 if arg=='--inherited-gpu-lease':
  gpu,fd=map(int,args[i+1].split(':')); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); gpus.append(gpu)
time.sleep(.1)
pathlib.Path(args[0]).write_text(json.dumps(dict(handoverId=token,agent=live.identity(os.getpid()),gpus=sorted(gpus))))
time.sleep(60)
'''
        def replacement(command,**kwargs):
            nonlocal child
            assert len(kwargs['pass_fds'])==2
            child=real_popen([sys.executable,'-c',code,str(output/'lease_handover_ready.json'),*command[2:]],**kwargs)
            return child
        monkeypatch.setattr(refill.subprocess,'Popen',replacement)
        result=refill.handoff(queue,execute=True)
        assert result['agentPid']==child.pid and result['workersPreserved']==1
        assert worker.poll() is None and live.same_process(identity)
        assert int(progress.read_text())>original_count
        assert owner.wait(timeout=3)==-15
        for gpu in (0,1):
            with (tmp_path/f'gpu{gpu}').open('a') as competitor:
                with pytest.raises(BlockingIOError):fcntl.flock(competitor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipts=list((output/'lease_transfers').glob('*.json'))
        assert len(receipts)==1 and json.loads(receipts[0].read_text())['stage']=='ready'
    finally:
        for process in (child,owner,worker):
            if process is not None:
                if process.poll() is None:process.terminate()
                process.wait(timeout=5)
