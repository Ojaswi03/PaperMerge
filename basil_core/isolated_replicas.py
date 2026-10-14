"""Local Linux namespace RPC for replicated TRAIN computations.

No unsandboxed fallback. CPU only. JSON messages are size/deadline bounded.
This is local process isolation, not multi-host or distributed consensus.
"""
from __future__ import annotations

import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

from .replica_wire import MAX_FRAME,encode,decode,pack_arrays,unpack_vote
from .replicated_training_service import ReplicatedTrainingService
from .replica_resources import ResourceScope,require_host_headroom

ROOT=Path(__file__).resolve().parents[1]


def sandbox_command(fixture,identity,fault):
    if type(identity) is not int or not 0<=identity<10 or fault not in ('none','poison','withhold','exit','forged'):
        raise ValueError('Invalid endpoint configuration.')
    fixture=Path(fixture).resolve(strict=True)
    venv=Path(sys.prefix).resolve()
    command=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
        '--ro-bind','/usr','/usr','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',
        '--dir','/etc','--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache',
        '--proc','/proc','--dev','/dev','--tmpfs','/tmp',
        '--ro-bind',str(venv),str(venv),'--ro-bind',str(ROOT/'basil_core'),str(ROOT/'basil_core'),
        '--ro-bind',str(fixture),'/train.npz','--clearenv','--setenv','PATH','/usr/bin',
        '--setenv','HOME','/tmp','--setenv','CUDA_VISIBLE_DEVICES','-1',
        '--setenv','TF_ENABLE_ONEDNN_OPTS','0','--setenv','TF_DETERMINISTIC_OPS','1',
        '--setenv','TF_NUM_INTRAOP_THREADS','1','--setenv','TF_NUM_INTEROP_THREADS','1',
        '--setenv','OMP_NUM_THREADS','1','--setenv','OPENBLAS_NUM_THREADS','1',
        '--setenv','MPLCONFIGDIR','/tmp/matplotlib','--setenv','TF_CPP_MIN_LOG_LEVEL','3',
        str(venv/'bin/python'),'-I','-u','-c',
        f'import sys;sys.path.insert(0,{str(ROOT)!r});from basil_core.sandbox_replica import main;main({identity},"/train.npz",{fault!r})']
    return command


class SandboxEndpoint:
    def __init__(self,fixture,identity,*,fault='none',timeout=15.,startup_timeout=45.):
        self.identity=identity;self.timeout=timeout;self.process=None
        self.scope=ResourceScope.create()
        self.process=subprocess.Popen(self.scope.command(sandbox_command(fixture,identity,fault)),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
        try:
            self.bootstrap=self._read(startup_timeout)
            if self.bootstrap['identity']!=identity or self.bootstrap['gpuVisible']:
                raise ValueError('Invalid sandbox identity/device.')
            if self.bootstrap['netNamespace']==os.readlink('/proc/self/ns/net') or self.bootstrap['pidNamespace']==os.readlink('/proc/self/ns/pid'):
                raise ValueError('Namespace isolation was not established.')
            self.public_key=bytes.fromhex(self.bootstrap['publicKey'])
            if len(self.public_key)!=32:raise ValueError('Invalid public key.')
            self.resources=self.scope.verify()
        except BaseException:
            self.close();raise

    def _read(self,timeout):
        end=time.monotonic()+timeout;raw=bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout,selectors.EVENT_READ)
            while True:
                remaining=end-time.monotonic()
                if remaining<=0 or not selector.select(remaining):
                    raise TimeoutError('Replica response deadline exceeded.')
                block=os.read(self.process.stdout.fileno(),min(65536,MAX_FRAME+1-len(raw)))
                if not block:raise ValueError('Replica terminated or closed its response pipe.')
                raw.extend(block)
                if len(raw)>MAX_FRAME:raise ValueError('Replica response exceeds frame limit.')
                if b'\n' in raw:
                    if not raw.endswith(b'\n') or raw.count(b'\n')!=1:raise ValueError('Unsolicited replica frames.')
                    return decode(raw)

    def reply(self,request,params):
        if self.process is None:raise TimeoutError('Replica unavailable; restart requires a new membership session.')
        try:
            raw=encode(dict(request=request.__dict__,params=pack_arrays(params)))
            # Write in bounded nonblocking chunks too: a malicious peer may stop reading.
            fd=self.process.stdin.fileno();os.set_blocking(fd,False)
            deadline=time.monotonic()+self.timeout
            with selectors.DefaultSelector() as selector:
                selector.register(fd,selectors.EVENT_WRITE)
                position=0
                while position<len(raw):
                    remaining=deadline-time.monotonic()
                    if remaining<=0 or not selector.select(remaining):raise TimeoutError('Replica request deadline exceeded.')
                    position+=os.write(fd,raw[position:position+4096])
            response=self._read(max(.001,deadline-time.monotonic()))
            if 'error' in response:raise ValueError('Replica computation rejected: '+response.get('message',''))
            return unpack_vote(response['vote'])
        except (OSError,ValueError,KeyError,TypeError,OverflowError,RecursionError,TimeoutError) as error:
            self.close()
            if isinstance(error,TimeoutError):raise
            raise ValueError('Rejected/failed sandbox response.') from error

    def close(self):
        if self.process is not None:
            p=self.process;self.process=None
            if getattr(self,'scope',None) is not None:self.scope.stop()
            if p.poll() is None:p.kill()
            p.wait(timeout=10)
            p.stdin.close();p.stdout.close()


class SandboxTrainingService(ReplicatedTrainingService):
    def evaluate(self,request,params):
        result,proof=super().evaluate(request,params)
        proof.update(physicalRPC=True,logicalSimulation=False,
            isolation='Linux namespaces; read-only code/TRAIN; no network',
            transport='local pipes / bounded JSON',fullResearchReady=False)
        proof['replicaMemoryCapBytes']=2048*1024**2
        return result,proof


class IsolatedReplicaPool:
    def __init__(self,fixture,*,faults=None,timeout=15.):
        import fcntl
        self.memoryBudget=require_host_headroom()
        lock=ROOT/'gui/worker_state/cart_isolated_pool.lock'
        lock.parent.mkdir(parents=True,exist_ok=True)
        self.lease=lock.open('a')
        try:fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.lease.close()
            raise RuntimeError('An isolated replica pool is already active; refusing duplicate allocation.')
        self.endpoints={}
        try:
            for i in range(10):self.endpoints[i]=SandboxEndpoint(fixture,i,fault=(faults or {}).get(i,'none'),timeout=timeout)
            if len({e.bootstrap['dataRoot'] for e in self.endpoints.values()})!=1:
                raise ValueError('Replica TRAIN roots differ.')
            self.service=SandboxTrainingService(self.endpoints)
        except BaseException:
            self.close();raise

    def close(self):
        for e in self.endpoints.values():e.close()
        if getattr(self,'lease',None) is not None:
            self.lease.close();self.lease=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
