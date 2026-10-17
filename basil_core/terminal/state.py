"""Atomic operational snapshots, bounded logs and process identities."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import socket
import tempfile
import time


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(value,stream,allow_nan=False,sort_keys=True)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)


def read_json(path, default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def identity(pid=None):
    pid=os.getpid() if pid is None else pid
    try:
        tail=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return dict(pid=pid,start=tail[19],boot=boot,host=socket.gethostname())
    except (OSError,IndexError):return None


def alive(owner):
    if not isinstance(owner,dict) or type(owner.get('pid')) is not int:return False
    if owner.get('host')!=socket.gethostname():return None
    now=identity(owner['pid'])
    if now is None:
        return False if not Path(f"/proc/{owner['pid']}").exists() else None
    if now!=owner:return False
    try:return Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return None


@contextmanager
def lease(path):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError(f'Resource is already locked: {path.name}') from error
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def safe_text(value):
    text=str(value)
    text=re.sub(r'(?i)((?:api[_ -]?key|access[_ -]?token|password)["\s]*[:=]["\s]*)[^\s,"}]+',r'\1[redacted]',text)
    return ''.join(c for c in text if c in '\n\t' or ord(c)>=32 and ord(c)!=127)


def logger(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    result=logging.getLogger('papermerge.terminal.'+hashlib.sha256(str(path).encode()).hexdigest())
    result.propagate=False;result.setLevel(logging.INFO)
    if not result.handlers:
        handler=RotatingFileHandler(path,maxBytes=4*1024**2,backupCount=3)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        result.addHandler(handler)
    return result


def event(snapshot, message):
    snapshot.setdefault('activity',[]).append(dict(time=datetime.now(timezone.utc).isoformat(),
                                                  message=safe_text(message)))
    del snapshot['activity'][:-20]
    snapshot['last_event_wall']=time.time()


def inspect_state(directory):
    value=read_json(Path(directory)/'status.json',dict(status='IDLE',entries=[],activity=[]))
    if value.get('status') in ('RUNNING','STOPPING','VALIDATING') and alive(value.get('owner')) is not True:
        value['status']='FAILED'
        value['error']='Queue controller is absent or its identity cannot be verified; validate saved workers before resuming.'
        for entry in value.get('entries',[]):
            if entry['status'] in ('RUNNING','STOPPING'):
                entry['status']='BLOCKED' if alive(entry.get('worker')) is not False else 'FAILED'
    return value


def request_stop(directory):
    directory=Path(directory)
    snapshot=read_json(directory/'status.json')
    if not snapshot or snapshot.get('status') not in ('RUNNING','STOPPING','VALIDATING'):
        raise ValueError('No running queue is recorded in this state directory.')
    if alive(snapshot.get('owner')) is not True:
        raise ValueError('Queue owner is absent or unverifiable; no process was signaled.')
    atomic_json(directory/'stop.json',dict(run_id=snapshot['run_id'],requested=time.time()))
    return snapshot['run_id']
