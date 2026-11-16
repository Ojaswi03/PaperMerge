"""Bounded retries for transient campaign filesystem operations, outside training."""
import errno
import json
import os
from pathlib import Path
import time

TRANSIENT_ERRNOS={getattr(errno,name) for name in
    ('EIO','ESTALE','ETIMEDOUT','EAGAIN','EREMOTEIO') if hasattr(errno,name)}


def transient_io(error):return isinstance(error,OSError) and error.errno in TRANSIENT_ERRNOS


def retry_io(function,*args,attempts=4,**kwargs):
    for attempt in range(attempts):
        try:return function(*args,**kwargs)
        except OSError as error:
            if not transient_io(error) or attempt==attempts-1:raise
            time.sleep(.25*2**attempt)


def read_json(path):return retry_io(lambda:json.loads(Path(path).read_text()))


def write_json(path,value):
    path=Path(path)
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    def write():
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(payload);os.replace(temporary,path)
    retry_io(write)
