#!/usr/bin/env python3
"""Stream-check local final checkpoint CRCs; never modify any experiment file."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

EXPECTED={'nodeParams.npy','outbound.npy','senderRounds.npy','referenceParams.npy','metadata.npy'}


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def inspect(path):
    """Exercise ZIP's normal CRC checks on every member, bounded RAM."""
    result=dict(path=str(path),members={},errors=[])
    try:
        before=path.stat();result['bytes']=before.st_size
        with zipfile.ZipFile(path) as archive:
            names=archive.namelist()
            if set(names)!=EXPECTED or len(names)!=len(EXPECTED):
                result['errors'].append('Unexpected or missing checkpoint members.')
            for item in archive.infolist():
                try:
                    with archive.open(item) as member:
                        while member.read(1024*1024):pass
                    result['members'][item.filename]='CRC_OK'
                except (OSError,ValueError,EOFError,zipfile.BadZipFile) as error:
                    result['members'][item.filename]=f'{type(error).__name__}: {error}'
                    result['errors'].append(item.filename)
        after=path.stat()
        result['fileUnchanged']=(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
        if not result['fileUnchanged']:result['errors'].append('File changed during inspection.')
    except (OSError,ValueError,EOFError,zipfile.BadZipFile) as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    result['ok']=not result['errors']
    return result


def scan(queue,repeats=2,progress=print):
    if not 1<=repeats<=3:raise ValueError('Repeats must be between one and three.')
    host=socket.gethostname().split('.')[0]
    state=json.loads((Path(queue)/'fleet/state.json').read_text());rows=[];bad=[];started=time.monotonic()
    for run,entry in state['runs'].items():
        if entry['owner']!=host:continue
        path=Path(entry['location']['output'])/run/'checkpoint.npz'
        row=dict(runId=run,**inspect(path));rows.append(row)
        progress(f'{len(rows):3} {"OK" if row["ok"] else "FAILED"} {run}',flush=True)
        if row['ok']:continue
        row['repeatReads']=[]
        for attempt in range(repeats):
            check=dict(attempt=attempt+1)
            try:
                check['sha256Before']=digest(path);check['archive']=inspect(path);check['sha256After']=digest(path)
                check['stableChecksum']=check['sha256Before']==check['sha256After']
            except OSError as error:check['error']=f'{type(error).__name__}: {error}'
            row['repeatReads'].append(check)
        row['availableBackups']=[]
        for parent in ('optimization_boundary','checkpoint_retries'):
            backup=path.parent.parent/'operational_adapters'/parent/run/'checkpoint.npz'
            if backup.exists():row['availableBackups'].append(dict(kind=parent,**inspect(backup)))
        bad.append(row)
    return dict(host=host,checked=len(rows),initiallyPassed=len(rows)-len(bad),badCheckpoints=bad,
        elapsedSeconds=time.monotonic()-started,trainingModified=False,checkpointsModified=False,
        scope='Archive CRC integrity only; not fresh model inference or a diagnosis of the hardware/storage root cause.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True,type=Path);parser.add_argument('--repeats',type=int,choices=(1,2,3),default=2)
    args=parser.parse_args();print(json.dumps(scan(args.queue,args.repeats),indent=2),flush=True)
