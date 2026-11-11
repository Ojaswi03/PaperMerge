#!/usr/bin/env python3
"""Download compact final BASIL models over one SSH connection per server.

Run in the user's local terminal to allow the ordinary SSH password prompt.
The remote exporter only reads completed checkpoints; it uses no GPU or SGD.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import test_fedsharp_image as frozen

REMOTE=r'''
import ast,hashlib,io,json,struct,sys,zipfile
from pathlib import Path
root=Path(sys.argv[1]);runs=json.loads(sys.argv[2]);bundle=io.BytesIO()
def npy(raw):
    if raw[:6]!=b'\x93NUMPY':raise ValueError('Not a NumPy array')
    version=raw[6];width=2 if version==1 else 4
    length=int.from_bytes(raw[8:8+width],'little');offset=8+width+length
    header=ast.literal_eval(raw[8+width:offset].decode('latin1').strip())
    return header,memoryview(raw)[offset:]
def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_STORED) as outer:
    for run in runs:
        path=root/run/'checkpoint.npz';stat=path.stat()
        row=json.loads((root/run/'run.json').read_text())
        if row['status']!='COMPLETED':raise ValueError('Run is not complete')
        with zipfile.ZipFile(path) as z:
            model=z.read('nodeParams.npy');metadata=z.read('metadata.npy')
        mh,payload=npy(metadata)
        if mh['descr']!='|u1':raise ValueError('Unexpected metadata dtype')
        meta=json.loads(payload.tobytes());cfg=meta['contract']['config']
        if meta['contract']!=row['contract'] or meta['cursor']!=cfg['nodes']*500 or len(meta['roundMetrics'])!=500:
            raise ValueError('Final contract differs')
        h,payload=npy(model);n,d=h['shape']
        if n!=100 or h['descr']!='<f4' or h['fortran_order'] or len(payload)!=n*d*4:
            raise ValueError('Unexpected parameter representation')
        final=meta['telemetry'][-n:]
        if [v['node'] for v in final]!=list(range(n)):raise ValueError('Final client order differs')
        hashes=[hashlib.sha256(payload[i*d*4:(i+1)*d*4]).hexdigest() for i in range(n)]
        if hashes!=[v['trainedHash'] for v in final]:raise ValueError('Trained parameter hashes differ')
        compact=dict(format='PaperMerge-final-parameters-v1',contract=meta['contract'],cursor=meta['cursor'],
            finalRoundAccuracy=meta['roundMetrics'][-1]['nodeAccuracy'],finalModelHashes=hashes,
            originalCheckpointSha256=sha(path),trainingPerformed=False,sourceCheckpointModified=False)
        after=path.stat()
        if (stat.st_ino,stat.st_size,stat.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Checkpoint changed during export')
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            z.writestr('nodeParams.npy',model)
            z.writestr('model_metadata.json',json.dumps(compact,sort_keys=True).encode())
        outer.writestr(run+'/model.npz',b.getvalue())
        print('Verified compact model: '+run,file=sys.stderr,flush=True)
        del metadata,meta,model,payload,b
sys.stdout.buffer.write(bundle.getvalue())
'''


def unpack(archive,selected,destination):
    """Accept only requested model entries and their published frozen contracts."""
    import numpy as np
    expected={c['view']['runId']+'/model.npz':c for c in selected}
    with zipfile.ZipFile(archive) as outer:
        if set(outer.namelist())!=set(expected):raise ValueError('Unexpected downloaded bundle entries.')
        for rel,c in expected.items():
            payload=outer.read(rel)
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                if set(z.namelist())!={'nodeParams.npy','model_metadata.json'}:raise ValueError('Invalid compact model members.')
                meta=json.loads(z.read('model_metadata.json'))
            row=frozen.read(c['output']/c['view']['runId']/'run.json')
            if meta['contract']!=row['contract'] or meta['cursor']!=50000:raise ValueError('Published model identity differs.')
            with np.load(io.BytesIO(payload),allow_pickle=False) as z:params=z['nodeParams'].copy()
            if params.shape[0]!=100 or params.dtype!=np.float32 or not np.isfinite(params).all():raise ValueError('Invalid weights.')
            if [hashlib.sha256(v.tobytes()).hexdigest() for v in params]!=meta['finalModelHashes']:
                raise ValueError('Downloaded parameter hashes differ.')
            target=Path(destination)/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise FileExistsError('Preserve existing compact model: '+str(target))
            temporary=target.with_suffix('.tmp');temporary.write_bytes(payload);os.replace(temporary,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=ROOT/'.papermerge/models/final-500r')
    parser.add_argument('--user',default='os675524')
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    selected=frozen.select_local_runs(args.destination,['mnist','cifar10'],'hidden',2025,('basil','basil_acds'))
    groups={}
    for c in selected:
        if c['checkpoint'].is_file():
            print('Already local: '+c['view']['runId']);continue
        groups.setdefault(c['owner'],[]).append(c)
    if args.describe:
        print(json.dumps({host:[c['view']['runId'] for c in values] for host,values in groups.items()},indent=2));return
    args.destination.mkdir(parents=True,exist_ok=True)
    for host,values in groups.items():
        runids=[c['view']['runId'] for c in values]
        remote_root=f'/var/tmp/papermerge-1718278706/{host}_fedsharp-all-500r-724feac03d03'
        remote=shlex.join(['python3','-c',REMOTE,remote_root,json.dumps(runids)])
        command=['ssh','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=10',f'{args.user}@{host}.cs.ohio.edu',remote]
        archive=args.destination/f'.compact-{host}.download'
        print(f'Connecting to {host}; downloading model parameters only.',flush=True)
        with archive.open('wb') as stream:subprocess.run(command,stdout=stream,check=True)
        unpack(archive,values,args.destination);size=archive.stat().st_size;archive.unlink()
        print(f'{host}: downloaded and verified {len(values)} models, {size/1024**2:.1f} MiB.',flush=True)
    print('BASIL models are ready for local inference.',flush=True)


if __name__=='__main__':main()
