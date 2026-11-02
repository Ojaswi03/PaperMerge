#!/usr/bin/env python3
"""Read-only diagnosis of failed continuation imports; never repair or requeue."""
import argparse
import json
from pathlib import Path
import pickle
import socket
import sys
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def checkpoint(path,expected,contract,cursor):
    import numpy as np
    from basil_core.round_continuation import sha
    result=dict(path=str(path))
    try:
        result['sha256']=sha(path)
        result['matchesFrozenSeed']=result['sha256']==expected
        if not result['matchesFrozenSeed']:return result
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(archive['metadata'].tobytes())
        result.update(metadataReadable=True,originalContractMatches=meta.get('contract')==contract,
            boundaryMatches=meta.get('cursor')==cursor)
        result['stableChecksum']=sha(path)==result['sha256']
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as error:
        result['error']=f'{type(error).__name__}: {error}'
    return result


def dataset_reads(dataset,repeats=2):
    from basil_core import sharp_data,thesis_raw_data
    original=sharp_data.digest
    results=[]
    for attempt in range(repeats):
        observed=[]
        def record(array):
            digest=original(array);observed.append(digest);return digest
        row=dict(attempt=attempt+1)
        try:
            with patch.object(sharp_data,'digest',record):
                x,y=thesis_raw_data.training_data(dataset)
            del x,y
            row['verified']=True
        except (OSError,ValueError,EOFError,zipfile.BadZipFile,pickle.UnpicklingError) as error:
            row.update(verified=False,error=f'{type(error).__name__}: {error}')
        row['observedArrayHashes']=observed
        results.append(row)
    return dict(expected=thesis_raw_data.EXPECTED[dataset],reads=results)


def diagnose(queue,check_inputs=False):
    from basil_core import basil_sharp_thesis as study
    from basil_core.round_continuation import read,source_record
    host=socket.gethostname().split('.')[0]
    binding=read(queue/'host_bindings'/host/'owner.json');output=Path(binding['output'])
    plan=read(output/'plan.json')
    result=dict(host=host,producerUnchanged=study.source()==plan['source'],failures=[],trainingModified=False)
    datasets=set()
    for cfg in plan['configs']:
        path=output/cfg['runId']/'run.json'
        if not path.exists():continue
        row=read(path)
        if row.get('status')!='FAILED':continue
        item=plan['continuations'][cfg['runId']]
        entry=dict(runId=cfg['runId'],error=row.get('error'),completedActivations=row.get('completedActivations'))
        if check_inputs:
            old=source_record(item)
            paths=dict(shared=queue/item['seedRelativePath'],
                local=output/'seed_cache'/cfg['runId']/'checkpoint.npz')
            if item['sourceHost'].split('.')[0]==host:paths['original']=Path(item['originalCheckpoint'])
            entry['checkpoints']={name:checkpoint(path,item['seedSha256'],old['contract'],item['sourceCursor'])
                for name,path in paths.items()}
            datasets.add(cfg['dataset'])
        result['failures'].append(entry)
    if check_inputs:result['datasets']={name:dataset_reads(name) for name in sorted(datasets)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--check-inputs',action='store_true',help='Hash failed checkpoint copies, parse metadata, and verify two raw dataset reads on CPU.')
    args=parser.parse_args()
    print(json.dumps(diagnose(args.queue.resolve(),args.check_inputs),indent=2),flush=True)


if __name__=='__main__':main()
