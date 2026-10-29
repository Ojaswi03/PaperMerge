#!/usr/bin/env python3
"""Resolve 1--4 physical GPUs for launchers without changing experiment configs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def selection(inventory,*,count=None,ids=None,jobs=None,saved=None):
    available={r['index']:r for r in inventory}
    if not available:raise ValueError('No NVIDIA GPUs detected.')
    if count is not None and (type(count) is not int or not 1<=count<=4):
        raise ValueError('GPU count must be between 1 and 4.')
    if jobs is not None and (type(jobs) is not int or not 1<=jobs<=8):
        raise ValueError('Workers per GPU must be between 1 and 8.')
    explicit=None
    if ids is not None:
        parts=ids.split(',')
        if any(not s.strip().isdigit() for s in parts):raise ValueError('GPU IDs must be comma-separated nonnegative integers.')
        explicit=[int(s.strip()) for s in parts]
        if not 1<=len(explicit)<=4 or len(explicit)!=len(set(explicit)):
            raise ValueError('Choose between 1 and 4 distinct GPU IDs.')
        if count is not None and count!=len(explicit):raise ValueError('GPU count differs from the explicit GPU list.')
    if saved is not None:
        hardware={}
        for p in saved['profiles']:
            if p['gpu'] in hardware and hardware[p['gpu']]!=p['gpu_uuid']:
                raise ValueError('Conflicting GPUs in saved plan.')
            hardware[p['gpu']]=p['gpu_uuid']
        chosen=list(hardware)
        if not 1<=len(chosen)<=4:raise ValueError('Saved GPU count is outside launcher support.')
        if count is not None and count!=len(chosen):raise ValueError('Saved run uses a different GPU count; choose a fresh output directory.')
        if explicit is not None and explicit!=chosen:raise ValueError('Saved run uses different GPU IDs; choose a fresh output directory.')
        for gpu,uuid in hardware.items():
            if gpu not in available or available[gpu]['uuid']!=uuid:
                raise ValueError('Saved GPU hardware is absent or changed on this server; choose a fresh output directory.')
        saved_jobs=saved['jobsPerGpu']
        if jobs is not None and jobs!=saved_jobs:raise ValueError('Saved worker count differs; choose a fresh output directory.')
        jobs=saved_jobs
    else:
        chosen=explicit if explicit is not None else sorted(available)[:count or min(len(available),4)]
        if count is not None and len(chosen)!=count:
            raise ValueError(f'Requested {count} GPUs, but this server exposes only {len(available)}.')
    if any(g not in available for g in chosen):raise ValueError('A requested physical GPU ID is unavailable on this server.')
    return dict(gpus=chosen,gpuCount=len(chosen),jobsPerGpu=jobs if jobs is not None else 2,
        maximumWorkers=len(chosen)*(jobs if jobs is not None else 2),detectedGpuCount=len(available))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-count',type=int,choices=(1,2,3,4));p.add_argument('--gpus')
    p.add_argument('--jobs-per-gpu',type=int,choices=range(1,9));p.add_argument('--saved-plan',type=Path)
    p.add_argument('--format',choices=('json','shell'),default='json');a=p.parse_args()
    from basil_core.execution_profile import gpu_inventory
    try:
        saved=json.loads(a.saved_plan.read_text()) if a.saved_plan else None
        result=selection(gpu_inventory(),count=a.gpu_count,ids=a.gpus,jobs=a.jobs_per_gpu,saved=saved)
    except (ValueError,RuntimeError,KeyError,OSError) as error:p.exit(2,str(error)+'\n')
    if a.format=='shell':print(','.join(map(str,result['gpus'])),result['jobsPerGpu'])
    else:print(json.dumps(result,indent=2))


if __name__=='__main__':main()
