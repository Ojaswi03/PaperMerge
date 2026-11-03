#!/usr/bin/env python3
"""Small synthetic CPU benchmark; no TensorFlow, GPU, training or experiment I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def benchmark(dataset,iterations=3):
    import numpy as np
    from basil_core.sharp_protocol import link_vector
    from basil_core.sharp_math import peak_screen
    dimension={'mnist':52650,'cifar10':117706}[dataset];count=66
    base=np.random.default_rng(731).normal(0,.02,size=dimension).astype(np.float32)
    samples={'linkReconstruction':[],'screening':[]}
    for index in range(iterations):
        tick=time.perf_counter();cpu=time.process_time()
        vectors=np.stack([link_vector(base,731,index,sender,99,.01) for sender in range(count)])
        samples['linkReconstruction'].append((time.perf_counter()-tick,time.process_time()-cpu))
        tick=time.perf_counter();cpu=time.process_time()
        peak_screen(vectors,full_window=True)
        samples['screening'].append((time.perf_counter()-tick,time.process_time()-cpu))
    root=Path(__file__).resolve().parents[1]
    return dict(dataset=dataset,dimensions=dimension,candidates=count,iterations=iterations,
        synthetic=True,trainingModified=False,gpuUsed=False,tensorflowLoaded='tensorflow' in sys.modules,
        scope='CPU link reconstruction and screening only; not a full activation or a measured live bottleneck.',
        timings={key:dict(wallMedianSeconds=statistics.median(pair[0] for pair in values),
            cpuMedianSeconds=statistics.median(pair[1] for pair in values)) for key,values in samples.items()},
        sourceSha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('basil_core/sharp_protocol.py','basil_core/sharp_math.py','basil_core/research_protocol.py')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('mnist','cifar10'),default='cifar10')
    parser.add_argument('--iterations',type=int,choices=range(1,6),default=3)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    print(json.dumps(benchmark(args.dataset,args.iterations),indent=2),flush=True)


if __name__=='__main__':main()
