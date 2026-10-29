#!/usr/bin/env python3
"""Device-specific exactness gate for a compiled-transfer checkpoint continuation."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(output,gpu):
    import numpy as np
    from scripts.run_cart_v2_study import setup_gpu,runtime
    from scripts.run_cart_v2_transfer import program_hashes
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    from basil_core.sharp_math import hop_coefficient
    from basil_core.sharp_data import training_data,prepare_partition
    from basil_core.sharp_cart_v2_protocol import check_source
    plan=read_json(output/'plan.json');check_source(plan['source']);tf=setup_gpu();checks=[]
    chosen={}
    for cfg in plan['configs']:
        key=(cfg['dataset'],cfg['model'])
        if key not in chosen or cfg['switches']['noise_mitigation']:chosen[key]=cfg
    for cfg in chosen.values():
        old=SharpWorker(cfg,'GPU');fast=CompiledTransferWorker(cfg,'GPU')
        initial=old.export();np.testing.assert_array_equal(initial,fast.export())
        x,y=training_data(cfg['dataset']);chunks,_,_=prepare_partition(y,cfg)
        indices=chunks[0][:cfg['batchSize']];images=x[indices];labels=y[indices]
        rng=np.random.default_rng(20261028)
        candidates=initial[None,:]+rng.normal(0,.001,size=(66,old.dimension)).astype(np.float32)
        old.probe(candidates[0],images,labels);fast.probe(candidates[0],images,labels)
        times={'original':[],'compiled':[]}
        for repeat in range(3):
            scores={}
            order=[('original',old),('compiled',fast)]
            if repeat%2:order.reverse()
            for name,worker in order:
                started=time.perf_counter();scores[name]=[worker.probe(c,images,labels) for c in candidates]
                times[name].append(time.perf_counter()-started)
            np.testing.assert_array_equal(scores['original'],scores['compiled'])
            np.testing.assert_array_equal(old.export(),fast.export())
        batch=np.arange(len(labels))
        coefficient=hop_coefficient(cfg['sigma'],1,cfg['learningRate']) if cfg['switches']['noise_mitigation'] else 0.
        a,ar=old.train(initial,images,labels,0,0,coefficient,batch)
        b,br=fast.train(initial,images,labels,0,0,coefficient,batch)
        np.testing.assert_array_equal(a,b)
        if ar!=br:raise AssertionError('Compiled-transfer SGD telemetry differs.')
        durations={k:statistics.median(v) for k,v in times.items()}
        checks.append(dict(dataset=cfg['dataset'],model=cfg['model'],candidateLossesExact=True,
            trainedWeightsExact=True,trainingTelemetryExact=True,medianSeconds=durations,
            scoringSpeedup=durations['original']/durations['compiled']))
        print(f'GPU {gpu}: {cfg["dataset"]} transfer exact; scoring {checks[-1]["scoringSpeedup"]:.2f}x.',flush=True)
    write_json(output/f'compiled_transfer_verification_{gpu}.json',dict(status='PASSED',
        source=plan['source'],programSha256=program_hashes(),runtime=runtime(tf),checks=checks,
        officialTestUsed=False,researchRunsCompleted=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);a=p.parse_args();verify(a.output,a.gpu)
