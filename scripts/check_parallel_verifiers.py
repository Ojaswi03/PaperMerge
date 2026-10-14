#!/usr/bin/env python3
"""Bounded verifier-transport equivalence check; computes CE/gradients, zero SGD."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def check():
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',
        TF_DETERMINISTIC_OPS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PAPERMERGE_LIVE_TRACKING='disabled',WANDB_MODE='disabled')
    import numpy as np
    from basil_core.research_protocol import build_model,params_hash
    from basil_core.isolated_replicas import IsolatedReplicaPool
    from basil_core.parallel_verifier_service import ParallelVerifierService
    from basil_core.replicated_verification import ComputeRequest,digest_json
    from basil_core.verified_cart_runner import fingerprints
    from scripts.check_verified_cart_full_batch import health
    from scripts.run_noniid_verified_condition import atomic_json
    cache=ROOT/'experiments/cache/cifar10'
    x=np.load(cache/'x_train_float32.npy',mmap_mode='r',allow_pickle=False)
    y=np.load(cache/'y_train_int32.npy',mmap_mode='r',allow_pickle=False)
    model=build_model('basil_paper_cnn',2025)
    params=[w.numpy() for w in model.trainable_variables]
    rows=[]
    with tempfile.TemporaryDirectory(prefix='cart-parallel-check-') as tmp:
        fixture=Path(tmp)/'train.npz'
        np.savez(fixture,images=x,labels=y,synthetic=np.asarray(False))
        for case,faults in (('honest',{}),('four_faults',{0:'poison',1:'forged',5:'withhold',7:'exit'})):
            with IsolatedReplicaPool(fixture,faults=faults,timeout=15.) as pool, ParallelVerifierService(pool.endpoints) as concurrent:
                before=health(pool)
                owner=4
                for kind,count in (('selection_ce',512),('gradient',16)):
                    request=ComputeRequest('parallel_verifier_transport_check',0,owner,20,0,kind,
                        params_hash(params),pool.endpoints[0].bootstrap['dataRoot'],
                        tuple(map(int,np.flatnonzero(y==owner)[:count])),(7,8) if kind=='gradient' else ())
                    start=time.monotonic();serial,proof=pool.service.evaluate(request,params);serial_time=time.monotonic()-start
                    start=time.monotonic();parallel,other=concurrent.evaluate(request,params);parallel_time=time.monotonic()-start
                    assert serial==parallel
                    assert proof['resultHash']==other['resultHash']
                    assert proof['matchingSigners']==other['matchingSigners']
                    assert proof['signatures']==other['signatures']
                    assert proof['requiredMatches']==other['requiredMatches']==5
                    assert len(other['matchingSigners'])>=5
                    for a,b in zip(serial.arrays(),parallel.arrays()):np.testing.assert_array_equal(a,b)
                    rows.append(dict(case=case,kind=kind,samples=count,loss=serial.cross_entropy,
                        resultHash=other['resultHash'],matchingSigners=other['matchingSigners'],
                        signaturesIdentical=True,arraysIdentical=True,
                        serialSeconds=serial_time,parallelSeconds=parallel_time,
                        countersBefore=before,countersAfter=health(pool)))
    sources=dict(fingerprints(),**{'basil_core/parallel_verifier_service.py':hashlib.sha256((ROOT/'basil_core/parallel_verifier_service.py').read_bytes()).hexdigest()})
    result=dict(status='passed',trainingUpdates=0,testExamplesLoaded=0,sourceHashes=sources,
        checkerSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=rows,
        semantics='Nine distinct CPU computations; same canonical certificates/results; only RPC scheduling changed')
    atomic_json(ROOT/'gui/worker_state/CART_PARALLEL_VERIFIER_CHECK.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    for row in rows:print(json.dumps({k:v for k,v in row.items() if not k.startswith('counters')}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:check()
    else:print(json.dumps(dict(trainingUpdates=0,checksStarted=0)))


if __name__=='__main__':main()

