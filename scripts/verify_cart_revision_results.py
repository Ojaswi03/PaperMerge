#!/usr/bin/env python3
"""Independently evaluate saved revision models on their original TRAIN holdout.

Producer source snapshots are checked against their recorded hashes; the
unchanged base architecture/data implementation must match the current one.
This does not independently replay every optimizer step.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(root,require_complete=True):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core.sharp_cart_v2_protocol import fingerprint,validate_config
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_data import training_data,prepare_partition,digest
    from basil_core.sharp_protocol import load_state,evaluate,vector_hash
    root=Path(root);plan=read_json(root/'plan.json')
    if plan['baseSource']!=fingerprint():raise ValueError('Base model/data/physics implementation differs.')
    for name,value in plan['source'].items():
        if hashlib.sha256((root/'source_snapshots'/name).read_bytes()).hexdigest()!=value:raise ValueError('Producer source snapshot mismatch: '+name)
    checked=[];pending=[];data={}
    for cfg in plan['configs']:
        validate_config(cfg);folder=root/cfg['runId'];path=folder/'run.json'
        if not path.exists() or read_json(path)['status']!='COMPLETED':pending.append(cfg['runId']);continue
        row=read_json(path);contract=row['contract']
        if row['config']!=cfg or row['source']!=plan['source'] or row['methodRevision']!=plan['revision']:
            raise ValueError('Revision run identity differs.')
        if cfg['dataset'] not in data:data[cfg['dataset']]=training_data(cfg['dataset'])
        x,y=data[cfg['dataset']];chunks,held,audit=prepare_partition(y,cfg)
        expected_data=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
        if expected_data!=row['data'] or audit!=row['partition'] or audit!=plan['partitions'][cfg['runId']]:raise ValueError('Data identity differs.')
        worker=SharpWorker(cfg,'CPU')
        if vector_hash(worker.export())!=row['initialHash']:raise ValueError('Initialization differs; use matching frozen TensorFlow/NumPy versions.')
        if contract!=dict(config=cfg,methodRevision=plan['revision'],source=plan['source'],runtime=row['runtime'],
                initialHash=row['initialHash'],data=expected_data,partition=audit):raise ValueError('Checkpoint contract differs.')
        state=load_state(folder/'checkpoint.npz',contract,worker.dimension)
        if state['cursor']!=cfg['nodes']*cfg['rounds'] or len(state['roundMetrics'])!=cfg['rounds']:raise ValueError('Incomplete saved horizon.')
        if any(r['training']['optimizerSteps']!=cfg['localSteps'] for r in state['telemetry']):raise ValueError('Unexpected local SGD step count.')
        for record in state['telemetry']:
            route=record['selection'].get('continuity',{});qc=(record.get('cart') or {}).get('qualityCommit',{})
            if route.get('qualityScoreBoundApplied') and route.get('carrierScore') is not None and route['selectedScore']<route['carrierScore']-.01-1e-12:
                raise ValueError('Recorded ingress score invariant failed.')
            if qc.get('applied') and qc['committedScore']<qc['baselineScore']-.01-1e-12:raise ValueError('Recorded commit score invariant failed.')
        with np.load(folder/'metrics.npz',allow_pickle=False) as arrays:saved=arrays['roundNodeAccuracy'];saved_class=arrays['roundNodePerClassAccuracy']
        if not np.array_equal(saved,np.asarray([r['nodeAccuracy'] for r in state['roundMetrics']])):raise ValueError('Exported metrics differ from checkpoint.')
        fresh=[];fresh_class=[]
        for flat in state['nodeParams']:
            acc,pc=evaluate(worker,flat,x[held],y[held],cfg['classes']);fresh.append(acc);fresh_class.append(pc)
        error=float(np.max(np.abs(np.asarray(fresh)-saved[-1])));pc_error=float(np.max(np.abs(np.asarray(fresh_class)-saved_class[-1])))
        if error>1/len(held)+1e-9:raise ValueError('Fresh CPU accuracy differs by more than one held-out image per node.')
        counts=np.bincount(y[held],minlength=cfg['classes'])
        if np.any(np.abs(np.asarray(fresh_class)-saved_class[-1])>1/counts[None,:]+1e-9):raise ValueError('Fresh class accuracy exceeds one-image tolerance.')
        checked.append(dict(runId=cfg['runId'],maximumNodeAccuracyDifference=error,maximumPerClassAccuracyDifference=pc_error,
            finalAccuracy=float(np.mean(fresh)),optimizerSteps=row['optimizerSteps']))
        print('VERIFIED '+cfg['runId'],flush=True)
    if require_complete and pending:raise ValueError('Pending/incomplete revision runs remain.')
    receipt=dict(status='ALL_REVISION_MODELS_VERIFIED' if not pending else 'REVISION_SUBSET_VERIFIED',verified=checked,pending=pending,
        methodRevision=plan['revision'],source=plan['source'],officialTestUsed=False,
        verifierSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(root/'independent_validation.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-pending',action='store_true');a=p.parse_args();verify(a.output.resolve(),not a.allow_pending)
