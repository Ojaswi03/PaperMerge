#!/usr/bin/env python3
"""Independently infer checkpoint predictions on CPU; optional frozen TEST endpoint.

Default evaluates only the original development holdout. --official-test reads
the already-cached benchmark TEST split and is intended after method selection.
It never feeds TEST examples or predictions into training/admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def test_data(dataset):
    import numpy as np
    import pickle
    if dataset=='mnist':
        path=Path.home()/'.keras/datasets/mnist.npz'
        with np.load(path,allow_pickle=False) as a:x=a['x_test'].astype(np.float32)[...,None]/255.;y=a['y_test'].astype(np.int32)
    elif dataset=='cifar10':
        path=Path.home()/'.keras/datasets/cifar-10-batches-py-target/cifar-10-batches-py/test_batch'
        with path.open('rb') as f:a=pickle.load(f,encoding='bytes')
        x=a[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1).astype(np.float32)/255.;y=np.asarray(a[b'labels'],np.int32)
    else:raise ValueError('Explicit TEST loading currently supports MNIST and CIFAR-10.')
    return x,y,dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(root,official_test=False):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()
    from basil_core.cart_comparison_study import effective_config,partitions_for
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import load_state,evaluate,vector_hash
    from basil_core.sharp_data import training_data,digest
    from basil_core.cart_campaign_io import read_json,write_json
    root=Path(root);plan=read_json(root/'plan.json');verified=[];cache={};tests={}
    # Validate recorded producer files rather than requiring every current
    # orchestration/reporting file to remain unchanged forever.
    for name,value in plan['source'].items():
        path=root/'source_snapshots'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=value:raise ValueError('Producer source snapshot mismatch: '+name)
    for name in ('basil_core/sharp_worker.py','basil_core/models.py','basil_core/sharp_data.py','basil_core/acds.py','basil_core/cart_comparison_study.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=plan['source'][name]:raise ValueError('Required inference/data source changed: '+name)
    for view in plan['configs']:
        cfg=effective_config(view);folder=root/view['runId'];row=read_json(folder/'run.json')
        if row['status']!='COMPLETED' or row['config']!=view or row['source']!=plan['source']:raise ValueError('Incomplete/mismatched run.')
        if cfg['dataset'] not in cache:cache[cfg['dataset']]=training_data(cfg['dataset'])
        x,y=cache[cfg['dataset']];_,held,audit=partitions_for(view,y)
        data=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
        if data!=row['data'] or audit!=row['partition'] or audit!=plan['partitions'][view['runId']]:raise ValueError('Data/ACDS contract differs.')
        worker=SharpWorker(cfg,'CPU')
        if worker.dimension<1 or vector_hash(worker.export())!=row['initialHash']:raise ValueError('Initialization/runtime differs.')
        contract=dict(config=cfg,comparisonCase=view['comparisonCase'],methodRevision=plan['revision'],source=plan['source'],
            runtime=row['runtime'],initialHash=row['initialHash'],data=data,partition=audit)
        if contract!=row['contract']:raise ValueError('Checkpoint contract differs.')
        state=load_state(folder/'checkpoint.npz',contract,worker.dimension)
        if state['cursor']!=cfg['rounds']*cfg['nodes'] or len(state['roundMetrics'])!=cfg['rounds']:raise ValueError('Incomplete checkpoint.')
        if any(r['training']['optimizerSteps']!=cfg['localSteps'] for r in state['telemetry']):raise ValueError('Changed SGD budget.')
        with np.load(folder/'metrics.npz',allow_pickle=False) as z:a=z['roundNodeAccuracy'];pc=z['roundNodePerClassAccuracy']
        if not np.array_equal(a,np.asarray([m['nodeAccuracy'] for m in state['roundMetrics']])):raise ValueError('Checkpoint/metrics differ.')
        fresh=[evaluate(worker,p,x[held],y[held],cfg['classes']) for p in state['nodeParams']]
        values=np.asarray([v for v,_ in fresh]);classes=np.asarray([v for _,v in fresh])
        error=float(np.max(abs(values-a[-1])))
        if error>1/len(held)+1e-9:raise ValueError('CPU validation accuracy differs by more than one image/client.')
        counts=np.bincount(y[held],minlength=cfg['classes'])
        if np.any(abs(classes-pc[-1])>1/counts[None,:]+1e-9):raise ValueError('CPU per-class recall differs beyond one image.')
        result=dict(runId=view['runId'],maximumValidationAccuracyDifference=error,finalValidationAccuracy=float(values.mean()))
        if official_test:
            if cfg['dataset'] not in tests:tests[cfg['dataset']]=test_data(cfg['dataset'])
            tx,ty,identity=tests[cfg['dataset']]
            inference=[evaluate(worker,p,tx,ty,cfg['classes']) for p in state['nodeParams']]
            result['officialTest']=dict(data=identity,clientAccuracy=[v for v,_ in inference],
                clientClassRecall=[v.tolist() for _,v in inference],meanAccuracy=float(np.mean([v for v,_ in inference])))
        verified.append(result);print('Verified '+view['runId'],flush=True)
    receipt=dict(status='ALL_COMPARISON_MODELS_VERIFIED',verified=verified,source=plan['source'],officialTestUsed=official_test,
        verifierSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpretation='Fresh CPU inference, not replay of every training update.')
    name='official_test_evaluation.json' if official_test else 'independent_validation.json'
    destination=root/name
    if destination.exists() and read_json(destination)!=receipt:raise FileExistsError('Preserve existing inference receipt before a differing evaluation.')
    write_json(destination,receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--official-test',action='store_true');a=p.parse_args();verify(a.output.resolve(),a.official_test)
