#!/usr/bin/env python3
"""Independently reconstruct final TRAIN-holdout predictions from checkpoints.

Inference only, explicitly on CPU. No fitting, parameter selection, official
TEST loading, or changes to a running training cohort are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def compare_predictions(predictions,labels,saved_nodes,saved_classes):
    import numpy as np
    p=np.asarray(predictions);y=np.asarray(labels)
    if (p.ndim!=2 or p.shape[1]!=len(y) or not np.issubdtype(p.dtype,np.integer)
            or not np.issubdtype(y.dtype,np.integer) or set(y.tolist())!=set(range(10))
            or np.any(p<0) or np.any(p>=10)):
        raise ValueError('Ten supported classes and aligned integer predictions required.')
    nodes=(p==y[None,:]).mean(axis=1)
    classes=np.stack([(p[:,y==c]==c).mean(axis=1) for c in range(10)],axis=1)
    if (np.shape(saved_nodes)!=nodes.shape or np.shape(saved_classes)!=classes.shape
            or not np.allclose(nodes,saved_nodes,rtol=0,atol=1e-6)
            or not np.allclose(classes,saved_classes,rtol=0,atol=1e-6)):
        raise ValueError('Independent CPU predictions do not reproduce the saved node/class accuracies.')
    return dict(meanAccuracy=float(nodes.mean()),nodeAccuracies=nodes.tolist(),classAccuracies=classes.tolist())


def verify(root,run_id):
    import numpy as np
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    from basil_core.research_protocol import build_model,params_hash
    from scripts.run_factorial_study import data
    from scripts.run_cart_memory_smoke import save_json
    plan=json.loads((root/'plan.json').read_text())
    cfg=next(c for c in plan['configs'] if c['runId']==run_id)
    folder=root/run_id;run=json.loads((folder/'run.json').read_text())
    if (run['status']!='COMPLETED' or run['config']!=cfg or run['source']!=plan['source']
            or run.get('officialTestUsed') is not False or plan.get('officialTestUsed') is not False):
        raise ValueError('A completed frozen-plan run is required.')
    for name,digest in plan['source'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Current inference dependencies differ from the frozen source.')
    (_,_,images,labels),hashes=data()
    if hashes!=plan['data'] or run['data']!=hashes:raise ValueError('TRAIN holdout identity differs.')
    prepare_environment('CPU')
    import tensorflow as tf
    configure_tensorflow(tf,'CPU')
    with tf.device('/CPU:0'):
        model=build_model(cfg['modelArchitecture'],cfg['seed'])
        predict=tf.function(lambda x:model(x,training=False),reduce_retracing=True)
        predictions=[]
        with np.load(folder/'checkpoint.npz',allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata']))
            if (metadata['config']!=cfg or metadata['nextActivation']!=10*cfg['nRounds']
                    or metadata['implementation']!=plan['source']):
                raise ValueError('Checkpoint cursor/config differs.')
            for node,digest in enumerate(metadata['nodeParams']):
                params=[saved[f'{digest}_{i}'] for i in range(metadata['weights'][digest])]
                if params_hash(params)!=digest:raise ValueError('Corrupt checkpoint model.')
                model.set_weights(params)
                labels_out=np.concatenate([predict(images[i:i+128]).numpy().argmax(axis=1)
                    for i in range(0,len(images),128)])
                predictions.append(labels_out)
                print(f'{run_id}: reconstructed node {node} on CPU.',flush=True)
        with np.load(folder/'metrics.npz',allow_pickle=False) as saved:
            comparison=compare_predictions(predictions,labels,saved['roundNodeAccuracy'][-1],saved['perClass'][-1])
    target=root/'reports/holdout_recheck';target.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(target/f'{run_id}.npz',predictions=np.asarray(predictions),labels=labels)
    result=dict(status='independent_cpu_predictions_match',runId=run_id,device='CPU',
        tensorflow=tf.__version__,officialTestUsed=False,optimizerSteps=0,
        verificationSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**comparison)
    save_json(target/f'{run_id}.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--run-id',required=True)
    args=p.parse_args();result=verify(args.root,args.run_id)
    print(json.dumps(dict(status=result['status'],runId=result['runId'],meanAccuracy=result['meanAccuracy'])))


if __name__=='__main__':main()
