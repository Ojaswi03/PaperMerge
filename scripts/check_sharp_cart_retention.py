#!/usr/bin/env python3
"""Paired local-adaptation mechanism check on every completed clean node.

This isolates one additional five-step local visit from snapshot selection and
channel effects. It is not an end-to-end SHARP robustness or multi-seed result.
The saved clean node is the reference; no holdout enters fitting or acceptance.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    import tensorflow as tf
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_data import training_data,prepare_partition
    from basil_core.sharp_protocol import load_state,evaluate
    from basil_core.sharp_cart import transport,probes
    from basil_core.sharp_math import class_ce
    from basil_core.research_protocol import keyed_rng
    from scripts.run_sharp_study import check_source,write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--dataset',choices=('mnist','cifar10','cifar100'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    plan=json.loads((args.campaign/'plan.json').read_text());check_source(plan['source'])
    configs=[c for c in plan['configs'] if c['dataset']==args.dataset and c['partition']=='dirichlet' and c['condition']=='clean']
    if len(configs)!=1:raise ValueError('Select a completed campaign with exactly one clean Dirichlet seed for this dataset.')
    cfg=configs[0];folder=args.campaign/cfg['runId'];metadata=json.loads((folder/'run.json').read_text())
    if metadata['status']!='COMPLETED':raise ValueError('Clean source run is incomplete.')
    worker=SharpWorker(cfg,'CPU');contract={key:metadata[key] for key in ('config','source','data','partition','runtime','initialHash')}
    state=load_state(folder/'checkpoint.npz',contract,worker.dimension)
    x,y=training_data(cfg['dataset']);chunks,heldout,audit=prepare_partition(y,cfg)
    if audit!=contract['partition']:raise ValueError('Source partition changed.')
    records=[]
    for node,reference in enumerate(state['nodeParams']):
        local=chunks[node];lx=x[local];ly=y[local];fit,check=probes(ly,cfg['seed'],node)
        logits=worker.outputs(reference,lx[fit]);classes,losses=class_ce(logits,ly[fit])
        memory=dict(params=reference.copy(),classes=classes.tolist(),losses=losses.tolist(),
                    allowance=np.zeros(len(classes)).tolist())
        round_id=cfg['rounds']
        first=keyed_rng(cfg['seed'],'sharp_batch',round_id,node,0).choice(len(ly),min(cfg['batchSize'],len(ly)),replace=False)
        proposal,training=worker.train(reference,lx,ly,round_id,node,0.,first)
        final,_,evidence=transport(worker,proposal,memory,lx,ly,node,round_id*cfg['nodes']+node)
        before,_=evaluate(worker,reference,x[heldout],y[heldout],cfg['classes'])
        after,_=evaluate(worker,proposal,x[heldout],y[heldout],cfg['classes'])
        corrected,_=evaluate(worker,final,x[heldout],y[heldout],cfg['classes'])
        records.append(dict(node=node,referenceAccuracy=before,proposalAccuracy=after,cartAccuracy=corrected,
                            delta=corrected-after,cart=evidence,optimizerSteps=training['optimizerSteps']))
        if (node+1)%10==0:print(f'{args.dataset}: checked {node+1}/{cfg["nodes"]} clean receivers.',flush=True)
    result=dict(status='COMPLETED_MECHANISM_CHECK',source=plan['source'],sourceRun=cfg['runId'],
        dataset=cfg['dataset'],seed=cfg['seed'],nodes=cfg['nodes'],classes=cfg['classes'],
        sourceConfig=cfg,case='one_extra_five_step_local_adaptation_no_attack_no_channel',
        allReceiversIncluded=True,holdoutUsedForDecisions=False,officialTestUsed=False,rows=records,
        referenceMean=float(np.mean([r['referenceAccuracy'] for r in records])),
        proposalMean=float(np.mean([r['proposalAccuracy'] for r in records])),
        cartMean=float(np.mean([r['cartAccuracy'] for r in records])),
        nonzeroCorrections=sum(r['cart']['scale']>0 for r in records),
        improvedReceivers=sum(r['delta']>0 for r in records),worsenedReceivers=sum(r['delta']<0 for r in records),
        interpretation='Isolated retention check after a clean trained reference. Not an attack/noise benchmark or independent-seed replication.')
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source','sourceConfig','rows')},indent=2))


if __name__=='__main__':main()
