#!/usr/bin/env python3
"""Independent CPU inference and identity checks for the handoff diagnostic."""
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
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_data import training_data,prepare_partition,digest
    from basil_core.sharp_protocol import load_state,evaluate,vector_hash
    from basil_core.sharp_cart_v2_protocol import fingerprint
    plan=json.loads((root/'plan.json').read_text())
    if plan['source']!=fingerprint():raise ValueError('Underlying scientific implementation changed.')
    runner=hashlib.sha256((ROOT/'scripts/probe_cart_handoff.py').read_bytes()).hexdigest()
    if plan['diagnosticProgramSha256']!=runner:raise ValueError('Diagnostic intervention implementation changed.')
    x,y=training_data('cifar10');data=dict(imagesSha256=digest(x),labelsSha256=digest(y))
    checked=[];pending=[];initial=None;partition=None
    for case in plan['cases']:
        folder=root/case['name'];path=folder/'result.json'
        if not path.exists():pending.append(case['name']);continue
        row=json.loads(path.read_text());contract=row['contract'];cfg=case['config']
        if (row['case']!=case or contract['diagnostic']!=case or contract['config']!=cfg
                or contract['source']!=plan['source'] or contract['data']!=data
                or contract['diagnosticProgramSha256']!=runner):raise ValueError('Diagnostic identity mismatch.')
        worker=SharpWorker(cfg,'CPU');chunks,held,audit=prepare_partition(y,cfg)
        if audit!=contract['partition'] or vector_hash(worker.export())!=contract['initialHash']:
            raise ValueError('Data partition or initialization changed.')
        if initial is None:initial=contract['initialHash'];partition=audit
        if contract['initialHash']!=initial or audit!=partition:raise ValueError('Diagnostic arms are not paired.')
        state=load_state(folder/'checkpoint.npz',contract,worker.dimension)
        if state['cursor']!=cfg['nodes']*cfg['rounds']:raise ValueError('Diagnostic horizon incomplete.')
        if any(r['attackActive'] or r['cart'] is not None or r['training']['rlaCoefficient']!=0 for r in state['telemetry']):
            raise ValueError('Attack, CART or RLA contaminated the intended pre-attack ablation.')
        if case['forceBestSingle'] and any(r['selection']['selectedCount']!=1 for r in state['telemetry']):
            raise ValueError('Best-single intervention did not select one model.')
        stored=state['roundMetrics'][-1]['nodeAccuracy'];fresh=[]
        for params in state['nodeParams']:fresh.append(evaluate(worker,params,x[held],y[held],cfg['classes'])[0])
        difference=float(np.max(np.abs(np.asarray(fresh)-stored)))
        if difference>1/len(held)+1e-9:raise ValueError('Independent CPU inference differs by more than one image per node.')
        checked.append(dict(name=case['name'],maximumNodeAccuracyDifference=difference,pairedDataAndInitialization=True))
        print('VERIFIED '+case['name'],flush=True)
    if require_complete and pending:raise ValueError('Cases still pending: '+str(pending))
    result=dict(status='ALL_DIAGNOSTIC_CASES_VERIFIED' if not pending else 'DIAGNOSTIC_SUBSET_VERIFIED',
        cases=checked,pending=pending,officialTestUsed=False,verificationProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-pending',action='store_true');a=p.parse_args()
    verify(a.output.resolve(),require_complete=not a.allow_pending)
