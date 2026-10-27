#!/usr/bin/env python3
"""Independent CPU inference and telemetry audit of completed SHARP runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def verify(root,*,require_complete=False):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import tensorflow as tf
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_data import training_data,prepare_partition
    from basil_core.sharp_protocol import load_state,evaluate,vector_hash
    from basil_core.sharp_math import hop_coefficient
    from basil_core.research_protocol import resolve_attackers
    from scripts.run_sharp_study import check_source,write_json
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    plan=json.loads((root/'plan.json').read_text());check_source(plan['source'])
    for name,digest in plan.get('executionProgram',{}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Validation execution program changed after its plan was frozen.')
    reports=[];pending=[];rows={}
    for cfg in plan['configs']:
        folder=root/cfg['runId'];path=folder/'run.json'
        if not path.exists() or json.loads(path.read_text())['status']!='COMPLETED':
            pending.append(cfg['runId']);continue
        run=json.loads(path.read_text())
        if run['config']!=cfg or run['source']!=plan['source'] or run['data']!=plan['data'][cfg['dataset']]:
            raise ValueError('Completed run manifest differs from frozen plan.')
        worker=SharpWorker(cfg,'CPU')
        if vector_hash(worker.export())!=run['initialHash']:raise ValueError('CPU and GPU common initializations differ.')
        x,y=training_data(cfg['dataset']);chunks,heldout,audit=prepare_partition(y,cfg)
        if audit!=run['partition'] or audit!=plan['partitions'][cfg['runId']]:raise ValueError('Partition audit mismatch.')
        contract={key:run[key] for key in ('config','source','data','partition','runtime','initialHash')}
        state=load_state(folder/'checkpoint.npz',contract,worker.dimension)
        if state['cursor']!=cfg['nodes']*cfg['rounds']:raise ValueError('Completed run is short of its horizon.')
        with np.load(folder/'metrics.npz',allow_pickle=False) as arrays:
            accuracy=arrays['roundNodeAccuracy'];per_class=arrays['roundNodePerClassAccuracy']
        if accuracy.shape!=(cfg['rounds'],cfg['nodes']) or per_class.shape!=(cfg['rounds'],cfg['nodes'],cfg['classes']):
            raise ValueError('Metric dimensions do not match nodes/rounds/classes.')
        fresh=[];fresh_classes=[]
        for params in state['nodeParams']:
            a,pc=evaluate(worker,params,x[heldout],y[heldout],cfg['classes']);fresh.append(a);fresh_classes.append(pc)
        difference=float(np.max(np.abs(np.asarray(fresh)-accuracy[-1])))
        if difference>1/len(heldout)+1e-9:raise ValueError('Independent CPU final accuracy differs by more than one picture per node.')
        counts=np.bincount(y[heldout],minlength=cfg['classes'])
        discrepancies=np.sum(np.abs(np.rint(np.asarray(fresh_classes)*counts)-np.rint(per_class[-1]*counts)),axis=1)
        if discrepancies.max()>1:raise ValueError('Independent per-class correct counts disagree.')
        scales=[];active_gates=0;all_fail=0
        attackers=resolve_attackers(cfg['seed'],cfg['actualAttackers'],cfg['nodes'])
        post_selected=0;post_attackers=0
        for index,record in enumerate(state['telemetry']):
            if record['activation']!=index+1 or (record['round'],record['node'])!=divmod(index,cfg['nodes']):
                raise ValueError('Telemetry cursor/order mismatch.')
            training=record['training'];selection=record['selection']
            node=record['node'];round_id=record['round']
            allowed=[(node-distance)%cfg['nodes'] for distance in range(1,cfg['window']+1)
                if round_id>0 or (node-distance)%cfg['nodes']<node]
            senders=selection['selectedSenders']
            if (len(set(senders))!=len(senders) or not set(senders).issubset(allowed)
                    or selection['availableCopies']!=len(allowed)
                    or selection['selectedCount']!=(len(senders) if allowed else 1)):
                raise ValueError('Selected senders violate the ring/window definition.')
            if (record['configuredByzantine']!=(node in attackers)
                    or record['attackActive']!=(node in attackers and round_id>=cfg['attackStart'])
                    or record['selectedConfiguredAttackers']!=sum(s in attackers for s in senders)):
                raise ValueError('Attack audit disagrees with the seeded adversary set.')
            if round_id>cfg['attackStart']:
                post_selected+=len(senders);post_attackers+=record['selectedConfiguredAttackers']
            if training['optimizerSteps']!=cfg['localSteps']:raise ValueError('Local steps differ from PDF/config.')
            eta=cfg['learningRate']/(1+cfg['learningRate']*record['round'])
            q=selection['selectedCount'] if cfg['hopDenominator']=='actual_selected' else cfg['byzantineBudget']
            expected=(hop_coefficient(cfg['sigma'],q,eta,cfg['curvature']) if cfg['method']=='sharp' else cfg['sigma']**2) if cfg['switches']['noise_mitigation'] else 0.
            if not np.isclose(training['rlaCoefficient'],expected,rtol=1e-12,atol=1e-15):raise ValueError('Wrong hop-aware RLA coefficient.')
            if not record['attackActive'] and record['outboundHash']!=record['trainedHash']:raise ValueError('Unrequested outbound mutation.')
            all_fail+=selection['screenAllFailFallback']
            cart=record['cart']
            if cart:
                if cart['backboneBeforeHash']!=cart['backboneAfterHash'] or not cart['fitCheckDisjoint']:
                    raise ValueError('CART backbone/probe invariant violated.')
                if len(cart['classGates'])!=cfg['classes']:raise ValueError('Wrong CART class count.')
                if cart['checkBefore'] is not None and cart['checkAfter']>cart['checkBefore']+1e-6:
                    raise ValueError('CART accepted a worse checked macro loss.')
                scales.append(cart['scale']);active_gates+=len(cart.get('activeGateClasses',[]))
        reports.append(dict(runId=cfg['runId'],dataset=cfg['dataset'],partition=cfg['partition'],seed=cfg['seed'],
            sigma=cfg['sigma'],condition=cfg['condition'],nodes=cfg['nodes'],rounds=cfg['rounds'],
            finalMean=float(np.mean(fresh)),finalWorstNode=float(np.min(fresh)),
            finalWorstClassMean=float(np.asarray(fresh_classes).mean(axis=0).min()),
            cpuGpuMaxAccuracyDifference=difference,nonzeroCartCorrections=sum(s>0 for s in scales),
            activeClassGates=active_gates,screenAllFailFallbacks=int(all_fail),
            resolvedAttackerIds=list(attackers),
            selectedAttackerFractionAfterAllSendersActive=post_attackers/post_selected if post_selected else None,
            checkpointComplete=True,officialTestUsed=False))
        rows[(cfg['dataset'],cfg['partition'],cfg['method'],cfg['seed'],cfg['sigma'],cfg['condition'])]=state['telemetry']
        print(f'VERIFIED {cfg["runId"]}: mean {np.mean(fresh):.2%}, CART commits {sum(s>0 for s in scales)}',flush=True)
    for key,history in rows.items():
        d,p,m,s,sigma,c=key
        if c.endswith('_cart') and (d,p,m,s,sigma,c[:-5]) in rows:
            control=rows[d,p,m,s,sigma,c[:-5]]
            n=next(r['nodes'] for r in reports if (r['dataset'],r['partition'],r['seed'],r['sigma'],r['condition'])==(d,p,s,sigma,c))
            for a,b in zip(history[:n],control[:n]):
                if (a['inputHash'],a['trainedHash'])!=(b['inputHash'],b['trainedHash']):
                    raise ValueError('Control/CART differ before a receiver has any historical memory.')
    if require_complete and pending:raise ValueError(f'{len(pending)} selected experiments are not complete.')
    result=dict(status='ALL_SELECTED_RUNS_VERIFIED' if not pending else 'COMPLETED_SUBSET_VERIFIED',
        verified=len(reports),pending=pending,rows=reports,officialTestUsed=False,
        verificationProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(root/'independent_validation.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    result=verify(args.output.resolve(),require_complete=args.require_complete)
    print(json.dumps(dict(status=result['status'],verified=result['verified'],pending=len(result['pending'])),indent=2))
