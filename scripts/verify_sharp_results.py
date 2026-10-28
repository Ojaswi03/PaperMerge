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
    plan=json.loads((root/'plan.json').read_text())
    methods={c['method'] for c in plan['configs']}
    guarded={c['method'] in ('sharp_guarded','sharp_cart_v2') for c in plan['configs']}
    if len(guarded)>1:raise ValueError('Original and guarded protocols require separate frozen plans.')
    if 'sharp_cart_v2' in methods:
        if methods!={'sharp_cart_v2'}:raise ValueError('CART v2 requires its own frozen plan.')
        from basil_core.sharp_cart_v2_protocol import check_source as check_v2,validate_config
        check_v2(plan['source'])
    elif guarded=={True}:
        from basil_core.sharp_guarded_protocol import check_source as check_guarded,validate_config
        protocol_source=dict(plan['source'])
        launcher='scripts/run_guarded_sharp_study.py'
        extra=protocol_source.pop(launcher,None)
        check_guarded(protocol_source)
        if extra is not None and hashlib.sha256((ROOT/launcher).read_bytes()).hexdigest()!=extra:
            raise ValueError('Guarded GPU launcher changed from the frozen plan.')
    else:
        from basil_core.sharp_study import validate_config
        check_source(plan['source'])
    for cfg in plan['configs']:validate_config(cfg)
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
        if cfg['method']=='sharp_cart_v2':
            from basil_core.research_protocol import keyed_rng
            from basil_core.attacks import applyAttack
            from basil_core.sharp_protocol import link_vector
            for sender,params in enumerate(state['nodeParams']):
                expected_outbound=params.copy();r=cfg['rounds']-1
                if sender in attackers and r>=cfg['attackStart']:
                    if cfg['attackType']=='hidden':
                        pieces=applyAttack(worker.tensors(params),'hidden',rng=keyed_rng(cfg['seed'],'sharp_attack',r,sender))
                        expected_outbound=np.concatenate([p.ravel() for p in pieces]).astype(np.float32)
                    else:
                        coordinate=int(keyed_rng(cfg['seed'],'sharp_spike',r,sender).integers(len(params)))
                        expected_outbound[coordinate]+=np.float32(20*max(float(np.sqrt(np.mean(params.astype(np.float64)**2))),.01)*np.sqrt(len(params)))
                np.testing.assert_array_equal(state['outbound'][sender],expected_outbound,
                    err_msg='Final outbound attack differs from the declared seeded transformation.')
            if cfg['sigma']:
                base=state['outbound'][0];r=cfg['rounds']-1
                first=link_vector(base,cfg['seed'],r,0,1,cfg['sigma'])
                repeat=link_vector(base,cfg['seed'],r,0,1,cfg['sigma'])
                other=link_vector(base,cfg['seed'],r,0,2,cfg['sigma'])
                np.testing.assert_array_equal(first,repeat)
                if np.array_equal(first,base) or np.array_equal(first,other):
                    raise ValueError('Declared channel is absent or reused between receivers.')
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
            if cfg['method'] in ('sharp_guarded','sharp_cart_v2') and cfg['switches']['byzantine_mitigation'] and allowed:
                raw=selection['rawSurvivorCount'];b=cfg['byzantineBudget']
                if selection.get('screenApplied'):
                    observed=sum(v<=selection['peakThreshold'] for v in selection['peakRatios'])
                else:observed=len(allowed)
                target=1 if raw<=b else min(b,raw-b)
                if (raw!=observed or selection['screenGuardFallback']!=(raw<=b)
                        or selection['targetCount']!=target or selection['selectedCount']>target
                        or selection['survivorUnattackedLowerBound']!=max(0,raw-b)):
                    raise ValueError('Guarded survivor count/fallback is inconsistent.')
            eta=cfg['learningRate']/(1+cfg['learningRate']*record['round'])
            q=selection['selectedCount'] if cfg['hopDenominator']=='actual_selected' else cfg['byzantineBudget']
            expected=(hop_coefficient(cfg['sigma'],q,eta,cfg['curvature']) if cfg['method'] in ('sharp','sharp_guarded','sharp_cart_v2') else cfg['sigma']**2) if cfg['switches']['noise_mitigation'] else 0.
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
                if cfg['method']=='sharp_cart_v2':
                    if cart['referenceMistakeTargets']!=0:raise ValueError('CART v2 copied a reference mistake.')
                    if cart['checkCorrectBefore'] is not None and np.any(np.asarray(cart['checkCorrectAfter'])<cart['checkCorrectBefore']):
                        raise ValueError('CART v2 decreased a checked class correct count.')
                    if cart['scale']>0 and not cart['repairedFitExamples']:
                        raise ValueError('CART v2 correction has no demonstrated forgetting target.')
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
        if cfg['method']=='sharp_cart_v2':reports[-1].update(finalAttackTransformVerified=True,
            channelReplayAndReceiverIndependenceVerified=bool(cfg['sigma']))
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
