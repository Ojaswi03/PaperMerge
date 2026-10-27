#!/usr/bin/env python3
"""Verify and aggregate completed factorial results at matching noise levels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def verify_checkpoint(folder,cfg,source,values,*,class_values=None):
    """Verify committed metrics, parameter hashes and every activation's switches."""
    import numpy as np
    from basil_core.protocol_checkpoint import _load_telemetry
    from basil_core.research_protocol import params_hash
    partition=json.loads((folder/'partition.json').read_text())
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as saved:
        cp=json.loads(str(saved['metadata']))
        if (cp['config']!=cfg or cp['implementation']!=source or
                cp['nextActivation']!=cfg['nRounds']*10 or cp['completedRounds']!=cfg['nRounds'] or
                cp['partitionHash']!=partition['hash'] or cp['initialHash']!=partition['initialModelHash']):
            raise ValueError('Checkpoint identity/cursor differs from the completed run.')
        np.testing.assert_array_equal(saved['round_node_accuracy'],values)
        if class_values is not None:
            np.testing.assert_array_equal(saved['round_node_per_class'],class_values)
        for digest,count in cp['weights'].items():
            params=[saved[f'{digest}_{i}'] for i in range(count)]
            if params_hash(params)!=digest or any(not np.isfinite(p).all() for p in params):
                raise ValueError('Checkpoint parameter hash/non-finite failure.')
        records=_load_telemetry(folder/'checkpoint.npz',cp)
        if cfg.get('cartVariant') in ('coherent','coherent_refresh','coherent_commit_ce','coherent_commit_error','coherent_qualified_error','coherent_projected_error','coherent_head_transport'):
            from basil_core.cart_feasible_worker import unpack_memory
            unpack_memory(cp['coherentCartMemory'],cp['coherentCartMemoryHash'],seed=cfg['seed'],
                cursor=cp['nextActivation'],unpack_params=lambda digest:
                    [saved[f'{digest}_{i}'].copy() for i in range(cp['weights'][digest])])
        elif cfg['cartEnabled']:
            from basil_core.cart_memory_study import validate_memory_checkpoint
            from basil_core.cart_local_memory import state_hash
            state=validate_memory_checkpoint(cp['cartMemory'],cp['nextActivation'],cfg['seed'])
            if state_hash(state)!=cp['cartMemoryHash']:raise ValueError('Corrupt CART memory.')
    if len(records)!=cfg['nRounds']*10:raise ValueError('Incomplete optimizer evidence.')
    for index,r in enumerate(records):
        round_id,node=divmod(index,10)
        attacked=cfg['attackHidden'] and node in cfg['resolvedAttackerIds'] and round_id>=cfg['attackStartRound']
        if ((r['round'],r['nodeId'])!=(round_id,node) or r['attackActive']!=attacked or
                r['channelNoiseActive']!=cfg['useChannelNoise'] or r['ebmMode']!=cfg['ebmMode']):
            raise ValueError('Activation environment/defense switches differ from the plan.')
        if bool(r.get('cartMemory'))!=cfg['cartEnabled']:raise ValueError('Missing or unexpected CART evidence.')
        if cfg.get('logitCalibrationEnabled'):
            from basil_core.dirichlet_calibration import calibration_offsets
            expected_offsets=calibration_offsets(r['localClassCounts'],method='prior_adjustment')
            if not np.array_equal(r['logitCalibrationOffsets'],expected_offsets):
                raise ValueError('Training-prior offsets differ from declared receiver TRAIN counts.')
        elif any(r.get('logitCalibrationOffsets',[])):raise ValueError('A hidden prior correction was enabled.')
        candidates=r['candidates']
        if cfg['snapshotSelection']:
            if len(candidates)!=5:raise ValueError('BASIL selection evidence is incomplete.')
            minimum=min(c['receiverLocalLoss'] for c in candidates)
            selected=min((c for c in candidates if c['receiverLocalLoss']<=minimum+1e-8),
                key=lambda c:(node-c['senderId'])%10)
            if (r['inputSourceNode'],r['inputSnapshotRound'])!=(selected['senderId'],selected['snapshotRound']):
                raise ValueError('BASIL did not select the measured receiver-local CE minimum.')
        elif candidates or r['inputSourceNode']!=(node-1)%10:
            raise ValueError('Clean/untreated handoff unexpectedly used mitigation selection.')
        samples=partition['counts'][node]
        sizes=[cfg['batchSize']]*(samples//cfg['batchSize'])+([samples%cfg['batchSize']] if samples%cfg['batchSize'] else [])
        if (len(r['epochs'])!=5 or any(e['samplesSeen']!=samples or e['batchSizes']!=sizes for e in r['epochs'])
                or r['optimizerSteps']!=5*len(sizes) or len(r['optimizerTelemetry'])!=5*len(sizes)):
            raise ValueError('A local epoch/update was skipped.')
        coefficient=cfg['channelNoiseSigmaAbsolute']**2 if cfg['ebmMode']=='gradient_norm_objective' else 0.
        for step in r['optimizerTelemetry']:
            offset_norm=float(np.linalg.norm(expected_offsets)) if cfg.get('logitCalibrationEnabled') else 0.
            if (not math.isclose(step['ebmCoefficient'],coefficient,rel_tol=1e-5,abs_tol=1e-10)
                    or step.get('snapshotAnchorMu',0.)!=0
                    or 'calibrationOffsetNorm' in step and not math.isclose(step['calibrationOffsetNorm'],offset_norm,rel_tol=1e-5,abs_tol=1e-7)):
                raise ValueError('Unexpected EBM coefficient, anchor or calibration.')
        if cfg.get('cartVariant') in ('coherent','coherent_refresh','coherent_commit_ce','coherent_commit_error','coherent_qualified_error','coherent_projected_error','coherent_head_transport'):
            from basil_core.cart_feasible_memory import within_envelope
            memory=r['cartMemory']
            if cfg.get('cartGuardGranularity')=='classifier_transport':
                evidence=memory['transport']
                if (memory['guardMode']!='classifier_transport' or memory['guardEnabled'] or memory['stepGuards']
                        or memory['inputReferenceHash']!=r['inputHash']
                        or memory['committedOutputHash']!=r['trainedOutputHash']
                        or evidence['backboneBeforeHash']!=evidence['backboneAfterHash']
                        or evidence['scale'] not in cfg['cartTransportScales']
                        or any(not 0<=g<=1 for g in evidence['classGates'])
                        or evidence['checkBefore'] is not None and evidence['checkAfter']>evidence['checkBefore']+1e-6
                        or 'fittedObjective' in evidence and evidence['fittedObjective']>evidence['zeroObjective']+1e-6):
                    raise ValueError('Classifier transport changed the backbone/input or failed its local acceptance checks.')
            if cfg.get('cartReferenceQualification'):
                from basil_core.cart_reference_qualification import qualified_error_reference
                before=memory['activeReferenceRisks']
                expected_qualified=qualified_error_reference(before) if before is not None else False
                if (memory['guardEnabled']!=expected_qualified or memory['referenceWasQualified']!=expected_qualified
                        or memory['referenceQualified']!=qualified_error_reference(memory['referenceLosses'])):
                    raise ValueError('CART retained an unqualified anchor or skipped a qualified guard.')
            if memory['guardEnabled']:
                if cfg.get('cartGuardGranularity')=='activation_boundary':
                    if (memory.get('guardMode')!='activation_boundary' or memory.get('riskMetric')!=cfg['cartRiskMetric']
                            or memory.get('inputRestorationEnabled') is not False or memory['stepGuards']
                            or memory['inputReferenceHash']!=r['inputHash']
                            or memory['committedOutputHash']!=r['trainedOutputHash']
                            or not within_envelope(memory['outputBridge']['classLosses'],memory['activeBounds'])):
                        raise ValueError('A CART commit changed selection or violated its declared class risk bound.')
                    if cfg.get('cartCommitGeometry'):
                        projection=memory['outputBridge']['projection']
                        if projection.get('status')=='not_needed':
                            if memory['proposalOutputHash']!=memory['committedOutputHash']:
                                raise ValueError('An unchanged proposal has a different committed model.')
                        elif (projection['selectedSquaredDistance']>projection['scalarSquaredDistance']+1e-8
                                or not 0<=projection['proposalFidelity']<=1+1e-8
                                or projection['proposalFidelity']+1e-6<projection['scalarScale']):
                            raise ValueError('Class-direction repair lost more of the proposal than its scalar comparator.')
                elif (len(memory['stepGuards'])!=r['optimizerSteps'] or any(
                        not within_envelope(s['classLosses'],memory['activeBounds'])
                        for s in memory['stepGuards'])):
                    raise ValueError('A protected step violates its actual class-loss bound.')
    return partition


def report(root,*,partial=False,plot=False,completed_only=False):
    import numpy as np
    from basil_core.factorial_study import aggregate_seed_results,validate_config as validate_factorial
    from scripts.run_cart_memory_smoke import save_json
    plan=json.loads((root/'plan.json').read_text())
    for name,digest in plan['source'].items():
        raw=(root/'source_snapshots'/digest).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError(f'Corrupt archived source: {name}')
    if completed_only and not partial:raise ValueError('A completed subset cannot be a full-study report.')
    rows=[];partitions={};initials={};runtimes=[];incomplete=[]
    for cfg in plan['configs']:
        if cfg.get('lossStudyRevision')=='cart_coherent_channel_pilot_v1':
            from basil_core.cart_coherent_study import validate_config
            validate_config(cfg)
        else:validate_factorial(cfg)
        folder=root/cfg['runId']
        if completed_only and not (folder/'run.json').exists():
            incomplete.append(dict(runId=cfg['runId'],status='PENDING'))
            continue
        metadata=json.loads((folder/'run.json').read_text())
        if (metadata['config']!=cfg or metadata['source']!=plan['source'] or metadata['data']!=plan['data']
                or metadata.get('officialTestUsed') is not False):
            raise ValueError('Result source, TRAIN data or condition differs from its frozen plan.')
        if metadata['status']!='COMPLETED':
            if completed_only:
                incomplete.append(dict(runId=cfg['runId'],status=metadata['status']))
                continue
            raise ValueError('Study is incomplete; no pooled result emitted.')
        with np.load(folder/'metrics.npz',allow_pickle=False) as saved:
            values=saved['roundNodeAccuracy'].copy()
            class_values=saved['perClass'].copy()
        if values.shape!=(cfg['nRounds'],10):raise ValueError('Incomplete round metrics.')
        if (class_values.shape!=(cfg['nRounds'],10,10) or not np.isfinite(class_values).all()
                or np.any(class_values<0) or np.any(class_values>1)
                or not np.allclose(class_values.mean(axis=2),values,rtol=0,atol=1e-6)):
            raise ValueError('Class metrics disagree with the balanced TRAIN holdout accuracy.')
        if not np.allclose(values.mean(axis=1),metadata['roundMean'],rtol=0,atol=1e-6):
            raise ValueError('Stored round means differ from node metrics.')
        partition=verify_checkpoint(folder,cfg,plan['source'],values,class_values=class_values)
        key=(cfg['partitionStrategy'],cfg['seed'])
        if key in partitions and partitions[key]!=partition:
            raise ValueError('Paired conditions use different partitions or initialization.')
        partitions[key]=partition
        if cfg['seed'] in initials and initials[cfg['seed']]!=partition['initialModelHash']:
            raise ValueError('Paired seeds use different initial models.')
        initials[cfg['seed']]=partition['initialModelHash']
        runtime=metadata['runtime']
        if (runtime['device']!='GPU' or runtime['tensorFloat32'] is not False or
                runtime['deterministicOps']!='1'):
            raise ValueError('Unverified arithmetic policy or CPU fallback.')
        signature={k:runtime[k] for k in ('tensorflow','numpy','software','gpuMemoryLimitMiB','intraOpThreads','interOpThreads','oneDnn')}
        signature['gpuModel']=runtime['gpu']['name']
        runtimes.append(signature)
        rows.append(dict(config=cfg,status=metadata['status'],roundNodeAccuracy=values))
    if any(r!=runtimes[0] for r in runtimes[1:]):raise ValueError('Hardware/software policies differ across the study.')
    rounds={c['nRounds'] for c in plan['configs']}
    if len(rounds)!=1:raise ValueError('Cannot mix different training horizons.')
    horizon=rounds.pop()
    if partial:
        result=dict(status='completed_subset_verified' if incomplete else 'selected_runs_verified',
            completeStudy=False,completeSelectedPlan=not incomplete,pendingOrIncomplete=incomplete,officialTestUsed=False,
            runs=[dict(config=r['config'],curve=r['roundNodeAccuracy'].mean(axis=1).tolist(),
                finalMean=float(r['roundNodeAccuracy'][-1].mean()),
                iidCleanTargetMet=bool(r['roundNodeAccuracy'][-1].mean()>=.9)
                if r['config']['conditionId']=='clean' and r['config']['partitionStrategy']=='iid' else None) for r in rows])
        save_json(root/'reports/selected_runs.json',result)
        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            figure,ax=plt.subplots(figsize=(9,5))
            for row in result['runs']:
                cfg=row['config'];ax.plot(range(1,horizon+1),row['curve'],label=f'{cfg["partitionStrategy"]} S{cfg["seed"]} {cfg["conditionId"]} sigma={cfg["channelNoiseSigmaAbsolute"]:g}')
            ax.axhline(.9,color='black',linestyle=':',label='IID clean target (90%)')
            ax.set(xlabel='Completed round',ylabel='Mean node TRAIN-holdout accuracy',ylim=(0,1))
            ax.legend(fontsize=8);figure.tight_layout();figure.savefig(root/'reports/selected_curves.png',dpi=150);plt.close(figure)
        return result
    summaries=aggregate_seed_results(rows,plan['sigmas'],endpoint_start=max(0,horizon-10),expected_rounds=horizon)
    grouped={(r['partition'],r['condition'],r['sigma']):r for r in summaries}
    paired=[]
    for partition in ('iid','dirichlet'):
        for sigma in plan['sigmas']:
            control=np.asarray(grouped[partition,'attack_noise_basil_ebm',sigma]['seedEndpoints'])
            cart=np.asarray(grouped[partition,'attack_noise_basil_ebm_cart',sigma]['seedEndpoints'])
            clean=np.asarray(grouped[partition,'clean',0.]['seedEndpoints'])
            difference=cart-control;gap=cart-clean
            paired.append(dict(partition=partition,sigma=sigma,seeds=[2025,2026,2027],
                cartMinusBothDefensesPerSeed=difference.tolist(),cartDeltaMean=float(difference.mean()),
                cartDeltaSampleStd=float(difference.std(ddof=1)),positiveSeeds=int((difference>0).sum()),
                fullStackMinusCleanPerSeed=gap.tolist(),fullStackCleanGapMean=float(gap.mean()),
                fullStackCleanGapSampleStd=float(gap.std(ddof=1))))
    result=dict(status='all_expected_seeds_verified',officialTestUsed=False,
        aggregation='node mean per round; fixed last-ten-round mean per seed; mean and sample SD across three matching seeds',
        groups=summaries,pairedCartEffects=paired)
    save_json(root/'reports/seed_summary.json',result)
    for row in summaries:
        print(f'{row["partition"]} {row["condition"]} sigma={row["sigma"]:g}: '
            f'{row["mean"]:.2%} +/- {row["sampleStd"]*100:.2f} percentage points (seed SD)')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--partial',action='store_true',help='Verify only selected development runs; emit no three-seed/full-study summary.')
    p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    result=report(args.root,partial=args.partial,plot=args.plot)
    if args.partial:
        print(json.dumps(dict(status=result['status'],completeStudy=False,
            runs=[dict(runId=r['config']['runId'],finalMean=r['finalMean']) for r in result['runs']]),indent=2))
    else:print(json.dumps(dict(status=result['status'],groups=len(result['groups']),pairedCartEffects=result['pairedCartEffects']),indent=2))


if __name__=='__main__':main()
