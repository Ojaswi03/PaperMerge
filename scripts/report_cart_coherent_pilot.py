#!/usr/bin/env python3
"""Read-only paired development evidence; no missing-seed pooling or test use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def report(root,*,plot=False,completed_only=False):
    import numpy as np
    from scripts.report_factorial_study import report as verify_selected
    from scripts.run_cart_memory_smoke import save_json
    from basil_core.protocol_checkpoint import _load_telemetry
    result=verify_selected(root,partial=True,plot=plot,completed_only=completed_only)
    rows=[]
    for run in result['runs']:
        cfg=run['config'];folder=root/cfg['runId']
        metadata=json.loads((folder/'run.json').read_text())
        row=dict(partition=cfg['partitionStrategy'],seed=cfg['seed'],sigma=cfg['channelNoiseSigmaAbsolute'],
            case=cfg['coherentCase'],objective=cfg.get('coherentTrainingObjective','raw_ce'),
            variant=cfg['cartVariant'],runId=cfg['runId'],
            curve=run['curve'],finalMean=run['finalMean'],elapsedSeconds=metadata['elapsedSeconds'])
        if (folder/'metrics.npz').is_file():
            with np.load(folder/'metrics.npz',allow_pickle=False) as saved:
                per_class=saved['perClass'][-1].mean(axis=0)
            row.update(finalClassMeanAccuracy=per_class.tolist(),finalWorstClassMeanAccuracy=float(per_class.min()))
        if cfg['cartVariant'] in ('coherent','coherent_refresh','coherent_commit_ce','coherent_commit_error','coherent_qualified_error','coherent_projected_error','coherent_head_transport'):
            with np.load(folder/'checkpoint.npz',allow_pickle=False) as saved:cp=json.loads(str(saved['metadata']))
            records=_load_telemetry(folder/'checkpoint.npz',cp)
            memories=[r['cartMemory'] for r in records]
            guarded=[m for m in memories if m['guardEnabled']]
            steps=[s for m in guarded for s in m['stepGuards']]
            row.update(protectedSteps=len(steps),
                meanAcceptedStepScale=float(np.mean([s['scale'] for s in steps])) if steps else None,
                zeroStepFraction=float(np.mean([s['scale']==0 for s in steps])) if steps else None,
                inputRepairFraction=float(np.mean([m['inputBridge']['scale']<1 for m in guarded])) if guarded else None,
                referenceAdmissions=sum(m['referenceAdmitted'] for m in memories),
                measuredProtectedViolations=0)
            if cfg.get('cartGuardGranularity')=='classifier_transport':
                evidence=[m['transport'] for m in memories]
                row.pop('measuredProtectedViolations',None)
                row.update(transportActivations=sum(e['scale']>0 for e in evidence),
                    meanTransportScale=float(np.mean([e['scale'] for e in evidence])),
                    referenceUses=sum(e['referenceUsed'] for e in evidence),backboneChanges=0,
                    meanLocalSelectionCEChange=float(np.mean([e['checkAfter']-e['checkBefore']
                        for e in evidence if e['checkBefore'] is not None])) if any(e['checkBefore'] is not None for e in evidence) else None)
            if cfg.get('cartGuardGranularity')=='activation_boundary':
                row.update(riskMetric=cfg['cartRiskMetric'],protectedActivations=len(guarded),
                    inputRepairFraction=None,
                    meanAcceptedOutputScale=float(np.mean([m['outputBridge']['scale'] for m in guarded])) if guarded else None,
                    zeroOutputFraction=float(np.mean([m['outputBridge']['scale']==0 for m in guarded])) if guarded else None)
            if cfg.get('cartReferenceQualification'):
                row.update(qualifiedReferenceVisits=sum(m['referenceQualified'] for m in memories),
                    firstProtectedActivation=next((i for i,m in enumerate(memories) if m['guardEnabled']),None))
            if cfg.get('cartCommitGeometry'):
                evidence=[m['outputBridge']['projection'] for m in guarded]
                row.pop('meanAcceptedOutputScale',None)
                row.update(projectedActivations=sum(p['family']=='projected' for p in evidence),
                    meanProposalFidelity=float(np.mean([p['proposalFidelity'] for p in evidence])) if evidence else None,
                    scalarCounterfactualMeanScale=float(np.mean([p.get('scalarScale',1.) for p in evidence])) if evidence else None,
                    projectionSolverFallbacks=sum(p.get('solver',{}).get('status')=='fallback_zero' for p in evidence))
        rows.append(row)
    lookup={(r['partition'],r['seed'],r['sigma'],r['case'],r['objective'],r['variant']):r for r in rows}
    pairs=[];unpaired=[]
    for row in rows:
        if row['variant']=='none':continue
        key=(row['partition'],row['seed'],row['sigma'],row['case'],row['objective'],'none')
        if key not in lookup:
            if completed_only:
                unpaired.append(row['runId']);continue
            raise ValueError('Matched no-CART control is missing.')
        control=lookup[key]
        clean=lookup.get((row['partition'],row['seed'],0.,'clean',row['objective'],'none'))
        pairs.append(dict(partition=row['partition'],seed=row['seed'],sigma=row['sigma'],case=row['case'],objective=row['objective'],
            variant=row['variant'],finalDeltaPercentagePoints=100*(row['finalMean']-control['finalMean']),
            meanTrajectoryDeltaPercentagePoints=100*float(np.mean(np.asarray(row['curve'])-control['curve'])),
            finalGapToCleanPercentagePoints=100*(row['finalMean']-clean['finalMean']) if clean else None,
            runtimeRatio=row['elapsedSeconds']/control['elapsedSeconds']))
    groups=[]
    keys=sorted({(p['partition'],p['sigma'],p['case'],p['objective'],p['variant']) for p in pairs})
    for key in keys:
        matching=[p for p in pairs if (p['partition'],p['sigma'],p['case'],p['objective'],p['variant'])==key]
        if len({p['seed'] for p in matching})!=len(matching):raise ValueError('Duplicate paired seed evidence.')
        if {p['seed'] for p in matching}=={2025,2026,2027}:
            differences=np.asarray([p['finalDeltaPercentagePoints'] for p in matching])
            selected=[r for r in rows if (r['partition'],r['sigma'],r['case'],r['objective'],r['variant'])==key]
            finals=np.asarray([r['finalMean'] for r in selected])
            control_finals=np.asarray([lookup[r['partition'],r['seed'],r['sigma'],r['case'],r['objective'],'none']['finalMean']
                for r in selected])
            groups.append(dict(partition=key[0],sigma=key[1],case=key[2],objective=key[3],variant=key[4],
                seeds=[p['seed'] for p in matching],pairedFinalDeltaMeanPercentagePoints=float(differences.mean()),
                pairedFinalDeltaSampleStdPercentagePoints=float(differences.std(ddof=1)),
                variantFinalMean=float(finals.mean()),variantFinalSampleStd=float(finals.std(ddof=1)),
                controlFinalMean=float(control_finals.mean()),controlFinalSampleStd=float(control_finals.std(ddof=1)),
                positiveSeeds=int((differences>0).sum())))
            gaps=[p['finalGapToCleanPercentagePoints'] for p in matching]
            if all(gap is not None for gap in gaps):
                groups[-1].update(finalGapToCleanMeanPercentagePoints=float(np.mean(gaps)),
                    finalGapToCleanSampleStdPercentagePoints=float(np.std(gaps,ddof=1)))
    status='completed_paired_subset_verified' if result.get('pendingOrIncomplete') or unpaired else 'selected_paired_pilots_verified'
    output=dict(status=status,rows=rows,pairedEffects=pairs,threeSeedGroups=groups,
        completeSelectedPlan=result.get('completeSelectedPlan',True),
        pendingOrIncomplete=result.get('pendingOrIncomplete',[]),unpairedCompletedRuns=unpaired,
        officialTestUsed=False,noveltyEstablished=False,fullStudyCompleted=False)
    save_json(root/'reports/coherent_pilot_summary.json',output)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--plot',action='store_true')
    p.add_argument('--completed-only',action='store_true',help='Verify completed development results and explicitly list unfinished jobs.')
    args=p.parse_args()
    result=report(args.root,plot=args.plot,completed_only=args.completed_only)
    print(json.dumps(dict(status=result['status'],completeSelectedPlan=result['completeSelectedPlan'],
        verifiedRuns=len(result['rows']),unfinishedRuns=len(result['pendingOrIncomplete']),unpairedRuns=len(result['unpairedCompletedRuns']),
        rows=[{k:v for k,v in r.items() if k not in ('curve','finalClassMeanAccuracy')} for r in result['rows']],
        pairedEffects=result['pairedEffects'],threeSeedGroups=result['threeSeedGroups']),indent=2))


if __name__=='__main__':main()
