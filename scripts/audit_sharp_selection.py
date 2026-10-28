#!/usr/bin/env python3
"""Read-only audit of SHARP screening, actual attacked copies and ancestry.

Reads the last atomic checkpoint, including during training. Ground-truth
attacker labels are used only after the fact; no training decision changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def analyze(cfg,telemetry,metrics):
    latest={};rounds={};accuracy={m['round']:m['meanAccuracy'] for m in metrics}
    for index,record in enumerate(telemetry):
        r,node=divmod(index,cfg['nodes'])
        if (record['round'],record['node'],record['activation'])!=(r,node,index+1):
            raise ValueError('Telemetry is not a complete ordered activation history.')
        selection=record['selection'];selected=selection['selectedSenders']
        available=[(node-d)%cfg['nodes'] for d in range(1,cfg['window']+1)
            if (node-d)%cfg['nodes'] in latest]
        if (len(set(selected))!=len(selected) or not set(selected).issubset(available)
                or selection['availableCopies']!=len(available)):
            raise ValueError('Selected copies disagree with the historical sender window.')
        row=rounds.setdefault(r,dict(roundIndex=r,displayRound=r+1,meanAccuracy=accuracy.get(r),
            activations=0,selectedCopies=0,selectedDirectlyAttackedCopies=0,
            activationsSelectingDirectAttack=0,inputsWithAttackAncestry=0,
            screenedActivations=0,allFailFallbacks=0,selectedCounts=[],
            screenedUnattackedAvailable=0,screenedUnattackedRawPass=0,screenedUnattackedRetained=0,
            screenedAttackedAvailable=0,screenedAttackedRawPass=0,screenedAttackedRetained=0,
            selectedDirectAttackLosses=[],bestUnexposedLosses=[],firstDirectAdmission=None))
        direct=sum(latest[s]['direct'] for s in selected)
        exposed=any(latest[s]['exposed'] for s in selected)
        row['activations']+=1;row['selectedCopies']+=len(selected)
        row['selectedDirectlyAttackedCopies']+=direct
        row['activationsSelectingDirectAttack']+=bool(direct)
        row['inputsWithAttackAncestry']+=exposed
        row['selectedCounts'].append(selection['selectedCount'])
        retained=set(available)
        if selection.get('screenApplied'):
            ratios=selection['peakRatios'];threshold=selection['peakThreshold']
            if len(ratios)!=len(available) or not math.isfinite(threshold):
                raise ValueError('Peak evidence does not match the available copies.')
            passed={s for s,value in zip(available,ratios) if math.isfinite(value) and value<=threshold}
            fallback=selection['screenAllFailFallback']
            if fallback!= (not passed):raise ValueError('Peak fallback evidence is inconsistent.')
            retained=set(available) if fallback else passed
            if not set(selected).issubset(retained):raise ValueError('A rejected copy was selected.')
            row['screenedActivations']+=1;row['allFailFallbacks']+=fallback
            for s in available:
                prefix='screenedAttacked' if latest[s]['direct'] else 'screenedUnattacked'
                row[prefix+'Available']+=1;row[prefix+'RawPass']+=s in passed
                row[prefix+'Retained']+=s in retained
        if direct and row['firstDirectAdmission'] is None:
            row['firstDirectAdmission']=dict(activation=index+1,node=node,roundIndex=r,
                availableUnattacked=sum(not latest[s]['direct'] for s in available),
                availableAttacked=sum(latest[s]['direct'] for s in available),
                retainedUnattacked=sum(not latest[s]['direct'] for s in retained),
                retainedAttacked=sum(latest[s]['direct'] for s in retained),
                selectedDirectlyAttacked=direct,selectedCopies=len(selected),
                targetCount=selection.get('targetCount'))
        unexposed=[]
        for candidate in selection.get('ranking',[]):
            s=candidate['sender'];loss=candidate['loss']
            if s not in latest:raise ValueError('Ranking contains an unavailable sender.')
            if loss is None or not math.isfinite(loss):continue
            if s in selected and latest[s]['direct']:row['selectedDirectAttackLosses'].append(loss)
            if not latest[s]['exposed']:unexposed.append(loss)
        if unexposed:row['bestUnexposedLosses'].append(min(unexposed))
        latest[node]=dict(direct=bool(record['attackActive']),
            exposed=bool(record['attackActive'] or exposed))
    for row in rounds.values():
        ratio=lambda a,b:row[a]/row[b] if row[b] else None
        row['selectedDirectAttackFraction']=ratio('selectedDirectlyAttackedCopies','selectedCopies')
        row['attackAncestryInputFraction']=ratio('inputsWithAttackAncestry','activations')
        for group in ('Unattacked','Attacked'):
            row[f'{group.lower()}RawPassFraction']=ratio('screened'+group+'RawPass','screened'+group+'Available')
            row[f'{group.lower()}RetainedFraction']=ratio('screened'+group+'Retained','screened'+group+'Available')
        row['meanSelectedCount']=statistics.mean(row.pop('selectedCounts'))
        for key in ('selectedDirectAttackLosses','bestUnexposedLosses'):
            values=row.pop(key);row[key+'Mean']=statistics.mean(values) if values else None
    return list(rounds.values())


def audit(output,*,dataset='mnist',partition='iid',seed=2025,condition='attack_defense',sigma=None):
    # Only NumPy is needed for the uint8 metadata member; no model arrays are read.
    import numpy as np
    output=Path(output);plan=json.loads((output/'plan.json').read_text())
    configs=[c for c in plan['configs'] if c['dataset']==dataset and c['partition']==partition
        and c['seed']==seed and c['condition']==condition and (sigma is None or c['sigma']==sigma)]
    if not configs:raise ValueError('No matching configuration in this frozen plan.')
    reports=[]
    for cfg in configs:
        folder=output/cfg['runId'];path=folder/'checkpoint.npz'
        run=json.loads((folder/'run.json').read_text()) if (folder/'run.json').exists() else {'status':'PENDING'}
        if not path.exists():
            reports.append(dict(config=cfg,status=run['status'],checkpointAvailable=False));continue
        with np.load(path,allow_pickle=False) as archive:metadata=json.loads(archive['metadata'].tobytes())
        contract=metadata['contract']
        if contract['config']!=cfg or contract['source']!=plan['source']:
            raise ValueError('Checkpoint differs from its frozen plan.')
        if metadata['cursor']!=len(metadata['telemetry']):raise ValueError('Checkpoint telemetry is incomplete.')
        reports.append(dict(config=cfg,status=run['status'],checkpointAvailable=True,
            checkpointActivations=metadata['cursor'],liveReportedActivations=run.get('completedActivations'),
            sourceSha256=plan['source'],dataIdentity=contract['data'],
            partitionHash=contract['partition']['partitionHash'],
            auditProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            rounds=analyze(cfg,metadata['telemetry'],metadata['roundMetrics']),
            officialTestUsed=False,interpretation='Read-only post-hoc transmission provenance. Attack ancestry is exposure, not proof of a lasting harmful effect. Unattacked copies can have attacked ancestors.'))
    return reports


def text_report(reports,*,all_rounds=False):
    percent=lambda x:f'{100*x:.2f}' if x is not None else '--'
    lines=[]
    for report in reports:
        cfg=report['config'];lines.append(f'{cfg["dataset"]}/{cfg["partition"]} seed{cfg["seed"]} '
            f'{cfg["condition"]} sigma{cfg["sigma"]:g} | {report["status"]}')
        if not report['checkpointAvailable']:
            lines.append('No checkpoint committed yet.');continue
        lines.append(f'Checkpoint A{report["checkpointActivations"]}; live counter A{report["liveReportedActivations"]}.')
        first=next((r['firstDirectAdmission'] for r in report['rounds'] if r['firstDirectAdmission']),None)
        if first:
            lines.append(f'First direct attack admission: A{first["activation"]}, node{first["node"]}, '
                f'round{first["roundIndex"]+1}; before screen {first["availableUnattacked"]} unattacked/'
                f'{first["availableAttacked"]} attacked; retained {first["retainedUnattacked"]} unattacked/'
                f'{first["retainedAttacked"]} attacked.')
        lines.append('Round Acc% SelectedAttack% InputAncestry% RetainUnattacked% RetainAttacked% AllFail MeanQ')
        rows=report['rounds'];last=rows[-1]['roundIndex'] if rows else -1
        for row in rows:
            r=row['roundIndex']
            if not all_rounds and not (cfg['attackStart']-2<=r<=cfg['attackStart']+6 or r>=last-2):continue
            lines.append(f'{row["displayRound"]:5} {percent(row["meanAccuracy"]):>6} '
                f'{percent(row["selectedDirectAttackFraction"]):>15} {percent(row["attackAncestryInputFraction"]):>14} '
                f'{percent(row["unattackedRetainedFraction"]):>17} {percent(row["attackedRetainedFraction"]):>15} '
                f'{row["allFailFallbacks"]:7} {row["meanSelectedCount"]:5.1f}')
        lines.append('Retention rates use full-window screens and include the recorded all-fail fallback.')
        lines.append('Attack labels are post-hoc audit information, never inputs to training decisions.')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',default='mnist');parser.add_argument('--partition',default='iid')
    parser.add_argument('--seed',type=int,default=2025);parser.add_argument('--condition',default='attack_defense')
    parser.add_argument('--sigma',type=float);parser.add_argument('--json',action='store_true')
    parser.add_argument('--all-rounds',action='store_true')
    args=parser.parse_args()
    try:reports=audit(args.output,dataset=args.dataset,partition=args.partition,seed=args.seed,
        condition=args.condition,sigma=args.sigma)
    except (OSError,ValueError,KeyError) as error:parser.exit(2,f'Audit could not read checkpoint: {error}\n')
    print(json.dumps(reports,indent=2,sort_keys=True) if args.json else text_report(reports,all_rounds=args.all_rounds))


if __name__=='__main__':main()
