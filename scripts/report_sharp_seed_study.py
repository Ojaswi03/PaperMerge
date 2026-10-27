#!/usr/bin/env python3
"""Combine verified CPU validation campaigns, never mixing noise levels.

Reports configured last-ten-round, final and attack-active endpoints together.
Requires all three seeds and all selected cases before reporting any group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def summarize(campaigns):
    import numpy as np
    from scripts.run_sharp_study import check_source
    from basil_core.sharp_study import validate_config
    expected=(2025,2026,2027);rows=[];source=None;programs={};identities=set()
    for campaign in campaigns:
        campaign=Path(campaign)
        plan=json.loads((campaign/'plan.json').read_text());check_source(plan['source'])
        if source is None:source=plan['source']
        if source!=plan['source']:raise ValueError('Do not combine source revisions.')
        checked=json.loads((campaign/'independent_validation.json').read_text())
        if checked['status']!='ALL_SELECTED_RUNS_VERIFIED' or checked['verified']!=len(plan['configs']):
            raise ValueError('Every selected run must be independently verified.')
        programs.update(plan.get('executionProgram',{}))
        for cfg in plan['configs']:
            validate_config(cfg)
            key=(cfg['dataset'],cfg['partition'],cfg['method'],cfg['nodes'],cfg['rounds'],cfg['model'],
                 cfg['dirichletAlpha'],cfg['attackStart'],cfg['sigma'],cfg['condition'],cfg['seed'])
            if key in identities:raise ValueError('Duplicate case/seed/noise result; do not count reruns as independent seeds.')
            identities.add(key)
            if cfg['trainPerClass'] is not None or cfg['validationPerClass'] is not None:
                raise ValueError('This replication report requires full fitting/reporting pools.')
            folder=campaign/cfg['runId'];run=json.loads((folder/'run.json').read_text())
            if run['status']!='COMPLETED' or run['config']!=cfg:raise ValueError('Incomplete or mismatched result.')
            with np.load(folder/'metrics.npz',allow_pickle=False) as f:
                accuracy=f['roundNodeAccuracy'];per_class=f['roundNodePerClassAccuracy']
            if accuracy.shape!=(cfg['rounds'],cfg['nodes']) or not np.isfinite(accuracy).all():
                raise ValueError('Invalid replication dimensions.')
            if cfg['attackStart']>=cfg['rounds']:raise ValueError('No attack-active rounds available.')
            validation=next(v for v in checked['rows'] if v['runId']==cfg['runId'])
            rows.append(dict(dataset=cfg['dataset'],partition=cfg['partition'],method=cfg['method'],
                nodes=cfg['nodes'],rounds=cfg['rounds'],modelTag=cfg['model'],alpha=cfg['dirichletAlpha'],
                attackStart=cfg['attackStart'],sigma=cfg['sigma'],condition=cfg['condition'],seed=cfg['seed'],
                runId=cfg['runId'],primary=float(accuracy[-10:].mean()),final=float(accuracy[-1].mean()),
                attackActive=float(accuracy[cfg['attackStart']:].mean()),
                finalWorstNode=float(accuracy[-1].min()),finalWorstClassMean=float(per_class[-1].mean(axis=0).min()),
                curve=accuracy.mean(axis=1).tolist(),independentValidation=validation,
                initialHash=run['initialHash'],partitionHash=run['partition']['partitionHash'],
                dataIdentity=run['data'],runtime=run['runtime'],config=cfg))
    grouping=lambda r:(r['dataset'],r['partition'],r['method'],r['nodes'],r['rounds'],r['modelTag'],r['alpha'],r['attackStart'],r['sigma'],r['condition'])
    keys=sorted({grouping(r) for r in rows});groups=[];paired=[]
    for key in keys:
        selected=sorted([r for r in rows if grouping(r)==key],key=lambda r:r['seed'])
        if tuple(r['seed'] for r in selected)!=expected:raise ValueError('Every reported condition/noise requires all three seeds.')
        settings={json.dumps({k:v for k,v in r['config'].items() if k not in ('seed','runId')},sort_keys=True) for r in selected}
        if len(settings)!=1:raise ValueError('Seed configurations do not match beyond seed/run ID.')
        if any(r['runtime'].get('device')!='/CPU:0' for r in selected):
            raise ValueError('This report is explicitly CPU validation; do not mix execution devices.')
        stats={}
        for endpoint in ('primary','final','attackActive'):
            values=np.asarray([r[endpoint] for r in selected])
            stats[endpoint]=dict(mean=float(values.mean()),sampleStd=float(values.std(ddof=1)),seedValues=values.tolist())
        groups.append(dict(dataset=key[0],partition=key[1],method=key[2],nodes=key[3],rounds=key[4],
            sigma=key[8],condition=key[9],seeds=list(expected),statistics=stats))
        if key[-1].endswith('_cart'):
            reference_key=(*key[:-1],key[-1][:-5])
            baseline=sorted([r for r in rows if grouping(r)==reference_key],key=lambda r:r['seed'])
            if tuple(r['seed'] for r in baseline)!=expected:raise ValueError('Missing paired non-CART seed.')
            for a,b in zip(selected,baseline):
                strip=lambda r:{k:v for k,v in r['config'].items() if k not in ('seed','runId','condition','cart','switches')}
                switches=lambda r:{k:v for k,v in r['config']['switches'].items() if k not in ('name','cart')}
                if (strip(a)!=strip(b) or switches(a)!=switches(b) or a['initialHash']!=b['initialHash']
                        or a['partitionHash']!=b['partitionHash'] or a['dataIdentity']!=b['dataIdentity']
                        or a['runtime']!=b['runtime']):
                    raise ValueError('CART/non-CART configurations or identities are not paired.')
            effects={}
            for endpoint in ('primary','final','attackActive'):
                delta=np.asarray([a[endpoint]-b[endpoint] for a,b in zip(selected,baseline)])
                effects[endpoint]=dict(mean=float(delta.mean()),sampleStd=float(delta.std(ddof=1)),
                    seedDeltas=delta.tolist(),positiveSeeds=int((delta>0).sum()))
            paired.append(dict(dataset=key[0],partition=key[1],nodes=key[3],rounds=key[4],sigma=key[8],
                seeds=list(expected),effects=effects))
    return dict(status='COMPLETE_VERIFIED_SELECTED_SEED_STUDY',sourceSha256=source,executionPrograms=programs,
        aggregationProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,groups=groups,pairedCartEffects=paired,officialTestUsed=False,
        interpretation='Descriptive three-seed development evidence, explicit CPU execution; no significance, novelty certification or general robustness claim.',
        aggregation='Within each seed average nodes and the fixed endpoint rounds, then average identical dataset/split/method/condition/sigma across seeds. Never average sigmas together.',
        endpoints=dict(primary='last ten rounds',attackActive='round index attackStart through the final round',final='final round'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaigns',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.campaigns)
    from scripts.run_sharp_study import write_json
    write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],runs=len(result['rows']),groups=len(result['groups']),
        pairedCartEffects=result['pairedCartEffects']),indent=2))
