#!/usr/bin/env python3
"""Report independently verified, matched guarded CART pilots across folders.

Keeps individual seed/noise pairs separate; does not treat nodes as seed repeats.
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
    from basil_core.sharp_guarded_protocol import check_source,validate_config
    rows=[];source=None;identities=set();pairs=[]
    for campaign in map(Path,campaigns):
        plan=json.loads((campaign/'plan.json').read_text());check_source(plan['source'])
        if source is None:source=plan['source']
        if source!=plan['source']:raise ValueError('Do not combine training source revisions.')
        checked=json.loads((campaign/'independent_validation.json').read_text())
        if checked['status']!='ALL_SELECTED_RUNS_VERIFIED' or checked['verified']!=len(plan['configs']):
            raise ValueError('All selected runs must be independently verified.')
        programs=plan.get('executionProgram',{})
        for name,value in programs.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:
                raise ValueError('Execution program differs from the frozen plan.')
        for cfg in plan['configs']:
            validate_config(cfg)
            if cfg['condition'] not in ('attack_noise_defense_rla','attack_noise_defense_rla_cart'):
                raise ValueError('This pilot compares the joint stack with and without CART.')
            if cfg['runId'] in identities:raise ValueError('Repeated run must not be counted twice.')
            identities.add(cfg['runId']);folder=campaign/cfg['runId']
            run=json.loads((folder/'run.json').read_text())
            if run['status']!='COMPLETED' or run['config']!=cfg or run['source']!=source:
                raise ValueError('Incomplete or mismatched result.')
            if cfg['rounds']<10 or cfg['attackStart']>=cfg['rounds']:
                raise ValueError('Pilot requires ten endpoint rounds and an active attack.')
            with np.load(folder/'metrics.npz',allow_pickle=False) as archive:
                accuracy=archive['roundNodeAccuracy']
            if accuracy.shape!=(cfg['rounds'],cfg['nodes']) or not np.isfinite(accuracy).all():
                raise ValueError('Invalid accuracy dimensions.')
            telemetry=json.loads((folder/'activation_telemetry.json').read_text())
            validation=next(v for v in checked['rows'] if v['runId']==cfg['runId'])
            rows.append(dict(campaign=campaign.name,runId=cfg['runId'],config=cfg,
                initialHash=run['initialHash'],partitionHash=run['partition']['partitionHash'],
                dataIdentity=run['data'],runtime=run['runtime'],executionPrograms=programs,
                continuations=run.get('continuations',[]),independentValidation=validation,
                primary=float(accuracy[-10:].mean()),final=float(accuracy[-1].mean()),
                attackActive=float(accuracy[cfg['attackStart']:].mean()),
                curve=accuracy.mean(axis=1).tolist(),telemetry=telemetry))
    key=lambda r:(r['config']['dataset'],r['config']['partition'],r['config']['seed'],r['config']['sigma'])
    for identity in sorted({key(r) for r in rows}):
        selected=[r for r in rows if key(r)==identity]
        controls=[r for r in selected if not r['config']['cart']]
        carts=[r for r in selected if r['config']['cart']]
        if len(controls)!=1 or len(carts)!=1:raise ValueError('Each pilot requires exactly one matched control and CART arm.')
        control,cart=controls[0],carts[0]
        strip=lambda r:{k:v for k,v in r['config'].items() if k not in ('runId','baseRunId','condition','cart','switches')}
        switches=lambda r:{k:v for k,v in r['config']['switches'].items() if k not in ('name','cart')}
        if (strip(control)!=strip(cart) or switches(control)!=switches(cart)
                or any(control[k]!=cart[k] for k in ('initialHash','partitionHash','dataIdentity','runtime'))):
            raise ValueError('Control and CART are not paired beyond their declared CART switch.')
        n=control['config']['nodes']
        for a,b in zip(control['telemetry'][:n],cart['telemetry'][:n]):
            if (a['inputHash'],a['trainedHash'])!=(b['inputHash'],b['trainedHash']):
                raise ValueError('Arms differ before any receiver has historical CART memory.')
        if len(control['telemetry'])!=n*control['config']['rounds'] or len(cart['telemetry'])!=len(control['telemetry']):
            raise ValueError('Incomplete paired telemetry.')
        pairs.append(dict(dataset=identity[0],partition=identity[1],seed=identity[2],sigma=identity[3],
            controlRunId=control['runId'],cartRunId=cart['runId'],firstRingHashesEqual=True,
            effects={endpoint:cart[endpoint]-control[endpoint] for endpoint in ('primary','final','attackActive')},
            nonzeroCartCorrections=sum(bool(t.get('cart')) and t['cart']['scale']>0 for t in cart['telemetry'])))
    for row in rows:row.pop('telemetry')
    return dict(status='VERIFIED_PAIRED_GUARDED_CART_PILOT',sourceSha256=source,rows=rows,pairs=pairs,
        aggregationProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),officialTestUsed=False,
        endpoints=dict(primary='mean across nodes and the final ten rounds',final='final round mean across nodes',
            attackActive='mean across nodes and zero-based round attackStart through the final round'),
        interpretation='Individual matched development pairs. A one-seed pilot is not an across-seed estimate; all positive and negative effects are retained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaigns',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from scripts.run_sharp_study import write_json
    result=summarize(args.campaigns);write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],pairs=result['pairs']),indent=2))
