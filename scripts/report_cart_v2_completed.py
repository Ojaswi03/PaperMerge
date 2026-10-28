#!/usr/bin/env python3
"""Read completed CART v2 pilot groups without waiting for unrelated jobs.

Never writes campaign files. Means and paired effects require all three seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def summarize(output,*,datasets=None,conditions=None,sigma=None):
    import numpy as np
    from basil_core.sharp_cart_v2_protocol import check_source,validate_config
    output=Path(output);plan=json.loads((output/'plan.json').read_text());check_source(plan['source'])
    expected=(2025,2026,2027);buckets={};identities=set();completed={}
    for cfg in plan['configs']:
        validate_config(cfg)
        if datasets and cfg['dataset'] not in datasets:continue
        if conditions and cfg['condition'] not in conditions:continue
        if sigma is not None and cfg['sigma'] not in (0.,sigma):continue
        basis={k:v for k,v in cfg.items() if k not in ('seed','runId','baseRunId','guardedRunId')}
        key=json.dumps(basis,sort_keys=True)
        identity=(key,cfg['seed'])
        if identity in identities:raise ValueError('Duplicate seed configuration.')
        identities.add(identity);group=buckets.setdefault(key,dict(config=basis,seeds={}))
        path=output/cfg['runId']/'run.json'
        run=json.loads(path.read_text()) if path.exists() else {'status':'PENDING'}
        row=dict(seed=cfg['seed'],runId=cfg['runId'],status=run['status'])
        group['seeds'][cfg['seed']]=row
        if run['status']!='COMPLETED':continue
        if (run['config']!=cfg or run['source']!=plan['source'] or run['data']!=plan['data'][cfg['dataset']]
                or run['partition']!=plan['partitions'][cfg['runId']]
                or run['completedActivations']!=cfg['nodes']*cfg['rounds']):
            raise ValueError('Completed run differs from its frozen plan or horizon.')
        with np.load(path.parent/'metrics.npz',allow_pickle=False) as arrays:a=arrays['roundNodeAccuracy']
        if (a.shape!=(cfg['rounds'],cfg['nodes']) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1)):
            raise ValueError('Invalid completed metric array.')
        row.update(primary=float(a[-min(10,cfg['rounds']):].mean()),final=float(a[-1].mean()))
        completed[cfg['runId']]=run
    if not buckets:raise ValueError('No configurations match the requested subset.')
    groups=[];pending=[]
    for key,g in sorted(buckets.items()):
        rows=[g['seeds'].get(s,dict(seed=s,status='ABSENT_FROM_PLAN')) for s in expected]
        if set(g['seeds'])!=set(expected) or any(r['status']!='COMPLETED' for r in rows):
            pending.append(dict(config=g['config'],seeds=rows));continue
        stats={name:dict(mean=float(np.mean([r[name] for r in rows])),
            sampleStd=float(np.std([r[name] for r in rows],ddof=1))) for name in ('primary','final')}
        groups.append(dict(config=g['config'],seeds=rows,statistics=stats))
    paired=[];unpaired=[]
    for cart in groups:
        if not cart['config']['cart']:continue
        baseline=dict(cart['config']);baseline['cart']=False
        baseline['condition']=baseline['condition'][:-5]
        candidate=[g for g in groups if g['config']['dataset']==baseline['dataset']
            and g['config']['partition']==baseline['partition'] and g['config']['sigma']==baseline['sigma']
            and g['config']['condition']==baseline['condition']]
        if len(candidate)!=1:
            unpaired.append(dict(dataset=baseline['dataset'],partition=baseline['partition'],sigma=baseline['sigma']));continue
        control=candidate[0]
        strip=lambda c:{k:v for k,v in c.items() if k not in ('condition','cart','switches')}
        switches=lambda c:{k:v for k,v in c['switches'].items() if k not in ('name','cart')}
        if strip(cart['config'])!=strip(control['config']) or switches(cart['config'])!=switches(control['config']):
            raise ValueError('Control and CART settings differ beyond CART.')
        for c,b in zip(cart['seeds'],control['seeds']):
            left,right=completed[c['runId']],completed[b['runId']]
            if any(left[k]!=right[k] for k in ('initialHash','data','partition')):
                raise ValueError('Control/CART data, partition or initialization are not paired.')
            for k in ('tensorflow','numpy','device','tf32'):
                if left['runtime'].get(k)!=right['runtime'].get(k):raise ValueError('Paired numerical runtimes differ.')
        effects={}
        for name in ('primary','final'):
            values=[c[name]-b[name] for c,b in zip(cart['seeds'],control['seeds'])]
            effects[name]=dict(mean=float(np.mean(values)),sampleStd=float(np.std(values,ddof=1)),seedDeltas=values)
        paired.append(dict(dataset=baseline['dataset'],partition=baseline['partition'],sigma=baseline['sigma'],effects=effects))
    return dict(status='DESCRIPTIVE_COMPLETED_THREE_SEED_GROUPS',groups=groups,pendingGroups=pending,
        pairedCartEffects=paired,unpairedCartGroups=unpaired,sourceSha256=plan['source'],
        reportProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),officialTestUsed=False,
        independentInferencePerformed=False,interpretation='Read-only metric summary, not independent inference verification. Only complete three-seed groups are averaged. Noise levels stay separate; pending groups are listed explicitly.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--datasets',nargs='+');p.add_argument('--conditions',nargs='+');p.add_argument('--sigma',type=float)
    p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    output=args.output.resolve();target=args.report.resolve()
    if target==output or output in target.parents:raise ValueError('Write the report outside the campaign directory.')
    result=summarize(output,datasets=args.datasets,conditions=args.conditions,sigma=args.sigma)
    if target.exists():raise FileExistsError('Use a fresh report path to preserve earlier summaries.')
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Completed three-seed groups; descriptive reporting, not independent inference.')
    for g in result['groups']:
        c=g['config'];s=g['statistics']
        print(f'{c["dataset"]}/{c["partition"]} sigma={c["sigma"]:g} {c["condition"]}: '
              f'final {100*s["final"]["mean"]:.2f}% +/- {100*s["final"]["sampleStd"]:.2f}; '
              f'primary {100*s["primary"]["mean"]:.2f}%')
    for pair in result['pairedCartEffects']:
        print(f'CART effect {pair["dataset"]}/{pair["partition"]} sigma={pair["sigma"]:g}: '
              f'final {100*pair["effects"]["final"]["mean"]:+.3f} points; '
              f'primary {100*pair["effects"]["primary"]["mean"]:+.3f} points')
    print(f'Pending groups: {len(result["pendingGroups"])}; unpaired CART groups: {len(result["unpairedCartGroups"])}; report: {target}')


if __name__=='__main__':main()
