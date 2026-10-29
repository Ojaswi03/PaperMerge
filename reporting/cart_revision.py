"""Revision-aware, complete-seed replacement of CART curves and averages.

Archived inputs remain immutable. Replacement requires matching frozen physics,
data, initialization, numerical runtime and all three seeds of each group.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def basis(cfg):return {k:v for k,v in cfg.items() if k not in ('seed','runId','baseRunId','guardedRunId')}
def group_key(cfg):return json.dumps(basis(cfg),sort_keys=True)


def collect(root,reference=None):
    import numpy as np
    from basil_core.cart_revision_study import check_source
    from basil_core.sharp_cart_v2_protocol import validate_config
    from basil_core.cart_campaign_io import read_json
    root=Path(root);plan=read_json(root/'plan.json');check_source(plan['source'])
    buckets={};input_hashes={};runs={}
    for cfg in plan['configs']:
        validate_config(cfg);folder=root/cfg['runId'];row=read_json(folder/'run.json')
        if (row['status']!='COMPLETED' or row['config']!=cfg or row['source']!=plan['source'] or
                row['methodRevision']!=plan['revision'] or row['completedActivations']!=cfg['nodes']*cfg['rounds']):
            raise ValueError('Revision report requires complete, identically frozen runs.')
        if row['data']!=plan['data'][cfg['dataset']] or row['partition']!=plan['partitions'][cfg['runId']]:
            raise ValueError('Revision data/partition identity mismatch.')
        with np.load(folder/'metrics.npz',allow_pickle=False) as arrays:a=arrays['roundNodeAccuracy']
        if a.shape!=(cfg['rounds'],cfg['nodes']) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):
            raise ValueError('Invalid complete accuracy array.')
        key=group_key(cfg);group=buckets.setdefault(key,dict(config=basis(cfg),seeds={},revision=plan['revision']))
        if cfg['seed'] in group['seeds']:raise ValueError('Duplicate revision seed.')
        group['seeds'][cfg['seed']]=dict(seed=cfg['seed'],runId=cfg['runId'],curve=a.mean(axis=1),
            primary=float(a[-min(10,cfg['rounds']):].mean()),final=float(a[-1].mean()))
        input_hashes[str((folder/'metrics.npz').resolve())]=hashlib.sha256((folder/'metrics.npz').read_bytes()).hexdigest()
        runs[cfg['runId']]=row
    groups=[];rows=[];reference_hashes={};paired=[]
    if reference:
        from reporting.cart_v2_campaign import collect as old_collect
        reference=Path(reference);old_summary,old_groups,old_rows,old_hashes,_=old_collect(reference)
        old_plan=read_json(reference/'plan.json')
        if old_plan['source']!=plan['baseSource']:raise ValueError('Reference baseline science differs.')
        groups=[dict(g,revision='CART v2 / unchanged baseline') for g in old_groups];rows=list(old_rows)
        reference_hashes={str((reference/run/'metrics.npz').resolve()):value for run,value in old_hashes.items()}
        for key,g in buckets.items():
            if set(g['seeds'])!={2025,2026,2027}:raise ValueError('Latest replacement requires all three seeds; never mix revisions.')
            matches=[v for v in old_groups if group_key(v['config'])==key]
            if len(matches)!=1:raise ValueError('No exact full-horizon reference group; produce standalone plots or supply matching controls.')
            # The matched old CART group already has a paired old no-CART control.
            original=next(v for v in old_summary['groups'] if group_key(v['config'])==key)
            for old_seed in original['seeds']:
                old=read_json(reference/old_seed['runId']/'run.json');new=runs[g['seeds'][old_seed['seed']]['runId']]
                if any(old[k]!=new[k] for k in ('data','partition','initialHash')):raise ValueError('Unpaired seed data or initialization.')
                for field in ('tensorflow','numpy','device','tf32'):
                    if old['runtime'].get(field)!=new['runtime'].get(field):raise ValueError('Reference numerical runtime differs.')
            groups=[v for v in groups if group_key(v['config'])!=key]
            rows=[v for v in rows if not (v['dataset']==g['config']['dataset'] and v['partition']==g['config']['partition']
                and v['sigma']==g['config']['sigma'] and v['condition']==g['config']['condition'])]
    for key,g in buckets.items():
        seeds=[g['seeds'][s] for s in sorted(g['seeds'])];curves=np.asarray([s['curve'] for s in seeds])
        groups.append(dict(config=g['config'],mean=curves.mean(axis=0),sd=curves.std(axis=0,ddof=1) if len(seeds)>1 else np.zeros(curves.shape[1]),
            revision=g['revision'],seedCount=len(seeds)))
        rows.extend(dict(dataset=g['config']['dataset'],partition=g['config']['partition'],sigma=g['config']['sigma'],
            condition=g['config']['condition'],seed=s['seed'],primary=s['primary'],final=s['final'],revision=g['revision']) for s in seeds)
        if reference and g['config']['cart']:
            controls=[v for v in old_summary['groups'] if v['config']['dataset']==g['config']['dataset']
                and v['config']['partition']==g['config']['partition'] and v['config']['sigma']==g['config']['sigma']
                and v['config']['condition']==g['config']['condition'][:-5]]
            if len(controls)!=1:raise ValueError('Missing compatible no-CART control.')
            control={s['seed']:s for s in controls[0]['seeds']};effects={}
            for endpoint in ('primary','final'):
                values=[s[endpoint]-control[s['seed']][endpoint] for s in seeds]
                effects[endpoint]=dict(mean=float(np.mean(values)),sampleStd=float(np.std(values,ddof=1)),seedDeltas=values)
            paired.append(dict(dataset=g['config']['dataset'],partition=g['config']['partition'],sigma=g['config']['sigma'],effects=effects))
    return plan,groups,rows,{**reference_hashes,**input_hashes},paired


def generate(root,destination,reference=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reporting.cart_v2_campaign import LABELS
    root=Path(root).resolve();dest=Path(destination).resolve()
    if dest==root or any(dest==root/c['runId'] or root/c['runId'] in dest.parents for c in json.loads((root/'plan.json').read_text())['configs']):
        raise ValueError('Figures cannot replace training inputs.')
    if reference and (Path(reference).resolve()==dest or Path(reference).resolve() in dest.parents):
        raise ValueError('Cannot overwrite the archived reference study.')
    plan,groups,rows,hashes,paired=collect(root,reference)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cart-chart-',dir=dest.parent) as temporary:
        stage=Path(temporary);files=[]
        def save(fig,name):
            for ext in ('png','pdf','svg'):
                filename=f'{name}.{ext}';fig.savefig(stage/filename,dpi=180,bbox_inches='tight');files.append(filename)
            plt.close(fig)
        for dataset in sorted({g['config']['dataset'] for g in groups}):
            sigmas=sorted({g['config']['sigma'] for g in groups if g['config']['dataset']==dataset and g['config']['sigma']>0}) or [0.]
            for sigma in sigmas:
                chosen=[g for g in groups if g['config']['dataset']==dataset and g['config']['sigma'] in (0.,sigma)]
                partitions=sorted({g['config']['partition'] for g in chosen},key=lambda s:(s!='iid',s))
                for kind in ('learning_curves','final_bars','primary_bars'):
                    fig,axes=plt.subplots(1,len(partitions),figsize=(7*len(partitions),6),sharey=True,squeeze=False)
                    for ax,partition in zip(axes[0],partitions):
                        panel=sorted([g for g in chosen if g['config']['partition']==partition],key=lambda g:list(LABELS).index(g['config']['condition']))
                        if len({g['config']['rounds'] for g in panel})!=1:raise ValueError('Never compare different horizons in one panel.')
                        for j,g in enumerate(panel):
                            cfg=g['config'];label=LABELS[cfg['condition']]
                            if cfg['cart']:label+=' ['+('new '+plan['variant'] if g['revision']==plan['revision'] else 'v2')+']'
                            color=plt.get_cmap('tab10')(list(LABELS).index(cfg['condition']))
                            if kind=='learning_curves':
                                x=np.arange(1,len(g['mean'])+1);ax.plot(x,100*g['mean'],label=label,color=color)
                                ax.fill_between(x,100*(g['mean']-g['sd']),100*(g['mean']+g['sd']),color=color,alpha=.12)
                            else:
                                subset=[r for r in rows if r['dataset']==dataset and r['partition']==partition and r['sigma']==cfg['sigma'] and r['condition']==cfg['condition']]
                                endpoint='final' if kind=='final_bars' else 'primary';v=[r[endpoint] for r in subset]
                                ax.bar(j,100*np.mean(v),yerr=100*np.std(v,ddof=1) if len(v)>1 else 0,color=color,label=label,capsize=3)
                        ax.set(title=partition.upper(),ylim=(0,100),ylabel='Mean node TRAIN-holdout accuracy (%)');ax.grid(alpha=.2)
                        if kind=='learning_curves':
                            ax.axvline(panel[0]['config']['attackStart']+1,color='black',ls='--',lw=1);ax.set_xlabel('Completed ring round')
                        else:ax.set_xticks(range(len(panel)),[LABELS[g['config']['condition']] for g in panel],rotation=30,ha='right',fontsize=7)
                        ax.legend(fontsize=6,loc='best')
                    fig.suptitle(f'{dataset.upper()} | sigma={sigma:g} | {plan["revision"]}')
                    fig.text(.5,.005,'Complete selected seed groups; SD is not a confidence interval. Official TEST unused.',ha='center',fontsize=8)
                    fig.tight_layout(rect=(0,.04,1,.95));save(fig,f'{dataset}_sigma_{sigma:g}_{kind}')
        columns=['dataset','partition','sigma','condition','seed','primary','final','revision']
        with (stage/'seed_metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows({k:r.get(k,'CART v2 / unchanged baseline') for k in columns} for r in rows)
        files.append('seed_metrics.csv')
        with (stage/'round_averages.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['dataset','partition','sigma','condition','round','meanAccuracy','sampleStd','revision'])
            for g in groups:
                cfg=g['config']
                for i,(mean,sd) in enumerate(zip(g['mean'],g['sd']),1):writer.writerow([cfg['dataset'],cfg['partition'],cfg['sigma'],cfg['condition'],i,float(mean),float(sd),g['revision']])
        files.append('round_averages.csv')
        summary=dict(revision=plan['revision'],pairedCartEffects=paired,seeds=rows,
            reference=str(Path(reference).resolve()) if reference else None,independentInferencePerformed=False,
            replacedGroups=len({group_key(c) for c in plan['configs']}))
        (stage/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');files.append('summary.json')
        manifest=dict(revision=plan['revision'],source=plan['source'],inputMetricSha256=hashes,files=files,
            outputSha256={name:hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in files},
            reportingProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (stage/'artifacts_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        backup=dest.with_name(dest.name+'.previous-'+str(time.time_ns())) if dest.exists() else None
        if backup:dest.rename(backup)
        try:shutil.move(str(stage),str(dest))
        except BaseException:
            if backup:backup.rename(dest)
            raise
        return manifest
