"""Seed-averaged CART campaign charts from complete, frozen saved metrics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

LABELS={
    'clean':'Clean', 'attack_only':'Attack only', 'attack_defense':'Attack + Byzantine defense',
    'noise_only':'Noise only','noise_rla':'Noise + RLA','attack_noise':'Attack + noise',
    'attack_noise_defense':'Attack + noise + Byzantine defense',
    'attack_noise_rla':'Attack + noise + RLA',
    'attack_noise_defense_rla':'Attack + noise + both defenses',
    'attack_noise_defense_rla_cart':'Attack + noise + both defenses + CART',
}


def collect(root):
    import numpy as np
    from scripts.report_cart_v2_completed import summarize
    root=Path(root);summary=summarize(root)
    if summary['pendingGroups'] or summary['unpairedCartGroups']:
        raise ValueError('Final figures require every selected three-seed group and CART control to complete.')
    plan=json.loads((root/'plan.json').read_text());rows=[];inputs={};groups=[]
    for group in summary['groups']:
        cfg=group['config'];curves=[]
        for seed in group['seeds']:
            path=root/seed['runId']/'metrics.npz'
            with np.load(path,allow_pickle=False) as archive:accuracy=archive['roundNodeAccuracy']
            curves.append(accuracy.mean(axis=1))
            inputs[seed['runId']]=hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(dict(dataset=cfg['dataset'],partition=cfg['partition'],sigma=cfg['sigma'],
                condition=cfg['condition'],seed=seed['seed'],primary=seed['primary'],final=seed['final']))
        array=np.asarray(curves)
        groups.append(dict(config=cfg,mean=array.mean(axis=0),sd=array.std(axis=0,ddof=1)))
    if not groups:raise ValueError('No complete three-seed results to plot.')
    layouts={}
    for dataset in sorted({g['config']['dataset'] for g in groups}):
        subset=[g for g in groups if g['config']['dataset']==dataset]
        sigmas=sorted({g['config']['sigma'] for g in subset if g['config']['sigma']>0}) or [0.]
        for sigma in sigmas:
            selected=[g for g in subset if g['config']['sigma'] in (0.,sigma)]
            layout=[]
            for partition in sorted({g['config']['partition'] for g in selected},key=lambda v:(v!='iid',v)):
                panel=[g for g in selected if g['config']['partition']==partition]
                if len({g['config']['rounds'] for g in panel})!=1 or len({g['config']['attackStart'] for g in panel})!=1:
                    raise ValueError('Never mix horizons or attack onsets in a figure.')
                conditions=[g['config']['condition'] for g in panel]
                if len(set(conditions))!=len(conditions):raise ValueError('Ambiguous repeated condition in a plot panel.')
                panel.sort(key=lambda g:list(LABELS).index(g['config']['condition']))
                layout.append((partition,panel))
            layouts[(dataset,sigma)]=layout
    return summary,groups,rows,inputs,layouts


def generate(root,destination):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    root=Path(root).resolve();destination=Path(destination).resolve()
    summary,groups,rows,inputs,layouts=collect(root)
    # Protect every training input and preserve earlier chart exports.
    if destination==root or any(destination==p or p in destination.parents for p in (root/g for g in inputs)):
        raise ValueError('Figures cannot replace the campaign or a run directory.')
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Use a fresh figure directory.')
    destination.mkdir(parents=True,exist_ok=True)
    colors={name:plt.get_cmap('tab10')(i) for i,name in enumerate(LABELS)}
    files=[]
    def save(fig,name):
        for ext in ('png','pdf','svg'):
            path=destination/f'{name}.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight')
            if ext=='svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
            files.append(path.name)
        plt.close(fig)
    for (dataset,sigma),layout in layouts.items():
        stem=f'{dataset}_sigma_{sigma:g}';panels=len(layout)
        fig,axes=plt.subplots(1,panels,figsize=(7*panels,5),sharey=True,squeeze=False)
        for ax,(partition,selected) in zip(axes[0],layout):
            for g in selected:
                cfg=g['config'];x=np.arange(1,cfg['rounds']+1);mean=100*g['mean'];sd=100*g['sd']
                ax.plot(x,mean,label=LABELS[cfg['condition']],color=colors[cfg['condition']],linewidth=1.8)
                ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=colors[cfg['condition']],alpha=.10)
            cfg=selected[0]['config']
            if cfg['attackStart']<cfg['rounds']:ax.axvline(cfg['attackStart']+1,color='black',ls='--',lw=1,label='Attack starts')
            ax.set(title=f'{"IID" if partition=="iid" else "Non-IID"} | {cfg["nodes"]} clients',
                xlabel='Completed ring round',ylim=(0,100));ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0,0].set_ylabel('Mean node TRAIN-holdout accuracy (%)')
        fig.suptitle(f'{dataset.upper()} | absolute Gaussian sigma={sigma:g} | mean of three seeds')
        fig.text(.5,.005,'Shading: +/- 1 sample SD across seeds; not a confidence interval. Official TEST unused.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.03,1,.94));save(fig,stem+'_learning_curves')
        primary_rounds=min(10,layout[0][1][0]['config']['rounds'])
        for endpoint,title in (('final','Final round'),('primary',f'Final {primary_rounds} rounds')):
            fig,axes=plt.subplots(1,panels,figsize=(7*panels,6),sharey=True,squeeze=False)
            for ax,(partition,selected) in zip(axes[0],layout):
                stats=[]
                for g in selected:
                    cfg=g['config']
                    s=next(v['statistics'][endpoint] for v in summary['groups'] if v['config']==cfg)
                    stats.append((100*s['mean'],100*s['sampleStd']))
                means=[s[0] for s in stats];sds=[s[1] for s in stats];x=np.arange(len(selected))
                ax.bar(x,means,yerr=sds,capsize=3,color=[colors[g['config']['condition']] for g in selected])
                for index,(mean,sd) in enumerate(stats):ax.text(index,min(103,mean+sd+1),f'{mean:.2f}',ha='center',fontsize=8)
                labels=[LABELS[g['config']['condition']].replace(' + ',' +\n') for g in selected]
                ax.set_xticks(x,labels,rotation=20,ha='right',fontsize=7)
                ax.set(title='IID' if partition=='iid' else 'Non-IID',ylim=(0,108));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            axes[0,0].set_ylabel('Mean node TRAIN-holdout accuracy (%)')
            fig.suptitle(f'{dataset.upper()} | sigma={sigma:g} | {title} average')
            fig.text(.5,.005,'Error bars: +/- 1 sample SD across three seeds. Negative and chance-level results are retained.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.03,1,.94));save(fig,stem+f'_{endpoint}_bars')
    for name,data in [('seed_metrics.csv',rows),('round_averages.csv',[
            dict(dataset=g['config']['dataset'],partition=g['config']['partition'],sigma=g['config']['sigma'],
                condition=g['config']['condition'],round=index+1,mean=float(mean),sampleStd=float(sd))
            for g in groups for index,(mean,sd) in enumerate(zip(g['mean'],g['sd']))])]:
        with (destination/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        files.append(name)
    summary['averaging']='Nodes within each round, then three seed means at the same dataset/split/condition/sigma. Final and last-ten-round bars are separate.'
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');files.append('summary.json')
    manifest=dict(status='FIGURES_COMPLETE',files=files,inputMetricSha256=inputs,
        reportingProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputSha256={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in files},
        independentInferencePerformed=False,officialTestUsed=False)
    (destination/'artifacts_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
