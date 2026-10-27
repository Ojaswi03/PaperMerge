"""Reproduce the bounded retention figure from committed numerical receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def plot_retention(receipt,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    receipt=Path(receipt);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results=json.loads(receipt.read_text())['results']
    fig,axes=plt.subplots(1,len(results),figsize=(10.5,4.5),sharey=True)
    for ax,row in zip(axes,results):
        if (not row['allReceiversIncluded'] or row['holdoutUsedForDecisions'] or row['officialTestUsed']
                or row['case']!='one_extra_five_step_local_adaptation_no_attack_no_channel'
                or len(row['rows'])!=row['nodes']):
            raise ValueError('Unexpected retention experiment scope.')
        for node in row['rows']:
            if abs(node['delta']-(node['cartAccuracy']-node['proposalAccuracy']))>1e-12:
                raise ValueError('Inconsistent recorded improvement.')
        values=[100*row[k] for k in ('referenceMean','proposalMean','cartMean')]
        labels=['Older reference','After local visit','After CART']
        bars=ax.bar(labels,values,color=['#56738f','#d59249','#258d7d'],width=.63)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+1.3,f'{value:.3f}%',ha='center',fontsize=9)
        delta=values[-1]-values[1]
        ax.set_title(f'{row["dataset"].upper()} | {row["nodes"]} clients\nCART recovery: +{delta:.3f} percentage points',fontsize=11)
        ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.tick_params(axis='x',labelsize=9)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean balanced holdout accuracy (%)')
    fig.suptitle('CART retention after one additional five-step local visit',fontsize=14)
    fig.text(.5,.025,'No attack or channel noise in this isolated check. One seed per dataset; client outcomes are not independent seed replications.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.92))
    for extension in ('svg','pdf','png'):fig.savefig(output/f'cart_retention.{extension}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    (output/'plot_provenance.json').write_text(json.dumps(dict(
        sourceReceipt=str(receipt),sourceSha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        programSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Isolated local adaptation, one seed per dataset, no attack/noise; no significance claim.'),indent=2)+'\n')


def plot_seed_study(receipt,output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    receipt=Path(receipt);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=json.loads(receipt.read_text());rows=data['rows']
    levels=sorted({r['sigma'] for r in rows if r['sigma']>0})
    fig,axes=plt.subplots(1,len(levels),figsize=(5*len(levels),4.5),sharey=True,squeeze=False)
    for ax,sigma in zip(axes[0],levels):
        for condition,label,color in [('clean','Clean','#606060'),
                ('attack_noise_defense_rla','SHARP','#537aa3'),
                ('attack_noise_defense_rla_cart','SHARP + CART','#228a77')]:
            selected=sorted([r for r in rows if r['condition']==condition and
                (r['sigma']==sigma or condition=='clean')],key=lambda r:r['seed'])
            if [r['seed'] for r in selected]!=[2025,2026,2027]:raise ValueError('Plot requires three matched seeds per group.')
            curves=100*np.asarray([r['curve'] for r in selected]);rounds=np.arange(1,curves.shape[1]+1)
            mean=curves.mean(axis=0);sd=curves.std(axis=0,ddof=1)
            ax.plot(rounds,mean,label=label,color=color,linewidth=1.8)
            ax.fill_between(rounds,mean-sd,mean+sd,color=color,alpha=.12)
        start=selected[0]['attackStart']+1
        ax.axvline(start,color='#aa6633',linestyle='--',linewidth=1,label='Attack starts (index 20)')
        ax.set(title=f'Absolute Gaussian sigma = {sigma:g}',xlabel='Completed ring round',ylim=(0,100))
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('Mean TRAIN-holdout accuracy (%)')
    axes[0,-1].legend(fontsize=8,loc='lower right')
    fig.suptitle('MNIST | 100 clients | Dirichlet 0.5 | three seeds | explicit CPU validation',fontsize=11)
    fig.text(.5,.015,'Shading: ±1 sample SD across seeds, not a confidence interval. Official TEST was not used.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.93))
    for ext in ('svg','pdf','png'):fig.savefig(output/f'mnist_seed_study.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    (output/'seed_plot_provenance.json').write_text(json.dumps(dict(sourceReceipt=str(receipt),
        sourceSha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        programSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,default=Path('docs/validation/SHARP_CART_RETENTION_20261027.json'))
    p.add_argument('--output',type=Path,default=Path('docs/figures/sharp-study'))
    p.add_argument('--seed-study',type=Path)
    args=p.parse_args();plot_retention(args.receipt,args.output)
    if args.seed_study:plot_seed_study(args.seed_study,args.output)
