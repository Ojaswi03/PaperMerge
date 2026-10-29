#!/usr/bin/env python3
"""Build the exact figure filenames used by the supplied updated thesis.

Copies verified combined plots and derives class/worst-client diagnostics from
archived arrays. Does not train models or change either archived result export.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def build(destination):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=ROOT/'docs/results/cart-v2-72-100r-20261028'
    revised=ROOT/'docs/results/cart-admission-cifar6-100r-20261028'
    destination=Path(destination).resolve()
    for archive in (original,revised):
        if destination==archive or archive in destination.parents:raise ValueError('Cannot write into archived results.')
    destination.mkdir(parents=True,exist_ok=True);provenance=[];files=[]
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def verify(p,archive):
        manifest=json.loads((archive/'SHA256.json').read_text())
        if manifest[str(p.relative_to(archive))]!=sha(p):raise ValueError('Archived input checksum mismatch: '+str(p))
    plans={p:json.loads((p/'plan.json').read_text()) for p in (original,revised)}
    conditions=[('clean','Clean'),('attack_noise','Attack + noise'),
        ('attack_noise_defense','Selection defense'),('attack_noise_rla','Noise mitigation'),
        ('attack_noise_defense_rla','SHARP'),('attack_noise_defense_rla_cart','SHARP + CART')]
    for ds,kind in [('mnist','learning_curves'),('cifar10','learning_curves'),('cifar10','primary_bars'),('cifar10','final_bars')]:
        name=f'combined_revised_{ds}_sigma_0.01_{kind}'
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
        for ax,partition in zip(axes,('iid','dirichlet')):
            for j,(condition,label) in enumerate(conditions):
                root=revised if ds=='cifar10' and condition.endswith('_cart') else original
                configs=[c for c in plans[root]['configs'] if c['dataset']==ds and c['partition']==partition and c['condition']==condition]
                if len(configs)!=3 or {c['seed'] for c in configs}!={2025,2026,2027}:raise ValueError('Complete three-seed figure group required.')
                curves=[]
                for c in sorted(configs,key=lambda c:c['seed']):
                    path=root/c['runId']/'metrics.npz';verify(path,root)
                    with np.load(path,allow_pickle=False) as z:a=z['roundNodeAccuracy']
                    if a.shape!=(100,100) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('Invalid plot accuracy array.')
                    curves.append(a.mean(axis=1))
                    provenance.append(dict(figure=name+'.pdf',input=str(path.relative_to(ROOT)),sha256=sha(path),description=label+' '+partition))
                a=100*np.asarray(curves);color=plt.get_cmap('tab10')(j)
                if kind=='learning_curves':
                    x=np.arange(1,101);mean=a.mean(axis=0);sd=a.std(axis=0,ddof=1)
                    ax.plot(x,mean,label=label,color=color,lw=1.7);ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.13)
                else:
                    values=a[:,-1] if kind=='final_bars' else a[:,-10:].mean(axis=1)
                    ax.bar(j,values.mean(),yerr=values.std(ddof=1),capsize=3,color=color)
            ax.set(title='IID' if partition=='iid' else 'Non-IID',ylim=(0,100))
            ax.grid(axis='y',alpha=.2)
            if kind=='learning_curves':
                ax.axvline(21,color='black',ls='--',lw=1,label='Attack onset');ax.set_xlabel('Round');ax.legend(fontsize=7,loc='best')
            else:ax.set_xticks(range(len(conditions)),[v for _,v in conditions],rotation=30,ha='right',fontsize=8)
        axes[0].set_ylabel('Mean client validation accuracy (%)')
        fig.suptitle(ds.upper().replace('CIFAR10','CIFAR-10'))
        fig.tight_layout()
        for ext in ('pdf','svg','png'):fig.savefig(destination/f'{name}.{ext}',dpi=180,bbox_inches='tight');files.append(f'{name}.{ext}')
        plt.close(fig)
    methods=[('Clean',original,'clean'),('SHARP',original,'attack_noise_defense_rla'),
        ('CART repair',original,'attack_noise_defense_rla_cart'),('CART admission + commit',revised,'attack_noise_defense_rla_cart')]
    class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    seed_rows=[]
    for partition in ('iid','dirichlet'):
        recall=[];worst=[]
        for label,root,condition in methods:
            configs=[c for c in plans[root]['configs'] if c['dataset']=='cifar10' and c['partition']==partition and c['condition']==condition]
            if {c['seed'] for c in configs}!={2025,2026,2027} or len(configs)!=3:raise ValueError('Complete three-seed group required.')
            pc=[];low=[]
            for c in sorted(configs,key=lambda c:c['seed']):
                run=json.loads((root/c['runId']/'run.json').read_text());path=root/c['runId']/'metrics.npz';verify(path,root)
                if run['status']!='COMPLETED':raise ValueError('Incomplete thesis input.')
                with np.load(path,allow_pickle=False) as a:accuracy=a['roundNodeAccuracy'];classes=a['roundNodePerClassAccuracy']
                if accuracy.shape!=(100,100) or classes.shape!=(100,100,10) or not np.isfinite(classes).all():raise ValueError('Unexpected metric shape.')
                if np.any(classes<0) or np.any(classes>1):raise ValueError('Invalid class recall.')
                pc.append(classes[-1].mean(axis=0));low.append(float(accuracy[-1].min()))
                for idx,value in enumerate(pc[-1]):seed_rows.append(dict(partition=partition,method=label,seed=c['seed'],
                    className=class_names[idx],classRecall=float(value),worstClient=low[-1]))
                provenance.append(dict(figure=f'cifar10_{partition}_class_recall_recomputed.pdf',input=str(path.relative_to(ROOT)),sha256=sha(path),description=label))
                if partition=='dirichlet':provenance.append(dict(figure='cifar10_dirichlet_worst_client_recomputed.pdf',input=str(path.relative_to(ROOT)),sha256=sha(path),description=label))
            recall.append(np.mean(pc,axis=0));worst.append(low)
        fig,ax=plt.subplots(figsize=(9,3.8));matrix=100*np.asarray(recall)
        image=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
        ax.set_xticks(range(10),class_names,rotation=35,ha='right');ax.set_yticks(range(4),[m[0] for m in methods])
        for i in range(4):
            for j in range(10):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='black' if matrix[i,j]>55 else 'white')
        ax.set_title('CIFAR-10 '+('IID' if partition=='iid' else 'Dirichlet non-IID')+' | final class recall')
        fig.colorbar(image,ax=ax,label='Recall (%)');fig.tight_layout()
        name=f'cifar10_{partition}_class_recall_recomputed'
        for ext in ('pdf','svg','png'):fig.savefig(destination/f'{name}.{ext}',dpi=180,bbox_inches='tight');files.append(f'{name}.{ext}')
        plt.close(fig)
        if partition=='dirichlet':
            fig,ax=plt.subplots(figsize=(7.5,4));v=100*np.asarray(worst)
            ax.bar(range(4),v.mean(axis=1),yerr=v.std(axis=1,ddof=1),capsize=4,color=['#888888','#1f77b4','#ff7f0e','#2ca02c'])
            for i in range(4):ax.scatter([i-.08,i,i+.08],v[i],s=18,color='black',zorder=3)
            ax.set_xticks(range(4),[m[0] for m in methods],rotation=20,ha='right')
            ax.set(ylim=(0,100),ylabel='Worst-client accuracy (%)',title='CIFAR-10 non-IID | round 100')
            ax.grid(axis='y',alpha=.2);fig.text(.5,.01,'Mean and sample SD across three seeds; dots show individual seeds.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.04,1,1));name='cifar10_dirichlet_worst_client_recomputed'
            for ext in ('pdf','svg','png'):fig.savefig(destination/f'{name}.{ext}',dpi=180,bbox_inches='tight');files.append(f'{name}.{ext}')
            plt.close(fig)
    with (destination/'figure_provenance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['figure','input','sha256','description']);w.writeheader();w.writerows(provenance)
    with (destination/'diagnostic_seed_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(seed_rows[0]));w.writeheader();w.writerows(seed_rows)
    files+=['figure_provenance.csv','diagnostic_seed_metrics.csv']
    # Explain similar MNIST outcome bars without pretending the owner
    # distributions are identical. Every dot is a client, not a replication.
    dominance=[];partition_rows=[]
    for partition in ('iid','dirichlet'):
        values=[]
        configs=[c for c in plans[original]['configs'] if c['dataset']=='mnist' and c['partition']==partition and c['condition']=='clean']
        for c in configs:
            path=original/c['runId']/'run.json';verify(path,original)
            audit=json.loads(path.read_text())['partition'];hist=np.asarray(audit['classCounts']);p=hist/hist.sum(axis=1,keepdims=True)
            values.extend(100*p.max(axis=1))
            partition_rows.append(dict(partition=partition,seed=c['seed'],partitionHash=audit['partitionHash'],
                meanDominantClassFraction=float(p.max(axis=1).mean()),meanClassesPresent=float((hist>0).sum(axis=1).mean())))
            provenance.append(dict(figure='mnist_partition_class_balance.pdf',input=str(path.relative_to(ROOT)),sha256=sha(path),description=partition))
        dominance.append(values)
    fig,ax=plt.subplots(figsize=(6.5,4));ax.boxplot(dominance,tick_labels=['IID','Non-IID'],showmeans=True)
    ax.set(ylim=(0,100),ylabel='Largest local class share (%)',title='MNIST client data distributions');ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('pdf','svg','png'):fig.savefig(destination/f'mnist_partition_class_balance.{ext}',dpi=180,bbox_inches='tight');files.append(f'mnist_partition_class_balance.{ext}')
    plt.close(fig)
    with (destination/'mnist_partition_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(partition_rows[0]));w.writeheader();w.writerows(partition_rows)
    files.append('mnist_partition_audit.csv')
    # Include inputs of the extra partition plot in the same provenance table.
    with (destination/'figure_provenance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['figure','input','sha256','description']);w.writeheader();w.writerows(provenance)
    manifest=dict(inputMetricAndFigureProvenance=provenance,officialTestUsed=False,independentInferencePerformed=False,
        outputSha256={n:sha(destination/n) for n in files},generatorSha256=sha(Path(__file__)))
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,default=ROOT/'docs/thesis-figures/cart-20261029')
    print(build(p.parse_args().destination))
