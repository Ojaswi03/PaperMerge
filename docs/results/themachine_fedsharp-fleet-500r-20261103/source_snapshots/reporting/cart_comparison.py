"""Complete-group mechanism/ACDS comparisons, with explicit extra-data labels."""
import csv
import hashlib
import json
from pathlib import Path


def collect(root):
    import numpy as np
    from basil_core.cart_comparison_study import check_source,effective_config
    root=Path(root);plan=json.loads((root/'plan.json').read_text());check_source(plan['source'])
    rows=[];curves=[];hashes={}
    for view in plan['configs']:
        cfg=effective_config(view);folder=root/view['runId'];run=json.loads((folder/'run.json').read_text())
        if (run['status']!='COMPLETED' or run['config']!=view or run['source']!=plan['source'] or
                run['methodRevision']!=plan['revision'] or run['completedActivations']!=cfg['nodes']*cfg['rounds']):
            raise ValueError('Comparison figures require every selected case to complete.')
        if run['data']!=plan['data'][cfg['dataset']] or run['partition']!=plan['partitions'][view['runId']]:
            raise ValueError('Comparison data identity differs.')
        path=folder/'metrics.npz'
        with np.load(path,allow_pickle=False) as z:a=z['roundNodeAccuracy']
        if a.shape!=(cfg['rounds'],cfg['nodes']) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):
            raise ValueError('Invalid complete metric array.')
        case=view['comparisonCase'];row=dict(dataset=cfg['dataset'],partition=cfg['partition'],seed=cfg['seed'],
            arm=case['arm'],environment=case['environment'],sigma=cfg['sigma'],rounds=cfg['rounds'],nodes=cfg['nodes'],
            attackStart=cfg['attackStart'],localSteps=cfg['localSteps'],alpha=cfg['dirichletAlpha'],
            primary=float(a[-min(10,len(a)):].mean()),final=float(a[-1].mean()),
            revision=plan['revision'],shareFraction=case['shareFraction'],runId=view['runId'],
            dataSharing=case['arm']=='acds',initialHash=run['initialHash'],
            ownerPartitionHash=run['partition']['partitionHash'],holdoutHash=run['partition']['holdoutHash'],
            runtime={k:run['runtime'].get(k) for k in ('tensorflow','numpy','device','tf32')})
        rows.append(row);curves.append(a.mean(axis=1));hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return plan,rows,curves,hashes


def _reference(root,requested):
    """Read frozen exports without requiring their producer to be current code."""
    import numpy as np
    root=Path(root);plan=json.loads((root/'plan.json').read_text());rows=[];curves=[];hashes={}
    for cfg in plan['configs']:
        matches=[r for r in requested if r['dataset']==cfg['dataset'] and r['partition']==cfg['partition']
            and r['seed']==cfg['seed'] and r['environment']=='joint' and cfg['sigma'] in (0.,r['sigma'])]
        if not matches:continue
        if cfg['condition'] not in ('clean','attack_noise_defense_rla','attack_noise_defense_rla_cart'):continue
        folder=root/cfg['runId'];run=json.loads((folder/'run.json').read_text())
        if run['status']!='COMPLETED' or run['config']!=cfg or run['source']!=plan['source']:raise ValueError('Incomplete reference.')
        for r in matches:
            for field,key in [('rounds','rounds'),('nodes','nodes'),('attackStart','attackStart'),('localSteps','localSteps'),('dirichletAlpha','alpha')]:
                if cfg[field]!=r[key]:raise ValueError('Reference physics/horizon differs: '+field)
            if any(run[k]!=v for k,v in [('initialHash',r['initialHash'])]):raise ValueError('Reference initialization differs.')
            if run['partition']['partitionHash']!=r['ownerPartitionHash'] or run['partition']['holdoutHash']!=r['holdoutHash']:
                raise ValueError('Reference owner data/holdout differs.')
            if {k:run['runtime'].get(k) for k in r['runtime']}!=r['runtime']:raise ValueError('Reference numerical runtime differs.')
        path=folder/'metrics.npz'
        with np.load(path,allow_pickle=False) as a:values=a['roundNodeAccuracy']
        if values.shape!=(cfg['rounds'],cfg['nodes']) or not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
            raise ValueError('Invalid reference metric array.')
        arm='saved_clean' if cfg['condition']=='clean' else ('saved_sharp' if not cfg['cart'] else 'saved_cart')
        label=plan.get('revision','original_v2')
        rows.append(dict(matches[0],arm=arm,revision=label,shareFraction=0.,dataSharing=False,runId=cfg['runId'],
            primary=float(values[-min(10,len(values)):].mean()),final=float(values[-1].mean())))
        curves.append(values.mean(axis=1));hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for ds,part in {(r['dataset'],r['partition']) for r in rows}:
        for arm in {r['arm'] for r in rows if r['dataset']==ds and r['partition']==part}:
            actual={r['seed'] for r in rows if r['dataset']==ds and r['partition']==part and r['arm']==arm}
            expected={r['seed'] for r in requested if r['dataset']==ds and r['partition']==part and r['environment']=='joint'}
            if actual!=expected:raise ValueError('Reference requires complete selected seed groups.')
    return rows,curves,hashes


def generate(root,destination,reference=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from basil_core.cart_comparison_study import LABELS
    root=Path(root).resolve();dest=Path(destination).resolve()
    plan,rows,curves,hashes=collect(root)
    refs=[] if reference is None else ([reference] if isinstance(reference,(str,Path)) else reference)
    for ref in refs:
        ref=Path(ref).resolve()
        if dest==ref or ref in dest.parents:raise ValueError('Cannot overwrite archived references.')
        extra,lines,inputs=_reference(ref,rows);rows+=extra;curves+=lines;hashes.update(inputs)
    if dest==root or any(root/c['runId']==dest or root/c['runId'] in dest.parents for c in plan['configs']):
        raise ValueError('Figure destination overlaps training inputs.')
    if dest.exists() and any(dest.iterdir()):
        # Never leave stale extra charts mixed with the new comparison.
        dest=dest.with_name(dest.name+'-'+__import__('time').strftime('%Y%m%d-%H%M%S'))
        if dest.exists():raise FileExistsError('Comparison figure destination already exists.')
    dest.mkdir(parents=True,exist_ok=True)
    labels=dict(LABELS,saved_clean='Clean (saved)',saved_sharp='SHARP (saved)',saved_cart='CART (saved)')
    groups={}
    for row,curve in zip(rows,curves):
        key=tuple(row[k] for k in ('dataset','partition','environment','sigma','arm','revision','shareFraction'))
        group=groups.setdefault(key,[])
        if row['seed'] in {r['seed'] for r,_ in group}:raise ValueError('Duplicate seed/arm in comparison.')
        group.append((row,curve))
    files=[]
    for dataset,env,sigma in sorted({(r['dataset'],r['environment'],r['sigma']) for r in rows}):
        chosen=[(k,v) for k,v in groups.items() if (k[0],k[2],k[3])==(dataset,env,sigma)]
        partitions=sorted({k[1] for k,_ in chosen},key=lambda p:p!='iid')
        for kind in ('learning_curves','final_bars','primary_bars'):
            fig,axes=plt.subplots(1,len(partitions),figsize=(8*len(partitions),6),squeeze=False,sharey=True)
            for ax,part in zip(axes[0],partitions):
                panel=[(k,v) for k,v in chosen if k[1]==part]
                tick_labels=[]
                for j,(key,items) in enumerate(panel):
                    arm=key[4];label=labels[arm]
                    if arm=='acds':label=f'SHARP + ACDS ({100*key[6]:g}% data)'
                    if arm=='saved_cart':label='SHARP + CART (reference)'
                    tick_labels.append(label)
                    a=np.asarray([v for _,v in items]);sd=a.std(axis=0,ddof=1) if len(a)>1 else np.zeros(a.shape[1])
                    color=plt.get_cmap('tab20')(j%20)
                    if kind=='learning_curves':
                        x=np.arange(1,a.shape[1]+1);ax.plot(x,100*a.mean(axis=0),label=label,color=color)
                        ax.fill_between(x,100*(a.mean(axis=0)-sd),100*(a.mean(axis=0)+sd),alpha=.1,color=color)
                    else:
                        endpoint='final' if kind=='final_bars' else 'primary';v=[r[endpoint] for r,_ in items]
                        ax.bar(j,100*np.mean(v),yerr=100*np.std(v,ddof=1) if len(v)>1 else 0,capsize=3,color=color,label=label)
                ax.set(title='IID' if part=='iid' else 'Non-IID',ylim=(0,100),ylabel='Mean client validation accuracy (%)')
                if kind=='learning_curves':
                    if env in ('joint','attack_only'):ax.axvline(panel[0][1][0][0]['attackStart']+1,color='black',ls='--')
                    ax.set_xlabel('Completed ring round');ax.legend(fontsize=7)
                else:ax.set_xticks(range(len(panel)),tick_labels,rotation=35,ha='right',fontsize=7)
                ax.grid(axis='y',alpha=.2)
            fig.suptitle(f'{dataset.upper()} | {env} | sigma={sigma:g}')
            fig.text(.5,.01,'Error bands/bars: sample SD across seeds. ACDS shares training examples; CART uses local data.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.04,1,.95))
            for ext in ('pdf','svg','png'):
                name=f'{dataset}_{env}_sigma_{sigma:g}_{kind}.{ext}';fig.savefig(dest/name,dpi=160,bbox_inches='tight');files.append(name)
            plt.close(fig)
    compact=[{k:v for k,v in r.items() if k!='runtime'} for r in rows]
    with (dest/'seed_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    files.append('seed_metrics.csv')
    (dest/'summary.json').write_text(json.dumps(dict(revision=plan['revision'],seeds=compact,
        independentInferencePerformed=False,additionalDataBaseline='ACDS shares raw fitting data before parameter attacks; no data poisoning or anonymity guarantee.'),indent=2)+'\n')
    files.append('summary.json')
    manifest=dict(revision=plan['revision'],source=plan['source'],inputMetricSha256=hashes,
        outputSha256={n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in files},destination=str(dest))
    (dest/'artifacts_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
