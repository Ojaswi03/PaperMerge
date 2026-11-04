"""Six IID/eight non-IID accuracy curves from complete, matched run archives."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(root):
    import numpy as np
    from basil_core import basil_sharp_thesis as study
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text())
    if plan['revision']!=study.REVISION:raise ValueError('Expected the matched BASIL/SHARP study.')
    hashes={'plan.json':sha(root/'plan.json')}
    for name,expected in plan['source'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid source path.')
        path=root/'source_snapshots'/relative
        if path.is_symlink() or sha(path)!=expected:raise ValueError('Source snapshot mismatch: '+name)
        hashes['source_snapshots/'+name]=expected
    rows=[];seen=set();grouped=defaultdict(list)
    for view in plan['configs']:
        cfg=study.effective_config(view);case=view['comparisonCase'];run=view['runId']
        if run in seen or Path(run).name!=run:raise ValueError('Duplicate/invalid run identity.')
        seen.add(run)
        options=SimpleNamespace(datasets=[cfg['dataset']],partitions=[cfg['partition']],seeds=[cfg['seed']],
            sigma=case['protocolConfig']['sigma'],nodes=cfg['nodes'],rounds=cfg['rounds'],
            local_steps=cfg['localSteps'],alpha=cfg['dirichletAlpha'],attack_start=cfg['attackStart'],
            attack_type=cfg['attackType'],gaussian_std=cfg.get('gaussianAttackStd',1.),
            environments=plan.get('environments',study.ENVIRONMENTS),
            families=plan.get('families',study.FAMILIES))
        expected=next(v for v in study.selected(options) if study.variant(v)==study.variant(view))
        if expected!=view:raise ValueError('Run differs from the declared figure matrix.')
        folder=root/run;record=json.loads((folder/'run.json').read_text())
        if (record['status']!='COMPLETED' or record['config']!=view or record['source']!=plan['source'] or
                record['methodRevision']!=plan['revision'] or record['completedActivations']!=cfg['nodes']*cfg['rounds'] or
                record['optimizerSteps']!=cfg['nodes']*cfg['rounds']*cfg['localSteps'] or
                record['data']!=plan['data'][cfg['dataset']] or record['partition']!=plan['partitions'][run]):
            raise ValueError('Incomplete or mismatched thesis run.')
        with np.load(folder/'metrics.npz',allow_pickle=False) as archive:accuracy=archive['roundNodeAccuracy']
        if accuracy.shape!=(cfg['rounds'],cfg['nodes']) or not np.isfinite(accuracy).all() or np.any(accuracy<0) or np.any(accuracy>1):
            raise ValueError('Invalid accuracy array.')
        for name in ('run.json','metrics.npz'):hashes[run+'/'+name]=sha(folder/name)
        item=dict(dataset=cfg['dataset'],partition=cfg['partition'],seed=cfg['seed'],family=case['family'],
            environment=case['environment'],addon=case['addon'],variant=case['arm'],runId=run,
            curve=accuracy.mean(axis=1),rounds=cfg['rounds'],nodes=cfg['nodes'],attackStart=cfg['attackStart'],
            attackType=cfg['attackType'])
        rows.append(item);grouped[cfg['dataset'],cfg['partition']].append(item)
    seeds=sorted({r['seed'] for r in rows})
    if len({r['attackType'] for r in rows})!=1:
        raise ValueError('One declared attack type is required per campaign.')
    if plan.get('attackType',rows[0]['attackType'])!=rows[0]['attackType']:
        raise ValueError('Campaign attack type differs from its configurations.')
    environments=tuple(plan.get('environments',study.ENVIRONMENTS))
    for (dataset,split),items in grouped.items():
        kinds={r['variant'] for r in items}
        families=plan.get('families',study.FAMILIES)
        wanted={f'{family}_{env}' for family in families for env in environments}
        if split=='dirichlet' and 'joint' in environments:
            wanted.update('basil_acds' if family=='basil' else 'sharp_cart' for family in families)
        if kinds!=wanted:raise ValueError('Missing requested accuracy curves.')
        if len({(r['rounds'],r['nodes'],r['attackStart']) for r in items})!=1:raise ValueError('Unmatched protocols.')
        for kind in kinds:
            if sorted(r['seed'] for r in items if r['variant']==kind)!=seeds:raise ValueError('Unmatched seed groups.')
    exported=root/'SHA256.json'
    if exported.exists():
        saved=json.loads(exported.read_text())
        if any(saved.get(n)!=value for n,value in hashes.items()):raise ValueError('Export checksum differs.')
    return plan,rows,hashes


def generate(root,destination):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,destination=Path(root).resolve(),Path(destination).resolve()
    if destination==root or destination in root.parents:raise ValueError('Separate figure directory required.')
    plan,rows,inputs=collect(root)
    manifest=destination/'figures_manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['inputSha256']!=inputs or any(sha(destination/n)!=v for n,v in saved['outputSha256'].items()):
            raise ValueError('Saved thesis figures differ; preserve them and use a new directory.')
        return saved
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Choose an empty figure directory.')
    destination.mkdir(parents=True,exist_ok=True);outputs=[]
    colors=dict(clean='#0072B2',attack_only='#D55E00',joint='#7B3294',noise_only='#009E73',extra='#00866A')
    labels=dict(clean='clean',attack_only='attack only',joint='attack + noise',noise_only='noise only')
    attacked=any(r['environment'] in ('attack_only','joint') for r in rows)
    style={'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'axes.titlesize':11,
        'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'basil-sharp-thesis',
        'figure.facecolor':'white','savefig.facecolor':'white'}
    with plt.rc_context(style):
        for dataset,split in sorted({(r['dataset'],r['partition']) for r in rows}):
            items=[r for r in rows if (r['dataset'],r['partition'])==(dataset,split)]
            fig,ax=plt.subplots(figsize=(7.3,4.8));legend=[]
            for family in ('basil','sharp'):
                for addon,env in [(None,'clean'),(None,'attack_only'),(None,'joint'),(None,'noise_only'),('acds' if family=='basil' else 'cart','joint')]:
                    selected=[r for r in items if (r['family'],r['environment'],r['addon'])==(family,env,addon)]
                    if not selected:continue
                    values=np.asarray([r['curve'] for r in selected])*100;x=np.arange(1,values.shape[1]+1)
                    method=('BASIL + ACDS' if addon else 'BASIL') if family=='basil' else ('FedSharp+' if addon else 'FedSharp')
                    name=method+': '+labels[env]
                    color=colors['extra' if addon else env]
                    pattern=dict(color=color,linestyle=(0,(1.1,2.1)) if family=='basil' else '-',
                        marker='D' if addon else dict(clean='o',attack_only='s',joint='^',noise_only='v')[env],
                        linewidth=1.65 if not addon else 1.95,markerfacecolor='white',markeredgewidth=.85)
                    offset=3*(3 if addon else ('clean','attack_only','joint','noise_only').index(env))+(family=='sharp')
                    ax.plot(x,values.mean(axis=0),label=name,markersize=3.4,markevery=(offset,12),
                        **pattern)
                    if len(values)>1:
                        sd=values.std(axis=0,ddof=1);mean=values.mean(axis=0)
                        ax.fill_between(x,np.clip(mean-sd,0,100),np.clip(mean+sd,0,100),color=color,alpha=.09,linewidth=0)
            count=items[0]['rounds'];onset=items[0]['attackStart']+1
            if attacked and onset<=count:ax.axvline(onset,color='#999999',linestyle=':',linewidth=.8)
            ax.set(title=('MNIST' if dataset=='mnist' else 'CIFAR-10')+' ('+('IID' if split=='iid' else 'non-IID')+')'
                    +(' / '+items[0]['attackType'].capitalize()+' attack' if attacked else ' / No attack'),
                xlabel='Communication round',ylabel='Validation accuracy (%)',xlim=(1,max(2,count)),ylim=(0,100))
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#DDDDDD',linewidth=.5)
            handles,names=ax.get_legend_handles_labels()
            fig.legend(handles,names,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.015),columnspacing=1.4)
            fig.subplots_adjust(left=.10,right=.985,top=.90,bottom=.30 if split!='iid' else .255)
            stem=f'{dataset}_{"iid" if split=="iid" else "non_iid"}_accuracy'
            for ext in ('svg','png'):
                name=stem+'.'+ext;fig.savefig(destination/name,dpi=600,bbox_inches='tight');outputs.append(name)
            plt.close(fig)
    with (destination/'accuracy_by_round.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['dataset','partition','method','condition','addon','seed','round','accuracy'])
        for row in rows:
            for r,value in enumerate(row['curve'],1):writer.writerow([row['dataset'],'IID' if row['partition']=='iid' else 'non-IID',
                row['family'].upper(),labels[row['environment']],row['addon'] or '',row['seed'],r,float(value)])
    outputs.append('accuracy_by_round.csv')
    (destination/'captions.txt').write_text('Mean client-model validation accuracy over communication rounds. '
        'Lines average independent seeds after averaging clients within each run; shading is sample SD across seeds. '
        +('BASIL uses dotted lines; FedSharp and FedSharp+ use solid lines. '
          'Circles denote clean, squares attack only, triangles attack plus noise, and diamonds the non-IID extensions. '
          'The vertical dotted line marks attack onset in attacked conditions. Attack type: '+rows[0]['attackType']+'. '
          if attacked else 'No Byzantine attack is active; downward triangles denote noise only. ')+
        'Only conditions declared in this campaign are plotted. '
        +('ACDS shares 5% of fitting examples. FedSharp uses checked pooling consistently, including FedSharp+.\n'
          if attacked else 'Channel noise is absolute independent Gaussian parameter noise on each link.\n'))
    outputs.append('captions.txt')
    result=dict(configurations=len(rows),inputSha256=inputs,plotterSha256=sha(Path(__file__)),
        outputSha256={n:sha(destination/n) for n in outputs},svgFiles=[n for n in outputs if n.endswith('.svg')])
    manifest.write_text(json.dumps(result,indent=2)+'\n');return result
