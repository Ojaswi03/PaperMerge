"""Audited FedSharp thesis figures, separate from the frozen HPC producer."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
ENVIRONMENTS=(('clean',None),('attack_only',None),('joint',None))
COLORS={'clean':'#0072B2','attack_only':'#D55E00','joint':'#7B3294','extra':'#00866A'}
CONDITIONS={'clean':'clean','attack_only':'attack only','joint':'attack + noise'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(family,addon):
    if family=='basil':return 'BASIL + ACDS' if addon else 'BASIL'
    return 'FedSharp+' if addon else 'FedSharp'


def audit(root):
    import numpy as np
    from reporting.basil_sharp_thesis import collect
    root=Path(root).resolve()
    manifest=json.loads((root/'SHA256.json').read_text())
    for name,expected in manifest.items():
        path=root/name
        if not path.resolve().is_relative_to(root) or path.is_symlink() or sha(path)!=expected:
            raise ValueError('Export checksum/path mismatch: '+name)
    plan,rows,inputs=collect(root)
    if len(rows)!=84 or {r['dataset'] for r in rows}!={'mnist','cifar10'}:
        raise ValueError('Full 84-configuration, two-dataset matrix required.')
    if len({(r['dataset'],r['partition']) for r in rows})!=4:
        raise ValueError('Four dataset/partition groups required.')
    receipt=json.loads((root/'independent_validation.json').read_text())
    if (receipt['status']!='ALL_COMPARISON_MODELS_VERIFIED' or receipt['source']!=plan['source']
            or receipt.get('officialTestUsed') is not False):
        raise ValueError('Matching independent development-holdout inference required.')
    inference={v['runId']:v for v in receipt['verified']}
    if len(receipt['verified'])!=84 or set(inference)!={r['runId'] for r in rows}:
        raise ValueError('Inference receipt does not cover every run exactly once.')
    groups=defaultdict(list);seed_rows=[];activations=0;steps=0
    for row in rows:
        folder=root/row['runId'];saved=json.loads((folder/'run.json').read_text())
        cfg=saved['config'];case=cfg['comparisonCase']
        if (cfg['nodes'],cfg['rounds'],cfg['localSteps'],cfg['seed']) not in {
                (100,100,5,s) for s in (2025,2026,2027)}:
            raise ValueError('Unexpected clients/rounds/steps/seed.')
        with np.load(folder/'metrics.npz',allow_pickle=False) as archive:
            a=archive['roundNodeAccuracy'];pc=archive['roundNodePerClassAccuracy']
            honest=archive['roundHonestMeanAccuracy']
        if pc.shape!=(100,100,10) or honest.shape!=(100,):
            raise ValueError('Metric shape mismatch.')
        if any(not np.isfinite(v).all() or np.any(v<0) or np.any(v>1) for v in (a,pc,honest)):
            raise ValueError('Nonfinite or out-of-range metrics.')
        # This source's holdout has 100 examples per class.
        np.testing.assert_allclose(pc.mean(axis=2),a,atol=1e-12,rtol=0)
        observed=inference[row['runId']]
        if (observed['maximumValidationAccuracyDifference']>.001+1e-9 or
                abs(observed['finalValidationAccuracy']-a[-1].mean())>.001+1e-9):
            raise ValueError('Independent inference exceeds its one-image tolerance.')
        activations+=saved['completedActivations'];steps+=saved['optimizerSteps']
        row.update(method=label(case['family'],case['addon']))
        record=dict(dataset=row['dataset'],partition='IID' if row['partition']=='iid' else 'non-IID',
            method=row['method'],condition=CONDITIONS[case['environment']],seed=cfg['seed'],runId=row['runId'],
            finalAccuracy=100*float(a[-1].mean()),primaryAccuracy=100*float(a[-10:].mean()),
            honestFinalAccuracy=100*float(honest[-1]),worstClient=100*float(a[-1].min()),
            worstMeanClassRecall=100*float(pc[-1].mean(axis=0).min()))
        seed_rows.append(record)
        groups[row['dataset'],row['partition'],row['variant']].append((row,record))
    summary=[]
    for (dataset,partition,variant),members in sorted(groups.items()):
        if sorted(r['seed'] for r,_ in members)!=[2025,2026,2027]:
            raise ValueError('Every curve requires three paired seed records.')
        item=dict(dataset=dataset,partition='IID' if partition=='iid' else 'non-IID',variant=variant,
            method=members[0][1]['method'],condition=members[0][1]['condition'])
        for metric in ('finalAccuracy','primaryAccuracy','honestFinalAccuracy','worstClient','worstMeanClassRecall'):
            values=[record[metric] for _,record in members]
            item[metric]=float(np.mean(values));item[metric+'SD']=float(np.std(values,ddof=1))
        summary.append(item)
    before_attack=[]
    for dataset in ('mnist','cifar10'):
        for partition in ('iid','dirichlet'):
            for family in ('basil','sharp'):
                clean={r['seed']:r for r,_ in groups[dataset,partition,family+'_clean']}
                attacked={r['seed']:r for r,_ in groups[dataset,partition,family+'_attack_only']}
                difference=max(float(np.max(np.abs(clean[s]['curve'][:20]-attacked[s]['curve'][:20]))) for s in clean)
                before_attack.append(dict(dataset=dataset,partition=partition,family=family,
                    maximumMeanAccuracyDifferenceBeforeAttack=difference))
    result=dict(status='VERIFIED_84_CONFIGURATION_EXPORT',configurations=84,checkedFiles=len(manifest),
        activations=activations,optimizerSteps=steps,independentlyEvaluatedModels=8400,
        maximumClientInferenceDifference=max(v['maximumValidationAccuracyDifference'] for v in inference.values()),
        evaluation='1000-example stratified original-TRAIN development holdout',
        independentVerification='Saved HPC fresh-CPU inference receipt checked; checkpoints are excluded from the compact export.',
        sourceArchive=str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
        archiveManifestSha256=sha(root/'SHA256.json'),inputSha256=inputs,
        preAttackCleanAttackChecks=before_attack,summary=summary)
    return rows,seed_rows,result


def render(destination,rows):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    destination=Path(destination);outputs=[];curve_manifest={}
    style={'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,
        'axes.labelsize':10.5,'axes.titlesize':12,'legend.fontsize':9,
        'axes.linewidth':.75,'xtick.labelsize':9,'ytick.labelsize':9,
        'svg.fonttype':'path','svg.hashsalt':'fedsharp-thesis-20261030',
        'figure.facecolor':'white','savefig.facecolor':'white'}
    with plt.rc_context(style):
        for dataset in ('mnist','cifar10'):
            for partition in ('iid','dirichlet'):
                items=[r for r in rows if (r['dataset'],r['partition'])==(dataset,partition)]
                fig,ax=plt.subplots(figsize=(7.6,5.2))
                fig.subplots_adjust(left=.115,right=.985,top=.915,bottom=.285 if partition=='dirichlet' else .245)
                inset=ax.inset_axes([.41,.13,.56,.43]) if dataset=='mnist' else None
                curves=[];detail_values=[]
                for family in ('basil','sharp'):
                    specs=list(ENVIRONMENTS)+([('joint','acds' if family=='basil' else 'cart')] if partition=='dirichlet' else [])
                    for env,addon in specs:
                        selected=sorted((r for r in items if (r['family'],r['environment'],r['addon'])==(family,env,addon)),key=lambda r:r['seed'])
                        if len(selected)!=3:raise ValueError('Incomplete curve.')
                        values=np.asarray([r['curve'] for r in selected])*100
                        mean=values.mean(axis=0);sd=values.std(axis=0,ddof=1);x=np.arange(1,101)
                        title=label(family,addon)+' — '+CONDITIONS[env]
                        color=COLORS['extra' if addon else env]
                        line=(0,(4.5,2.3)) if family=='basil' else '-'
                        marker=('D' if family=='basil' else '^') if addon else ('s' if family=='basil' else 'o')
                        kwargs=dict(color=color,linestyle=line,linewidth=1.65 if not addon else 1.95)
                        ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=color,alpha=.105,linewidth=0,zorder=1)
                        # Clean and attack-only trajectories can nearly coincide.
                        # Stagger markers so both remain identifiable without
                        # shifting or smoothing their measured values.
                        offset=(2*('clean','attack_only','joint').index(env)+(0 if family=='basil' else 1))*2
                        ax.plot(x,mean,label=title,marker=marker,markersize=3.4,markerfacecolor='white',
                            markeredgewidth=.85,markevery=(offset,12),zorder=3,**kwargs)
                        ax.plot(x,mean,linestyle='none',marker=marker,markersize=3.4,
                            markerfacecolor='white',markeredgecolor=color,markeredgewidth=.85,
                            markevery=(offset,12),zorder=5)
                        if inset:
                            inset.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=color,alpha=.10,linewidth=0)
                            inset.plot(x,mean,**dict(kwargs,linewidth=1.15))
                            detail_values.extend([mean[20:]-sd[20:],mean[20:]+sd[20:]])
                        curves.append(dict(label=title,seeds=[r['seed'] for r in selected],
                            runIds=[r['runId'] for r in selected],meanPercent=mean.tolist(),sampleSDPercent=sd.tolist()))
                ax.axvline(21,color='#666666',linestyle=(0,(1.5,2.5)),linewidth=.8,zorder=0)
                ax.set(title=('MNIST' if dataset=='mnist' else 'CIFAR-10')+' — '+('IID' if partition=='iid' else 'non-IID'),
                    xlabel='Communication round',ylabel='Mean client validation accuracy (%)',xlim=(1,100),
                    ylim=(0,100 if dataset=='mnist' else 50))
                ax.set_xticks([1,20,40,60,80,100]);ax.yaxis.set_major_locator(MultipleLocator(20 if dataset=='mnist' else 10))
                ax.spines[['top','right']].set_visible(False)
                ax.grid(axis='y',color='#DFE3E8',linewidth=.55,zorder=0)
                ax.tick_params(direction='out',length=3.5,width=.7)
                if inset:
                    lo=5*math.floor(float(np.min(detail_values))/5)
                    hi=min(100,5*math.ceil(float(np.max(detail_values))/5))
                    inset.set(xlim=(21,100),ylim=(lo,hi))
                    inset.set_xticks([21,60,100]);inset.set_yticks(np.arange(lo,hi+1,5))
                    inset.tick_params(labelsize=7,length=2,width=.6,pad=2)
                    inset.set_title('Detail: rounds 21–100',fontsize=8,pad=4,
                        bbox=dict(facecolor='white',edgecolor='none',pad=1))
                    inset.grid(axis='y',color='#E6E8EC',linewidth=.4)
                    for spine in inset.spines.values():spine.set_color('#999999');spine.set_linewidth(.55)
                handles,names=ax.get_legend_handles_labels()
                legend=fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.52,.025),ncol=2,
                    frameon=False,columnspacing=1.8,handlelength=2.7,handletextpad=.65,labelspacing=.75)
                stem=f'{dataset}_{"iid" if partition=="iid" else "non_iid"}_accuracy'
                fig.canvas.draw()
                renderer=fig.canvas.get_renderer();box=legend.get_window_extent(renderer)
                if box.x0<0 or box.x1>fig.bbox.x1:raise ValueError('Legend exceeds figure width.')
                if box.y1>ax.get_window_extent(renderer).y0-22:raise ValueError('Legend overlaps axis labels.')
                for suffix in ('svg','png'):
                    name=stem+'.'+suffix
                    fig.savefig(destination/name,dpi=600,metadata={'Creator':'PaperMerge verified FedSharp thesis figures'})
                    outputs.append(name)
                curve_manifest[stem]=curves
                plt.close(fig)
    return outputs,curve_manifest


def review(root,destination):
    root=Path(root).resolve();destination=Path(destination).resolve()
    if destination==root or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError('Review destination must be separate from the immutable archive.')
    rows,seeds,evidence=audit(root)
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Use an empty review destination.')
    destination.mkdir(parents=True,exist_ok=True)
    outputs,curves=render(destination,rows)
    # Publication deliverables are vector SVGs, PNGs and compact evidence tables.
    for name,records in (('seed_endpoints.csv',seeds),('summary_endpoints.csv',evidence['summary'])):
        with (destination/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        outputs.append(name)
    with (destination/'accuracy_by_round.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['dataset','partition','method','condition','seed','round','accuracyPercent'])
        for r in rows:
            for number,value in enumerate(r['curve'],1):writer.writerow([r['dataset'],
                'IID' if r['partition']=='iid' else 'non-IID',r['method'],CONDITIONS[r['environment']],r['seed'],number,float(value)*100])
    outputs.append('accuracy_by_round.csv')
    (destination/'captions.txt').write_text(
        'Mean accuracy of 100 separately evaluated client models on the fixed 1,000-example development holdout, '
        'plotted over 100 complete ring rounds. Lines show the mean across seeds 2025, 2026 and 2027; '
        'shading shows sample standard deviation across the three seed means. '
        'Every recorded round is plotted without smoothing. The dotted vertical line marks attack onset at displayed round 21. '
        'BASIL curves are dashed with square markers, FedSharp curves are solid with circular markers; '
        'the non-IID extensions use diamond/triangle markers. MNIST includes a labelled detail inset for rounds 21–100. '
        'CIFAR-10 axes span 0–50%; MNIST axes span 0–100%. '
        'The non-IID partition uses Dirichlet concentration 0.5. '
        'Noisy cases use absolute Gaussian coordinate standard deviation 0.01. '
        'BASIL + ACDS shares 5% of fitting examples; FedSharp+ uses the complete declared class-aware extension. '
        'FedSharp denotes legacy SHARP; FedSharp+ denotes its full legacy CART extension.\n')
    outputs.append('captions.txt')
    evidence.update(reviewerSha256=sha(Path(__file__)),curves=curves,
        svgFiles=[n for n in outputs if n.endswith('.svg')],outputSha256={n:sha(destination/n) for n in outputs},
        thesisImagesReplaced=False)
    (destination/'review.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence
