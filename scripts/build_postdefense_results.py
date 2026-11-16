#!/usr/bin/env python3
"""Reproduce the current accuracy/training-CE tables and high-resolution figures."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = dict(E1='hal9k_fedsharp-fleet-500r-20261103', E2='themachine_fedsharp-fleet-500r-20261103',
    E4='themachine_fedsharp-plain-clean-500r-20261104', E5='themachine_basil-plain-clean-500r-20261104',
    E6='themachine_fedsharp-noise-only-500r-20261114')
METHODS = ['BASIL', 'BASIL + ACDS', 'FedSharp', 'FedSharp+']
CONDITIONS = [('clean','none','Clean'),('noise_only','none','Noise only'),
    ('attack_only','gaussian','Attack only: Gaussian'),('attack_only','hidden','Attack only: Hidden'),
    ('joint','gaussian','Attack + noise: Gaussian'),('joint','hidden','Attack + noise: Hidden')]
SPLITS = [('mnist','iid'),('mnist','dirichlet'),('cifar10','iid'),('cifar10','dirichlet')]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load():
    manifests, provenance = {}, {}
    for key, folder in ARCHIVES.items():
        root = ROOT/'docs/results'/folder
        manifests[key] = read(root/'SHA256.json')
        for name, expected in manifests[key].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or (root/name).is_symlink() or sha(root/name)!=expected:
                raise ValueError('Source export checksum/path mismatch: '+name)
        provenance[str((root/'SHA256.json').relative_to(ROOT))] = sha(root/'SHA256.json')
    selected = list(csv.DictReader((ROOT/'docs/thesis/results-evidence-20261104/current_runs.csv').open()))
    noise = read(ROOT/'docs/results'/ARCHIVES['E6']/'plan.json')
    for view in noise['configs']:
        selected.append(dict(archive='E6',runId=view['runId'],dataset=view['dataset'],partition=view['partition'],
            method='FedSharp+' if view['comparisonCase']['addon'] else 'FedSharp',
            environment='noise_only',attack='none',seed=view['seed']))
    if len(selected)!=168 or len({r['runId'] for r in selected})!=168:
        raise ValueError('Expected the current 168 unique runs.')
    groups, per_run, arrays = defaultdict(list), [], {}
    for case in selected:
        folder = ROOT/'docs/results'/ARCHIVES[case['archive']]/case['runId']
        row, mechanism = read(folder/'run.json'), read(folder/'mechanisms.json')
        cfg = row['contract']['config']
        if (row['status']!='COMPLETED' or row['completedRounds']!=500 or row['completedActivations']!=50000
                or row['optimizerSteps']!=250000 or cfg['nodes']!=100 or cfg['localSteps']!=5
                or mechanism['source']!=row['source'] or mechanism['runId']!=case['runId']):
            raise ValueError('Incomplete or foreign run/mechanism identity.')
        with np.load(folder/'metrics.npz',allow_pickle=False) as archive:
            accuracy = archive['roundNodeAccuracy'].copy()
        loss = np.asarray([r['meanLocalTrainCE'] for r in mechanism['rounds']],dtype=float)
        if (accuracy.shape!=(500,100) or loss.shape!=(500,) or not np.isfinite(accuracy).all()
                or not np.isfinite(loss).all() or np.any(loss<0) or np.any((accuracy<0)|(accuracy>1))
                or any(r['displayRound']!=i+1 or r['activations']!=100 for i,r in enumerate(mechanism['rounds']))):
            raise ValueError('Invalid accuracy or training-CE series.')
        if case['environment']=='clean' and (cfg['actualAttackers'],cfg['sigma'],cfg['proximalMu'],cfg['window'])!=(0,0,0,1):
            raise ValueError('Expected the current matched direct-transfer clean control.')
        if case['environment']=='noise_only' and (cfg['actualAttackers']!=0 or cfg['sigma']!=.01):
            raise ValueError('Noise-only physical environment differs.')
        key = tuple(case[k] for k in ('dataset','partition','method','environment','attack'))
        member = dict(seed=int(case['seed']),runId=case['runId'],accuracy=accuracy.mean(axis=1)*100,loss=loss)
        groups[key].append(member)
        arrays[case['runId']] = (accuracy,loss)
        per_run.append(dict(dataset=case['dataset'],partition=case['partition'],method=case['method'],
            environment=case['environment'],attack=case['attack'],seed=int(case['seed']),runId=case['runId'],
            accuracy491To500Percent=float(accuracy[-10:].mean()*100),trainingCE491To500=float(loss[-10:].mean()),
            recorded500RoundSeconds=row['elapsedSeconds'],averageRoundSeconds=row['elapsedSeconds']/500))
    summary, curves = [], []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda r:r['seed'])
        if [r['seed'] for r in members]!=[2025,2026,2027]:
            raise ValueError('Every group requires exactly the three original seeds.')
        accuracy,loss = (np.asarray([r[k] for r in members]) for k in ('accuracy','loss'))
        a,l = accuracy[:,-10:].mean(axis=1),loss[:,-10:].mean(axis=1)
        identity = dict(zip(('dataset','partition','method','environment','attack'),key))
        summary.append(dict(**identity,accuracyMeanPercent=float(a.mean()),accuracySampleSdPercent=float(a.std(ddof=1)),
            trainingCEMean=float(l.mean()),trainingCESampleSd=float(l.std(ddof=1))))
        curves.append(dict(**identity,runIds=[r['runId'] for r in members],
            accuracyMean=accuracy.mean(axis=0).tolist(),accuracySampleSd=accuracy.std(axis=0,ddof=1).tolist(),
            trainingCEMean=loss.mean(axis=0).tolist(),trainingCESampleSd=loss.std(axis=0,ddof=1).tolist()))
    if len(summary)!=56:
        raise ValueError('Expected 56 three-seed groups.')
    for dataset,split in SPLITS:
        for seed in (2025,2026,2027):
            pair = [r for r in per_run if (r['dataset'],r['partition'],r['seed'],r['environment'])==(dataset,split,seed,'clean')]
            for i in (0,1):
                np.testing.assert_array_equal(arrays[pair[0]['runId']][i],arrays[pair[1]['runId']][i])
    return summary,per_run,curves,dict(inputManifests=provenance,manifestedFilesVerified=sum(map(len,manifests.values())),
        currentRuns=168,groups=56,seedCountPerGroup=3,lossField='mechanisms.json: rounds[].meanLocalTrainCE',
        lossScope='Mean local SGD cross-entropy; not held-out validation/test loss or the full regularized objective.',
        plusLossScope='FedSharp+ training CE uses prior-adjusted training logits.',originalMeasurementsModified=False)


def display(dataset, split):
    return ('MNIST' if dataset=='mnist' else 'CIFAR-10')+' '+('IID' if split=='iid' else 'non-IID')


def style(curve):
    method,env = curve['method'],curve['environment']
    color = {'clean':'#0072B2','attack_only':'#D55E00','joint':'#7B3294','noise_only':'#198754'}[env]
    marker = {'clean':'o','attack_only':'s','joint':'^','noise_only':'v'}[env]
    if method=='BASIL + ACDS':
        color,marker = '#008B8B','D'
    if method=='FedSharp+':
        color,marker = ('#B34B8C','P') if env=='joint' else ('#7B4F22','X')
    return dict(color=color,marker=marker,linestyle=(0,(1.1,2.1)) if method.startswith('BASIL') else '-',
                markerfacecolor='white',markeredgewidth=.9,linewidth=1.8)


def figure(curves, dataset, split, attack, metric, destination):
    available = [c for c in curves if (c['dataset'],c['partition'])==(dataset,split)
        and c['attack']==('none' if c['environment'] in ('clean','noise_only') else attack)]
    env_order = ['clean','attack_only','noise_only','joint']
    available.sort(key=lambda c:(env_order.index(c['environment']),METHODS.index(c['method'])))
    fig, ax = plt.subplots(figsize=(9.2,7.3))
    metric_key = 'accuracy' if metric=='accuracy' else 'trainingCE'
    x = np.arange(1,501)
    all_bounds = []
    for index,curve in enumerate(available):
        mean,sd = (np.asarray(curve[metric_key+k]) for k in ('Mean','SampleSd'))
        lower,upper = np.maximum(0,mean-sd),mean+sd
        if metric=='accuracy':
            upper = np.minimum(100,upper)
        all_bounds.append((lower,upper))
        label = curve['method']+': '+{'clean':'clean','attack_only':'attack only','joint':'attack + noise','noise_only':'noise only'}[curve['environment']]
        appearance = style(curve)
        ax.fill_between(x,lower,upper,color=appearance['color'],alpha=.10,linewidth=0)
        ax.plot(x,mean,label=label,markersize=4.5,markevery=(index*4,60),**appearance)
        np.testing.assert_array_equal(ax.lines[-1].get_ydata(),mean)
    ax.set_xlabel('Communication round',fontsize=12)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#DFE3E8',linewidth=.55)
    ax.tick_params(labelsize=10.5)
    ax.set_xlim(1,500)
    ax.set_xticks([1,100,200,300,400,500])
    ax.axvline(21,color='#777777',linestyle=(0,(2,3)),linewidth=.85)
    ax.set_ylabel('Mean client validation accuracy (%)' if metric=='accuracy' else 'Mean local training CE',fontsize=12.5)
    if metric=='accuracy':
        ax.set_ylim(0,100)
    else:
        upper = max(v[1].max() for v in all_bounds)
        ax.set_ylim(0,max(.1,upper*1.06))
    title = display(dataset,split)+' | '+attack.capitalize()+' attack comparison'
    fig.suptitle(title,fontsize=15,y=.97)
    handles,names = ax.get_legend_handles_labels()
    columns = 2
    nrows = int(np.ceil(len(handles)/columns))
    order = [r*columns+c for c in range(columns) for r in range(nrows) if r*columns+c<len(handles)]
    legend = fig.legend([handles[i] for i in order],[names[i] for i in order],ncol=columns,
        loc='upper center',bbox_to_anchor=(.5,.245),frameon=False,fontsize=13,
        handlelength=2.7,columnspacing=1.5,labelspacing=.65,borderaxespad=0)
    fig.canvas.draw()
    legend_height = legend.get_window_extent(fig.canvas.get_renderer()).height/fig.dpi
    total_height = 4.4+1.55+legend_height
    fig.set_size_inches(9.2,total_height,forward=True)
    fig._suptitle.set_y(1-.15/total_height)
    bottom = (legend_height+.65)/total_height
    fig.subplots_adjust(left=.11,right=.985,bottom=bottom,top=1-.9/total_height)
    legend.set_bbox_to_anchor((.5,(legend_height+.15)/total_height),transform=fig.transFigure)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = legend.get_window_extent(renderer)
    if not (box.x0>0 and box.x1<fig.bbox.x1 and box.y0>0):
        raise ValueError('Legend extends beyond the figure.')
    if box.padded(3).overlaps(ax.get_tightbbox(renderer)):
        raise ValueError('Legend overlaps an axis label.')
    title_box = fig._suptitle.get_window_extent(renderer)
    if not (title_box.x0>=0 and title_box.x1<=fig.bbox.x1 and title_box.y1<=fig.bbox.y1):
        raise ValueError('Figure title extends beyond the canvas.')
    if len(fig.axes)!=1:
        raise ValueError('One plot per image is required.')
    stem = f"{dataset}_{'iid' if split=='iid' else 'non_iid'}_{attack}_{metric}"
    for suffix in ('png','svg'):
        path = destination/(stem+'.'+suffix)
        metadata = dict(Creator='PaperMerge research figures',Title=title)
        if suffix=='svg':
            metadata['Date'] = None
        fig.savefig(path,dpi=600,metadata=metadata)
        if suffix=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    return dict(stem=stem,curves=len(available),legendFontPoints=13,legendPosition='below',
        axesPerFigure=1,detailPanel=None,curvesChecked=True,legendClearanceChecked=True,
        runIds=[run for curve in available for run in curve['runIds']],pngDpi=600)


def comparison_tables(summary):
    lookup = {(r['dataset'],r['partition'],r['method'],r['environment'],r['attack']):r for r in summary}
    lines = []
    for dataset,split in SPLITS:
        lines += ['### '+display(dataset,split),'']
        for metric,sd,title,precision in [('accuracyMeanPercent','accuracySampleSdPercent','Validation accuracy (%)',3),
                                          ('trainingCEMean','trainingCESampleSd','Local training CE',4)]:
            lines += ['**'+title+'**','', '| Condition | '+' | '.join(METHODS)+' |',
                '| --- | ---: | ---: | ---: | ---: |']
            for env,attack,label in CONDITIONS:
                cells = [label]
                available = [lookup[dataset,split,m,env,attack] for m in METHODS if (dataset,split,m,env,attack) in lookup]
                best = max((r[metric] for r in available),default=0)
                for method in METHODS:
                    r = lookup.get((dataset,split,method,env,attack))
                    value = '--' if r is None else f"{r[metric]:.{precision}f} ± {r[sd]:.{precision}f}"
                    if r and metric=='accuracyMeanPercent' and len(available)>1 and abs(r[metric]-best)<1e-12:
                        value = '**'+value+'**'
                    cells.append(value)
                lines.append('| '+' | '.join(cells)+' |')
            lines.append('')
    return lines


def update_readme(summary):
    path = ROOT/'README.md'
    original = path.read_text()
    prefix = original.split('## Current validation study')[0] if '## Current validation study' in original else original.split('## Current 500-round validation results')[0]
    test = original.split('## Official test evaluation',1)[1].split('## Trained models',1)[0]
    test_rows = {}
    for folder in ('full-official-test-20261110','basil-full-test-20261110'):
        data = read(ROOT/'docs/results/official-test-20261110'/folder/'results.json')
        for row in data['results']:
            test_rows[row['dataset'],row['method']] = row['allClients']
    table = ['| Dataset / metric | '+' | '.join(METHODS)+' |','| --- | ---: | ---: | ---: | ---: |']
    for dataset in ('mnist','cifar10'):
        name = 'MNIST' if dataset=='mnist' else 'CIFAR-10'
        table.append('| '+name+' test accuracy | '+' | '.join(f"{100*test_rows[dataset,m]['meanAccuracy']:.3f}%" for m in METHODS)+' |')
        table.append('| '+name+' raw-logit test CE | '+' | '.join(f"{test_rows[dataset,m]['meanCrossEntropyLoss']:.6f}" for m in METHODS)+' |')
    start = test.index('| Dataset')
    end = test.index('\n\n',start)
    test = test[:start]+'\n'.join(table)+test[end:]
    tail = '## Trained models'+original.split('## Trained models',1)[1]
    lines = ['## Current 500-round validation results','',
        'The current study contains **168 runs in 56 three-seed groups**: MNIST and CIFAR-10, IID and non-IID, '
        'with seeds 2025, 2026 and 2027. The latest 18 noise-only runs are complete and independently verified on the producing server. '
        'The dataset includes 24 clean, 48 attack-only, 72 attack-and-noise and 24 noise-only runs.','',
        '**Accuracy and local-training cross-entropy (CE) are shown in separate tables as mean ± sample SD across three seeds.** '
        'Each seed endpoint averages rounds 491 to 500 and all 100 clients. These are development-holdout accuracies, not official test results.','',
        '**Loss scope:** the exports do not contain validation loss for every run. CE below is the saved local-training loss '
        '(`mechanisms.json`, `meanLocalTrainCE`), averaged over five SGD steps per activation, then clients and rounds. '
        'It excludes the added gradient-norm and proximal penalties. FedSharp+ CE uses prior-adjusted training logits; '
        'other methods use their recorded training logits. These local losses are diagnostics, not a common raw-logit validation/test loss '
        'and should not be used alone to rank the methods. The separate official-test table reports raw-logit test CE where available.','',
        'Gaussian and Hidden identify the Byzantine attack. Attack conditions use 33 attackers starting in round 21. '
        'Noise conditions use channel sigma 0.01; noise-only has zero attackers. Clean BASIL and FedSharp accept the immediate predecessor '
        'without defenses and use mu zero. Their twelve paired accuracy and training-CE trajectories match exactly.','',
        '[All 168 per-run values](docs/figures/postdefense_results/per_run.csv) | '
        '[All 56 group summaries](docs/figures/postdefense_results/group_summary.csv) | '
        '[Source and figure checks](docs/figures/postdefense_results/audit.json)','',*comparison_tables(summary),
        '`--` means no matching case in the current matrix, not zero accuracy. Bold indicates the highest observed accuracy mean in a row, '
        'not statistical significance. Training CE is not ranked. BASIL + ACDS has only non-IID attack-and-noise cases. '
        'BASIL noise-only covers MNIST only. FedSharp+ has non-IID attack-and-noise and noise-only cases; there is no FedSharp+ clean or IID case.','',
        'BASIL performs better in the measured attack-only cases. FedSharp generally performs better when attack and noise occur together; '
        'CIFAR-10 non-IID Hidden is the near-equal exception for the base methods. FedSharp+ performs best on CIFAR-10 non-IID '
        'attack-and-noise and noise-only, while base FedSharp is slightly higher on MNIST non-IID. These are descriptive comparisons, '
        'not significance claims or universal guarantees. ACDS uses additional fitting-data access. Clean-versus-noisy comparisons '
        'also change the defensive policy, so their gaps cannot be attributed only to the environment.','',
        'The latest noise-only export is [here](docs/results/themachine_fedsharp-noise-only-500r-20261114/). '
        'One CIFAR-10 FedSharp+ run resumed from activation 41,523 after a CUDA timeout; it completed and passed fresh CPU verification. '
        'Timing values in the per-run CSV exclude recovery downtime. Historical timing comparisons reflect different shared-server loads. '
        'Use the [automatic launcher](docs/FEDSHARP_NOISE_ONLY_AUTO.md) for reproduction.','',
        '## Updated postdefense plots','',
        'Each image contains one plot covering all 500 unsmoothed rounds, with three-seed means and sample SD bands. '
        'Legends are below the plot. Accuracy and training-CE previews are shown side by side and wrap below each other on narrow screens. '
        'Click a preview to open the full-resolution PNG. '
        'Clean and noise-only references appear in both attack contexts. The context label applies only to the attacked curves. '
        'Training-CE figures use the loss definition above. PNG is 600 DPI and SVG is native vector.','']
    base = 'docs/figures/postdefense_results/'
    for dataset,split in SPLITS:
        for attack in ('gaussian','hidden'):
            stem = f"{dataset}_{'iid' if split=='iid' else 'non_iid'}_{attack}"
            title = display(dataset,split)+' | '+attack.capitalize()+' attack conditions'
            lines += ['### '+title,'','<p>',
                f'  <a href="{base}{stem}_accuracy.png"><img src="{base}{stem}_accuracy.png" width="420" alt="{title}: validation accuracy"></a>',
                f'  <a href="{base}{stem}_training_loss.png"><img src="{base}{stem}_training_loss.png" width="420" alt="{title}: local training CE"></a>',
                '</p>','',
                f'[Accuracy SVG]({base}{stem}_accuracy.svg) | [Training CE SVG]({base}{stem}_training_loss.svg)','']
    new = prefix+'\n'.join(lines)+'\n## Official test evaluation'+test+tail
    if '- [Postdefense figures and tables]' not in new:
        new = new.replace('## Figures and thesis material\n','## Figures and thesis material\n\n- [Postdefense figures and tables](docs/figures/postdefense_results/): current 168-run accuracy and training-CE figures, PNG/SVG.\n')
    path.write_text(new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/figures/postdefense_results')
    parser.add_argument('--update-readme',action='store_true')
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    summary,per_run,curves,audit = load()
    write_csv(destination/'group_summary.csv',summary)
    write_csv(destination/'per_run.csv',per_run)
    (destination/'plot_data.json').write_text(json.dumps(curves,indent=2,allow_nan=False)+'\n')
    audit.update(rendererSha256=sha(Path(__file__)),figures={})
    with plt.rc_context({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':12,
                         'svg.fonttype':'path','svg.hashsalt':'postdefense-20261116','axes.unicode_minus':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'}):
        for dataset,split in SPLITS:
            for attack in ('gaussian','hidden'):
                for metric in ('accuracy','training_loss'):
                    result = figure(curves,dataset,split,attack,metric,destination)
                    audit['figures'][result['stem']] = result
                    print('Rendered '+result['stem'],flush=True)
    (destination/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    notes = ['# Postdefense results','',
        'Current 168 runs, 56 three-seed groups, 500 rounds. Accuracy is development-holdout accuracy. '
        'Loss is local training CE, not held-out validation/test CE or the full regularized objective. '
        'FedSharp+ training CE uses prior-adjusted logits. Endpoints average rounds 491 to 500.','',
        'Each context has separate validation-accuracy and training-CE figures in 600-DPI PNG and native-vector SVG. '
        'Each image contains one full-range plot, without a zoom panel. All legends are below. '
        'Clean and noise-only references are repeated across the two attack contexts.','',
        '`group_summary.csv` contains all 56 group summaries; `per_run.csv` contains all 168 individual values and timings. '
        '`plot_data.json` records the exact plotted means/SDs and run identities. '
        '`audit.json` links the five source manifests and records figure checks.','',*comparison_tables(summary)]
    (destination/'README.md').write_text('\n'.join(notes).rstrip()+'\n')
    if args.update_readme:
        update_readme(summary)
    manifest = {str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='SHA256.json'}
    (destination/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Completed {len(audit["figures"])} figures, two formats each; {len(summary)} groups.',flush=True)


if __name__=='__main__':
    main()
