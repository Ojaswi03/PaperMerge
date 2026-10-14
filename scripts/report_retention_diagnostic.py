#!/usr/bin/env python3
"""Compare the two completed clean retention controls, without training."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def review_control(directory, *, weighted=False):
    import numpy as np
    from reporting.noniid_study_plots import load_run
    metadata, metrics, records = load_run(directory)
    cfg = metadata['config']; rounds = cfg['nRounds']
    steps = [s for r in records for s in r['optimizerTelemetry']]
    with (Path(directory)/'round_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    expected = rounds*10
    checks = dict(
        complete=metadata['status']=='completed' and metadata['completedRounds']==rounds
            and metadata['completedActivations']==expected and len(records)==expected,
        measuredRounds=[int(r['round']) for r in rows]==list(range(rounds)),
        diagnosticLabel=metadata['researchValid'] is False,
        cloudDisabled=metadata['wandbEnabled'] is False,
        finiteMetrics=all(np.isfinite(v).all() for v in metrics.values() if v.dtype.kind in 'fci'),
        finiteSteps=all(all(np.isfinite(s[k]) for k in ('baseCrossEntropy', 'baseGradientNorm',
            'robustObjectiveGradientNorm', 'snapshotDistanceSquared', 'snapshotAnchorPenalty',
            'anchorGradientNorm', 'anchorCorrectionRatio', 'ebmObjective')) for s in steps),
        fiveFullEpochs=all(r['optimizerSteps']==50 and len(r['epochs'])==5
            and all(e['samplesSeen']==5000 and e['batchSizes']==[512]*9+[392]
                    for e in r['epochs']) for r in records),
        frozenReference=all(r['snapshotAnchorReferenceHash']==r['inputHash'] for r in records),
        rollingMemory=all(r['memorySenderIds']==[(r['nodeId']-d)%10 for d in range(5,0,-1)]
            and r['memorySnapshotRounds']==[r['round'] if s<r['nodeId'] else r['round']-1
                                            for s in r['memorySenderIds']] for r in records),
        localMinimumSelected=all(r['selectedCandidateLoss']<=min(c['receiverLocalLoss']
            for c in r['candidates'])+1e-8 for r in records),
        noAttackOrNoise=all(not r['attackActive'] and not r['channelNoiseActive']
            and all(l['noiseL2Norm']==0 for l in r['outgoingLinks']) for r in records),
        noEbm=all(s['ebmCoefficient']==0 and s['ebmPenalty']==0 and s['ebmCorrectionNorm']==0 for s in steps),
        correctObjective=all(np.isclose(s['ebmObjective'],s['baseCrossEntropy']+s['snapshotAnchorPenalty'],rtol=1e-5,atol=1e-6)
            and np.isclose(s['snapshotAnchorMu'],cfg['snapshotAnchorMu'],rtol=1e-6,atol=1e-12)
            and np.isclose(s['snapshotAnchorPenalty'],cfg['snapshotAnchorMu']/2*s['weightedSnapshotDistanceSquared' if weighted else 'snapshotDistanceSquared'],rtol=1e-5,atol=1e-6)
            for s in steps),
        allOptimizerSteps=len(steps)==expected*50,
        changedParameters=all(r['trainedOutputHash']!=r['inputHash'] for r in records),
    )
    for column, key in [('full_test_accuracy','fullTestAccuracy'),('mean_node_accuracy','averageAccuracy')]:
        checks[column+'CsvMatches']=len(rows)==rounds and np.allclose(
            [float(r[column]) for r in rows], metrics[key], rtol=0, atol=1e-8)
    states = metrics['roundNodePerClassAccuracy']
    own = np.diagonal(states,axis1=1,axis2=2).mean(axis=1)
    other = (states.sum(axis=(1,2))-np.diagonal(states,axis1=1,axis2=2).sum(axis=1))/90
    means = metrics['averageAccuracy']
    window = min(5, rounds)
    return dict(control=cfg['controlId'],rounds=rounds,checks=checks,technicalPass=all(checks.values()),
        finalNode9=float(metrics['fullTestAccuracy'][-1]),finalMean=float(means[-1]),
        finalWorst=float(metrics['worstNodeAccuracy'][-1]),finalOwnClass=float(own[-1]),
        finalOtherClass=float(other[-1]),firstWindowMean=float(means[:window].mean()),
        lastWindowMean=float(means[-window:].mean()),
        meanOtherClassForgetting=float(np.mean([r['meanOtherClassForgetting'] for r in records])),
        meanAnchorRatio=float(np.mean([s['anchorCorrectionRatio'] for s in steps])),
        meanAnchorPenalty=float(np.mean([s['snapshotAnchorPenalty'] for s in steps])),
        meanLocalCe=float(np.mean([s['baseCrossEntropy'] for s in steps])),
        meanDistanceSquared=float(np.mean([s['snapshotDistanceSquared'] for s in steps])),
        runtimeSeconds=metadata['runtimeSeconds'],peakGpuMiB=metadata['memorySafety']['maximumGpuPeakMiB'],
        memoryPressureEvents=metadata['memorySafety']['pressureEvents'],
        ownClassCollapse=bool(own[-1]>.95 and other[-1]<.05),
        partitionHash=metadata['partitionHash'],initialModelHash=metadata['initialModelHash'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, choices=[5,30], default=30)
    args = parser.parse_args()
    from scripts.prepare_retention_diagnostic import diagnostic_configs
    from reporting.noniid_study_plots import load_run, plt, save
    configs = diagnostic_configs(args.rounds)
    directories = [ROOT/c['resultDirectory'] for c in configs]
    reviews = [review_control(d) for d in directories]
    if any(load_run(d)[0]['config']!=cfg for d,cfg in zip(directories,configs)):
        raise ValueError('Executed configuration differs from the predeclared control.')
    paired = len({r['partitionHash'] for r in reviews})==len({r['initialModelHash'] for r in reviews})==1
    technical = paired and all(r['technicalPass'] for r in reviews)
    output = ROOT/f'newResults/nonIID/retention_diagnostic/comparison_{args.rounds}r'
    plots = ROOT/f'newPlots/nonIID/retention_diagnostic/comparison_{args.rounds}r'
    output.mkdir(parents=True,exist_ok=True); plots.mkdir(parents=True,exist_ok=True)
    for key,label,name in [('averageAccuracy','Mean node accuracy','mean_node_accuracy_comparison'),
        ('fullTestAccuracy','Node 9 full-test accuracy','full_test_accuracy_comparison'),
        ('worstNodeAccuracy','Worst node accuracy','worst_node_accuracy_comparison')]:
        fig, ax = plt.subplots(figsize=(10,5))
        for directory,color,style,title in zip(directories,['#2475bc','#e68613'],['-','--'],['CE only','CE + fixed anchor']):
            _,metrics,_=load_run(directory)
            ax.plot(metrics[key],color=color,linestyle=style,label=title)
        ax.set(xlabel='Full ring round (zero-based)',ylabel=label,ylim=(0,1))
        ax.legend(); save(fig,plots/f'{name}.png')
    lines = [f'# Clean one-class retention diagnostic: {args.rounds} rounds','',
        'Full data, five full local epochs, strict BASIL handoff, no attack/noise/W&B. These are mechanism diagnostics, not a new production campaign. No test-driven coefficient tuning, early stopping or 500-round extension occurred.','',
        '| Control | Contracts | Node 9 final | Mean final | Worst final | Own class | Other classes | Runtime (min) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in reviews:
        values=' | '.join(f"{100*r[k]:.4f}%" for k in ['finalNode9','finalMean','finalWorst','finalOwnClass','finalOtherClass'])
        lines.append(f"| {r['control']} | {'PASS' if r['technicalPass'] else 'FAIL'} | {values} | {r['runtimeSeconds']/60:.1f} |")
    lines += ['', '| Control | First 5 mean | Last 5 mean | Mean other-class forgetting (pp) | Mean anchor/CE gradient ratio | Mean anchor penalty | Mean local CE | Peak allocator MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in reviews:
        lines.append(f"| {r['control']} | {100*r['firstWindowMean']:.4f}% | {100*r['lastWindowMean']:.4f}% | {100*r['meanOtherClassForgetting']:.4f} | {r['meanAnchorRatio']:.6g} | {r['meanAnchorPenalty']:.6g} | {r['meanLocalCe']:.6g} | {r['peakGpuMiB']:.1f} |")
    lines += ['', '## Interpretation','',
        f"Anchor minus CE final mean accuracy: {100*(reviews[1]['finalMean']-reviews[0]['finalMean']):+.4f} percentage points. Anchor minus CE final other-class accuracy: {100*(reviews[1]['finalOtherClass']-reviews[0]['finalOtherClass']):+.4f} percentage points.", '',
        f"Execution and pairing contracts: {'PASS' if technical else 'FAIL'}. Own-class prediction collapse: CE={reviews[0]['ownClassCollapse']}, anchor={reviews[1]['ownClassCollapse']}.", '',
        'Own-class collapse means final own-class accuracy above 95% with other-class accuracy below 5%; this is a descriptive flag, not a software correctness or production acceptance threshold. Lower forgetting does not establish better retention if incoming knowledge was already absent. One seed and no attack/noise here limit generalization. The existing Hidden transformation remains unchanged for future threat comparisons.', '',
        'CE is implemented using the same anchor worker with μ=0, verified to give ordinary CE loss/gradients/updates. The other control uses μ=0.01; neither uses EBM or legacy lambda. Five epochs, initialization, partition, augmentation, selection and schedule are paired. A class-preservation extension would require a separately documented mechanism; unchanged failed baselines are not retuned until their test curves look good.', '',
        '## Loss and measured mechanism','',
        '`CE control: L = CE`. `Anchor control: L = CE + (0.01/2) ||θ − θ_selected||²`. The anchor reference is the complete received snapshot, frozen for all five epochs; its gradient is `0.01(θ − θ_selected)`. No EBM, extra lambda, ACDS, clipping, weight decay, momentum or averaging is introduced.', '',
        f"The anchor-gradient norm averaged {100*reviews[1]['meanAnchorRatio']:.4f}% of the CE-gradient norm. Mean squared snapshot distance was {reviews[0]['meanDistanceSquared']:.6f} for CE and {reviews[1]['meanDistanceSquared']:.6f} for anchoring. These are averages over evolving trajectories, not a paired identical-start one-step comparison.", '',
        'The regularizer operated mathematically as intended, but parameter proximity is not an explicit guarantee that previous-class predictions remain correct. Ordinary CE on a one-class batch also admits a constant-class solution. If both controls show own-class collapse, repeating unchanged runs until one looks better is not a scientifically justified fix. A new retention mechanism must be explicitly specified and labelled separately; test accuracy must not choose its coefficients.', '',
        '## Comparison plots','']
    for name in ('mean_node_accuracy_comparison','full_test_accuracy_comparison','worst_node_accuracy_comparison'):
        relative=os.path.relpath(plots/f'{name}.png',output)
        lines += [f'![{name.replace("_"," ")}](<{relative}>)','']
    confusion=ROOT/configs[1]['plotDirectory']/f'confusions_r{args.rounds-1:02d}_node0.png'
    lines += ['### Representative before/after-epoch confusion matrices','',
        'Fixed-anchor control, final round, Node 0. Rows are true classes; columns are predictions. Values are normalized within each true class (0–1). A bright vertical column indicates constant-class predictions, not successful ten-class discrimination. These are evaluation-only observations and do not feed the loss or selection.', '',
        f'![Frozen-anchor before and after five local epochs](<{os.path.relpath(confusion,output)}>)','']
    lines += ['## Contracts','', '| Check | CE | Anchor |','|---|---|---|']
    for key in sorted(reviews[0]['checks']):
        lines.append('| '+key+' | '+' | '.join('PASS' if r['checks'][key] else 'FAIL' for r in reviews)+' |')
    (output/'RETENTION_DIAGNOSTIC_REVIEW.md').write_text('\n'.join(lines)+'\n')
    payload=dict(technicalPass=technical,paired=paired,reviews=reviews,queuePrepared=False,wandbEnabled=False)
    (output/'review.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
