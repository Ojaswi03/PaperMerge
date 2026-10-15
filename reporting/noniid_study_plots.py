"""Measured one-class trajectories and the four-condition continuation review."""
from __future__ import annotations
import csv
import gzip
import json
import os
from pathlib import Path
# Direct-script imports must resolve the project before lazy helpers.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basil_core.lazy_imports import lazy_pyplot
plt = lazy_pyplot(backend='Agg')
from basil_core.research_protocol import CLASS_NAMES

COLORS = {'A':'#2475bc', 'B':'#e68613', 'C':'#bc3f52', 'D':'#20845b'}
STYLES = {'A':('-', 'o', (0,5)), 'B':('--','s',(1,5)), 'C':(':','^',(2,5)), 'D':('-.','D',(3,5))}


def load_run(directory):
    import numpy as np
    directory = Path(directory)
    metadata = json.loads((directory/'run.json').read_text())
    with np.load(directory/'metrics.npz', allow_pickle=False) as saved:
        metrics = {key: saved[key] for key in saved.files}
    with gzip.open(directory/'activation_telemetry.json.gz', 'rt') as handle:
        records = json.load(handle)
    return metadata, metrics, records


def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def generate_condition_plots(directory, output):
    import numpy as np
    from reporting.sequential_basil_plots import generate_run_plots
    metadata, metrics, records = load_run(directory)
    if metadata['status'] != 'completed': raise ValueError('Plots require a completed run.')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    # Retain the established BASIL memory, noise and five-epoch diagnostics.
    generated = generate_run_plots(directory, output/'diagnostics')['generated']
    cfg = metadata['config']; letter = cfg['conditionId']
    objective=cfg.get('effectiveLocalObjective',cfg.get('localObjective',''))
    method_label={'class_gradient_importance_anchor':'Class-aware importance anchor',
        'cross_entropy':'CE only','snapshot_anchor_only':'Uniform snapshot anchor',
        'logit_prior_snapshot_anchor':'Local-prior CE + snapshot anchor',
        'logit_prior_ebm_snapshot_anchor':'Local-prior CE + source EBM + snapshot anchor',
        'source_ebm_snapshot_anchor':'Source EBM + uniform snapshot anchor'}.get(objective,letter)
    def mark(ax):
        if cfg['attackHidden']: ax.axvline(20, ls=':', color='#333', label='Hidden begins R20')
    for key, label, name in [
        ('fullTestAccuracy','Node 9 full-test accuracy','full_test_accuracy_vs_round'),
        ('averageAccuracy','Mean node full-test accuracy','mean_node_accuracy_vs_round'),
        ('worstNodeAccuracy','Worst node full-test accuracy','worst_node_accuracy_vs_round'),
        ('fullTestLoss','Node 9 full-test cross-entropy','full_test_loss_vs_round')]:
        fig, ax = plt.subplots(figsize=(9,4)); ax.plot(metrics[key], color=COLORS[letter], label=method_label)
        distribution='Dirichlet α=0.2' if cfg.get('partitionStrategy')=='dirichlet' else 'one-class'
        mark(ax); ax.set(xlabel='Full ring round (zero-based)', ylabel=label, title=f'{method_label}: {cfg["nRounds"]}-round full-data {distribution} run')
        if key != 'fullTestLoss': ax.set_ylim(0,1)
        ax.legend(); path=output/f'{name}.png'; save(fig,path); generated.append(str(path))
    fig, ax = plt.subplots(figsize=(10,5))
    for i, name in enumerate(CLASS_NAMES): ax.plot(metrics['fullTestPerClassAccuracy'][:,i], label=name)
    mark(ax); ax.set(xlabel='Round', ylabel='Node 9 class accuracy', ylim=(0,1)); ax.legend(ncol=2, fontsize=8)
    save(fig, output/'per_class_accuracy_vs_round.png')
    rounds = cfg['nRounds']
    component_fields=[('snapshotDistanceSquared','Distance squared from frozen received snapshot'),
                         ('snapshotAnchorPenalty','Snapshot anchor penalty'),
                         ('anchorCorrectionRatio','Anchor-gradient / CE-gradient norm'),
                         ('ebmPenalty','Source EBM penalty'),
                         ('ebmCorrectionRatio','EBM-correction / CE-gradient norm')]
    if objective=='class_gradient_importance_anchor':
        component_fields += [('importanceWeightMean','Mean class-importance protection weight'),
            ('importanceWeightMax','Maximum class-importance protection weight'),
            ('weightedSnapshotDistanceSquared','Class-importance-weighted snapshot distance squared')]
    for field, title in component_fields:
        values = np.asarray([np.mean([s[field] for s in r['optimizerTelemetry']]) for r in records]).reshape(rounds,10).mean(axis=1)
        fig, ax = plt.subplots(figsize=(9,4)); ax.plot(values, color=COLORS[letter]); mark(ax)
        ax.set(xlabel='Round', ylabel=title); save(fig,output/f'{field}_vs_round.png')
    # Before + all five local epochs, never inferred from final states.
    retained = np.concatenate([metrics['beforeTrainingPerClassAccuracy'][:,:,None,:], metrics['epochPerClassAccuracy']],axis=2)
    for node in (0,3,9):
        for r in sorted(set([0, min(20,rounds-1), rounds-1])):
            fig, ax = plt.subplots(figsize=(9,4))
            for c, name in enumerate(CLASS_NAMES): ax.plot(range(6), retained[r,node,:,c], marker='o', label=name)
            ownership='' if cfg.get('partitionStrategy')=='dirichlet' else f', own class {CLASS_NAMES[node]}'
            ax.set(xticks=range(6),xticklabels=['Before','E1','E2','E3','E4','E5'],ylabel='Class accuracy',ylim=(0,1),title=f'{method_label} — R{r}, Node {node}{ownership}')
            ax.legend(ncol=2,fontsize=7); save(fig,output/f'class_retention_r{r:02d}_node{node}.png')
    confusions = []
    for path in Path(directory).glob('audit*/confusions.json'): confusions.extend(json.loads(path.read_text()))
    lookup = {(e['round'],e['node'],e['stage']):np.asarray(e['confusion']) for e in confusions}
    for node in (0,3,9):
        for r in sorted(set([0, min(20,rounds-1),rounds-1])):
            fig, axes = plt.subplots(2,3,figsize=(12,7))
            for stage, ax in enumerate(axes.flat):
                counts=lookup.get((r,node,stage))
                if counts is None: raise ValueError(f'Missing confusion R{r}, Node {node}, stage {stage}')
                image=ax.imshow(counts/counts.sum(axis=1,keepdims=True),vmin=0,vmax=1,cmap='magma')
                ax.set(title='Before' if stage==0 else f'Epoch {stage}',xlabel='Predicted class',ylabel='True class')
            fig.suptitle(f'{method_label} — R{r}, Node {node}: evaluation-only confusion matrices')
            save(fig,output/f'confusions_r{r:02d}_node{node}.png')
    return {'plotRoot':str(output),'generated':generated}


def review_run(directory):
    import numpy as np
    metadata, m, records = load_run(directory)
    cfg=metadata['config']; rounds=cfg['nRounds']; checks={}
    checks['completed50Rounds']=metadata['status']=='completed' and rounds==50 and metadata['completedRounds']==50
    checks['500Activations']=len(records)==500 and metadata['completedActivations']==500
    checks['finiteMetrics']=all(np.isfinite(v).all() for v in m.values() if v.dtype.kind in 'fci')
    with (Path(directory)/'round_metrics.csv').open() as handle: csv_rows=list(csv.DictReader(handle))
    checks['contiguousMeasuredRounds']=[int(r['round']) for r in csv_rows]==list(range(50))
    checks['fiveFullEpochsAndPartialBatch']=all(len(r['epochs'])==5 and r['optimizerSteps']==50 and
        all(e['samplesSeen']==5000 and e['batchSizes']==[512]*9+[392] for e in r['epochs']) for r in records)
    checks['exactFrozenSelectedReference']=all(r['snapshotAnchorReferenceHash']==r['inputHash'] for r in records)
    checks['rollingFivePredecessorMemory']=all(
        r['memorySenderIds']==[(r['nodeId']-distance)%10 for distance in range(5,0,-1)] and
        r['memorySnapshotRounds']==[r['round'] if sender < r['nodeId'] else r['round']-1 for sender in r['memorySenderIds']]
        for r in records)
    checks['receiverLocalMinimumSelected']=all(r['selectedCandidateLoss'] <= min(c['receiverLocalLoss'] for c in r['candidates'])+1e-8 for r in records)
    checks['correctDelayedAttack']=all(r['attackActive']==bool(cfg['attackHidden'] and r['nodeId'] in cfg['resolvedAttackerIds'] and r['round']>=20) for r in records)
    checks['correctChannel']=all(r['channelNoiseActive']==bool(cfg['useChannelNoise']) and
        all(link['coordinateSigma']==cfg['channelNoiseSigmaAbsolute'] for link in r['outgoingLinks']) for r in records)
    if cfg['useChannelNoise']:
        checks['independentNoisyLinks']=all(len({link['noiseHash'] for link in r['outgoingLinks']})==5 and
            all(link['noiseL2Norm']>0 for link in r['outgoingLinks']) for r in records)
    else:
        checks['zeroNoiseOnCleanLinks']=all(link['noiseL2Norm']==0 for r in records for link in r['outgoingLinks'])
    steps=[s for r in records for s in r['optimizerTelemetry']]
    checks['finiteStepComponents']=all(all(np.isfinite(s[k]) for k in ('baseCrossEntropy','gradientNormSquared',
        'ebmObjective','ebmPenalty','snapshotAnchorPenalty','snapshotDistanceSquared','baseGradientNorm',
        'robustObjectiveGradientNorm','ebmCorrectionNorm','anchorGradientNorm')) for s in steps)
    expected=.0001 if cfg['conditionId']=='D' else 0.
    checks['correctEbmCoefficient']=all(np.isclose(s['ebmCoefficient'],expected,rtol=1e-6,atol=1e-12) for s in steps)
    checks['objectiveIsSumOfTerms']=all(np.isclose(s['ebmObjective'],s['baseCrossEntropy']+s['ebmPenalty']+s['snapshotAnchorPenalty'],rtol=1e-5,atol=1e-6) for s in steps)
    checks['fixedMuAndNoLambda']=cfg['snapshotAnchorMu']==.01 and metadata['legacyLambdaIgnored']
    checks['noCloudTracking']=metadata['wandbEnabled'] is False
    checks['parameterUpdatesPresent']=all(r['inputHash'] != r['trainedOutputHash'] for r in records)
    class_states=m['roundNodePerClassAccuracy']; diagonal=np.diagonal(class_states,axis1=1,axis2=2).mean(axis=1)
    other=(class_states.sum(axis=(1,2))-np.diagonal(class_states,axis1=1,axis2=2).sum(axis=1))/90
    mean=m['averageAccuracy']; first=float(mean[:10].mean()); late=float(mean[-10:].mean())
    # Descriptive review, not a test-set-tuned hyperparameter selection rule.
    collapsed=bool(diagonal[-1]>.95 and other[-1]<.05)
    selected_corrupted=sum(r['selectedSenderByzantine'] and r['inputSnapshotRound']>=20 for r in records)
    return dict(condition=cfg['conditionId'],checks=checks,technicalPass=all(checks.values()),
        finalFullTest=float(m['fullTestAccuracy'][-1]),finalMean=float(mean[-1]),finalWorst=float(m['worstNodeAccuracy'][-1]),
        bestMean=float(mean.max()),first10Mean=first,last10Mean=late,lateChangePp=100*(late-first),
        finalOwnClass=float(diagonal[-1]),finalOtherClass=float(other[-1]),constantOwnClassCollapse=collapsed,
        meanOtherClassForgetting=float(np.mean([r['meanOtherClassForgetting'] for r in records])),
        meanAnchorRatio=float(np.mean([s['anchorCorrectionRatio'] for s in steps])),
        meanEbmRatio=float(np.mean([s['ebmCorrectionRatio'] for s in steps])),
        actuallyCorruptedSelections=int(selected_corrupted),snapshotSelections=len(records),
        postAttackWindowSelections=sum(r['round']>=20 for r in records) if cfg['attackHidden'] else 0,
        corruptedSelectionRate=selected_corrupted/len(records),runtimeSeconds=metadata['runtimeSeconds'],
        partitionHash=metadata['partitionHash'],initialModelHash=metadata['initialModelHash'])


def generate_study_report(directories, output, plot_output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    plot_output=Path(plot_output); plot_output.mkdir(parents=True,exist_ok=True)
    reviews=[review_run(d) for d in directories]
    if [r['condition'] for r in reviews] != list('ABCD'): raise ValueError('Four ordered runs required.')
    paired=len({r['partitionHash'] for r in reviews})==len({r['initialModelHash'] for r in reviews})==1
    _,_,c_records=load_run(directories[2]); _,_,d_records=load_run(directories[3])
    noise_paired=all([l['noiseHash'] for l in c['outgoingLinks']]==[l['noiseHash'] for l in d['outgoingLinks']]
        for c,d in zip(c_records,d_records)) and len(c_records)==len(d_records)==500
    paired=paired and noise_paired
    technical=paired and all(r['technicalPass'] for r in reviews)
    # A correct implementation can still fail the scientific continuation review.
    promising=all(r['last10Mean']>r['first10Mean'] and not r['constantOwnClassCollapse'] for r in reviews)
    status='ELIGIBLE FOR MANUAL 500-ROUND PREPARATION' if technical and promising else 'DO NOT PREPARE 500-ROUND QUEUE'
    for key,name,label in [('fullTestAccuracy','full_test_accuracy','Node 9 test accuracy'),
        ('averageAccuracy','mean_node_accuracy','Mean node test accuracy'),('worstNodeAccuracy','worst_node_accuracy','Worst node test accuracy')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for directory in directories:
            meta,m,_=load_run(directory); c=meta['config']['conditionId']; style,marker,every=STYLES[c]
            ax.plot(m[key],color=COLORS[c],linestyle=style,marker=marker,markevery=every,markersize=4,label=c)
        ax.axvline(20,color='#333',ls=':',label='Attack begins B/C/D');ax.set(xlabel='Round',ylabel=label,ylim=(0,1));ax.legend()
        save(fig,plot_output/f'ABCD_{name}_comparison.png')
    lines=['# Non-IID four-condition 50-round review','',
        'Full data; five full epochs; fixed μ=0.01; W&B disabled. No 500-round research runs started.','',
        '| Condition | Technical checks | Node 9 final | Mean final | Worst final | Own class | Other classes | First 10 mean | Last 10 mean | Change (pp) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in reviews:
        values=[100*r[k] for k in ['finalFullTest','finalMean','finalWorst','finalOwnClass','finalOtherClass','first10Mean','last10Mean']]
        lines.append(f"| {r['condition']} | {'PASS' if r['technicalPass'] else 'FAIL'} | "+' | '.join(f'{v:.2f}%' for v in values)+f" | {r['lateChangePp']:+.2f} |")
    lines += ['', '| Condition | Other-class forgetting (pp) | Mean anchor/CE gradient ratio | Mean EBM/CE gradient ratio | Corrupted selections / total | Runtime (min) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in reviews: lines.append(f"| {r['condition']} | {100*r['meanOtherClassForgetting']:.2f} | {r['meanAnchorRatio']:.6g} | {r['meanEbmRatio']:.6g} | {r['actuallyCorruptedSelections']}/{r['snapshotSelections']} | {r['runtimeSeconds']/60:.1f} |")
    lines += ['', '## Continuation decision','',f'**{status}**','',
        'Technical validity requires 50 contiguous measured rounds, 500 activations, finite losses/gradients, 50 updates per activation, the 392-example partial batch, unchanged frozen selected references, and correct objective sums. Learning is reviewed separately: last-ten vs first-ten mean accuracy and own-class prediction collapse. This is a descriptive continuation review, not a promised accuracy threshold or proof that 500 rounds will learn.',
        '', '## Interpretation limits','',
        'A/B/C all use CE plus the same anchor; D adds variance-based full EBM. C–D compares EBM under matched noise. There is no unanchored CE control here, so these four runs alone cannot isolate the causal benefit of anchoring. A single seed is not statistical significance. Global test data are evaluation-only; no sigma, μ, learning-rate, snapshot-selection or training rule is adapted to these results.',
        '', 'The current project Hidden transformation is retained by explicit operator choice. It is not an exact reproduction of the source paper’s gradient-aggregator attack; a source-faithful model-space adaptation awaits advisor approval.','']
    lines += ['## What the measured trends mean','',
        'All four conditions completed their execution contracts, but all four ended with own-class specialization. The fixed anchor did not preserve useful ten-class knowledge in these runs. This is consistent with both catastrophic forgetting (loss of incoming class knowledge during local training) and constant-class prediction collapse (almost all test images assigned the active node’s class). Before/after-epoch confusion matrices make that distinction visible.', '',
        f"A’s last-ten minus first-ten mean accuracy changed by only {reviews[0]['lateChangePp']:.4f} percentage points. Its final other-class accuracy was {100*reviews[0]['finalOtherClass']:.4f}%. More iterations are not justified by a clear learning trend here; no protocol or hyperparameter has been changed to force a better result.", '',
        f"The mean D EBM-correction/CE-gradient ratio was {100*reviews[3]['meanEbmRatio']:.5f}%. The second-order correction was present, but small. The C–D final mean accuracy difference was {100*(reviews[3]['finalMean']-reviews[2]['finalMean']):+.4f} percentage points. A collapsed clean baseline limits interpretation: this does not establish universal EBM ineffectiveness or isolate noise mitigation independently of forgetting.", '',
        'Smaller measured forgetting alone is not better retention if the incoming model already lacks other-class knowledge. There is also no unanchored control, so these results do not isolate the causal effect of the anchor.', '',
        '## Snapshot Selection under one-class data','',
        '| Condition | Actually corrupted selections in R20–49 | Rate in that window |',
        '|---|---:|---:|']
    for r in reviews[1:]:
        denominator=r['postAttackWindowSelections']
        rate=100*r['actuallyCorruptedSelections']/denominator if denominator else 0
        lines.append(f"| {r['condition']} | {r['actuallyCorruptedSelections']}/{denominator} | {rate:.2f}% |")
    _,_,b_records=load_run(directories[1])
    example=next((r for r in b_records if r['round']==20 and r['nodeId']==1),None)
    if example is not None:
        lines += ['', 'Example: B, R20, receiver Node 1. Every candidate was scored on the same receiver-local batch; no global accuracy or sender telemetry controlled selection.', '',
            '| Sender | Snapshot round | Actually corrupted | Receiver-local CE | Selected |',
            '|---:|---:|---:|---:|---|']
        for candidate in example['candidates']:
            sender=candidate['senderId']; snapshot_round=candidate['snapshotRound']
            corrupted=candidate['senderByzantine'] and snapshot_round>=20
            lines.append(f"| {sender} | {snapshot_round} | {'Yes' if corrupted else 'No'} | {candidate['receiverLocalLoss']:.6f} | {'Yes' if sender==example['inputSourceNode'] else 'No'} |")
        lines += ['', 'A corrupted snapshot can have lower loss on the receiver’s single class than honest snapshots specialized to other classes. Following the minimum-local-loss rule is not a software defect; this experiment does not demonstrate successful Byzantine filtering under extreme one-class data. Attacker-designated snapshots from before R20 were still uncorrupted.','']
    lines += ['## Exact objectives','',
        'A/B/C: `L = CE + (0.01/2) · ‖θ − θ_selected‖²`. D: `L = CE + 0.0001 · ‖∇CE‖² + (0.01/2) · ‖θ − θ_selected‖²`. The selected received snapshot remains frozen across all five epochs. D applies `∇CE + 0.0002 H_CE ∇CE + 0.01(θ − θ_selected)`, not scalar gradient scaling.','',
        '## Metric definitions','',
        '`Node 9 final` is the last trained Node-9 model evaluated before its outbound attack/noise. `Mean final` averages all ten trained node models; `Worst final` is their minimum. `Own class` averages each node’s accuracy on its assigned class; `Other classes` averages the other nine classes across all nodes. Forgetting is before-minus-after class accuracy, so positive values indicate knowledge loss. Ratios compare actual gradient-component norms with the CE-gradient norm; they do not change optimization.','',
        '## Comparison trajectories','']
    for name in ('full_test_accuracy','mean_node_accuracy','worst_node_accuracy'):
        relative=os.path.relpath(plot_output/f'ABCD_{name}_comparison.png',output)
        lines += [f'![{name.replace("_"," ")}](<{relative}>)','']
    lines += ['## Pairing and per-condition contracts','',f'Identical partition and initial weights: {paired}. Identical keyed C/D absolute-noise draws: {noise_paired}.','',
        '| Check | A | B | C | D |','|---|---|---|---|---|']
    for name in sorted({k for r in reviews for k in r['checks']}):
        lines.append('| '+name+' | '+' | '.join('N/A' if name not in r['checks'] else 'PASS' if r['checks'][name] else 'FAIL' for r in reviews)+' |')
    (output/'NONIID_50_ROUND_REVIEW.md').write_text('\n'.join(lines)+'\n')
    (output/'continuation_review.json').write_text(json.dumps(dict(status=status,paired=paired,channelNoiseDrawsPaired=noise_paired,technicalPass=technical,promisingTrend=promising,reviews=reviews),indent=2)+'\n')
    return dict(status=status,technicalPass=technical,promisingTrend=promising)
