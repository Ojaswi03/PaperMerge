#!/usr/bin/env python3
"""Analyze the completed class-aware diagnostic against preserved clean controls."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from basil_core.class_aware_retention import diagnostic_config, REVISION, state_hash
    from basil_core.protocol_checkpoint import load_checkpoint
    from basil_core.research_protocol import params_hash
    from scripts.prepare_retention_diagnostic import diagnostic_configs
    from scripts.report_retention_diagnostic import review_control
    from reporting.noniid_study_plots import load_run,plt,save
    configs=diagnostic_configs(30)+[diagnostic_config()]
    directories=[ROOT/c['resultDirectory'] for c in configs]
    runs=[load_run(d) for d in directories]
    for (metadata,_,_),config in zip(runs,configs):
        if metadata['config']!=config:raise ValueError('Saved configuration differs from predeclared diagnostic.')
    reviews=[review_control(d,weighted=index==2) for index,d in enumerate(directories)]
    metadata,metrics,records=runs[-1]
    steps=[s for r in records for s in r['optimizerTelemetry']]
    coverage={}
    provenance=True
    transmissions={}
    handoff=True
    for record in records:
        state=record['classRetention']
        key=(record['inputSnapshotRound'],record['inputSourceNode'])
        if record['inputSnapshotRound']<0:
            expected=state_hash(None)
            input_hash=metadata['initialModelHash']
        else:
            expected=coverage[key]['outputStateHash']
            input_hash=transmissions[key][record['nodeId']]
        handoff=handoff and record['inputHash']==input_hash
        provenance=provenance and state['inputStateHash']==expected
        provenance=provenance and state['protectedClasses']==[c for c in state['inputClasses'] if c!=record['nodeId']]
        provenance=provenance and state['outputClasses']==sorted(set(state['inputClasses'])|{record['nodeId']})
        coverage[(record['round'],record['nodeId'])]=state
        transmissions[(record['round'],record['nodeId'])]={l['receiverId']:l['transmittedHash'] for l in record['outgoingLinks']}
    restored=load_checkpoint(directories[-1]/'checkpoint.npz',configs[-1],
        partition_hash=metadata['partitionHash'],initial_hash=metadata['initialModelHash'])
    checkpoint_ok=restored['next_activation']==300 and restored['completed_rounds']==30
    for memory in restored['memories']:
        for snapshot in memory.candidates():
            checkpoint_ok=checkpoint_ok and state_hash(snapshot.retention_state)==coverage[(snapshot.round_id,snapshot.sender_id)]['outputStateHash']
    checks=dict(classStateMatchesSelectedSource=bool(provenance),checkpointRestoresAllMemories=bool(checkpoint_ok),
        exactIncomingTransmissionHash=bool(handoff),
        localSamples5000=all(r['localSampleCount']==5000 for r in records),
        parameterCount117706=all(l['parameterCount']==117706 for r in records for l in r['outgoingLinks']),
        fixedRoundDecay=all(np.isclose(r['learningRate'],.05/(1+.05*r['round']),rtol=0,atol=1e-12) for r in records),
        activationTensorShape=metrics['activationPerClassAccuracy'].shape==(30,10,10),
        fiveEpochTensorShape=metrics['epochPerClassAccuracy'].shape==(30,10,5,10),
        importanceBounds=all(1<=s['importanceWeightMean']<=s['importanceWeightMax']<=101 for s in steps),
        correctClassSampleVisits=all(r['classRetention']['sampleVisits']==25000 for r in records),
        noRawDataSharing=all(r['classRetention']['rawDataShared'] is False for r in records),
        firstActivationMatchesUniformBaseline=records[0]['inputHash']==runs[1][2][0]['inputHash'] and
            records[0]['trainedOutputHash']==runs[1][2][0]['trainedOutputHash'],
        partitionPaired=len({r['partitionHash'] for r in reviews})==1,
        initializationPaired=len({r['initialModelHash'] for r in reviews})==1)
    reviews[-1]['checks'].update(checks)
    reviews[-1]['technicalPass']=all(reviews[-1]['checks'].values())
    aggregate=dict(meanImportanceWeight=float(np.mean([s['importanceWeightMean'] for s in steps])),
        maxImportanceWeight=float(max(s['importanceWeightMax'] for s in steps)),
        meanProtectedClasses=float(np.mean([len(r['classRetention']['protectedClasses']) for r in records])),
        finalClassCoverage=[r['classRetention']['outputClasses'] for r in records[-10:]])
    aggregate['furthestPredecessorSelections']=sum(r['inputSourceNode']==(r['nodeId']-5)%10 for r in records)
    aggregate['lateFurthestPredecessorSelections']=sum(r['inputSourceNode']==(r['nodeId']-5)%10 for r in records if r['round']>=20)
    confusions=[]
    for path in directories[-1].glob('audit*/confusions.json'):confusions.extend(json.loads(path.read_text()))
    stages={e['stage']:np.asarray(e['confusion']) for e in confusions if e['round']==29 and e['node']==0}
    aggregate['finalNode0PredictionStages']=[dict(stage=stage,
        dominantClass=int(stages[stage].sum(axis=0).argmax()),
        predictionShare=float(stages[stage].sum(axis=0).max()/stages[stage].sum())) for stage in range(6)]
    finals={e['node']:np.asarray(e['confusion']) for e in confusions if e['round']==29 and e['stage']==5}
    aggregate['finalConstantOwnClassPredictions']=len(finals)==10 and all(
        counts[:,node].sum()==counts.sum() for node,counts in finals.items())
    output=ROOT/'newResults/nonIID/class_aware_retention/comparison_30r'
    plots=ROOT/'newPlots/nonIID/class_aware_retention/comparison_30r'
    output.mkdir(parents=True,exist_ok=True);plots.mkdir(parents=True,exist_ok=True)
    titles=['CE only','Uniform anchor (mu=0.01)','Class-aware importance anchor (mu=0.01)']
    colors=['#2475bc','#e68613','#20845b'];styles=['-', '--','-.']
    plot_names=[]
    for key,ylabel,name in [('averageAccuracy','Mean node full-test accuracy','mean_node_accuracy_comparison'),
        ('fullTestAccuracy','Node 9 full-test accuracy','full_test_accuracy_comparison'),
        ('worstNodeAccuracy','Worst node full-test accuracy','worst_node_accuracy_comparison')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for (_,m,_),title,color,style in zip(runs,titles,colors,styles):
            ax.plot(range(30),m[key],color=color,ls=style,label=title)
        ax.set(xlabel='Full ring round (zero-based)',ylabel=ylabel,ylim=(0,1));ax.legend()
        save(fig,plots/f'{name}.png');plot_names.append(name)
    for field,ylabel,name in [('importanceWeightMean','Mean frozen importance weight','importance_weight_mean'),
        ('importanceWeightMax','Maximum frozen importance weight','importance_weight_max')]:
        values=np.array([np.mean([s[field] for s in r['optimizerTelemetry']]) for r in records]).reshape(30,10).mean(axis=1)
        fig,ax=plt.subplots(figsize=(10,4));ax.plot(values,color=colors[-1])
        ax.set(xlabel='Round',ylabel=ylabel);save(fig,plots/f'{name}.png');plot_names.append(name)
    fig,ax=plt.subplots(figsize=(10,4))
    counts=np.array([len(r['classRetention']['protectedClasses']) for r in records]).reshape(30,10)
    im=ax.imshow(counts.T,aspect='auto',vmin=0,vmax=9,cmap='viridis')
    ax.set(xlabel='Round',ylabel='Receiver node',title='Previously seen classes protected in selected snapshot')
    fig.colorbar(im,ax=ax,label='Class count');save(fig,plots/'protected_class_coverage.png');plot_names.append('protected_class_coverage')
    technical=all(r['technicalPass'] for r in reviews)
    payload=dict(technicalPass=technical,revision=REVISION,reviews=reviews,importance=aggregate,
        queuePrepared=False,wandbEnabled=False,hyperparametersSelectedFromTest=False)
    lines=['# Clean class-aware retention diagnostic: 30 full-data rounds','',
        'This is a separately labelled method, compared with two preserved completed controls. No threat matrix, 500-round queue, W&B run, test-driven coefficient selection or early stopping was executed.','',
        '| Method | Contracts | Node 9 final | Mean node final | Worst node final | Own class | Other classes | Runtime (min) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for title,r in zip(titles,reviews):
        values=' | '.join(f"{100*r[k]:.4f}%" for k in ['finalNode9','finalMean','finalWorst','finalOwnClass','finalOtherClass'])
        lines.append(f"| {title} | {'PASS' if r['technicalPass'] else 'FAIL'} | {values} | {r['runtimeSeconds']/60:.1f} |")
    lines+=['','| Method | First 5 mean | Last 5 mean | Mean other-class forgetting (pp) | Anchor/CE gradient ratio | Mean anchor penalty | Peak GPU allocator (MiB) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for title,r in zip(titles,reviews):
        lines.append(f"| {title} | {100*r['firstWindowMean']:.4f}% | {100*r['lastWindowMean']:.4f}% | {100*r['meanOtherClassForgetting']:.4f} | {r['meanAnchorRatio']:.6g} | {r['meanAnchorPenalty']:.6g} | {r['peakGpuMiB']:.1f} |")
    delta=100*(reviews[2]['finalMean']-reviews[1]['finalMean'])
    other_delta=100*(reviews[2]['finalOtherClass']-reviews[1]['finalOtherClass'])
    lines+=['','## Measured interpretation','',
        f"Class-aware minus uniform-anchor final mean: {delta:+.4f} pp; final other-class accuracy: {other_delta:+.4f} pp. All class-aware contracts: {'PASS' if reviews[2]['technicalPass'] else 'FAIL'}.", '',
        f"Descriptive own-class collapse flags: CE={reviews[0]['ownClassCollapse']}, uniform={reviews[1]['ownClassCollapse']}, class-aware={reviews[2]['ownClassCollapse']}. The flag means own-class >95% and other-class <5%, not a software failure threshold.", '',
        f"Every measured class-aware round mean was between {100*metrics['averageAccuracy'].min():.4f}% and {100*metrics['averageAccuracy'].max():.4f}%. Final confusion matrices confirm constant own-class prediction at all ten nodes: {aggregate['finalConstantOwnClassPredictions']}.", '',
        f"Mean squared parameter drift was {reviews[2]['meanDistanceSquared']:.6f}, versus {reviews[1]['meanDistanceSquared']:.6f} under the uniform anchor (a descriptive {100*(1-reviews[2]['meanDistanceSquared']/reviews[1]['meanDistanceSquared']):.2f}% reduction across evolving trajectories). The anchor/CE gradient norm ratio averaged {reviews[2]['meanAnchorRatio']:.6f}, versus {reviews[1]['meanAnchorRatio']:.6f}; the new correction was not numerically absent. Neither the smaller drift nor stronger correction preserved other-class predictions.", '',
        'If other-class predictions remain absent, this does not solve forgetting even if the penalty and state transport are correct. Mean forgetting can decrease just because incoming knowledge was already lost. Single-seed differences are descriptive, not statistical significance or a universal method ranking.', '',
        '## Why class coverage did not become class knowledge','',
        f"BASIL selected the fifth/farthest stored counterclockwise predecessor in {aggregate['furthestPredecessorSelections']}/300 decisions. During rounds 20–29 it did so in {aggregate['lateFurthestPredecessorSelections']}/100 decisions. The late source mapping was `[5,6,7,8,9,0,1,2,3,4]`, forming opposite-node model-history pairs, although all ten nodes still activated in the required logical ring order. The receiver-local minimum-loss rule was followed; selection was not changed using global accuracy or class coverage.", '',
        'The selected trajectories therefore carried only partial class histories, mostly opposite-class pairs. A class ID in an importance summary certifies where gradients were collected, not successful retention of that class. The selected parameter reference was already specialized to another class; a weighted distance penalty around it does not restore separate older-class solutions that are absent from those parameters. Five epochs of ordinary one-class CE still admit a constant-class predictor. These measured observations explain why smaller parameter movement is insufficient here; they are not a claim that all importance-based continual-learning methods must fail.', '',
        'Final-round Node 0 switches from dog to airplane by E1 and remains there through E5. This demonstrates overwriting of incoming class performance and constant-class collapse. Epoch count was not reduced in response.', '',
        '| Evaluation stage | Dominant predicted class | Share of all test predictions |',
        '|---|---|---:|',
        *[f"| {'Before' if s['stage']==0 else 'Epoch '+str(s['stage'])} | {['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'][s['dominantClass']]} | {100*s['predictionShare']:.2f}% |" for s in aggregate['finalNode0PredictionStages']], '',
        '## Exact loss and limitations','',
        '`L = CE + (0.01/2) sum_q Omega[q] (theta[q] - theta_selected[q])^2`. `Omega = 1 + mean normalized gradient importance over received other classes`, or 1 with no prior other class. Its derivative is `grad CE + 0.01 Omega (theta - theta_selected)`.', '',
        'Importance comes from squared minibatch-mean ordinary CE gradients along local training, normalized by their global coordinate mean and saturated at 100. It is NOT the true Fisher, NOT exact EWC, and NOT a class-accuracy oracle. The unit baseline preserves the approved uniform anchor and only the diagonal protection rule changes. The complete selected reference and diagonal remain frozen for all five epochs.', '',
        f"Mean importance weight={aggregate['meanImportanceWeight']:.6g}; maximum={aggregate['maxImportanceWeight']:.6g}; mean protected class count={aggregate['meanProtectedClasses']:.4f}. Coverage follows the selected model's history, not a central registry.", '',
        '| Final node | Classes with carried gradient summaries (not certified knowledge) |',
        '|---:|---|',
        *[f"| {node} | {', '.join(map(str,classes))} |" for node,classes in enumerate(aggregate['finalClassCoverage'])], '',
        'Source motivation: [EWC Equation (3), Kirkpatrick et al.](https://arxiv.org/html/1612.00796v2). The estimator, normalization, selected-snapshot anchor, and transport are explicitly our adaptations. No raw images or global test metrics determine importance. No ACDS, averaging, clipping, momentum, weight decay, EBM or legacy lambda is introduced.', '',
        'The auxiliary summaries add communication/storage and have no formal privacy or Byzantine-authentication guarantee. This study has no attack/noise and cannot validate threat robustness. The existing Hidden transformation is unchanged. No 500-round queue is prepared; unchanged baseline failures are not retuned until their test curves look good.', '',
        '## Decision','',
        'The software and mathematical contracts passed, but this clean class-aware method did not demonstrate ten-class retention in the approved 30-round diagnostic. **NOT READY FOR A 500-ROUND NON-IID STUDY WITH THIS METHOD.** No new attack/noise matrix was started after this failed learning-trend gate; the existing completed evidence remains available. This is a scientific negative result, not a fabricated software failure for falling below the 50% research target.', '',
        '## Evidence plots','']
    for name in plot_names:
        lines += [f'![{name.replace("_"," ")}](<{os.path.relpath(plots/f"{name}.png",output)}>)','']
    confusion=ROOT/configs[-1]['plotDirectory']/'confusions_r29_node0.png'
    lines+=['### Before and after each local epoch: final-round Node 0','',
        f'![Evaluation-only confusion matrices](<{os.path.relpath(confusion,output)}>)','',
        '## Contracts','', '| Check | CE | Uniform | Class-aware |','|---|---|---|---|']
    for key in sorted(set().union(*(r['checks'] for r in reviews))):
        statuses=['PASS' if r['checks'].get(key) is True else 'FAIL' if key in r['checks'] else 'Not applicable' for r in reviews]
        lines.append('| '+key+' | '+' | '.join(statuses)+' |')
    (output/'CLASS_AWARE_RETENTION_REVIEW.md').write_text('\n'.join(lines)+'\n')
    (output/'review.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
    if not technical:raise SystemExit('A diagnostic evidence contract failed.')


if __name__=='__main__':main()
