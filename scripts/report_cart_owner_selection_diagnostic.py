#!/usr/bin/env python3
"""Validate and plot saved three-round SS-on/off CART results; never trains."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RESULT=ROOT/'newResults/nonIID/cart_owner_gradients/clean_three_round_selection_pair_v1'
PLOTS=ROOT/'newPlots/nonIID/cart_owner_gradients/clean_three_round_selection_pair_v1'


def summarize(records,control, *, selection_rule=None, rounds=3):
    import numpy as np
    from basil_core.cart_owner_gradient_worker import available_owner_classes
    from basil_core.research_protocol import basil_round_learning_rate
    from scripts.run_cart_owner_selection_diagnostic import CONTROLS
    if selection_rule is not None and selection_rule not in ('receiverLocalLoss','balancedTrainingLoss'):
        raise ValueError('Unknown explicit selection scoring rule.')
    if type(rounds) is not int or rounds not in (3, 30):
        raise ValueError('Unsupported diagnostic report horizon.')
    if (selection_rule is None and control not in CONTROLS) or len(records)!=rounds*10:
        raise ValueError('Thirty complete activations per valid control required.' if rounds == 3
                         else 'Three hundred complete activations per valid control required.')
    responses=0;traffic=0;rows=[];initial_selections=0
    for activation,r in enumerate(records):
        round_id,node=divmod(activation,10)
        if (r['round'],r['node'],r['condition'],r['steps'])!=(round_id,node,control,50):raise ValueError('Activation order/steps differ.')
        if r['inputHash']!=r['snapshotAnchorReferenceHash']:raise ValueError('Strict frozen-anchor handoff differs.')
        if r['learningRate']!=basil_round_learning_rate(.05,round_id):raise ValueError('Round-decay schedule differs.')
        if len(r['epochs'])!=5 or any(e['samplesSeen']!=5000 or e['batchSizes']!=[512]*9+[392] for e in r['epochs']):
            raise ValueError('Five complete local epochs differ.')
        selection=r['selection'];chosen=next(d for d in selection if
            (d['senderId'],d['snapshotRound'])==(r['selectedSender'],r['selectedSnapshotRound']))
        if control=='basil_ss' or selection_rule is not None:
            score=selection_rule or 'receiverLocalLoss'
            minimum=min(d[score] for d in selection)
            tied=[d for d in selection if d[score]<=minimum+1e-8]
            expected=min(tied,key=lambda d:(node-d['senderId'])%10)
            if (chosen['senderId'],chosen['snapshotRound'])!=(expected['senderId'],expected['snapshotRound']):
                raise ValueError('Receiver-local minimum/tie rule differs.')
        else:
            if r['selectedSender']!=(node-1)%10 or r['selectedSnapshotRound']!=(round_id if node else round_id-1):
                raise ValueError('Immediate predecessor identity/age differs.')
            if activation and r['inputHash']!=records[activation-1]['outputHash']:
                raise ValueError('Immediate predecessor weight hash differs.')
        initial_selections+=r['selectedSnapshotRound']==-1
        old=available_owner_classes(round_id,node,diagnostic_rounds=rounds)
        if r['olderClasses']!=list(old) or len(r['ownerGradients'])!=50 or len(r['optimizerTelemetry'])!=50:
            raise ValueError('Missing, future or wrong owner responses.')
        for step,owner in enumerate(r['ownerGradients']):
            if owner['round']!=round_id or owner['receiver']!=node or owner['optimizerStep']!=step:
                raise ValueError('Owner gradient step context differs.')
            classes=sorted((*old,node))
            if owner['availableClasses']!=classes or owner['classWeights']!={str(c):1/len(classes) for c in classes}:
                raise ValueError('Equal class weighting differs.')
            if [o['owner'] for o in owner['owners']]!=list(old) or any(
                o['support']!=16 or len(set(o['trainingIndices']))!=16 or
                o['parameterHash']!=owner['modelHash'] for o in owner['owners']):
                raise ValueError('Owner freshness/support differs.')
            if owner['ownerOptimizerSteps'] or owner['modelAveraging'] or owner['rawImagesReturned']:
                raise ValueError('Unexpected helper/model averaging operation.')
            responses+=owner['extraOwnerResponses'];traffic+=owner['responseCoordinateBytes']
        for s in r['optimizerTelemetry']:
            if not all(np.isfinite(v) for v in s.values() if isinstance(v,(int,float))):raise ValueError('Non-finite step metric.')
            if s['ebmCoefficient']!=0 or s['ebmPenalty']!=0 or s['configuredSigma']!=0:raise ValueError('Unexpected noise/EBM.')
        if node==9:
            current=records[activation-9:activation+1]
            acc=[n['globalTestEvaluationOnly']['accuracy'] for n in current]
            train=[n['epochs'][-1]['trainingAnalysis']['perClassAccuracy'] for n in current]
            rows.append(dict(round=round_id,full_test_accuracy=acc[-1],mean_node_accuracy=float(np.mean(acc)),
                worst_node_accuracy=min(acc),full_test_loss=r['globalTestEvaluationOnly']['crossEntropy'],
                own_class_training_probe_accuracy=float(np.mean([train[n][n] for n in range(10)])),
                other_class_training_probe_accuracy=float(np.mean([np.delete(train[n],n).mean() for n in range(10)])),
                **{f'class_{c}_accuracy':v for c,v in enumerate(r['globalTestEvaluationOnly']['perClassAccuracy'])}))
    if responses!=2250+(rounds-1)*4500:raise ValueError('Expected 2250 first-round responses plus 4500 per subsequent round.')
    final=records[-10:]
    return dict(control=control,rounds=rows,optimizerSteps=rounds*500,helperOptimizerSteps=0,
        extraOwnerResponses=responses,responseCoordinateBytes=traffic,initialSnapshotSelections=initial_selections,
        finalPerNodeAccuracy=[r['globalTestEvaluationOnly']['accuracy'] for r in final],
        finalPerClassAccuracy=records[-1]['globalTestEvaluationOnly']['perClassAccuracy'],checksPassed=True)


def report():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.run_cart_owner_selection_diagnostic import CONTROLS,verify_configs
    from scripts.run_cart_feature_diagnostic import write_json
    run=json.loads((RESULT/'run.json').read_text())
    if run['status']!='completed' or run['activationsCompleted']!=60 or run['optimizerSteps']!=3000:
        raise ValueError('Paired run incomplete; no missing points fabricated.')
    if PLOTS.exists() or (RESULT/'summary.json').exists():raise ValueError('Refusing review overwrite.')
    summaries={};histories={};states={};source_data={}
    for control in CONTROLS:
        directory=RESULT/control;state=json.loads((directory/'run.json').read_text())
        if state['status']!='completed' or state['roundsCompleted']!=3 or state['optimizerSteps']!=1500:
            raise ValueError('Control incomplete.')
        records=json.loads((directory/'activations.json').read_text())
        summaries[control]=summarize(records,control);histories[control]=records;states[control]=state
        traces=[json.loads(l) for l in (directory/'audit/batch_trace.jsonl').read_text().splitlines()]
        if len(traces)!=1500:raise ValueError('Missing batch trace.')
        for i,t in enumerate(traces):
            owner=t['ownerGradientContext']
            if owner['modelHash']!=t['parameterHashBefore'] or any(o['parameterHash']!=t['parameterHashBefore'] for o in owner['owners']):
                raise ValueError('Stale owner hash in actual execution trace.')
            r,batch=divmod(i,50)
            if (t['round_id'],t['node_id'])!=divmod(r,10):raise ValueError('Batch-trace activation order differs.')
        source_data[control]=[(t['indicesHash'],t['augmentationKey'],t['imagesHash'],
            [(o['owner'],o['trainingIndices'],o['augmentationKey']) for o in t['ownerGradientContext']['owners']]) for t in traces]
        with np.load(directory/'checkpoint.npz',allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata']))
            if metadata['nextActivation']!=30 or len(metadata['snapshots'])!=50:raise ValueError('Incomplete final memory checkpoint.')
            from basil_core.research_protocol import params_hash
            for s in metadata['snapshots']:
                prefix=f"r{s['receiver']}s{s['sender']}"
                weights=[saved[f'{prefix}w{j}'] for j in range(10)]
                if s['round']!=2 or params_hash(weights)!=s['hash']:raise ValueError('Final memory checkpoint checksum differs.')
    a,b=(states[c] for c in CONTROLS)
    if any(a[k]!=b[k] for k in ('initialModelHash','partitionHash','sourceHashes','numericalExecution')):
        raise ValueError('Unpaired initialization/partition/source/device.')
    if source_data[CONTROLS[0]]!=source_data[CONTROLS[1]]:raise ValueError('Data/augmentation/helper sampling is not paired.')
    del source_data
    PLOTS.mkdir(parents=True)
    colors={'basil_ss':'#0072B2','immediate_predecessor':'#D55E00'}
    labels={'basil_ss':'CART + BASIL SS','immediate_predecessor':'CART + immediate predecessor (SS off)'}
    for metric,name in [('full_test_accuracy','full_test_accuracy_comparison'),
        ('mean_node_accuracy','mean_node_accuracy_comparison'),('worst_node_accuracy','worst_node_accuracy_comparison'),
        ('full_test_loss','full_test_ce_comparison')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for control in CONTROLS:
            rows=summaries[control]['rounds'];ax.plot([r['round'] for r in rows],[r[metric] for r in rows],
                marker='o',color=colors[control],label=labels[control])
        ax.set(xlabel='Full ring round (zero-based)',ylabel=metric,title='Three-round clean owner-gradient CART comparison')
        ax.set_xticks(range(3));
        if metric!='full_test_loss':ax.set_ylim(0,1)
        ax.legend();fig.tight_layout();fig.savefig(PLOTS/f'{name}.png',dpi=160);plt.close(fig)
    for control in CONTROLS:
        records=histories[control];path=PLOTS/control;path.mkdir()
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        matrix=np.array([r['globalTestEvaluationOnly']['perClassAccuracy'] for r in records])
        im=axes[0].imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='viridis')
        axes[0].set(xlabel='Class',ylabel='Sequential activation',title='Full-test class accuracy');axes[0].set_xticks(range(10))
        confusion=np.array(records[-1]['globalTestEvaluationOnly']['confusion'])
        axes[1].imshow(confusion/1000,vmin=0,vmax=1,cmap='viridis')
        axes[1].set(xlabel='Predicted class',ylabel='True class',title='Final Node-9 confusion (row proportions)')
        axes[1].set_xticks(range(10));axes[1].set_yticks(range(10));fig.colorbar(im,ax=axes.ravel().tolist())
        fig.suptitle(labels[control]);fig.savefig(path/'class_accuracy_and_final_confusion.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(13,5));epochs=[]
        for r in records:
            epochs.append(r['beforeTraining']['trainingAnalysis']['perClassAccuracy'])
            epochs.extend(e['trainingAnalysis']['perClassAccuracy'] for e in r['epochs'])
        im=ax.imshow(np.asarray(epochs).T,aspect='auto',vmin=0,vmax=1,cmap='viridis')
        ax.set(xlabel='Six stages/activation: before and epochs 1–5',ylabel='Class',title=labels[control]+' — TRAIN analysis only')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path/'five_epoch_training_class_retention.png',dpi=160);plt.close(fig)
        rows=summaries[control]['rounds']
        with (RESULT/control/'round_metrics.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(controls=summaries,pairedDataAndOwnerSamples=True,
        configurationDifferences=verify_configs(run['device']),batchTraceChecks=3000,
        checkpointSnapshotChecks=100,initialModelHash=a['initialModelHash'],partitionHash=a['partitionHash'])
    write_json(RESULT/'summary.json',summary)
    table=['| Round | SS Node 9 | Predecessor Node 9 | SS mean | Predecessor mean | SS worst | Predecessor worst |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for i in range(3):
        x,y=(summaries[c]['rounds'][i] for c in CONTROLS)
        table.append(f"| {i} | {100*x['full_test_accuracy']:.2f}% | {100*y['full_test_accuracy']:.2f}% | {100*x['mean_node_accuracy']:.2f}% | {100*y['mean_node_accuracy']:.2f}% | {100*x['worst_node_accuracy']:.2f}% | {100*y['worst_node_accuracy']:.2f}% |")
    text='# Three-round owner-gradient CART: Snapshot Selection ablation\n\n'
    text+='Both fresh controls completed 3 rounds, 30 full-data activations, 1,500 active SGD updates and 11,250 older-owner gradient responses. No helper SGD, attacks, noise, EBM, W&B or production queue.\n\n'
    text+='\n'.join(table)+'\n\n'
    text+='## Paired protocol and objective\n\n'
    text+='`mean_{c in available owners} CE_c(theta) + mu_epoch/2 * ||theta - theta_input||²`. Round 0 uses only already visited earlier owners; Rounds 1–2 use all ten classes including the current owner. Helper batches contain 16 training examples; the active node still completes five full passes over 5,000 examples with batch 512 and last batch 392. Strict parameter replacement, SGD reset, LR 0.05/(1+0.05r), S=5 and CART anchor cap 0.01 are shared.\n\n'
    text+='SS chooses minimum receiver-local CE with nearest-predecessor ties. The SS-off diagnostic chooses the immediate predecessor, not an average. Both evaluate candidates on the same receiver-local TRAIN batch. Anchor confidence uses the actual chosen input loss, so its numeric value can differ as a consequence of selection even though the controller rule is identical. This tests the whole selection-policy effect within CART, not an anchor-coefficient-isolated contrast.\n\n'
    text+='All 3,000 batch traces have current owner-gradient parameter hashes. Initial/partition hashes, device policy, shuffle/augmentation keys, owner indices/keys match across controls. Final evidence checkpoints contain 50 rolling snapshots/control, all Round 2, with 100 matching snapshot checks. No automatic resume was tested or offered. Test images are evaluation-only; the balanced 640-image TRAIN probe is not held out from ordinary SGD and never controls training/selection.\n\n'
    text+='## Final per-class accuracy\n\n| Class | BASIL SS | Immediate predecessor |\n|---|---:|---:|\n'
    names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    for c,name in enumerate(names):
        text+=f"| {name} | {100*summaries['basil_ss']['finalPerClassAccuracy'][c]:.2f}% | {100*summaries['immediate_predecessor']['finalPerClassAccuracy'][c]:.2f}% |\n"
    text+='\n## Execution\n\n| Control | Runtime (min) | Initial snapshots selected | Numerical failures | OOM | Peak TensorFlow VRAM (MiB) |\n|---|---:|---:|---:|---:|---:|\n'
    for c in CONTROLS:
        s=states[c];text+=f"| {c} | {s['runtimeSeconds']/60:.2f} | {summaries[c]['initialSnapshotSelections']} | {s['numericalFailures']} | {s['oomEvents']} | {s['memorySafety']['maximumGpuPeakMiB']:.2f} |\n"
    text+='\n## Limitations\n\nSingle seed, three-round diagnostic, additional clean owner-gradient supervision simulated in one process; not plain BASIL or a privacy/Byzantine guarantee. Finite execution is not learning success. No outcome-based tuning, thresholds or automatic longer-run promotion. The approved BASIL baseline and all earlier evidence remain unchanged.\n'
    (RESULT/'OWNER_SELECTION_REVIEW.md').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();report()
