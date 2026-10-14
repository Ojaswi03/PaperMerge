#!/usr/bin/env python3
"""Saved-data-only review of the bounded owner-gradient CART diagnostic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DEFAULT = ROOT/'newResults/nonIID/cart_owner_gradients/clean_one_traversal_16_v1'


def summarize(records):
    import numpy as np
    if len(records)!=10: raise ValueError('Ten complete activations required; never fill missing nodes.')
    rows = []; response_bytes = 0; responses = 0
    for node,record in enumerate(records):
        if record['node']!=node or record['round']!=0 or record['steps']!=50:
            raise ValueError('First traversal activation/optimizer counts differ.')
        if record['inputHash']!=record['snapshotAnchorReferenceHash']:
            raise ValueError('Selected snapshot/anchor hash mismatch.')
        if len(record['epochs'])!=5 or any(e['samplesSeen']!=5000 or e['batchSizes']!=[512]*9+[392] for e in record['epochs']):
            raise ValueError('True full-epoch contract failed.')
        if len(record['ownerGradients'])!=50 or record['helperOptimizerSteps']!=0 or record['rawImagesTransferred']:
            raise ValueError('Owner-response contract failed.')
        for step,owner in enumerate(record['ownerGradients']):
            if owner['optimizerStep']!=step or owner['availableClasses']!=list(range(node+1)):
                raise ValueError('Stale/missing/future owner context.')
            if [r['owner'] for r in owner['owners']]!=list(range(node)):
                raise ValueError('Older owner identities differ.')
            if any(r['support']!=16 or r['parameterHash']!=owner['modelHash'] for r in owner['owners']):
                raise ValueError('Helper batch or synchronized state differs.')
            if owner['ownerOptimizerSteps']!=0 or owner['modelAveraging'] or owner['rawImagesReturned']:
                raise ValueError('Gradient service changed execution semantics.')
            response_bytes+=owner['responseCoordinateBytes']; responses+=owner['extraOwnerResponses']
        for step in record['optimizerTelemetry']:
            if not all(np.isfinite(v) for v in step.values() if isinstance(v,(int,float))):
                raise ValueError('Non-finite optimizer telemetry.')
            if step['ebmCoefficient']!=0 or step['ebmPenalty']!=0: raise ValueError('Unexpected EBM.')
        before=record['beforeTraining']['trainingAnalysis']; after=record['epochs'][-1]['trainingAnalysis']
        classes=np.asarray(after['perClassAccuracy']); previous=np.asarray(before['perClassAccuracy'])
        rows.append(dict(node=node,globalTestAccuracy=record['globalTestEvaluationOnly']['accuracy'],
            trainingProbeAccuracy=after['accuracy'],ownClass=classes[node],
            otherClasses=float(np.mean(np.delete(classes,node))),
            meanOtherClassForgetting=float(np.mean(np.delete(previous-classes,node))),
            olderClassAccuracy=float(classes[:node].mean()) if node else None,
            selectedSender=record['selectedSender'],responseCoordinateBytes=record['responseCoordinateBytes']))
    accuracies=[r['globalTestAccuracy'] for r in rows]
    return dict(nodes=rows,node9FullTestAccuracy=accuracies[-1],meanNodeFullTestAccuracy=float(np.mean(accuracies)),
        worstNodeFullTestAccuracy=min(accuracies),finalPerClass=records[-1]['globalTestEvaluationOnly']['perClassAccuracy'],
        optimizerSteps=500,helperOptimizerSteps=0,extraOwnerResponses=responses,
        responseCoordinateBytes=response_bytes,rawImagesTransferred=False,checksPassed=True)


def report(result=DEFAULT):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.run_cart_feature_diagnostic import write_json
    result=Path(result).resolve()
    if result!=DEFAULT.resolve(): raise ValueError('Only the bounded authorized result path is supported.')
    run=json.loads((result/'run.json').read_text())
    if run['status']!='completed' or run['activationsCompleted']!=10 or run['optimizerSteps']!=500:
        raise ValueError('Learning run incomplete; no final result fabricated.')
    records=json.loads((result/'activations.json').read_text()); summary=summarize(records)
    plots=ROOT/'newPlots/nonIID/cart_owner_gradients/clean_one_traversal_16_v1'
    if plots.exists() or (result/'summary.json').exists(): raise ValueError('Refusing review overwrite.')
    plots.mkdir(parents=True)
    fig,ax=plt.subplots(figsize=(9,5))
    for key,label in [('globalTestAccuracy','Full test'),('trainingProbeAccuracy','TRAIN analysis probe')]:
        ax.plot(range(10),[r[key] for r in summary['nodes']],marker='o',label=label)
    ax.set(xlabel='Active node (first traversal)',ylabel='Accuracy',ylim=(0,1),title='Owner-gradient CART: bounded clean traversal')
    ax.axhline(.1,color='gray',linestyle=':',label='Balanced chance reference'); ax.legend(); fig.tight_layout()
    fig.savefig(plots/'accuracy_by_activation.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    matrix=np.array([r['globalTestEvaluationOnly']['perClassAccuracy'] for r in records])
    image=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='viridis')
    ax.set(xlabel='CIFAR-10 class',ylabel='Active node after training',title='Full-test node-by-class accuracy')
    ax.set_xticks(range(10)); ax.set_yticks(range(10)); fig.colorbar(image,ax=ax); fig.tight_layout()
    fig.savefig(plots/'node_class_accuracy.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    epoch_matrix=[]
    for r in records:
        epoch_matrix.append(r['beforeTraining']['trainingAnalysis']['perClassAccuracy'])
        epoch_matrix.extend(e['trainingAnalysis']['perClassAccuracy'] for e in r['epochs'])
    image=ax.imshow(np.asarray(epoch_matrix).T,aspect='auto',vmin=0,vmax=1,cmap='viridis')
    ax.set(xlabel='Six stages per node: before, then epochs 1–5',ylabel='Class',title='TRAIN-only analysis: class retention within five epochs')
    ax.set_yticks(range(10)); fig.colorbar(image,ax=ax); fig.tight_layout()
    fig.savefig(plots/'five_epoch_class_retention.png',dpi=160); plt.close(fig)
    write_json(result/'summary.json',summary)
    rows=['| Node | Full test | TRAIN probe | Own class (TRAIN) | Other classes (TRAIN) | Older classes (TRAIN) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary['nodes']:
        older='unavailable' if r['olderClassAccuracy'] is None else f"{100*r['olderClassAccuracy']:.2f}%"
        rows.append(f"| {r['node']} | {100*r['globalTestAccuracy']:.2f}% | {100*r['trainingProbeAccuracy']:.2f}% | {100*r['ownClass']:.2f}% | {100*r['otherClasses']:.2f}% | {older} |")
    text='# Owner-gradient CART: bounded clean learning review\n\n'
    text+='One full traversal, ten full-data activations, five full epochs each, 500 active SGD updates. No attacks/noise/EBM/W&B. Not a production or Byzantine-robustness result.\n\n'
    text+='\n'.join(rows)+'\n\n'
    text+=f"Node-9 full test: **{100*summary['node9FullTestAccuracy']:.2f}%**; mean node: **{100*summary['meanNodeFullTestAccuracy']:.2f}%**; worst node: **{100*summary['worstNodeFullTestAccuracy']:.2f}%**. Runtime: {run['runtimeSeconds']:.1f}s.\n\n"
    text+=f"Extra helper responses: {summary['extraOwnerResponses']}; gradient-coordinate response traffic: {summary['responseCoordinateBytes']/1024**2:.2f} MiB (excludes outbound parameter requests and serialization). Helpers performed zero SGD updates.\n\n"
    text+='The previous feature-replay/preservation comparison stayed at 10% after 30 rounds. This is a different auxiliary-supervision protocol and a one-traversal diagnostic, not a matched 30-round performance claim. Balanced TRAIN probes are analysis only, not held out from ordinary SGD. Test data never supply helper gradients, SS losses, anchor strength or stopping decisions.\n\n'
    text+='## Objective\n\n`mean_c CE_c(theta) + mu_epoch/2 * ||theta - theta_selected||²`, over the current class and already visited owner classes. Every helper response is computed at the active model’s current weights, with 16 owner-local augmented TRAIN examples. The selected starting snapshot remains immutable; only the active node applies the gradient.\n\n'
    text+='## Limits\n\nSingle seed; tiny helper minibatches; additional clean gradient communication; owner computation simulated in one process. No Byzantine filtering/privacy guarantee for helpers, no channel-noise evidence, no automatic resume or promotion, and no claim of attaining the 50% research target.\n'
    (result/'OWNER_GRADIENT_LEARNING_REVIEW.md').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--result',type=Path,default=DEFAULT)
    report(parser.parse_args().result)
