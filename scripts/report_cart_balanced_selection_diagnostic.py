#!/usr/bin/env python3
"""Validate and report the clean fixed-anchor selection pair. No training."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_records(records, control, *, rounds=3):
    import math
    from scripts.run_cart_balanced_selection_diagnostic import CONTROLS
    from scripts.report_cart_owner_selection_diagnostic import summarize
    if control not in CONTROLS:
        raise ValueError('Unknown clean control.')
    score = 'receiverLocalLoss' if control == CONTROLS[0] else 'balancedTrainingLoss'
    summary = summarize(records, control, selection_rule=score, rounds=rounds)
    for record in records:
        owners = sorted(record['olderClasses']+[record['node']])
        decision = record['selectionDecision']
        if decision['availableOwners'] != owners or decision['sourceSplit'] != 'train' or decision['selectedBy'] != score:
            raise ValueError('Wrong selection owner availability or evidence split.')
        if record['snapshotAnchorMu'] != .01 or len(record['cartControl']) != 6:
            raise ValueError('Fixed five-epoch anchor controller missing.')
        if any(c['mu'] != .01 or c['rule'] != 'fixed_frozen_selected_snapshot' for c in record['cartControl']):
            raise ValueError('Anchor is not fixed.')
        if any(not math.isclose(s['snapshotAnchorMu'], .01, rel_tol=1e-6) for s in record['optimizerTelemetry']):
            raise ValueError('Applied optimizer anchor differs.')
        for candidate in record['selection']:
            losses = candidate['ownerLosses']
            if sorted(map(int, losses)) != owners or not math.isclose(candidate['balancedTrainingLoss'],
                    math.fsum(losses.values())/len(owners), rel_tol=1e-12):
                raise ValueError('Balanced selection loss/coverage differs.')
            if candidate['receiverLocalLoss'] != losses[str(record['node'])]:
                raise ValueError('Receiver-local selection loss differs.')
        selected = next(s for s in record['selection'] if s['senderId'] == record['selectedSender'])
        if selected['parameterHash'] != record['inputHash']:
            raise ValueError('Selected parameters do not exactly initialize training.')
    return summary


def trace_signature(path, expected_steps):
    """Streaming digest of actual paired samples; never load a full trace in RAM."""
    import hashlib
    digest = hashlib.sha256()
    count = 0
    with path.open() as handle:
        for i, line in enumerate(handle):
            trace = json.loads(line)
            owner = trace['ownerGradientContext']
            if owner['modelHash'] != trace['parameterHashBefore'] or any(
                    o['parameterHash'] != trace['parameterHashBefore'] for o in owner['owners']):
                raise ValueError('Stale owner response.')
            if (trace['round_id'], trace['node_id']) != divmod(i//50, 10):
                raise ValueError('Invalid actual batch order.')
            signature = (trace['indicesHash'], trace['augmentationKey'], trace['imagesHash'],
                         [(o['owner'], o['trainingIndices'], o['augmentationKey']) for o in owner['owners']])
            digest.update(json.dumps(signature, separators=(',', ':')).encode()+b'\n')
            count += 1
    if count != expected_steps:
        raise ValueError('Missing actual optimizer traces.')
    return digest.hexdigest()


def load_records(directory, rounds, stream=False):
    if not stream:
        return json.loads((directory/'activations.json').read_text())
    paths = sorted((directory/'activations').glob('*.json'))
    if [p.name for p in paths] != [f'{i:04d}.json' for i in range(rounds*10)]:
        raise ValueError('Missing or extra activation evidence files.')
    return [json.loads(p.read_text()) for p in paths]


def report(*, study=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if study is None:
        from scripts import run_cart_balanced_selection_diagnostic as study
    RESULT, PLOTS, CONTROLS = study.RESULT, study.PLOTS, study.CONTROLS
    rounds = getattr(study, 'ROUNDS', 3)
    from basil_core.research_protocol import params_hash, keyed_rng
    from scripts.run_cart_feature_diagnostic import write_json
    run = json.loads((RESULT/'run.json').read_text())
    if run['status'] != 'completed' or run['activationsCompleted'] != rounds*20 or run['optimizerSteps'] != rounds*1000:
        raise ValueError('Incomplete pair; no missing results fabricated.')
    if PLOTS.exists() or (RESULT/'summary.json').exists():
        raise ValueError('Refusing report overwrite.')
    summaries, histories, states, paired = {}, {}, {}, []
    labels = np.load(ROOT/'experiments/cache/cifar10/y_train_int32.npy', mmap_mode='r')
    expected_probes = {str(c): keyed_rng(2025,'selection_batch',c).permutation(np.flatnonzero(labels==c))[:512].tolist()
                       for c in range(10)}
    for control in CONTROLS:
        directory = RESULT/control
        state = json.loads((directory/'run.json').read_text())
        if state['status'] != 'completed' or state['roundsCompleted'] != rounds:
            raise ValueError('Incomplete control.')
        records = load_records(directory, rounds, getattr(study, 'STREAM_ACTIVATIONS', False))
        summaries[control] = validate_records(records, control, rounds=rounds)
        histories[control], states[control] = records, state
        for r in records:
            if r['selectionDecision']['selectionProbeIndices'] != {
                    str(c): expected_probes[str(c)] for c in r['selectionDecision']['availableOwners']}:
                raise ValueError('Wrong deterministic TRAIN selection batch.')
        paired.append(trace_signature(directory/'audit/batch_trace.jsonl', rounds*500))
        with np.load(directory/'checkpoint.npz', allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            if metadata['nextActivation'] != rounds*10 or len(metadata['snapshots']) != 50:
                raise ValueError('Incomplete rolling-memory evidence.')
            for s in metadata['snapshots']:
                weights = [saved[f"r{s['receiver']}s{s['sender']}w{j}"] for j in range(10)]
                if s['round'] != rounds-1 or params_hash(weights) != s['hash']:
                    raise ValueError('Checkpoint snapshot checksum differs.')
    a,b = (states[c] for c in CONTROLS)
    if paired[0] != paired[1] or any(a[k] != b[k] for k in ('partitionHash','initialModelHash','sourceHashes','numericalExecution')):
        raise ValueError('Unpaired data, helpers, initialization or execution.')
    del paired
    PLOTS.mkdir(parents=True)
    names = ['Receiver-local SS','Balanced-owner TRAIN SS']
    colors = ['#0072B2','#D55E00']
    for metric in ('full_test_accuracy','mean_node_accuracy','worst_node_accuracy','full_test_loss'):
        fig, ax = plt.subplots(figsize=(8,4))
        for control,name,color in zip(CONTROLS,names,colors):
            rows = summaries[control]['rounds']
            scale = 1 if metric == 'full_test_loss' else 100
            ax.plot([r['round'] for r in rows],[scale*r[metric] for r in rows],marker='o',label=name,color=color)
        ax.set(xlabel='Full ring round (zero-based)',ylabel=metric+(' (CE)' if metric == 'full_test_loss' else ' (%)'),
               title=f'{rounds}-round clean non-IID CART; fixed μ=0.01')
        if metric != 'full_test_loss': ax.set_ylim(0,100)
        if rounds == 3: ax.set_xticks([0,1,2])
        ax.legend(); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(PLOTS/f'{metric}_comparison.png',dpi=160); plt.close(fig)
    for control,name in zip(CONTROLS,names):
        records = histories[control]
        path = PLOTS/control; path.mkdir()
        fig, ax = plt.subplots(figsize=(11,4))
        matrix = np.asarray([r['globalTestEvaluationOnly']['perClassAccuracy'] for r in records])
        im=ax.imshow(matrix.T,aspect='auto',vmin=0,vmax=1,cmap='viridis')
        ax.set(xlabel='Sequential activation index',ylabel='CIFAR-10 class',title=name+' — full TEST, evaluation only')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path/'activation_per_class_accuracy.png',dpi=160);plt.close(fig)
        fig, ax = plt.subplots(figsize=(7,5))
        im=ax.imshow(np.asarray([r['globalTestEvaluationOnly']['perClassAccuracy'] for r in records[-10:]]),vmin=0,vmax=1,cmap='viridis')
        ax.set(xlabel='Class',ylabel='Node',title=name+' — final node/class accuracy')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path/'final_node_class_accuracy.png',dpi=160);plt.close(fig)
        rows=summaries[control]['rounds']
        with (RESULT/control/'round_metrics.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        # TRAIN probes are not held out from ordinary SGD. They are diagnostic
        # only; full TEST is measured separately after each activation.
        epoch_rows = []
        for r in records:
            stages = [r['beforeTraining']['trainingAnalysis']]+[e['trainingAnalysis'] for e in r['epochs']]
            for stage, metrics in enumerate(stages):
                epoch_rows.append(dict(round=r['round'],node=r['node'],epoch=stage,
                    train_probe_accuracy=metrics['accuracy'],
                    **{f'class_{c}_accuracy':v for c,v in enumerate(metrics['perClassAccuracy'])}))
        with (RESULT/control/'epoch_training_probe_metrics.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(epoch_rows[0]));writer.writeheader();writer.writerows(epoch_rows)
        fig, ax = plt.subplots(figsize=(12,4))
        matrix = np.asarray([[e[f'class_{c}_accuracy'] for c in range(10)] for e in epoch_rows])
        im=ax.imshow(matrix.T,aspect='auto',vmin=0,vmax=1,cmap='viridis')
        ax.set(xlabel='Activation × 6 stages: before, epochs 1–5',ylabel='Class',
               title=name+' — TRAIN probe, NOT held-out TEST')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path/'epoch_training_probe_heatmap.png',dpi=160);plt.close(fig)
        fig, axes = plt.subplots(2,5,figsize=(15,6),sharex=True,sharey=True)
        for cls, ax in enumerate(axes.flat):
            ax.plot([r['round'] for r in rows],[100*r[f'class_{cls}_accuracy'] for r in rows])
            ax.set(title=f'Class {cls}',ylim=(0,100),xlabel='Round',ylabel='Test accuracy (%)')
        fig.suptitle(name+' — Node-9 full TEST');fig.tight_layout()
        fig.savefig(path/'per_class_accuracy_vs_round.png',dpi=160);plt.close(fig)
        fig, axes = plt.subplots(2,5,figsize=(15,6))
        for r, ax in zip(records[-10:],axes.flat):
            ax.imshow(r['globalTestEvaluationOnly']['confusion'],vmin=0,vmax=1000,cmap='Blues')
            ax.set(title=f"Node {r['node']}",xlabel='Predicted',ylabel='True')
        fig.suptitle(name+' — final TEST confusion counts (scale 0–1000)');fig.tight_layout()
        fig.savefig(path/'final_node_confusion_matrices.png',dpi=160);plt.close(fig)
    summary=dict(controls=summaries,pairedActualDataAndOwnerKeys=True,appliedFixedMu=.01,
                 batchTraceChecks=rounds*1000,checkpointSnapshotChecks=100,
                 partitionHash=a['partitionHash'],initialModelHash=a['initialModelHash'])
    write_json(RESULT/'summary.json',summary)
    text='# Clean non-IID CART: balanced Snapshot Selection\n\n'
    text+=f'{rounds}-round full-data diagnostic; not research-valid production evidence. No attack, noise, EBM, W&B or queue run. All five local epochs execute.\n\n'
    text+='| Round | Local SS Node 9 | Balanced SS Node 9 | Local mean | Balanced mean | Local worst | Balanced worst |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in range(rounds):
        x,y=(summaries[c]['rounds'][r] for c in CONTROLS)
        text+='| '+str(r)+' | '+' | '.join(f'{100*v:.2f}%' for v in (x['full_test_accuracy'],y['full_test_accuracy'],x['mean_node_accuracy'],y['mean_node_accuracy'],x['worst_node_accuracy'],y['worst_node_accuracy']))+' |\n'
    text+='\n## What changed and what stayed fixed\n\n'
    text+='Both optimize equal-class mean CE from current/local and available owner gradients, plus `0.01/2 * ||theta - selected_snapshot||²`. The reference is frozen for all five epochs. The new selector minimizes mean TRAIN CE across already-visited owners plus the active owner. The control minimizes active-receiver TRAIN CE. Both retain S=5 and nearest-predecessor ties, with complete selected-weight replacement. Neither averages models. This is an additional CART selection mechanism, **not the paper BASIL rule**.\n\n'
    text+=f'All {rounds*1000:,} executed steps, 100 cached-snapshot hashes, paired data/augmentation/helper keys and immutable anchors are checked. Each activation uses 5×5,000 local examples, 50 optimizer steps, and final batches of 392. Helper gradients use 16 TRAIN images per older owner per update, with no helper optimizer. Selection uses fixed unaugmented 512-image owner TRAIN probes; these are not held out from local SGD. R0 exposes only visited owners; subsequent rounds expose all ten. Full 10,000-image TEST evaluations never control training or selection.\n\n'
    text+='| CIFAR class | Local SS | Balanced SS |\n|---|---:|---:|\n'
    for c,name in enumerate(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']):
        text+=f"| {name} | {100*summaries[CONTROLS[0]]['finalPerClassAccuracy'][c]:.2f}% | {100*summaries[CONTROLS[1]]['finalPerClassAccuracy'][c]:.2f}% |\n"
    text+='\n| Control | Runtime (min) | Initial snapshots selected | NaN/Inf failures | OOM | Peak TF VRAM (MiB) |\n|---|---:|---:|---:|---:|---:|\n'
    for c in CONTROLS:
        s=states[c]
        text+=f"| {c} | {s['runtimeSeconds']/60:.2f} | {summaries[c]['initialSnapshotSelections']} | {s['numericalFailures']} | {s['oomEvents']} | {s['memorySafety']['maximumGpuPeakMiB']:.2f} |\n"
    text+='\n## Learning versus forgetting\n\nPeak values below describe the saved trajectory, not a selection/tuning criterion. A peak followed by a lower final value is loss of observed class performance; it alone does not establish the cause. TRAIN epoch probes and full TEST activation evaluations are different evidence.\n\n'
    text+='| Control | Class | Peak TEST (any trained activation) | Final Node-9 TEST | Largest non-owner within-activation TRAIN drop |\n|---|---:|---:|---:|---:|\n'
    for c in CONTROLS:
        records=histories[c]
        for cls in range(10):
            peak=max(r['globalTestEvaluationOnly']['perClassAccuracy'][cls] for r in records)
            final=records[-1]['globalTestEvaluationOnly']['perClassAccuracy'][cls]
            drop=max(r['beforeTraining']['trainingAnalysis']['perClassAccuracy'][cls]-
                     r['epochs'][-1]['trainingAnalysis']['perClassAccuracy'][cls] for r in records if r['node']!=cls)
            text+=f'| {c} | {cls} | {100*peak:.2f}% | {100*final:.2f}% | {100*drop:.2f} pp |\n'
    text+='\n## Limits\n\nA single seed does not establish universal improvement or statistical significance. This comparison isolates selection with fixed μ; older adaptive-anchor results are a different diagnostic. Owner helpers and selection are clean simulated auxiliary services, not a demonstrated Byzantine-secure protocol or privacy guarantee. The snapshot-only Hidden attack does not defend these added channels. Full A/B/C/D remains gated on a declared threat boundary. No test-driven tuning or automatic longer-run promotion is performed. Evidence checkpoints are saved after activations; automatic exact resume is not implemented for this diagnostic.\n'
    (RESULT/'BALANCED_SELECTION_REVIEW.md').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    report()
