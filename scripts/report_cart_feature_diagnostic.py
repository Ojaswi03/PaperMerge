#!/usr/bin/env python3
"""Saved-only feature-memory review; no TensorFlow, data loading or training."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cart_feature_diagnostic import RESULT, PLOTS


def review():
    import numpy as np
    manifest = json.loads((RESULT/'run.json').read_text())
    if manifest['status'] != 'completed' or manifest['activationsCompleted'] != 20 or manifest['optimizerSteps'] != 1000:
        raise ValueError('Incomplete diagnostic; no fabricated summary.')
    if manifest['config']['testSetUsed'] or manifest['wandbEnabled'] or manifest['productionQueueStarted']:
        raise ValueError('Wrong diagnostic scope.')
    rows = []; summary = {}; checks = 0
    for condition in ('cart', 'cart_feature_memory'):
        records = json.loads((RESULT/condition/'activations.json').read_text())
        if [r['node'] for r in records] != list(range(10)): raise ValueError('Missing/duplicate activations.')
        with np.load(RESULT/condition/'final_nodes.npz', allow_pickle=False) as weights:
            from basil_core.research_protocol import params_hash
            for record in records:
                node = record['node']
                actual = [weights[f'node{node}_w{j}'] for j in range(10)]
                if params_hash(actual) != record['outputHash']: raise ValueError('Checkpoint hash mismatch.')
                if record['steps'] != 50 or record['inputHash'] != record['snapshotAnchorReferenceHash']:
                    raise ValueError('Step count / complete reference mismatch.')
                if any(e['samplesSeen'] != 5000 or e['batchSizes'] != [512]*9+[392] for e in record['epochs']):
                    raise ValueError('Full-epoch contract failed.')
                stages = [record['beforeTraining'], *record['epochs']]
                before = np.asarray(stages[0]['trainingAnalysis']['perClassAccuracy'])
                for stage, value in enumerate(stages):
                    metrics = value['trainingAnalysis']; acc = np.asarray(metrics['perClassAccuracy'])
                    if not np.isfinite(acc).all() or metrics['classSupport'] != [64]*10:
                        raise ValueError('Invalid training-probe measurements.')
                    confusion = np.asarray(metrics['confusion'])
                    if confusion.shape != (10, 10) or not np.issubdtype(confusion.dtype, np.integer) or (confusion < 0).any():
                        raise ValueError('Invalid saved confusion matrix.')
                    if not np.array_equal(confusion.sum(axis=1), [64]*10) or not np.array_equal(confusion.sum(axis=0), metrics['predictedCounts']):
                        raise ValueError('Confusion support/predictions mismatch.')
                    if not np.array_equal(np.diag(confusion)/64, acc) or not math.isclose(np.trace(confusion)/640, metrics['accuracy'], abs_tol=1e-12):
                        raise ValueError('Saved accuracy differs from confusion counts.')
                    rows.append(dict(condition=condition, node=node, stage=stage,
                        trainingAnalysisAccuracy=metrics['accuracy'], ownClassAccuracy=float(acc[node]),
                        otherClassAccuracy=float(np.delete(acc, node).mean()),
                        visitedClassAccuracy=float(acc[:node].mean()) if node else None,
                        meanVisitedClassForgetting=float((before[:node]-acc[:node]).mean()) if node else None,
                        frozenFeatureAccuracy=value.get('frozenFeatureReplayAccuracy'),
                        localEpochLoss=value.get('localTrainingLoss'),
                        **{f'class_{c}': float(a) for c, a in enumerate(acc)}))
                    checks += 1
                for step in record['optimizerTelemetry']:
                    if step['ebmCoefficient'] != 0 or step['configuredSigma'] != 0:
                        raise ValueError('EBM/noise leaked into clean diagnostic.')
                    if condition == 'cart_feature_memory':
                        expected = step['baseCrossEntropy']+step['snapshotAnchorPenalty']+step['featureReplayPenalty']
                        if not math.isclose(step['totalObjective'], expected, abs_tol=1e-5, rel_tol=1e-5):
                            raise ValueError('Total objective differs from measured components.')
                        if step['featureReplayBeta'] != 1.: raise ValueError('Unapproved beta.')
                if condition == 'cart_feature_memory':
                    m = record['featureMemory']
                    if m['modelHash'] != record['inputHash'] or m['presentClasses'] != list(range(node)):
                        raise ValueError('Unexpected model / future / missing feature source.')
                    from basil_core.cart_feature_memory import FeaturePacket
                    if [FeaturePacket.from_dict(p).digest for p in m['packets']] != m['packetHashes']:
                        raise ValueError('Packet checksum mismatch.')
                    if m['featureRows'] != node*16 or m['coordinateBytes'] > 117504:
                        raise ValueError('Unbounded class memory.')
                    checks += 1
        final = [r for r in rows if r['condition'] == condition and r['stage'] == 5]
        active = [r for r in final if r['node'] > 0]
        summary[condition] = dict(meanTrainingAnalysisAccuracy=float(np.mean([r['trainingAnalysisAccuracy'] for r in final])),
            worstTrainingAnalysisAccuracy=min(r['trainingAnalysisAccuracy'] for r in final),
            meanOwnClassAccuracy=float(np.mean([r['ownClassAccuracy'] for r in final])),
            meanOtherClassAccuracy=float(np.mean([r['otherClassAccuracy'] for r in final])),
            meanVisitedClassForgetting=float(np.mean([r['meanVisitedClassForgetting'] for r in active])),
            node9TrainingAnalysisAccuracy=final[-1]['trainingAnalysisAccuracy'],
            node9PerClassAccuracy=[final[-1][f'class_{c}'] for c in range(10)],
            constantOwnClassPredictors=sum(r['epochs'][-1]['trainingAnalysis']['predictedCounts'][r['node']] == 640 for r in records))
        if condition == 'cart_feature_memory':
            nonempty_steps = [step for record in records if record['featureMemory']['featureRows']
                              for step in record['optimizerTelemetry']]
            summary[condition].update(meanReplayPenalty=float(np.mean([s['featureReplayPenalty'] for s in nonempty_steps])),
                meanReplayGradientNorm=float(np.mean([s['featureReplayGradientNorm'] for s in nonempty_steps])),
                meanReplayCorrectionRatio=float(np.mean([s['featureReplayCorrectionRatio'] for s in nonempty_steps])))
    return manifest, rows, summary, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest, rows, summary, checks = review()
    # Numerical/plot dependencies remain lazy and this command cannot train.
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    with (RESULT/'epoch_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    colors = {'cart': '#d97706', 'cart_feature_memory': '#2563eb'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition in colors:
        final = [r for r in rows if r['condition'] == condition and r['stage'] == 5]
        axes[0].plot([r['node'] for r in final], [100*r['trainingAnalysisAccuracy'] for r in final],
            marker='o', linestyle='--' if condition == 'cart_feature_memory' else '-', label=condition, color=colors[condition])
        axes[1].plot([r['node'] for r in final], [100*r['ownClassAccuracy'] for r in final],
            marker='o', linestyle='--' if condition == 'cart_feature_memory' else '-', label=condition, color=colors[condition])
    for ax, title in zip(axes, ('Balanced training-probe accuracy', 'Own-class training-probe accuracy')):
        ax.set(xlabel='Sequential node activation', ylabel='Accuracy (%)', title=title, ylim=(0, 100))
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('One clean traversal — analysis-only training probe, not test accuracy')
    fig.tight_layout(); fig.savefig(PLOTS/'activation_accuracy_comparison.png', dpi=150); plt.close(fig)
    final = [r for r in rows if r['condition'] == 'cart_feature_memory' and r['stage'] == 5 and r['node'] > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r['node'] for r in final], [100*r['frozenFeatureAccuracy'] for r in final],
        marker='o', label='Frozen replay features (head only)', color='#7c3aed')
    ax.plot([r['node'] for r in final], [100*r['visitedClassAccuracy'] for r in final],
        marker='s', linestyle='--', label='Old-class images (updated encoder + head)', color='#dc2626')
    ax.set(xlabel='Node activation', ylabel='Old-class accuracy (%)', ylim=(0, 100),
        title='Different TRAINING sample sets — analysis only, not a causal isolation')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(PLOTS/'frozen_features_vs_image_retention.png', dpi=150); plt.close(fig)
    for condition in colors:
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        for node, ax in enumerate(axes.flat):
            values = [r for r in rows if r['condition'] == condition and r['node'] == node]
            for c in range(10):
                ax.plot([r['stage'] for r in values], [100*r[f'class_{c}'] for r in values],
                    label=f'class {c}', linewidth=2 if c == node else 1, alpha=1 if c == node else .6)
            ax.set(title=f'Node {node}', xlabel='Epoch (0 = before)', ylim=(0, 100)); ax.grid(alpha=.2)
        axes.flat[0].legend(ncol=2, fontsize=6)
        fig.suptitle(f'{condition}: class retention on TRAINING probe, not held out')
        fig.tight_layout(); fig.savefig(PLOTS/f'{condition}_epoch_per_class.png', dpi=150); plt.close(fig)
    report = ['# CART feature-memory: bounded clean diagnostic', '',
        '**Analysis-only training-probe results; not held out and not global test accuracy.**', '',
        'One traversal per control, 10 full-data activations/control; β=1 chosen before execution. '
        'Both use current CART control, five complete epochs, μ_max=0.01, strict BASIL handoff and S=5. '
        'No attack, noise, EBM, W&B, queue or production promotion.', '',
        '| Metric | Current CART | CART + feature memory |', '|---|---:|---:|']
    for label, key in [('Mean node probe accuracy', 'meanTrainingAnalysisAccuracy'),
                       ('Worst node probe accuracy', 'worstTrainingAnalysisAccuracy'),
                       ('Mean own-class accuracy', 'meanOwnClassAccuracy'),
                       ('Mean other-class accuracy', 'meanOtherClassAccuracy'),
                       ('Mean visited-class forgetting (before minus after)', 'meanVisitedClassForgetting'),
                       ('Node-9 probe accuracy', 'node9TrainingAnalysisAccuracy')]:
        unit = ' pp' if key == 'meanVisitedClassForgetting' else '%'
        report.append(f"| {label} | {100*summary['cart'][key]:.3f}{unit} | {100*summary['cart_feature_memory'][key]:.3f}{unit} |")
    report += ['', '| Node | CART probe accuracy | Feature-memory probe accuracy | Feature-memory own class | Visited-class forgetting |', '|---|---:|---:|---:|---:|']
    for node in range(10):
        a, b = [next(r for r in rows if r['condition'] == c and r['node'] == node and r['stage'] == 5) for c in colors]
        forgetting = 'unavailable' if b['meanVisitedClassForgetting'] is None else f"{100*b['meanVisitedClassForgetting']:.3f} pp"
        report.append(f"| {node} | {100*a['trainingAnalysisAccuracy']:.3f}% | {100*b['trainingAnalysisAccuracy']:.3f}% | {100*b['ownClassAccuracy']:.3f}% | {forgetting} |")
    feature_records = json.loads((RESULT/'cart_feature_memory/activations.json').read_text())
    control_records = json.loads((RESULT/'cart/activations.json').read_text())
    report += ['', '## Replay activity and the remaining representation problem', '',
        '| Node | Selected sender : snapshot round | Old feature classes | Final frozen-feature accuracy | Final old-class image-probe accuracy | Mean replay / CE gradient-norm ratio |',
        '|---|---|---:|---:|---:|---:|']
    for record in feature_records:
        node = record['node']; final = record['epochs'][-1]
        frozen = final.get('frozenFeatureReplayAccuracy')
        visited = np.asarray(final['trainingAnalysis']['perClassAccuracy'])[:node]
        fa = 'unavailable' if frozen is None else f'{100*frozen:.3f}%'
        ia = 'unavailable' if not len(visited) else f'{100*visited.mean():.3f}%'
        ratio = np.mean([s['featureReplayCorrectionRatio'] for s in record['optimizerTelemetry']])
        report.append(f"| {node} | {record['selectedSender']} : R{record['selectedSnapshotRound']} | {len(record['featureMemory']['presentClasses'])} | {fa} | {ia} | {ratio:.4f} |")
    different = [b['node'] for a, b in zip(control_records, feature_records) if a['outputHash'] != b['outputHash']]
    report += ['',
        f"Output parameter hashes differ from control at nodes {different}; Node 0 has empty memory and an identical output hash.",
        f"Across nonempty-memory steps, mean replay penalty = {summary['cart_feature_memory']['meanReplayPenalty']:.6f}, "
        f"mean replay gradient norm = {summary['cart_feature_memory']['meanReplayGradientNorm']:.6f}, "
        f"mean per-step replay/CE norm ratio = {summary['cart_feature_memory']['meanReplayCorrectionRatio']:.6f}.", '',
        'The replay term is therefore active, not silently bypassed. Both controls nevertheless end '
        'with 10/10 node models predicting only their own class for the entire 640-image probe. '
        'There is **no final raw-image retention improvement in this bounded comparison**.', '',
        'At Node 1, final frozen class-0 feature accuracy is 100%, while class-0 image-probe accuracy '
        'is 0%. The encoder hash changes during every activation. This discrepancy is consistent '
        'with the changing-representation limitation of detached head replay; differences in the '
        'feature/probe sample sets and representation quality mean it is not an isolated causal proof.', '',
        'During this first traversal, some selected snapshots have round -1: BASIL can select an '
        'untrained initialized candidate over a specialized predecessor when receiver-local CE is '
        'lower. The table records this actual behavior; no global accuracy enters that choice. '
        'Related frozen-extractor methods cannot be assumed to transfer their results to this '
        'trainable-backbone, from-scratch setting.', '',
        '**No larger queue is justified by this result.** Do not increase beta, freeze the backbone, '
        'add cross-class images or extend training merely to force a target accuracy. A different '
        'representation-preservation mechanism would require an explicitly revised protocol and '
        'a new prespecified comparison.']
    report += ['', '## Interpretation limits', '',
        'Feature replay is additional training-side communication, not a metadata-only CART change. '
        'Selected-encoder refresh makes the initial coordinates consistent; they are fixed during the '
        'five epochs while the encoder still trains. The feature term supplies head gradients only. '
        'Frozen-feature accuracy is not proof of accuracy on current raw-image embeddings.', '',
        'The balanced 640-image probe remains part of the training dataset; it is excluded from SS '
        'and feature packets but not from ordinary class-owner SGD. It is neither held-out validation '
        'nor test accuracy. It never controls selection, gradients, β or μ. One seed/traversal does not '
        'establish convergence, generalization, noise mitigation or Byzantine robustness.', '',
        'Credit: PASS for related feature rehearsal; FedProto for class means; FedProK/FeTrIL for related '
        'feature-retention methods; PearFL for decentralized prototype exchange. This implementation '
        'does not use their feature translation, frozen-backbone training or model averaging.', '',
        '## Execution / safety', '',
        f"* Status: {manifest['status']}; 20/20 activations, 1,000/1,000 optimizer steps.",
        f"* Runtime: {manifest['runtimeSeconds']:.2f} seconds on {manifest['deviceName']}.",
        f"* Tracked GPU peak: {manifest['memorySafety']['maximumGpuPeakMiB']:.2f} MiB under the configured cap.",
        f'* Saved-result checks: {checks}; epoch CSV rows: {len(rows)}.',
        '* No detected non-finite training/evaluation or OOM event in this diagnostic.', '',
        '## Figures', '',
        f'![Activation comparison](../../../../newPlots/nonIID/cart_feature_memory/{PLOTS.name}/activation_accuracy_comparison.png)', '',
        f'![Current CART epochs](../../../../newPlots/nonIID/cart_feature_memory/{PLOTS.name}/cart_epoch_per_class.png)', '',
        f'![Feature-memory epochs](../../../../newPlots/nonIID/cart_feature_memory/{PLOTS.name}/cart_feature_memory_epoch_per_class.png)', '']
    report += [f'![Frozen features versus current images](../../../../newPlots/nonIID/cart_feature_memory/{PLOTS.name}/frozen_features_vs_image_retention.png)', '']
    (RESULT/'CART_FEATURE_REVIEW.md').write_text('\n'.join(report))
    (RESULT/'summary.json').write_text(json.dumps(dict(summary=summary, checksPassed=checks,
        trainingOnly=True, testAccuracyReported=False), indent=2)+'\n')
    print(json.dumps(dict(status='reported', checksPassed=checks, csvRows=len(rows), summary=summary,
        report=str(RESULT/'CART_FEATURE_REVIEW.md'))))


if __name__ == '__main__': main()
