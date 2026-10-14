#!/usr/bin/env python3
"""Report saved same-image audits/representation diagnostics; cannot train."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cart_representation_diagnostic import AUDIT, RESULT, PLOTS, output_paths


def validate_observation(observation):
    import numpy as np
    if not observation['trainingOnly'] or not observation['analysisOnly'] or observation['heldOut']:
        raise ValueError('Invalid same-image audit usage.')
    for metrics in observation['interventions'].values():
        confusion = np.asarray(metrics['confusion'])
        if confusion.shape != (10, 10) or not np.issubdtype(confusion.dtype, np.integer) or (confusion < 0).any():
            raise ValueError('Invalid same-image confusion matrix.')
        if not np.array_equal(confusion.sum(axis=1), [64]*10) or not np.array_equal(confusion.sum(axis=0), metrics['predictedCounts']):
            raise ValueError('Same-image support/prediction mismatch.')
        if not np.array_equal(np.diag(confusion)/64, metrics['perClassAccuracy']) or not math.isclose(np.trace(confusion)/640, metrics['accuracy'], abs_tol=1e-12):
            raise ValueError('Accuracy differs from saved confusion counts.')
    if len(observation['perClassDrift']) != 10 or any(not math.isfinite(d['meanSquaredFeatureDrift']) or d['meanSquaredFeatureDrift'] < 0 for d in observation['perClassDrift']):
        raise ValueError('Invalid saved representation drift.')


def audit_rows(records):
    import numpy as np
    rows = []
    for record in records:
        validate_observation(record)
        node = record['node']
        round_id = record.get('round', 0)
        visited = list(range(node)) if round_id == 0 else [c for c in range(10) if c != node]
        current = record['interventions']['current_encoder_current_head']
        old_encoder = record['interventions']['selected_encoder_current_head']
        old_head = record['interventions']['current_encoder_selected_head']
        reference = record['interventions']['selected_encoder_selected_head']
        per = np.asarray(current['perClassAccuracy'])
        rows.append(dict(condition=record['condition'], round=round_id, node=node, stage=record['stage'],
            accuracy=current['accuracy'], ownClassAccuracy=float(per[node]),
            otherClassAccuracy=float(np.delete(per, node).mean()),
            visitedClassAccuracy=float(per[visited].mean()) if visited else None,
            selectedEncoderCurrentHeadVisitedAccuracy=float(np.mean(np.asarray(old_encoder['perClassAccuracy'])[visited])) if visited else None,
            currentEncoderSelectedHeadVisitedAccuracy=float(np.mean(np.asarray(old_head['perClassAccuracy'])[visited])) if visited else None,
            selectedEncoderSelectedHeadVisitedAccuracy=float(np.mean(np.asarray(reference['perClassAccuracy'])[visited])) if visited else None,
            meanFeatureDrift=record['representation']['meanSampleFeatureDrift'],
            meanReferenceCosine=record['representation']['meanReferenceCosine'],
            meanOldSquaredFeatureDrift=float(np.mean([record['perClassDrift'][c]['meanSquaredFeatureDrift'] for c in visited])) if visited else None,
            **{f'class_{c}': float(a) for c, a in enumerate(per)}))
    return rows


def validate_round_metrics(records, data, rounds):
    """Cross-check plotted CSV values against measured activation evidence."""
    if [int(row['round']) for row in data] != list(range(rounds)):
        raise ValueError('Missing/duplicate round CSV points.')
    for round_id, row in enumerate(data):
        states = records[round_id*10:(round_id+1)*10]
        if len(states) != 10 or [(r['round'], r['node']) for r in states] != [(round_id, n) for n in range(10)]:
            raise ValueError('Round CSV requires ten measured node states.')
        final = states[-1]['globalTestEvaluationOnly']
        accuracies = [r['globalTestEvaluationOnly']['accuracy'] for r in states]
        expected = dict(full_test_accuracy=final['accuracy'],
            mean_node_accuracy=sum(accuracies)/10, worst_node_accuracy=min(accuracies),
            full_test_loss=final['crossEntropy'],
            **{f'class_{c}_accuracy': a for c, a in enumerate(final['perClassAccuracy'])})
        for key, value in expected.items():
            if not math.isfinite(float(row[key])) or not math.isclose(float(row[key]), value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Round {round_id} CSV {key} differs from measured activation evidence.')


def validate_fixed_gamma(configured, measured):
    import numpy as np
    # Training uses float32 TensorFlow scalars. Check the exact applied scalar,
    # not equality with the JSON/Python float64 calibration representation.
    if measured != float(np.float32(configured)):
        raise ValueError('Gamma adapted during learning.')


def report_saved_audit():
    import numpy as np
    source = json.loads((AUDIT/'audit.json').read_text())
    if source['status'] != 'completed' or source['optimizerSteps'] != 0 or source['trainingRunsStarted'] != 0 or source['testSetUsed']:
        raise ValueError('Saved audit is incomplete or has changed scope.')
    origin = ROOT/source['sourceRun']
    if hashlib.sha256((origin/'run.json').read_bytes()).hexdigest() != source['sourceManifestHash']:
        raise ValueError('Source run manifest changed after the read-only audit.')
    for path, digest in source['sourceArtifactHashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise ValueError('Source checkpoint/activation evidence changed after audit.')
    records = source['records']
    if len(records) != 40 or {(r['condition'], r['node'], r['stage']) for r in records} != {
        (c, n, s) for c in ('cart', 'cart_feature_memory') for n in range(10) for s in (0, 5)}:
        raise ValueError('Missing/duplicate saved observations.')
    identities = {(r['imageIndexHash'], r['imageHash'], r['labelsHash']) for r in records}
    if len(identities) != 1: raise ValueError('Audit image identity changed between comparisons.')
    rows = audit_rows(records)
    with (AUDIT/'same_image_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# Same-image CART representation audit', '',
        '**Zero additional optimizer steps. Balanced TRAINING probe, not held out or test accuracy.**', '',
        'Each exact selected model was reconstructed from canonical initialization or the saved predecessor state. '
        'Input/output hashes were checked. Every intervention below uses the same 640 training images. '
        'The live model is never assigned an intervention or updated.', '',
        '| Control | Node | Current encoder/current head: old classes | Selected encoder/current head: old classes | Current encoder/selected head: old classes | Feature drift L2/image |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['stage'] != 5 or row['node'] == 0: continue
        lines.append(f"| {row['condition']} | {row['node']} | {100*row['visitedClassAccuracy']:.3f}% | "
            f"{100*row['selectedEncoderCurrentHeadVisitedAccuracy']:.3f}% | "
            f"{100*row['currentEncoderSelectedHeadVisitedAccuracy']:.3f}% | {row['meanFeatureDrift']:.6f} |")
    improved = [r for r in rows if r['stage'] == 5 and r['node'] > 0 and
        r['selectedEncoderCurrentHeadVisitedAccuracy'] > r['visitedClassAccuracy']]
    lines += ['', '## Evidence and limits', '',
        f"Restoring selected-encoder features while retaining the current head raises visited-class accuracy in {len(improved)}/18 final node/control observations.", '',
        'This removes the previous different-sample-set confound. It characterizes the contribution of the encoder '
        'at the observed parameter states; it does not prove that encoder drift is the only cause, or that a local '
        'preservation penalty will solve missing-class learning. The selected models may themselves have weak '
        'representations, and the classifier also changes.', '',
        'Only before-training and after-epoch-5 weights exist for this saved comparison. '
        '**Epochs 1–4 cannot be reconstructed and are not fabricated.** '
        'The new opt-in diagnostic records all six stages directly if explicitly run.', '',
        'No beta/gamma/sigma was selected from this audit, and no test data, W&B or research queue was used.', '',
        '![Matched-image comparison](../../../../newPlots/nonIID/cart_representation/same_image_saved_v1/same_image_encoder_head_comparison.png)', '']
    (AUDIT/'SAME_IMAGE_REPRESENTATION_AUDIT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = ROOT/'newPlots/nonIID/cart_representation/same_image_saved_v1'
    plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for condition, ax in zip(('cart', 'cart_feature_memory'), axes):
        final = [r for r in rows if r['condition'] == condition and r['stage'] == 5 and r['node'] > 0]
        for key, label, color, style in (
            ('visitedClassAccuracy', 'Current encoder + current head', '#dc2626', '-'),
            ('selectedEncoderCurrentHeadVisitedAccuracy', 'Selected encoder + current head', '#7c3aed', '--'),
            ('currentEncoderSelectedHeadVisitedAccuracy', 'Current encoder + selected head', '#059669', ':')):
            ax.plot([r['node'] for r in final], [100*r[key] for r in final], label=label, color=color, linestyle=style, marker='o')
        ax.set(title=condition, xlabel='Node activation', ylabel='Visited-class accuracy (%)', ylim=(0, 100))
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle('IDENTICAL training images — analysis-only encoder/head interventions')
    fig.tight_layout(); fig.savefig(plots/'same_image_encoder_head_comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(dict(status='reported', observations=len(rows), sameImageIdentity=True,
        improvedEncoderInterventions=len(improved), trainingRunsStarted=0,
        report=str(AUDIT/'SAME_IMAGE_REPRESENTATION_AUDIT.md'))))


def review_learning():
    import numpy as np
    from basil_core.research_protocol import params_hash
    manifest = json.loads((RESULT/'run.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Incomplete learning diagnostic; no invented completion.')
    cfg = manifest['config']
    rounds = cfg['nRounds']
    if rounds not in {1, 30} or manifest['activationsCompleted'] != 20*rounds or manifest['optimizerSteps'] != 1000*rounds:
        raise ValueError('Incomplete learning diagnostic; no invented completion.')
    if cfg['testSetUsed'] != (rounds == 30) or cfg['researchValid'] or manifest['wandbEnabled'] or manifest['productionQueueStarted']:
        raise ValueError('Invalid diagnostic scope.')
    gamma = cfg['representationGamma']
    if not math.isfinite(gamma) or gamma <= 0: raise ValueError('Missing/invalid fixed gamma.')
    calibration = manifest.get('calibration')
    if calibration:
        probe = set(json.loads((RESULT/'partition_audit.json').read_text())['analysisProbe'])
        if set(calibration['trainingIndices']) & probe or not calibration['trainingOnly'] or calibration['testSetUsed']:
            raise ValueError('Gamma calibration used analysis/test data.')
        if not math.isclose(gamma*calibration['unitPreservationGradientNorm']/calibration['baseGradientNorm'], calibration['targetRatio'], rel_tol=1e-6):
            raise ValueError('Fixed gamma differs from predeclared calibration.')
        if calibration['targetRatio'] != cfg['representationCalibrationRatio'] or calibration['parameterHash'] != manifest['initialModelHash']:
            raise ValueError('Calibration differs from the declared rule/canonical initialization.')
    flat = []; all_records = {}; checks = 0
    for condition in cfg['controls']:
        records = json.loads((RESULT/condition/'activations.json').read_text())
        if [(r['round'], r['node']) for r in records] != [(r, n) for r in range(rounds) for n in range(10)]:
            raise ValueError('Missing/duplicate rounds/nodes.')
        all_records[condition] = records
        with np.load(RESULT/condition/'final_nodes.npz', allow_pickle=False) as saved:
            for record in records:
                node = record['node']
                if record['round'] == rounds-1 and params_hash([saved[f'node{node}_w{j}'] for j in range(10)]) != record['outputHash']:
                    raise ValueError('Final checkpoint checksum mismatch.')
                if record['steps'] != 50 or record['inputHash'] != record['snapshotAnchorReferenceHash']:
                    raise ValueError('Strict handoff/full-epoch contract failed.')
                if len(record['epochs']) != 5 or any(e['samplesSeen'] != 5000 or e['batchSizes'] != [512]*9+[392] for e in record['epochs']):
                    raise ValueError('Five full local epochs/partial batch violated.')
                expected_classes = list(range(node)) if record['round'] == 0 else [c for c in range(10) if c != node]
                if record['featureMemory']['presentClasses'] != expected_classes or record['featureMemory']['modelHash'] != record['inputHash']:
                    raise ValueError('Replay memory scope mismatch.')
                stages = [record['beforeTraining'], *record['epochs']]
                identity = {(s['sameImageAudit']['imageIndexHash'], s['sameImageAudit']['imageHash'], s['sameImageAudit']['labelsHash']) for s in stages}
                if len(identity) != 1: raise ValueError('Images changed across the five epochs.')
                for stage, value in enumerate(stages):
                    observation = dict(condition=condition, round=record['round'], node=node, stage=stage, **value['sameImageAudit'])
                    if observation['selectedParameterHash'] != record['inputHash']:
                        raise ValueError('Frozen teacher differs from actual BASIL-selected snapshot.')
                    flat.append(observation); checks += 1
                for step in record['optimizerTelemetry']:
                    if step['ebmCoefficient'] != 0 or step['configuredSigma'] != 0: raise ValueError('EBM/noise leaked into clean comparison.')
                    expected = step['baseCrossEntropy']+step['snapshotAnchorPenalty']+step['featureReplayPenalty']
                    if condition == 'feature_replay_preserve':
                        expected += step['representationPenalty']
                        validate_fixed_gamma(gamma, step['representationGamma'])
                    if not math.isclose(step['totalObjective'], expected, abs_tol=1e-5, rel_tol=1e-5):
                        raise ValueError('Saved complete objective differs from components.')
                if condition == 'feature_replay_preserve' and (not record['selectedTeacherFrozenForActivation'] or record['selectedTeacherEncoderHash'] != record['selectedEncoderHash']):
                    raise ValueError('Teacher/reference encoder identity mismatch.')
                if rounds == 30:
                    metrics = record['globalTestEvaluationOnly']
                    confusion = np.asarray(metrics['confusion'])
                    if confusion.shape != (10, 10) or not np.array_equal(confusion.sum(axis=1), [1000]*10):
                        raise ValueError('Full evaluation-only test support differs.')
                    if not np.array_equal(np.diag(confusion)/1000, metrics['perClassAccuracy']) or not math.isclose(np.trace(confusion)/10000, metrics['accuracy'], abs_tol=1e-12):
                        raise ValueError('Global test accuracy differs from confusion counts.')
                checks += 1
        if rounds == 30:
            with (RESULT/condition/'round_metrics.csv').open() as handle:
                validate_round_metrics(records, list(csv.DictReader(handle)), rounds)
            checks += rounds
    rows = audit_rows(flat)
    if len({(r['imageIndexHash'], r['imageHash'], r['labelsHash']) for r in flat}) != 1:
        raise ValueError('Probe identity differs across paired controls/activations.')
    summary = {}
    for condition in cfg['controls']:
        final = [r for r in rows if r['condition'] == condition and r['round'] == rounds-1 and r['stage'] == 5]
        active = [r for r in final if r['visitedClassAccuracy'] is not None]
        summary[condition] = dict(meanTrainingProbeAccuracy=float(np.mean([r['accuracy'] for r in final])),
            meanOwnClassAccuracy=float(np.mean([r['ownClassAccuracy'] for r in final])),
            meanOtherClassAccuracy=float(np.mean([r['otherClassAccuracy'] for r in final])),
            meanVisitedClassAccuracy=float(np.mean([r['visitedClassAccuracy'] for r in active])),
            meanOldSquaredFeatureDrift=float(np.mean([r['meanOldSquaredFeatureDrift'] for r in active])))
        if rounds == 30:
            last = all_records[condition][-10:]
            summary[condition].update(finalFullTestAccuracy=last[-1]['globalTestEvaluationOnly']['accuracy'],
                finalMeanNodeTestAccuracy=float(np.mean([r['globalTestEvaluationOnly']['accuracy'] for r in last])),
                finalWorstNodeTestAccuracy=min(r['globalTestEvaluationOnly']['accuracy'] for r in last),
                finalTestPerClassAccuracy=last[-1]['globalTestEvaluationOnly']['perClassAccuracy'])
    return manifest, rows, flat, all_records, summary, checks


def report_learning():
    import numpy as np
    manifest, rows, flat, records, summary, checks = review_learning()
    rounds = manifest['config']['nRounds']
    PLOTS.mkdir(parents=True, exist_ok=True)
    with (RESULT/'epoch_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'feature_replay': '#d97706', 'feature_replay_preserve': '#2563eb'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition in colors:
        final = [r for r in rows if r['condition'] == condition and r['round'] == rounds-1 and r['stage'] == 5]
        axes[0].plot([r['node'] for r in final], [100*r['accuracy'] for r in final], marker='o', label=condition, color=colors[condition])
        axes[1].plot([r['node'] for r in final], [r['meanFeatureDrift'] for r in final], marker='o', label=condition, color=colors[condition])
    for ax, title, ylabel in zip(axes, ('Final balanced TRAINING-probe accuracy', 'Same-image encoder feature drift'), ('Accuracy (%)', 'Mean drift L2/image')):
        ax.set(title=title, xlabel='Node activation', ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=7)
    axes[0].set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(PLOTS/'representation_comparison.png', dpi=150); plt.close(fig)
    for condition in colors:
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        for node, ax in enumerate(axes.flat):
            values = [r for r in rows if r['condition'] == condition and r['round'] == rounds-1 and r['node'] == node]
            for c in range(10): ax.plot([r['stage'] for r in values], [100*r[f'class_{c}'] for r in values], label=f'class {c}', alpha=1 if c == node else .6)
            ax.set(title=f'Node {node}', xlabel='Epoch (0 = before)', ylim=(0, 100)); ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f'{condition}: identical TRAINING images, analysis only')
        fig.tight_layout(); fig.savefig(PLOTS/f'{condition}_epoch_per_class.png', dpi=150); plt.close(fig)
        for node in (1, 9):
            stages = [r for r in flat if r['condition'] == condition and r['round'] == rounds-1 and r['node'] == node]
            fig, axes = plt.subplots(2, 3, figsize=(10, 6))
            for observation, ax in zip(stages, axes.flat):
                matrix = np.asarray(observation['interventions']['current_encoder_current_head']['confusion'])
                ax.imshow(matrix/64, vmin=0, vmax=1, cmap='Blues')
                ax.set(title=f"Epoch {observation['stage']}", xlabel='Predicted class', ylabel='True class')
            fig.suptitle(f'{condition}, Node {node}: TRAINING-probe confusion')
            fig.tight_layout(); fig.savefig(PLOTS/f'{condition}_node{node}_confusion_epochs.png', dpi=150); plt.close(fig)
    lines = ['# CART representation-preservation diagnostic', '',
        f'**{rounds} clean round(s)/control; same-image scores use a TRAINING probe, not held-out validation.**', '',
        '| Metric | Feature replay control | Feature replay + selected-encoder preservation |', '|---|---:|---:|']
    for label, key in (('Balanced probe accuracy', 'meanTrainingProbeAccuracy'), ('Own-class accuracy', 'meanOwnClassAccuracy'),
                       ('Other-class accuracy', 'meanOtherClassAccuracy'), ('Visited-class accuracy', 'meanVisitedClassAccuracy')):
        lines.append(f"| {label} | {100*summary['feature_replay'][key]:.3f}% | {100*summary['feature_replay_preserve'][key]:.3f}% |")
    if rounds == 30:
        for label, key in (('Final Node-9 full-test accuracy', 'finalFullTestAccuracy'),
                           ('Final mean-node full-test accuracy', 'finalMeanNodeTestAccuracy'),
                           ('Final worst-node full-test accuracy', 'finalWorstNodeTestAccuracy')):
            lines.append(f"| {label} | {100*summary['feature_replay'][key]:.3f}% | {100*summary['feature_replay_preserve'][key]:.3f}% |")
        lines += ['', 'Full-test metrics evaluate all 10,000 CIFAR test examples. They never control '
            'SGD, Snapshot Selection, mu, gamma, LR, horizon or stopping.', '']
    lines += ['', f"Fixed gamma: **{manifest['config']['representationGamma']:.9g}**; rule: `{manifest['config']['representationGammaRule']}`.",
        'Beta remains 1.0; the existing CART mu cap remains 0.01. No attacks, channel noise, EBM, averaging, clipping, weight decay or W&B in this learning comparison.', '',
        '| Control | Mean old-class squared feature drift |', '|---|---:|']
    for c in colors: lines.append(f"| {c} | {summary[c]['meanOldSquaredFeatureDrift']:.6f} |")
    delta = summary['feature_replay_preserve']['meanTrainingProbeAccuracy']-summary['feature_replay']['meanTrainingProbeAccuracy']
    lines += ['', f'Observed change in mean balanced probe accuracy: **{100*delta:+.3f} percentage points**.', '',
        'This single-seed clean diagnostic does not establish universal performance, long-run convergence, Byzantine safety, '
        'noise mitigation or a universal improvement. Lower feature drift alone is not a successful retention result. '
        'No automatic coefficient retuning or larger campaign follows.', '',
        f"Execution: {manifest['activationsCompleted']} activations, {manifest['optimizerSteps']} optimizer steps, "
        f"{manifest['runtimeSeconds']:.2f} seconds; saved-result checks: {checks}; epoch rows: {len(rows)}.", '',
        '## Figures', '', f'![Comparison](../../../../newPlots/nonIID/cart_representation/{PLOTS.name}/representation_comparison.png)', '']
    for c in colors:
        lines += [f'![{c} epochs](../../../../newPlots/nonIID/cart_representation/{PLOTS.name}/{c}_epoch_per_class.png)', '']
    if rounds == 30:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for condition in colors:
            with (RESULT/condition/'round_metrics.csv').open() as handle:
                data = list(csv.DictReader(handle))
            validate_round_metrics(records[condition], data, rounds)
            for ax, key, title in zip(axes, ('full_test_accuracy', 'mean_node_accuracy', 'worst_node_accuracy'),
                                     ('Node-9 full-test accuracy', 'Mean node full-test accuracy', 'Worst node full-test accuracy')):
                ax.plot([int(r['round']) for r in data], [100*float(r[key]) for r in data], label=condition, color=colors[condition])
                ax.set(title=title, xlabel='Full ring round', ylabel='Accuracy (%)', ylim=(0, 100)); ax.grid(alpha=.2)
            axes[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(PLOTS/'full_test_round_comparison.png', dpi=150); plt.close(fig)
        lines += [f'![Full-test trajectories](../../../../newPlots/nonIID/cart_representation/{PLOTS.name}/full_test_round_comparison.png)', '']
    (RESULT/'REPRESENTATION_PRESERVATION_REVIEW.md').write_text('\n'.join(lines))
    (RESULT/'summary.json').write_text(json.dumps(dict(summary=summary, checksPassed=checks,
        trainingOnly=rounds == 1, testAccuracyReported=rounds == 30), indent=2)+'\n')
    print(json.dumps(dict(status='reported', checksPassed=checks, csvRows=len(rows), summary=summary)))


def main():
    global RESULT, PLOTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning', action='store_true', help='Report completed learning evidence instead of the zero-training saved audit')
    parser.add_argument('--rounds', type=int, choices=[1, 30], default=1)
    args = parser.parse_args()
    RESULT, PLOTS = output_paths(args.rounds)
    report_learning() if args.learning else report_saved_audit()


if __name__ == '__main__': main()
