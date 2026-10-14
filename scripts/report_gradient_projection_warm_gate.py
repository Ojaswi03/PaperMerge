#!/usr/bin/env python3
"""Saved-only verification, plots and report for the three-round diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_gradient_projection_warm_gate import ROOT, RESULT, PLOTS, learning_screen, expected_candidates
from scripts.run_gradient_projection_gate import save_json


def review(directory=RESULT):
    import numpy as np
    directory = Path(directory)
    run = json.loads((directory/'run.json').read_text())
    if run['status'] != 'completed':
        raise ValueError('Partial diagnostics cannot be analyzed as complete.')
    config = run['config']
    checks = dict(sixtyActivations=run['completedActivations'] == 60,
        threeRoundsEach=run['completedRoundsByControl'] == {'ce': 3, 'projection': 3},
        diagnosticOnly=config['researchValid'] is False and config['trainingSamplesPerNode'] == 4500
            and config['validationSamplesPerClass'] == 500,
        cloudAndThreatsOff=run['wandbEnabled'] is False and config['liveTrackingEnabled'] is False
            and config['attackerCount'] == 0 and config['useChannelNoise'] is False,
        noEbmOrAnchor=config['ebmMode'] == 'none' and config['snapshotAnchorMu'] == 0.,
        unchangedProtocol=config['localEpochs'] == 5 and config['batchSize'] == 512
            and config['basilMemorySize'] == 5 and config['aggregationMode'] == 'strict_sequential_handoff'
            and config['projectionEnergy'] == .99 and config['momentum'] == config['weightDecayCoefficient'] == 0.
            and config['gradientClipNorm'] is None,
        unchangedSources=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h
            for p, h in run['implementationSha256'].items()))
    controls = {name: json.loads((directory/name/'activations.json').read_text()) for name in ('ce', 'projection')}
    round_rows, summaries, traces = [], {}, {}
    for name, records in controls.items():
        checks[name+'_orderedActivations'] = [(r['round'], r['nodeId']) for r in records] == [(r, n) for r in range(3) for n in range(10)]
        checks[name+'_fiveFullEpochs'] = all(r['steps'] == 45 and len(r['epochs']) == 5 and
            all(e['samplesSeen'] == 4500 and e['batchSizes'] == [512]*8+[404] for e in r['epochs']) for r in records)
        checks[name+'_decay'] = all(np.isclose(r['learningRate'], .05/(1+.05*r['round'])) for r in records)
        checks[name+'_rollingMemory'] = all([(d['senderId'], d['snapshotRound']) for d in r['candidates']] ==
            expected_candidates(r['round'], r['nodeId']) for r in records)
        selected_correctly = True
        hashes_match = True
        prior_outputs = {}
        for r in records:
            minimum = min(d['receiverLocalLoss'] for d in r['candidates'])
            best = min((d for d in r['candidates'] if d['receiverLocalLoss'] <= minimum+1e-8),
                       key=lambda d: (r['nodeId']-d['senderId']) % 10)
            selected_correctly &= r['selectedSender'] == best['senderId'] and r['selectedSnapshotRound'] == best['snapshotRound']
            hashes_match &= r['inputHash'] == r['selectedParamsHash'] == best['paramsHash']
            for d in r['candidates']:
                hashes_match &= d['paramsHash'] == (run['initialModelHash'] if d['snapshotRound'] == -1 else
                    prior_outputs[(d['snapshotRound'], d['senderId'])])
            prior_outputs[(r['round'], r['nodeId'])] = r['trainedOutputHash']
        checks[name+'_minimumLocalCeTieBreak'] = bool(selected_correctly)
        checks[name+'_strictHandoffAndCandidateHashes'] = bool(hashes_match)
        stages = [s for r in records for s in [r['beforeTraining'], *r['epochs']]]
        checks[name+'_balancedValidation'] = all(np.array(s['validationConfusion']).shape == (10, 10)
            and np.array_equal(np.sum(s['validationConfusion'], axis=1), [500]*10)
            and np.allclose(np.diag(s['validationConfusion'])/500, s['perClassValidationAccuracy'])
            and np.isclose(np.trace(s['validationConfusion'])/5000, s['globalValidationAccuracy']) for s in stages)
        steps = [s for r in records for s in r['optimizerTelemetry']]
        checks[name+'_finiteSteps'] = len(steps) == 1350 and all(np.isfinite([v for v in s.values()
            if isinstance(v, (float, int))]).all() for s in steps)
        checks[name+'_noEbmLeak'] = all(s['ebmCoefficient'] == s['ebmPenalty'] == s['ebmCorrectionNorm'] == 0. for s in steps)
        checks[name+'_orthogonality'] = all(s['projectionOrthogonalityResidual'] < 1e-5 for s in steps)
        with (directory/name/'audit/batch_trace.jsonl').open() as handle:
            traces[name] = [json.loads(line) for line in handle]
        checks[name+'_finiteParameterAndGradientAudits'] = len(traces[name]) == 1350 and all(
            stat['nan'] == stat['positiveInf'] == stat['negativeInf'] == 0
            for t in traces[name] for group in ('gradients', 'parameterStatsBefore', 'parameterStatsAfter') for stat in t[group])
        for round_id in range(3):
            rows = records[round_id*10:(round_id+1)*10]
            accuracies = np.array([r['epochs'][-1]['perClassValidationAccuracy'] for r in rows])
            other = np.array([np.delete(a, i) for i, a in enumerate(accuracies)])
            warm_steps = [s for r in rows for s in r['optimizerTelemetry']]
            before = np.array([r['beforeTraining']['perClassValidationAccuracy'] for r in rows])
            forgetting = np.array([np.delete(b-a, i) for i, (b, a) in enumerate(zip(before, accuracies))])
            round_rows.append(dict(condition=name, round=round_id,
                mean_node_validation_accuracy=float(accuracies.mean()),
                node9_validation_accuracy=float(accuracies[9].mean()),
                mean_own_class_accuracy=float(np.diag(accuracies).mean()),
                mean_other_class_accuracy=float(other.mean()),
                mean_other_class_forgetting=float(forgetting.mean()),
                own_class_above_5percent=int(sum(np.diag(accuracies) > .05)),
                states_with_two_other_classes_above_5percent=int(sum(np.sum(other > .05, axis=1) >= 2)),
                constant_class_states=int(sum(np.max(np.sum(r['epochs'][-1]['validationConfusion'], axis=0)) == 5000 for r in rows)),
                mean_applied_base_gradient_ratio=float(np.mean([s['robustObjectiveGradientNorm']/(s['baseGradientNorm']+1e-12) for s in warm_steps])),
                mean_removed_gradient_ratio=float(np.mean([s['projectionRemovedRatio'] for s in warm_steps])),
                protected_history_activations=sum(bool(r['gradientProjection']['inputClasses']) for r in rows)))
        summaries[name] = [r for r in round_rows if r['condition'] == name]
    checks['pairedDataOrderAndAugmentation'] = all(all(a[k] == b[k] for k in (
        'round_id', 'node_id', 'epoch', 'batch', 'imagesHash', 'labelsHash', 'indicesHash',
        'augmentationKey', 'augmentedImagesHash')) for a, b in zip(traces['ce'], traces['projection']))
    checks['warmedProjectionHistoryPresent'] = all(r['selectedSnapshotRound'] >= 0
        and r['gradientProjection']['inputClasses'] and all(d['representationClasses'] for d in r['candidates'])
        for r in controls['projection'][10:])
    checks['screenMatchesSaved'] = learning_screen(controls) == json.loads((directory/'learning_screen.json').read_text())
    old = ROOT/'newResults/nonIID/gradient_projection/clean_training_validation_gate_v1'
    cold_start = {}
    for name in controls:
        original = json.loads((old/name/'activations.json').read_text())
        cold_start[name] = sum(a['trainedOutputHash'] == b['trainedOutputHash'] for a, b in zip(original, controls[name][:10]))
    if not all(checks.values()):
        raise ValueError(f'Saved-result contracts failed: {[k for k,v in checks.items() if not v]}')
    return dict(checks=checks, summaries=summaries, roundMetrics=round_rows,
        screen=learning_screen(controls), firstRoundHashMatchesOriginal=cold_start,
        runtimeSeconds=run['runtimeSeconds'], memorySafety=run['memorySafety'],
        initialModelHash=run['initialModelHash'], partitionHash=run['partitionHash']), controls


def main():
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/papermerge-projection-warm-mpl')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result, controls = review()
    PLOTS.mkdir(parents=True, exist_ok=True)
    save_json(RESULT/'saved_result_verification.json', result)
    with (RESULT/'round_validation_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['roundMetrics'][0]))
        writer.writeheader(); writer.writerows(result['roundMetrics'])
    activation_rows = []
    for name, rows in controls.items():
        for r in rows:
            n = r['nodeId']
            before, after = r['beforeTraining'], r['epochs'][-1]
            b, a = np.array(before['perClassValidationAccuracy']), np.array(after['perClassValidationAccuracy'])
            activation_rows.append(dict(condition=name, round=r['round'], node=n,
                selected_sender=r['selectedSender'], selected_snapshot_round=r['selectedSnapshotRound'],
                validation_before=before['globalValidationAccuracy'], validation_after=after['globalValidationAccuracy'],
                own_before=float(b[n]), own_after=float(a[n]),
                other_before=float(np.delete(b, n).mean()), other_after=float(np.delete(a, n).mean()),
                other_forgetting=float(np.delete(b-a, n).mean()),
                own_accuracy_by_epoch=json.dumps([e['perClassValidationAccuracy'][n] for e in r['epochs']]),
                input_exposure_classes=json.dumps(r['gradientProjection']['inputClasses']),
                output_basis_ranks=json.dumps(r['gradientProjection']['ranks'])))
    with (RESULT/'activation_validation_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(activation_rows[0]))
        writer.writeheader(); writer.writerows(activation_rows)
    for name, rows in controls.items():
        trajectory = np.array([r['epochs'][-1]['perClassValidationAccuracy'] for r in rows])
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(trajectory.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        for boundary in (9.5, 19.5): ax.axvline(boundary, color='white', linestyle='--')
        ax.set(xlabel='Sequential activation index (round × 10 + node)', ylabel='CIFAR class',
            title=f'{name}: three-round held-out TRAINING-validation retention')
        fig.colorbar(im, ax=ax, label='Class accuracy'); fig.tight_layout()
        fig.savefig(PLOTS/f'{name}_activation_class_heatmap.png'); plt.close(fig)
        for round_id in range(3):
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(trajectory[10*round_id:10*(round_id+1)], vmin=0, vmax=1, cmap='viridis')
            ax.set(xlabel='CIFAR class', ylabel='Active node', title=f'{name}: R{round_id} validation-only node/class')
            fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(PLOTS/f'{name}_round{round_id}_node_class.png'); plt.close(fig)
        for round_id, node in ((1, 0), (2, 5), (2, 9)):
            r = rows[10*round_id+node]
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for ax, stage, title in zip(axes.flat, [r['beforeTraining'], *r['epochs']],
                    ['Before', 'Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']):
                ax.imshow(np.asarray(stage['validationConfusion'])/500, vmin=0, vmax=1, cmap='viridis')
                ax.set(title=title, xlabel='Prediction', ylabel='True class')
            fig.suptitle(f'{name}: R{round_id} Node {node}, validation only')
            fig.tight_layout(); fig.savefig(PLOTS/f'{name}_r{round_id}_node{node}_epoch_confusions.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, color in [('ce', '#0072B2'), ('projection', '#D55E00')]:
        for ax, field, title in zip(axes, ['mean_node_validation_accuracy', 'mean_own_class_accuracy', 'mean_other_class_accuracy'],
                                   ['Mean node validation', 'Own class', 'Other classes']):
            ax.plot(range(3), [r[field] for r in result['summaries'][name]], marker='o', label=name, color=color)
            ax.set(xlabel='Round (zero-based)', ylabel='Accuracy', title=title, ylim=(0, 1), xticks=range(3))
            ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS/'paired_round_validation_comparison.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, name in enumerate(('ce', 'projection')):
        for round_id, color in ((1, '#0072B2'), (2, '#D55E00')):
            own, other = [], []
            for stage in range(6):
                values = np.array([(r['beforeTraining'] if stage == 0 else r['epochs'][stage-1])[
                    'perClassValidationAccuracy'] for r in controls[name][10*round_id:10*(round_id+1)]])
                own.append(np.diag(values).mean())
                other.append(np.mean([np.delete(a, i) for i, a in enumerate(values)]))
            for ax, values, title in zip(axes[row], (own, other), ('Own class', 'Other classes')):
                ax.plot(range(6), values, marker='o', color=color, label=f'R{round_id}')
                ax.set(title=f'{name}: {title}', xlabel='Before (0) / completed local epoch',
                       ylabel='Mean validation accuracy', ylim=(0, 1), xticks=range(6))
                ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS/'warmed_epoch_class_retention.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    rows = controls['projection']
    axes[0].plot(range(30), [np.mean([s['robustObjectiveGradientNorm']/(s['baseGradientNorm']+1e-12)
        for s in r['optimizerTelemetry']]) for r in rows], color='#D55E00', label='Applied/base')
    axes[0].set(title='Projected gradient norm ratio', ylim=(0, 1.05), ylabel='Norm ratio')
    axes[1].plot(range(30), [len(r['gradientProjection']['inputClasses']) for r in rows],
        color='#0072B2', label='Selected input history')
    axes[1].set(title='Exposed classes (not knowledge)', ylim=(0, 10), ylabel='Class count')
    for i in range(5):
        axes[2].plot(range(30), [r['gradientProjection']['ranks'][i] for r in rows], label=f'Layer {i}')
    axes[2].set(title='Output basis ranks', ylabel='Rank')
    for ax in axes:
        ax.set_xlabel('Sequential activation index')
        ax.axvline(9.5, color='gray', linestyle='--'); ax.axvline(19.5, color='gray', linestyle='--')
        ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS/'projection_update_and_history_diagnostics.png'); plt.close(fig)
    table = ['| Control | Round | Mean node | Node 9 | Own class | Other classes | Constant-class states | Applied/base gradient |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['roundMetrics']:
        table.append(f'| {r["condition"]} | {r["round"]} | {r["mean_node_validation_accuracy"]:.3%} | '
            f'{r["node9_validation_accuracy"]:.3%} | {r["mean_own_class_accuracy"]:.3%} | '
            f'{r["mean_other_class_accuracy"]:.3%} | {r["constant_class_states"]}/10 | {r["mean_applied_base_gradient_ratio"]:.6f} |')
    details = ['| R | Node | Selected source | Exposed classes in selected state | Own before → after | Other before → after | Final basis ranks |',
               '|---:|---:|---|---|---:|---:|---|']
    for r in controls['projection'][10:]:
        before, after = r['beforeTraining']['perClassValidationAccuracy'], r['epochs'][-1]['perClassValidationAccuracy']
        n = r['nodeId']
        details.append(f'| {r["round"]} | {n} | {r["selectedSender"]}:R{r["selectedSnapshotRound"]} | '
            f'{r["gradientProjection"]["inputClasses"]} | {before[n]:.1%} → {after[n]:.1%} | '
            f'{np.mean(np.delete(before,n)):.1%} → {np.mean(np.delete(after,n)):.1%} | {r["gradientProjection"]["ranks"]} |')
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    final_table = ['| Class | CE final Node 9 | Projection final Node 9 |', '|---|---:|---:|']
    for i, name in enumerate(classes):
        final_table.append(f'| {name} | {controls["ce"][-1]["epochs"][-1]["perClassValidationAccuracy"][i]:.1%} | '
            f'{controls["projection"][-1]["epochs"][-1]["perClassValidationAccuracy"][i]:.1%} |')
    final_p, final_c = result['summaries']['projection'][-1], result['summaries']['ce'][-1]
    final_own = [r['epochs'][-1]['perClassValidationAccuracy'][r['nodeId']] for r in controls['projection'][-10:]]
    blocked = [i for i, accuracy in enumerate(final_own) if accuracy <= .05]
    node9 = controls['projection'][-1]['epochs'][-1]['perClassValidationAccuracy']
    missing = [classes[i] for i, accuracy in enumerate(node9) if accuracy == 0.]
    warm_steps = [s for r in controls['projection'][10:] for s in r['optimizerTelemetry']]
    applied_ratio = float(np.mean([s['robustObjectiveGradientNorm']/(s['baseGradientNorm']+1e-12) for s in warm_steps]))
    screen_status = 'PASSED' if result['screen']['passed'] else 'DID NOT PASS'
    text = f'''# Three-round warmed-memory CE versus projection diagnostic

**Completed: three rounds per control, 60 activations, 2,700 SGD updates.**
Runtime {result['runtimeSeconds']/60:.2f} minutes; GPU allocator peak
{result['memorySafety']['maximumGpuPeakMiB']:.2f} MiB, pressure events
{result['memorySafety']['pressureEvents']}. All {len(result['checks'])} saved-result contracts passed.
No attacks, channel noise, EBM, anchoring, W&B or queue execution.

## What this measures

All metrics here are **held-out training-validation accuracy, NOT CIFAR test accuracy**.
Per class: 4,500 SGD images and 500 reserved training-validation images.
Five complete epochs, batch 512, final batch 404, 45 SGD steps per activation.
This intentionally labelled diagnostic has researchValid=false; the full-data
protocol remains 5,000 samples, final batch 392 and 50 steps per activation.
Both fresh controls share the original split/initialization, data order and
stateless augmentation. The test set is never evaluated or used for selection.
Node 9 means the model just trained by Node 9; mean node averages ten trained
states after the corresponding round. Own/other means average each receiver's
own/other classes. Other-class forgetting is before minus after (signed).

R0 is cold start; **R1-R2 have five trained candidates in every memory**.
The unchanged BASIL minimum local-CE rule selects one complete snapshot plus
its basis state. No averaging, validation-guided selection or memory mixing.
Projection energy remains 0.99, with 32 representation images and 2,048 patches.
No thresholds, schedule or updates were tuned after viewing these results.
First-round parameter hashes matching the earlier one-traversal evidence:
{json.dumps(result['firstRoundHashMatchesOriginal'])} (out of ten per control).

## Round-level results

{chr(10).join(table)}

![Paired validation curves](../../../../newPlots/nonIID/gradient_projection/{PLOTS.name}/paired_round_validation_comparison.png)
![Projection class trajectory](../../../../newPlots/nonIID/gradient_projection/{PLOTS.name}/projection_activation_class_heatmap.png)
![CE class trajectory](../../../../newPlots/nonIID/gradient_projection/{PLOTS.name}/ce_activation_class_heatmap.png)

{chr(10).join(final_table)}

## Warmed-memory mechanism evidence

Exposure labels mean representation history, **not demonstrated class knowledge**.
The table shows whether a selected history learns the new class or merely
preserves a previous constant-class predictor. Saved per-step telemetry records
projection norm, applied/base-gradient ratio and orthogonality residual.
The applied/base ratio is a norm ratio, not the fraction of gradient coordinates.
All before/epoch-1..5 confusions are retained in activations.json, not just these examples.

{chr(10).join(details)}

![Final Node 9 epochs](../../../../newPlots/nonIID/gradient_projection/{PLOTS.name}/projection_r2_node9_epoch_confusions.png)
![Warmed epoch retention](../../../../newPlots/nonIID/gradient_projection/{PLOTS.name}/warmed_epoch_class_retention.png)
![Gradient and history diagnostics](../../../../newPlots/nonIID/gradient_projection/{PLOTS.name}/projection_update_and_history_diagnostics.png)

## Predeclared warmed-memory screen

```json
{json.dumps(result['screen'], indent=2)}
```

These conservative 5%/5pp/80%/50% validation screens were written before this
run to distinguish old-class preservation from new-class learning. They are
not mathematical correctness tests, significance tests or a 50% research
target. Passing would not prove a ten-class solution. This diagnostic cannot
automatically launch a longer run, even if the screen passes.

## Scientific boundaries and next decision

**Measured outcome: warmed-memory screen {screen_status}.**
Final mean node validation accuracy: projection {final_p['mean_node_validation_accuracy']:.3%}
versus CE {final_c['mean_node_validation_accuracy']:.3%} (difference
{100*(final_p['mean_node_validation_accuracy']-final_c['mean_node_validation_accuracy']):+.3f} pp).
Projection own-class mean was {final_p['mean_own_class_accuracy']:.3%}, with
nodes {blocked} at or below 5% own-class accuracy in the final traversal;
{final_p['constant_class_states']}/10 final states made the same class prediction
for every validation image. Final Node 9 had exactly zero validation accuracy
on {missing}. Mean warmed applied/base-gradient norm was {applied_ratio:.6f}.
Unlike the first-four-node cold-start screen, every warmed projection activation
selected trained history, so the reported suppression/retention is observed with
the projection mechanism actually active. This does not disentangle every
interaction with BASIL selection, which remains unchanged.

Projection is a GPM-inspired **biased, single-head adaptation**, not exact paper
GPM and not the gradient of a new regularized objective. It projects CE updates
as G_perp = G - Q(Q.T G), including biases. Bases are frozen per activation,
expanded only from receiver SGD images and travel with the selected snapshot.
BASIL can discard other histories by choosing a different candidate.
Protected parameter subspaces need not encode all-class discrimination.
Source/adaptation mathematics: [protocol guide](../../../../docs/GRADIENT_PROJECTION_RETENTION.md).
No raw examples travel; basis privacy/authentication guarantees are not established.

One seed, three rounds and a reserved training-validation split cannot justify
production performance or universal claims. If own-class learning is suppressed
despite warmed history, do not promote this unchanged variant to 30/500 rounds.
A mechanism with representative previous-class training signals would require
a separately agreed data/privacy design, not retuning against test accuracy.
CPU/GPU synthetic tolerances do not establish bit-identical execution or
multiprocessing reproducibility. Snapshot NPZ files are **not full resume checkpoints**.
Existing evidence was preserved; no previous result is extended by this run.

## Software verification

| Check | Status |
|---|---|
{chr(10).join(f'| {k} | PASS |' for k in result['checks'])}

## Test and evidence handoff

Actual compile/GPU/test commands and observed suite totals are documented in
the [warmed-diagnostic guide](../../../../docs/GRADIENT_PROJECTION_WARM_DIAGNOSTIC.md).
Those focused tests are separate from the 32 saved-result contracts above;
the legacy full-suite collection issue is not claimed fixed.
After execution, SHA-256 verification is recorded separately in
`evidence_integrity.json`: only the new diagnostic namespace is excluded,
not earlier projection/IID/nonIID results or papers. A saved-only report
regeneration does not rerun historical hash verification.
'''
    (RESULT/'GRADIENT_PROJECTION_WARM_REVIEW.md').write_text(text)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
