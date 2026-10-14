#!/usr/bin/env python3
"""Saved-only verification/plots of the bounded head/backbone diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_forgetting_mechanism_audit import RESULT, PLOTS, SOURCE, write_json


def review(result=RESULT):
    import numpy as np
    from basil_core.research_protocol import params_hash
    run = json.loads((result/'run.json').read_text())
    if run['status'] != 'completed' or run['optimizerSteps'] != 150 or run['controlsCompleted'] != ['ce', 'fixed_anchor', 'cart']:
        raise ValueError('Incomplete mechanism audit; no invented epochs or research continuation.')
    if hashlib.sha256((SOURCE/'run.json').read_bytes()).hexdigest() != run['sourceRunSHA256']:
        raise ValueError('Prior source evidence changed.')
    for path, expected in run['sourceFilesSHA256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Executed source changed: {path}')
    probe = json.loads((result/'probe.json').read_text())
    if probe['classSupport'] != [64]*10 or set(probe['indices']) & set(probe['selectionIndices']):
        raise ValueError('Unbalanced audit probe or overlap with local selection.')
    rows, records, checks, traces = [], {}, [], {}
    for name in run['controlsCompleted']:
        stages = json.loads((result/name/'stages.json').read_text())
        record = json.loads((result/name/'activation.json').read_text())
        if len(stages) != 6 or record['steps'] != 50 or len(record['epochs']) != 5:
            raise ValueError('Missing before/five-epoch evidence.')
        if record['snapshotAnchorReferenceHash'] != record['selectedHash'] or record['selectedHash'] != stages[0]['parameterHash']:
            raise ValueError('Handoff/anchor mismatch.')
        candidates = record['candidates']
        if [(d['senderId'], d['snapshotRound']) for d in candidates] != [(s, 2) for s in range(5, 10)]:
            raise ValueError('Wrong warmed predecessor memory.')
        minimum = min(d['receiverLocalLoss'] for d in candidates)
        winner = min((d for d in candidates if d['receiverLocalLoss'] <= minimum+1e-8), key=lambda d: (0-d['senderId']) % 10)
        if record['selectedSender'] != winner['senderId']:
            raise ValueError('Wrong receiver-local selection/tie rule.')
        for epoch in record['epochs']:
            if epoch['batchSizes'] != [512]*9+[392] or epoch['samplesSeen'] != 5000:
                raise ValueError('Incomplete local epoch or dropped last batch.')
        for stage_id, stage in enumerate(stages):
            if stage['stage'] != stage_id:
                raise ValueError('Duplicated/missing stage.')
            with np.load(result/name/f'epoch_{stage_id}_params.npz', allow_pickle=False) as payload:
                if params_hash([payload[f'w{i}'] for i in range(10)]) != stage['parameterHash']:
                    raise ValueError('Epoch parameter checksum differs.')
            for variant, metrics in stage['interventions'].items():
                matrix = np.asarray(metrics['confusion'])
                if not np.array_equal(matrix.sum(axis=1), np.full(10, 64)) or not np.isclose(np.trace(matrix)/640, metrics['accuracy']):
                    raise ValueError('Confusion/accuracy/support mismatch.')
                np.testing.assert_allclose(np.diag(matrix)/64, metrics['perClassAccuracy'])
                if not np.isfinite(metrics['crossEntropy']):
                    raise ValueError('Non-finite inference.')
                rows.append(dict(condition=name, stage=stage_id, intervention=variant,
                    accuracy=metrics['accuracy'], ownClassAccuracy=metrics['perClassAccuracy'][0],
                    otherClassAccuracy=float(np.mean(metrics['perClassAccuracy'][1:])),
                    crossEntropy=metrics['crossEntropy'], dominantPrediction=int(np.argmax(metrics['predictedCounts'])),
                    dominantPredictionFraction=max(metrics['predictedCounts'])/640,
                    backboneDrift=stage['parameterDrift']['backbone'], headDrift=stage['parameterDrift']['head'],
                    featureReferenceCosine=stage['representation']['meanReferenceCosine']))
        # Selected state counterfactual is fixed across all epochs.
        if any(s['interventions']['selected'] != stages[0]['interventions']['selected'] for s in stages):
            raise ValueError('Selected reference evaluation changed.')
        with (result/name/'audit/batch_trace.jsonl').open() as handle:
            trace = [json.loads(line) for line in handle]
        if len(trace) != 50:
            raise ValueError('Missing audited optimizer steps.')
        traces[name] = [tuple(json.dumps(s[k], sort_keys=True) for k in (
            'round_id', 'node_id', 'epoch', 'batch', 'indicesHash', 'augmentationKey', 'augmentedImagesHash')) for s in trace]
        for step in trace:
            if step['coefficient'] != 0 or step['ebmPenalty'] != 0 or step['ebmCorrectionNorm'] != 0:
                raise ValueError('EBM leaked into clean diagnostic.')
            for category in ('parameterStatsBefore', 'parameterStatsAfter', 'gradients'):
                if any(v['nan'] or v['positiveInf'] or v['negativeInf'] for v in step[category]):
                    raise ValueError('Non-finite audited update.')
        if name == 'cart':
            if record['registryTrainingSupport'] != [512]+[0]*9 or record['unverifiedLocalClasses'] != list(range(1, 10)):
                raise ValueError('Balanced audit probe fed CART control.')
        records[name] = dict(stages=stages, record=record)
        checks.extend([f'{name}: ordered six stages and hashed epoch weights',
            f'{name}: strict selected handoff/frozen reference, minimum local CE and S=5',
            f'{name}: 50 finite updates; five 5000-example epochs with final 392',
            f'{name}: all five counterfactuals contain 640 balanced training-probe predictions'])
    if not traces['ce'] == traces['fixed_anchor'] == traces['cart']:
        raise ValueError('Different actual augmentation/order across controls.')
    checks.extend(['Matched training order/augmentation across all 50 steps',
        'CART receives only class-0 support; balanced diagnostic scores cannot control mu',
        'Source evidence and executed implementation fingerprints verified'])
    return dict(status='passed', checkedContractGroups=len(checks), checks=checks, rows=rows), records, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verified, controls, run = review()
    write_json(RESULT/'verification.json', verified)
    with (RESULT/'intervention_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verified['rows'][0]))
        writer.writeheader(); writer.writerows(verified['rows'])
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/papermerge-forgetting-plots')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    variants = ('trained', 'trained_backbone_selected_head', 'selected_backbone_trained_head', 'trained_zero_output_bias')
    colors = ('#2563eb', '#d97706', '#9333ea', '#059669')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, (name, control) in zip(axes, controls.items()):
        for variant, color in zip(variants, colors):
            ax.plot(range(6), [100*s['interventions'][variant]['perClassAccuracy'][0] for s in control['stages']],
                '-o', label=variant.replace('_', ' '), color=color)
        ax.set(title=name, xlabel='Local epoch (0 = selected input)', ylim=(0, 100), xticks=range(6))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Class-0 training-audit accuracy (%)')
    axes[-1].legend(fontsize=7)
    fig.suptitle('Inference-only head/backbone interventions — not trained methods')
    fig.tight_layout(); fig.savefig(PLOTS/'head_backbone_interventions.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, control) in zip(axes, controls.items()):
        matrix = [s['interventions']['trained']['perClassAccuracy'] for s in control['stages']]
        im = ax.imshow(matrix, vmin=0, vmax=1, aspect='auto')
        ax.set(title=name, xlabel='Class', ylabel='Epoch (0 = before)', xticks=range(10), yticks=range(6))
    fig.colorbar(im, ax=axes.tolist(), label='Training-audit class accuracy')
    fig.savefig(PLOTS/'five_epoch_class_retention.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    for name, control in controls.items():
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for ax, stage in zip(axes.flat, control['stages']):
            ax.imshow(stage['interventions']['trained']['confusion'], vmin=0, vmax=64)
            ax.set(title=f"Epoch {stage['stage']}", xlabel='Prediction', ylabel='True class')
        fig.suptitle(f'{name}: training-probe confusions (64 examples/class, analysis only)')
        fig.tight_layout(); fig.savefig(PLOTS/f'{name}_confusions.png', dpi=160); plt.close(fig)
    text = ['# Head/backbone forgetting audit', '',
        '**Completed bounded diagnostic, not a research continuation.** One warmed Node-0 activation per '
        'existing CE/fixed-anchor/CART state; 150 total updates. Full 5000-example local class, five epochs, '
        'batch 512 with final 392, reset SGD, approved R3 learning rate. No attacks/noise/EBM/W&B/queue.', '',
        'Scores below use a balanced **640-example training-side audit probe**. These examples were not reserved '
        'from historical training: scores are NOT held-out validation or global test accuracy. The audit probe is '
        'disjoint from the receiver-local selection batch, and never supplied to SGD, selection or CART control. '
        'No test set was loaded in this diagnostic. The three starting states differ because they come from '
        'three previously trained controls; this is not a fresh common-initialization method comparison.', '',
        '## 1. Locate output collapse', '',
        '| Control | Epoch | Own class (%) | Other classes (%) | Dominant prediction | Fraction (%) | Feature cosine to selected | Backbone drift L2 | Head drift L2 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in verified['rows']:
        if row['intervention'] == 'trained':
            text.append(f"| {row['condition']} | {row['stage']} | {100*row['ownClassAccuracy']:.3f} | "
                f"{100*row['otherClassAccuracy']:.3f} | {row['dominantPrediction']} | {100*row['dominantPredictionFraction']:.3f} | "
                f"{row['featureReferenceCosine']:.6f} | {row['backboneDrift']:.6f} | {row['headDrift']:.6f} |")
    text.extend(['', '## Final-epoch counterfactuals', '',
        'A swap diagnoses compatibility with the reference, not an independently trained solution. '
        'Changing the head also changes which features it can use; these tests cannot prove features have retained all classes. '
        'Removing output bias does not freeze the head kernel or backbone.', '',
        '| Control | Inference-only intervention | Probe accuracy (%) | Own (%) | Other (%) | Dominant prediction |',
        '|---|---|---:|---:|---:|---:|'])
    for row in verified['rows']:
        if row['stage'] == 5:
            text.append(f"| {row['condition']} | {row['intervention']} | {100*row['accuracy']:.3f} | "
                f"{100*row['ownClassAccuracy']:.3f} | {100*row['otherClassAccuracy']:.3f} | {row['dominantPrediction']} |")
    text.extend(['', '## 2. Registry measurement', '',
        '| CART stage | Local class support | Supported retention gap | Applied mu |', '|---:|---|---:|---:|'])
    for c in controls['cart']['record']['cartControl']:
        text.append(f"| {c['stage']} | {c['supportedClasses']} only | {c['supportedGap']:.6f} | {c['mu']:.9f} |")
    text.extend(['', 'A carried class best is historical telemetry, not a current-model guarantee. '
        'A gap of zero only covers class 0 here; the other nine are unavailable to the controller. '
        'Our centralized probe sees them for diagnosis only. We do not pass it to CART.', '',
        '## Why ordinary CE encourages specialization', '',
        r'For logits $z$, $\partial CE/\partial z_c = p_c-\mathbf{1}[c=i]$ on class-i batches. '
        'This creates direct gradient pressure to raise the target logit and lower non-target logits; '
        'joint nonlinear parameter updates need not move every logit monotonically. The output-bias gradient is the batch mean '
        'of this quantity; the head-kernel gradient additionally depends on hidden features. Backbone gradients '
        'also change the representation. There is no missing-class supervised term in these batches. '
        'The measured single-class probe bias gradients and tensor drift are saved in stages.json and the batch audit.', '',
        '## Measured interpretation', '',
        'For each of these three warmed snapshots, the trained head predicts class 0 even on the '
        'selected backbone; the selected head predicts its old constant class even on the trained backbone. '
        'Zeroing the output bias still leaves every probe image assigned to class 0. Thus final-head '
        'replacement is sufficient to switch the constant prediction in this diagnostic, and bias alone '
        'does not explain the collapse. The backbone also changed; nonzero feature variance or high '
        'reference cosine does not prove useful ten-class discrimination. Restoring an already-collapsed '
        'selected head restores an old collapse, not cumulative learning.', '',
        '## 3. Bounded next adaptation — decision, not automatic promotion', '',
        'Do not extend the failed clean controls. A useful next CART change is **distributed training-side '
        'retention verification**: keep images private at their class-owning node, evaluate the current '
        'candidate on each owner\'s deterministic local probe, and return current per-class scores/support '
        'bound to the candidate parameter hash. This supplies a real forgetting measurement instead of '
        'treating historical maxima as current knowledge. It does not itself supply missing-class gradients '
        'or guarantee successful retention. Extra probe requests/score exchange would be a separately '
        'revisioned adaptation requiring explicit approval; BASIL selection stays receiver-local.', '',
        'Before any live controller, test score/hash binding, support coverage, staleness, malicious score '
        'claims and whether the measured signal detects lost classes. Do not claim remote Byzantine '
        'scores are independently verified. Fix the measurement first; then predeclare the controller '
        'and new-class/retention trade-off test using training-side validation. No higher mu was selected '
        'from these scores. No replay, ACDS, public data, classifier freezing or new loss was added.', '',
        '## 4. Keep EBM independent', '',
        r'$L=CE+\sigma_e^2\|\nabla CE\|_2^2+(\mu/2)\|\theta-\theta_{sel}\|_2^2$. '
        r'The EBM derivative is $g+2\sigma_e^2Hg$; the anchor adds $\mu(\theta-\theta_{sel})$. '
        'There is no extra lambda. EBM acts on the local CE gradient, not missing-class knowledge. '
        'A highly confident one-class predictor can have very small local CE gradients despite failing '
        'other classes. These clean executions intentionally use sigma=0 and contain no EBM. '
        'Mathematical/no-lambda contracts are tested separately; noise robustness is not established by this audit.', '',
        '## Verification and limits', '',
        f"Saved-evidence contract groups: **{verified['checkedContractGroups']} passed**. Runtime: "
        f"{run['runtimeSeconds']/60:.2f} minutes on {run['deviceName']}. GPU tracked allocator peak: "
        f"{run['memorySafety']['maximumGpuPeakMiB']:.2f} MiB; pressure events: {run['memorySafety']['pressureEvents']}.", '',
        'Single seed, one warmed receiver, small training probe, already-specialized incoming snapshots. '
        'No evidence here identifies a universal failure mechanism or establishes class discrimination '
        'in the hidden representation. Intermediate weights of the ORIGINAL three-round study were not '
        'saved: this is a new bounded activation, not a fabricated reconstruction of those epochs. '
        'Mixed head/backbone inference is not a proposed handoff/averaging/training algorithm.', '',
        '## Figures', '',
        '![Head/backbone inference](../../../../newPlots/nonIID/forgetting_audit/warmed_node0_head_backbone_v1/head_backbone_interventions.png)', '',
        '![Five-epoch retention](../../../../newPlots/nonIID/forgetting_audit/warmed_node0_head_backbone_v1/five_epoch_class_retention.png)', '',
        'See each control\'s *_confusions.png, raw stages.json, activation.json, epoch NPZs, '
        'audit/batch_trace.jsonl and intervention_metrics.csv.'])
    (RESULT/'FORGETTING_MECHANISM_REVIEW.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='passed', checkedContractGroups=verified['checkedContractGroups'],
        report=str(RESULT/'FORGETTING_MECHANISM_REVIEW.md'), researchCampaignsStarted=0)))


if __name__ == '__main__':
    main()
