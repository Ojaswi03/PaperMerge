#!/usr/bin/env python3
"""Verify and explain the completed projection gate; saved results only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULT = ROOT/'newResults/nonIID/gradient_projection/clean_training_validation_gate_v1'


def constant_class_count(rows):
    import numpy as np
    return int(sum(np.max(np.sum(r['epochs'][-1]['validationConfusion'], axis=0)) == 5000 for r in rows))


def review(directory=RESULT):
    import numpy as np
    from scripts.run_gradient_projection_gate import gate_decision
    directory = Path(directory)
    manifest = json.loads((directory/'run.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Refusing to analyze a partial gate as completed.')
    controls = {c: json.loads((directory/c/'activations.json').read_text()) for c in ('ce', 'projection')}
    config = manifest['config']
    checks = dict(completed=manifest['completedActivations'] == 20,
        heldOutTrainingValidation=config['trainingSamplesPerNode'] == 4500 and config['validationSamplesPerClass'] == 500,
        fiveEpochs=config['localEpochs'] == 5, batch512=config['batchSize'] == 512,
        researchInvalid=config['researchValid'] is False, cloudDisabled=manifest['wandbEnabled'] is False,
        noThreats=config['attackerCount'] == 0 and config['useChannelNoise'] is False,
        noEbmOrAnchor=config['ebmMode'] == 'none' and config['snapshotAnchorMu'] == 0.,
        sameHandoffAndMemory=config['aggregationMode'] == 'strict_sequential_handoff' and config['basilMemorySize'] == 5,
        implementationUnchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
            for name, digest in manifest['implementationSha256'].items()))
    summaries = {}
    for name, rows in controls.items():
        prefix = name + '_'
        checks[prefix+'tenOrderedActivations'] = len(rows) == 10 and [r['nodeId'] for r in rows] == list(range(10))
        checks[prefix+'fiveFullEpochs'] = all(r['steps'] == 45 and len(r['epochs']) == 5 and
            all(e['samplesSeen'] == 4500 and e['batchSizes'] == [512]*8+[404] for e in r['epochs']) for r in rows)
        checks[prefix+'localCeSelection'] = all(next(d['receiverLocalLoss'] for d in r['candidates']
            if d['senderId'] == r['selectedSender']) <= min(d['receiverLocalLoss'] for d in r['candidates']) + 1e-8 for r in rows)
        stages = [[r['beforeTraining'], *r['epochs']] for r in rows]
        checks[prefix+'validationCounts'] = all(np.array(s['validationConfusion']).shape == (10, 10) and
            np.array_equal(np.sum(s['validationConfusion'], axis=1), [500]*10) for group in stages for s in group)
        checks[prefix+'validationMetricsAgree'] = all(np.allclose(s['perClassValidationAccuracy'],
            np.diag(s['validationConfusion'])/500) and np.isclose(s['globalValidationAccuracy'],
            np.trace(s['validationConfusion'])/5000) for group in stages for s in group)
        steps = [s for r in rows for s in r['optimizerTelemetry']]
        checks[prefix+'finiteSteps'] = all(np.isfinite([v for v in s.values() if isinstance(v, (float, int))]).all() for s in steps)
        checks[prefix+'noEbmLeak'] = all(s['ebmCoefficient'] == s['ebmPenalty'] == s['ebmCorrectionNorm'] == 0. for s in steps)
        checks[prefix+'orthogonality'] = all(s['projectionOrthogonalityResidual'] < 1e-5 for s in steps)
        own = [r['epochs'][-1]['perClassValidationAccuracy'][r['nodeId']] for r in rows]
        other = [np.mean([a for c, a in enumerate(r['epochs'][-1]['perClassValidationAccuracy']) if c != r['nodeId']]) for r in rows]
        output = rows[-1]['epochs'][-1]
        active_steps = [s for r in rows if r['gradientProjection']['inputClasses'] for s in r['optimizerTelemetry']]
        summaries[name] = dict(meanValidationAccuracy=float(np.mean([r['epochs'][-1]['globalValidationAccuracy'] for r in rows])),
            endOfTraversalValidationAccuracy=output['globalValidationAccuracy'],
            meanOwnClassAccuracy=float(np.mean(own)), meanOtherClassAccuracy=float(np.mean(other)),
            ownClassAccuracyByNode=own,
            constantClassStates=constant_class_count(rows),
            initialSnapshotSelections=sum(r['selectedSnapshotRound'] == -1 for r in rows),
            activationsWithProtectedHistory=sum(bool(r['gradientProjection']['inputClasses']) for r in rows),
            meanRemovedGradientRatio=float(np.mean([s['projectionRemovedRatio'] for s in active_steps])) if active_steps else None,
            meanAppliedToBaseGradientRatio=float(np.mean([s['robustObjectiveGradientNorm']/(s['baseGradientNorm']+1e-12) for s in active_steps])) if active_steps else None,
            maxOrthogonalityResidual=max(s['projectionOrthogonalityResidual'] for s in steps))
    first_empty = [i for i, r in enumerate(controls['projection']) if not r['gradientProjection']['inputClasses']]
    checks['emptyHistoryExactPairedParameters'] = all(controls['projection'][i]['trainedOutputHash'] ==
        controls['ce'][i]['trainedOutputHash'] for i in first_empty)
    decision = gate_decision(controls)
    checks['savedDecisionMatches'] = decision == json.loads((directory/'gate_decision.json').read_text())
    if not all(checks.values()):
        raise ValueError(f'Saved gate verification failed: {[k for k,v in checks.items() if not v]}')
    return dict(checks=checks, summaries=summaries, decision=decision,
        runtimeSeconds=manifest['runtimeSeconds'], memorySafety=manifest['memorySafety']), controls


def main():
    import os
    os.environ['MPLCONFIGDIR'] = '/tmp/papermerge-projection-mpl'
    result, controls = review()
    from scripts.run_gradient_projection_gate import save_json
    save_json(RESULT/'saved_result_verification.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plots = ROOT/'newPlots/nonIID/gradient_projection/clean_training_validation_gate_v1'
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    activation = controls['projection'][5]
    for ax, stage, title in zip(axes.flat, [activation['beforeTraining'], *activation['epochs']],
                               ['Before', 'Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']):
        ax.imshow(np.array(stage['validationConfusion'])/500, vmin=0, vmax=1, cmap='viridis')
        ax.set(title=title, xlabel='Predicted class', ylabel='True class')
    fig.suptitle('Projection, Node 5: protected history is active (validation only)')
    fig.tight_layout(); fig.savefig(plots/'projection_node5_epoch_confusions.png'); plt.close(fig)
    path = RESULT/'GRADIENT_PROJECTION_GATE_REVIEW.md'
    original = path.read_text().split('\n## Saved-result verification and mechanism analysis\n')[0].replace('nan%', '—')
    summary_rows = ['| Control | Mean node validation | Own-class mean | Other-class mean | Constant-class states |',
                    '|---|---:|---:|---:|---:|']
    for name, data in result['summaries'].items():
        summary_rows.append(f'| {name} | {data["meanValidationAccuracy"]:.2%} | {data["meanOwnClassAccuracy"]:.2%} | {data["meanOtherClassAccuracy"]:.2%} | {data["constantClassStates"]}/10 |')
    p = result['summaries']['projection']
    node7 = controls['projection'][7]['epochs'][-1]['perClassValidationAccuracy']
    node9 = controls['projection'][9]['epochs'][-1]['perClassValidationAccuracy']
    zero_own = [i for i, value in enumerate(p['ownClassAccuracyByNode']) if value == 0.]
    screen_status = 'passed' if result['decision']['passed'] else 'did not pass'
    continuation = ('The gate alone does not authorize a production run. Complete full-data checkpoint integration before a separately verified 30-round diagnostic.'
        if result['decision']['passed'] else 'Do not promote this unchanged method to a 30-/500-round campaign.')
    text = f'''
## Saved-result verification and mechanism analysis

**{len(result['checks'])}/{len(result['checks'])} saved-result contracts passed.**
These software/math checks are separate from the learning-trend screen, which {screen_status}.
Runtime: {result['runtimeSeconds']/60:.2f} minutes. Peak GPU allocator:
{result['memorySafety']['maximumGpuPeakMiB']:.2f} MiB. Pressure events:
{result['memorySafety']['pressureEvents']}. No recorded numerical failure/OOM.

{chr(10).join(summary_rows)}

Projection selected initialization snapshots {p['initialSnapshotSelections']}/10
times and received protected history in {p['activationsWithProtectedHistory']}/10
activations. Empty-history output hashes matched paired CE exactly. The
cold-start gate's early lack of retention therefore does not isolate a failure
of projection itself: BASIL's unchanged local-CE rule selected parameters with
no prior representation history.

Where protected history was present, the mean removed-gradient norm ratio was
{p['meanRemovedGradientRatio']:.6f}, and the mean applied/base-gradient norm
ratio was {p['meanAppliedToBaseGradientRatio']:.6f}. These are norm ratios,
not percentages of parameter coordinates. Orthogonality residual peaked at
{p['maxOrthogonalityResidual']:.3g}. No gradient clipping or forced multiplier
was used. Inspect the per-node table and epoch confusion matrices: preserving
a previous constant-class prediction while failing the active class is not
learning a ten-class classifier. An increase in the 'other-class mean' can
merely reflect predicting one older class instead of the current class.

![Node 5 protected-history epochs](../../../../newPlots/nonIID/gradient_projection/clean_training_validation_gate_v1/projection_node5_epoch_confusions.png)

### Partial retention is real but incomplete

Node 7 retained {node7[1]:.1%} automobile validation accuracy while attaining {node7[7]:.1%}
horse accuracy. Node 9 finished with {node9[1]:.1%} automobile and {node9[7]:.1%} horse accuracy;
{sum(value == 0. for value in node9)} classes had zero accuracy, with truck accuracy {node9[9]:.1%}. These are
held-out training-validation observations, NOT full CIFAR test results.
Thus the late improvement is limited retention, not ten-class
learning. Nodes {zero_own} ended with 0% own-class validation accuracy.
No second traversal was started to seek a more favorable outcome.

### Decision

The combined BASIL-plus-projection cold-start learning screen {screen_status}.
{continuation} This
single-seed gate is a negative result for this adaptation/setup, not a
universal failure claim about GPM. It distinguishes two obstacles: BASIL
can discard specialized histories during local selection, and projection
can preserve an already collapsed predictor while suppressing new learning.
No selection rule, threshold, epochs, LR or representation threshold was
changed after observing the result. A different warm-start/retention mechanism
would require a separately declared diagnostic and approval.

### Contract details

| Check | Status |
|---|---|
{chr(10).join(f'| {name} | PASS |' for name in result['checks'])}
'''
    path.write_text(original+text)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
