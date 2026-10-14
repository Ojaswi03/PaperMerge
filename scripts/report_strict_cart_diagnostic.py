#!/usr/bin/env python3
"""Verify saved bounded CART evidence and generate comparison tables/plots."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_strict_cart_diagnostic import RESULT, PLOTS, expected_candidates
from scripts.run_gradient_projection_gate import save_json


def review():
    import numpy as np
    run = json.loads((RESULT/'run.json').read_text())
    if run['status'] != 'completed' or run['completedActivations'] != 90:
        raise ValueError('Incomplete diagnostic; no invented results.')
    source_resolution = {}
    for path, expected in run['implementationSha256'].items():
        current = ROOT/path
        archived = RESULT/'execution_sources'/path
        source = current if hashlib.sha256(current.read_bytes()).hexdigest() == expected else archived
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Executed training source unavailable or changed: {path}')
        source_resolution[path] = str(source.relative_to(ROOT))
    controls = {}; traces = {}; rows = []; checks = []
    for name in ('ce', 'fixed_anchor', 'cart'):
        records = json.loads((RESULT/name/'activations.json').read_text())
        if len(records) != 30: raise ValueError('Missing activations.')
        seen = {}; data_keys = []
        for index, record in enumerate(records):
            r, node = divmod(index, 10)
            if (record['round'], record['nodeId']) != (r, node): raise ValueError('Round duplication/order.')
            candidates = record['candidates']
            if [(s['senderId'], s['snapshotRound']) for s in candidates] != expected_candidates(r, node):
                raise ValueError('Rolling memory mismatch.')
            minimum = min(d['receiverLocalLoss'] for d in candidates)
            chosen = min((d for d in candidates if d['receiverLocalLoss'] <= minimum+1e-8),
                         key=lambda d: (node-d['senderId']) % 10)
            if chosen['senderId'] != record['selectedSender']: raise ValueError('Wrong SS winner.')
            for d in candidates:
                expected = seen[(d['snapshotRound'], d['senderId'])] if d['snapshotRound'] >= 0 else run['initialModelHash']
                if expected != d['paramsHash']: raise ValueError('Candidate hash differs from actual sender output.')
            if record['selectedParamsHash'] != chosen['paramsHash'] or record['snapshotAnchorReferenceHash'] != chosen['paramsHash']:
                raise ValueError('Strict handoff/frozen reference mismatch.')
            seen[(r, node)] = record['trainedOutputHash']
            if record['steps'] != 50 or len(record['epochs']) != 5: raise ValueError('Epoch/step count.')
            for e in record['epochs']:
                if e['samplesSeen'] != 5000 or e['batchSizes'] != [512]*9+[392]: raise ValueError('Dropped partial batch.')
            for stage in [record['beforeTraining']]+record['epochs']:
                matrix = np.asarray(stage['testConfusion'])
                if matrix.shape != (10, 10) or not np.array_equal(matrix.sum(1), np.full(10, 1000)):
                    raise ValueError('Test evaluation coverage.')
                if not np.isclose(stage['globalTestAccuracy'], np.trace(matrix)/10000): raise ValueError('Confusion/accuracy mismatch.')
                np.testing.assert_allclose(stage['perClassTestAccuracy'], np.diag(matrix)/1000)
            steps = record['optimizerTelemetry']
            if any(s['ebmCoefficient'] != 0 or not np.isfinite(list(s.values())).all() for s in steps):
                raise ValueError('EBM leaked into clean control or non-finite telemetry.')
            if name == 'cart':
                from basil_core.class_registry import ClassRegistry
                if r > 0 and record['registryBefore'] != records[(r-1)*10+node]['registryAfter']:
                    raise ValueError('Logical-node registry continuity failed.')
                previous = record['selectedSnapshotRound']
                carried = records[previous*10+record['selectedSender']]['outputRegistry'] if previous >= 0 else ClassRegistry().toDict()
                if record['receivedRegistry'] != carried:
                    raise ValueError('Received registry is not bound to the selected snapshot.')
                if len(record['cartControl']) != 6 or record['unverifiedLocalClasses'] != [c for c in range(10) if c != node]:
                    raise ValueError('Unsupported registry evidence used.')
                for step in steps:
                    if not np.isclose(step['snapshotAnchorMu'], record['cartControl'][step['epoch']-1]['mu'], atol=1e-9):
                        raise ValueError('Applied coefficient differs from local control.')
                from basil_core.strict_cart import anchor_strength
                for control in record['cartControl']:
                    ref = np.array([np.nan if a is None else a for a in control['referenceAccuracy']])
                    cur = np.array([np.nan if a is None else a for a in control['currentLocalAccuracy']])
                    expected_control = anchor_strength(.01, record['selectedReceiverLocalLoss'], ref, cur,
                                                       record['registryTrainingSupport'])
                    for key in ('mu', 'confidence', 'supportedGap'):
                        if not np.isclose(control[key], expected_control[key]):
                            raise ValueError('Controller differs from declared training-side formula.')
        with (RESULT/name/'audit/batch_trace.jsonl').open() as f:
            trace = [json.loads(line) for line in f]
        if len(trace) != 1500: raise ValueError('Missing batch audits.')
        for step in trace:
            data_keys.append(tuple(json.dumps(step[k], sort_keys=True) for k in
                ('round_id', 'node_id', 'epoch', 'batch', 'imagesHash', 'labelsHash', 'indicesHash', 'augmentationKey', 'augmentedImagesHash')))
            for key in ('parameterStatsBefore', 'parameterStatsAfter', 'gradients'):
                if any(t['nan'] or t['positiveInf'] or t['negativeInf'] for t in step[key]):
                    raise ValueError('Non-finite audited update.')
        traces[name] = data_keys
        # Round-trip saved model/registry evidence for all fifty snapshots.
        from basil_core.research_protocol import params_hash
        with np.load(RESULT/name/'snapshot_memory.npz', allow_pickle=False) as payload:
            metadata = json.loads(str(payload['metadata']))
            if len(metadata) != 50: raise ValueError('Missing memory snapshots.')
            if len({(m['receiver'], m['sender']) for m in metadata}) != 50:
                raise ValueError('Duplicate memory sender entries.')
            for m in metadata:
                prefix = f"r{m['receiver']}s{m['sender']}"
                weights = [payload[f'{prefix}w{j}'] for j in range(10)]
                if params_hash(weights) != m['hash']: raise ValueError('Checkpoint model hash mismatch.')
                if m['round'] != 2 or m['hash'] != seen[(2, m['sender'])]:
                    raise ValueError('Checkpoint is stale or differs from actual trained output.')
                if m['sender'] not in [s for s, _ in expected_candidates(3, m['receiver'])]:
                    raise ValueError('Checkpoint sender not a predecessor.')
                if name == 'cart' and m['registry'] is None: raise ValueError('Lost carried registry.')
                if name == 'cart' and m['registry'] != records[20+m['sender']]['outputRegistry']:
                    raise ValueError('Checkpoint registry differs from bound sender output.')
        for r in range(3):
            group = records[r*10:(r+1)*10]
            classes = np.array([a['epochs'][-1]['perClassTestAccuracy'] for a in group])
            before = np.array([a['beforeTraining']['perClassTestAccuracy'] for a in group])
            off = ~np.eye(10, dtype=bool)
            rows.append(dict(condition=name, round=r,
                meanNodeAccuracy=float(np.mean([a['epochs'][-1]['globalTestAccuracy'] for a in group])),
                node9Accuracy=group[-1]['epochs'][-1]['globalTestAccuracy'],
                worstNodeAccuracy=min(a['epochs'][-1]['globalTestAccuracy'] for a in group),
                ownClassAccuracy=float(np.diag(classes).mean()), otherClassAccuracy=float(classes[off].mean()),
                otherClassForgetting=float((before-classes)[off].mean()),
                constantPredictionStates=int(sum(max(np.asarray(a['epochs'][-1]['testConfusion']).sum(0)) == 10000 for a in group)),
                meanAnchorMu=float(np.mean([s['snapshotAnchorMu'] for a in group for s in a['optimizerTelemetry']])),
                meanSnapshotDistanceSquared=float(np.mean([a['optimizerTelemetry'][-1]['snapshotDistanceSquared'] for a in group]))))
        controls[name] = records
        checks.extend([f'{name}: 30 ordered activations and 1500 finite audited updates',
            f'{name}: strict snapshot hashes, minimum local CE/tie rule, rolling S=5',
            f'{name}: five full 5000-sample epochs, final 392-sample batches',
            f'{name}: all six evaluations contain 10000 balanced test images',
            f'{name}: frozen anchor and final 50-snapshot checkpoint hashes'])
    if not (traces['ce'] == traces['fixed_anchor'] == traces['cart']): raise ValueError('Unpaired training order/augmentation.')
    checks.extend(['Identical actual training data/shuffle/augmentation across all 1500 paired steps',
                   'CART coefficient matches training-only control each epoch; nine absent classes unavailable',
                   'Logical-node registry continuity and selected/checkpoint model-registry binding',
                   'Executed source fingerprints verified against current files or immutable execution archive'])
    return dict(status='passed', checks=checks, checkedContracts=len(checks), roundMetrics=rows,
                executedSourceResolution=source_resolution), controls


def main():
    import numpy as np
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/papermerge-cart-mpl')
    verified, controls = review()
    save_json(RESULT/'saved_result_verification.json', verified)
    rows = verified['roundMetrics']
    with (RESULT/'round_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    colors = dict(ce='#2463eb', fixed_anchor='#e58900', cart='#8c36bb')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in controls:
        subset = [r for r in rows if r['condition'] == name]
        for ax, metric, title in zip(axes, ('meanNodeAccuracy', 'ownClassAccuracy', 'otherClassAccuracy'),
            ('Mean node full-test accuracy', 'Own-class accuracy', 'Other-class accuracy')):
            ax.plot(range(3), [100*r[metric] for r in subset], '-o', color=colors[name], label=name)
            ax.set(title=title, xlabel='Zero-based ring round', ylabel='Accuracy (%)', xticks=[0, 1, 2])
            ax.set_ylim(0, 100)
            ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(PLOTS/'paired_round_accuracy.png', dpi=160); plt.close(fig)
    for name, records in controls.items():
        classes = np.array([r['epochs'][-1]['perClassTestAccuracy'] for r in records])
        fig, ax = plt.subplots(figsize=(11, 4)); im = ax.imshow(classes.T*100, vmin=0, vmax=100, aspect='auto', origin='lower')
        ax.set(title=f'{name}: full-test class accuracy after each activation', xlabel='Activation index', ylabel='Class')
        fig.colorbar(im, ax=ax, label='Accuracy (%)'); fig.tight_layout()
        fig.savefig(PLOTS/f'{name}_activation_per_class.png', dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(classes[-10:]*100, vmin=0, vmax=100)
        ax.set(title=f'{name}: final node-by-class test accuracy', xlabel='Test class', ylabel='Trained node',
               xticks=range(10), yticks=range(10))
        fig.colorbar(im, ax=ax, label='Accuracy (%)'); fig.tight_layout()
        fig.savefig(PLOTS/f'{name}_final_node_by_class.png', dpi=160); plt.close(fig)
        chosen = records[20]
        stages = [chosen['beforeTraining']]+chosen['epochs']
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for k, (ax, stage) in enumerate(zip(axes.ravel(), stages)):
            im = ax.imshow(np.array(stage['testConfusion'])/1000, vmin=0, vmax=1)
            ax.set(title=f'R2 Node0: {"before" if k == 0 else "epoch " + str(k)}', xlabel='Predicted class', ylabel='True class')
        fig.suptitle(name); fig.tight_layout(); fig.savefig(PLOTS/f'{name}_five_epoch_confusions.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, records in controls.items():
        epoch_own = []; epoch_other = []
        for stage_id in range(6):
            values = np.array([(r['beforeTraining'] if stage_id == 0 else r['epochs'][stage_id-1])['perClassTestAccuracy'] for r in records[10:]])
            own = np.array([values[j, r['nodeId']] for j, r in enumerate(records[10:])])
            epoch_own.append(100*own.mean()); epoch_other.append(100*(values.sum()-own.sum())/(20*9))
        ax.plot(range(6), epoch_own, '-o', color=colors[name], label=f'{name} own class')
        ax.plot(range(6), epoch_other, '--s', color=colors[name], label=f'{name} other classes')
    ax.set(title='Warmed R1–R2: before and after each complete local epoch', xlabel='Local epoch (0 = before)', ylabel='Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.legend(ncol=2); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(PLOTS/'warmed_five_epoch_retention.png', dpi=160); plt.close(fig)
    final = {r['condition']: r for r in rows if r['round'] == 2}
    gain = 100*(final['cart']['meanNodeAccuracy'] - final['ce']['meanNodeAccuracy'])
    text = ['# Strict CART: completed full-data diagnostic', '',
        'Three fresh GPU controls; 3 full rounds, 10 one-class nodes, 5000 training samples/node, five complete epochs, '
        'batch 512 (last 392), 50 updates/activation. All 10000 test images are evaluation-only. No attacks, noise, EBM, GPM or W&B.', '',
        '| Control | Round | Mean node (%) | Node 9 (%) | Worst (%) | Own class (%) | Other classes (%) | Other-class forgetting (pp) | Mean μ | Final-step drift² |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['condition']} | {r['round']} | {100*r['meanNodeAccuracy']:.3f} | {100*r['node9Accuracy']:.3f} | "
            f"{100*r['worstNodeAccuracy']:.3f} | {100*r['ownClassAccuracy']:.3f} | {100*r['otherClassAccuracy']:.3f} | "
            f"{100*r['otherClassForgetting']:.3f} | {r['meanAnchorMu']:.6g} | {r['meanSnapshotDistanceSquared']:.6g} |")
    text += ['', '## Final class retention by node', '',
        '| Control | Node | Own-class accuracy (%) | Mean other-class accuracy (%) | Full-test accuracy (%) |',
        '|---|---:|---:|---:|---:|']
    for name, records in controls.items():
        for record in records[20:]:
            node = record['nodeId']; scores = record['epochs'][-1]['perClassTestAccuracy']
            text.append(f"| {name} | {node} | {100*scores[node]:.3f} | "
                f"{100*(sum(scores)-scores[node])/9:.3f} | {100*record['epochs'][-1]['globalTestAccuracy']:.3f} |")
    cart_steps = [s for r in controls['cart'] for s in r['optimizerTelemetry']]
    cart_controls = [c for r in controls['cart'] for c in r['cartControl'][:5]]
    text += ['', '## Measured CART control', '',
        f"Applied μ range: {min(s['snapshotAnchorMu'] for s in cart_steps):.8g} to "
        f"{max(s['snapshotAnchorMu'] for s in cart_steps):.8g}. "
        f"Nonzero supported gaps: {sum(c['supportedGap'] > 0 for c in cart_controls)}/{len(cart_controls)} epoch controls. "
        f"Mean anchor-gradient/CE-gradient norm ratio: {np.mean([s['anchorCorrectionRatio'] for s in cart_steps]):.8g}. "
        'This ratio is telemetry only; it never changes the algorithm.']
    registry = controls['cart'][-1]['outputRegistry']
    current_classes = controls['cart'][-1]['epochs'][-1]['perClassTestAccuracy']
    text += ['', '## Why a registry is not the model\'s current knowledge', '',
        'Historical registry scores below were measured on local TRAINING probes at their source. '
        'Current class scores below are evaluation-only TEST measurements of the final Node-9 model. '
        'These are deliberately different quantities: a carried historical success is not a certificate of current retention.', '',
        '| Class | Historical local registry best (%) | Source | Measurement round | Probe support | Current Node-9 test accuracy (%) | Verifiable locally by Node 9? |',
        '|---|---:|---:|---:|---:|---:|---|']
    names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for c, label in enumerate(names):
        best = f"{100*registry['class_best_acc'][c]:.3f}" if registry['class_best_support'][c] > 0 else 'Unavailable'
        text.append(f"| {label} | {best} | {registry['class_best_source'][c]} | {registry['class_best_round'][c]} | "
            f"{registry['class_best_support'][c]} | {100*current_classes[c]:.3f} | {'Yes' if c == 9 else 'No'} |")
    text += ['', f'CART final mean-node difference from paired CE: **{gain:+.3f} percentage points**.',
        'This is a single-seed, three-round mechanism diagnostic, not evidence of universal improvement. '
        'No accuracy target is a software assertion; no test score selected μ or changed the method.', '',
        '## Framework and mathematics', '',
        'See [strict CART protocol](../../../../docs/STRICT_CART_PROTOCOL.md). '
        'The registry travels with the selected snapshot; carried historical claims are distinct from receiver-verified evidence. '
        'Only the local training probe controls μ. The anchor remains the complete selected received snapshot for all five epochs.', '',
        '```text\nq = 1 / (1 + selected receiver-local CE)\n'
        'gap_e = mean(max(0, verified reference class accuracy - current local class accuracy))\n'
        'mu_e = 0.01 * min(1, q + gap_e), over locally supported classes only\n'
        'L = CE + sigma_e^2 * ||gradient CE||^2 + (mu_e/2) * ||theta - theta_selected||^2\n'
        'Clean diagnostic: sigma_e = 0; mu_e is fixed within each epoch.\n```', '',
        'This confidence-plus-gap rule is an explicitly new adaptation, not an exact legacy CART or paper reproduction. '
        'The supported reference target is capped by what the exact selected model reproduces locally; a registry best is not current knowledge. '
        'A one-class receiver cannot verify the other nine classes.', '',
        '## Verification', '', f"Saved-evidence checks: **{verified['checkedContracts']} passed**; "
        'all 4500 updates finite, complete handoff/candidate hashes, rolling memory, identical actual input/augmentation traces. '
        'Model/registry snapshot files are evidence only—not a resumable production checkpoint. '
        'Separate CPU/GPU synthetic verification covers CE/EBM plus anchor and ignored lambda; it does not claim bitwise CPU/GPU equivalence.', '',
        '## Figures', '', '![Paired rounds](../../../../newPlots/nonIID/strict_cart/clean_full_data_three_round_v1/paired_round_accuracy.png)', '',
        '![Warmed epochs](../../../../newPlots/nonIID/strict_cart/clean_full_data_three_round_v1/warmed_five_epoch_retention.png)', '',
        '![CART class trajectory](../../../../newPlots/nonIID/strict_cart/clean_full_data_three_round_v1/cart_activation_per_class.png)', '',
        '![CART confusion matrices](../../../../newPlots/nonIID/strict_cart/clean_full_data_three_round_v1/cart_five_epoch_confusions.png)', '',
        '![Final CART node-by-class matrix](../../../../newPlots/nonIID/strict_cart/clean_full_data_three_round_v1/cart_final_node_by_class.png)', '',
        '## Limits', '', 'Clean-only execution does not establish noisy-channel EBM or Byzantine robustness. '
        'Remote class records provide no raw examples or supervised gradients for absent classes. '
        'No longer run, production queue or W&B run was started automatically.']
    text += ['', '## Learning conclusion', '',
        '| Control | Final constant-class states | Final mean own-class accuracy (%) | Final mean other-class accuracy (%) |',
        '|---|---:|---:|---:|']
    for name, r in final.items():
        text.append(f"| {name} | {r['constantPredictionStates']}/10 | {100*r['ownClassAccuracy']:.3f} | {100*r['otherClassAccuracy']:.3f} |")
    if final['cart']['ownClassAccuracy'] > .99 and final['cart']['otherClassAccuracy'] < .001:
        text += ['', 'CART still specializes almost exclusively to the active local class. '
            'The measured clean diagnostic does **not** support a claim that this adaptation solves extreme one-class forgetting. '
            'No production continuation is recommended from these data alone. Correct protocol execution is distinct from successful class retention.']
    else:
        text += ['', 'Inspect own-class learning and other-class retention together; a higher mean alone can hide nodes '
            'that fail to learn their class. The table provides observed evidence only, not statistical significance or automatic promotion.']
    (RESULT/'STRICT_CART_REVIEW.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='verified', checkedContracts=verified['checkedContracts'], final=final)), flush=True)


if __name__ == '__main__': main()
