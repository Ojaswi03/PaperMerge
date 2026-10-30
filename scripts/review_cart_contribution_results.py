#!/usr/bin/env python3
"""Audit the full export and create plain-language thesis review figures."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def review(root, destination, figures=True):
    import numpy as np
    from reporting.cart_publication_curves import collect, generate, LABELS, sha, safe_relative
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError('Use a separate review destination.')
    checksums = json.loads((root / 'SHA256.json').read_text())
    for name, expected in checksums.items():
        path = root / safe_relative(name)
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('Export checksum mismatch: ' + name)
    plan, records, inputs = collect(root)
    receipt = json.loads((root / 'independent_validation.json').read_text())
    if receipt['status'] != 'ALL_COMPARISON_MODELS_VERIFIED' or receipt['source'] != plan['source']:
        raise ValueError('Matching independent inference receipt is required.')
    verified = {row['runId']: row for row in receipt['verified']}
    if len(verified) != len(records) or set(verified) != {r['runId'] for r in records}:
        raise ValueError('Independent inference does not cover every configuration.')
    groups, seeds = defaultdict(list), []
    for row in records:
        path = root / row['runId']
        saved = json.loads((path / 'run.json').read_text())
        view = saved['config']
        if saved['optimizerSteps'] != view['nodes'] * view['rounds'] * view['localSteps']:
            raise ValueError('Optimizer-step count changed.')
        with np.load(path / 'metrics.npz', allow_pickle=False) as archive:
            accuracy = archive['roundNodeAccuracy']
            recall = archive['roundNodePerClassAccuracy']
            honest = archive['roundHonestMeanAccuracy']
        classes = view['classes']
        if (recall.shape != (*accuracy.shape, classes) or honest.shape != (view['rounds'],)
                or not np.isfinite(recall).all() or not np.isfinite(honest).all()
                or np.any(recall < 0) or np.any(recall > 1) or np.any(honest < 0) or np.any(honest > 1)):
            raise ValueError('Invalid recall or honest-client metric series.')
        # The saved reporting holdout is class-balanced, so macro recall equals accuracy.
        np.testing.assert_allclose(recall.mean(axis=2), accuracy, rtol=0, atol=1e-12)
        inferred = verified[row['runId']]
        if inferred['maximumValidationAccuracyDifference'] > .001 + 1e-9:
            raise ValueError('Independent inference exceeds one validation image per client.')
        if abs(inferred['finalValidationAccuracy'] - accuracy[-1].mean()) > .001 + 1e-9:
            raise ValueError('Independent mean inference differs.')
        values = dict(dataset=row['dataset'], partition='IID' if row['partition']=='iid' else 'non-IID',
            method=LABELS[row['arm']], variant=row['arm'], seed=row['seed'], runId=row['runId'],
            finalAccuracy=100 * float(accuracy[-1].mean()),
            primaryAccuracy=100 * float(accuracy[-10:].mean()),
            worstClient=100 * float(accuracy[-1].min()),
            worstMeanClassRecall=100 * float(recall[-1].mean(axis=0).min()),
            honestFinalAccuracy=100 * float(honest[-1]))
        seeds.append(values)
        groups[(values['dataset'], values['partition'], values['variant'])].append(values)
    summary = []
    for key, rows in sorted(groups.items()):
        if sorted(r['seed'] for r in rows) != [2025, 2026, 2027]:
            raise ValueError('Three matched seeds required for the full review.')
        row = dict(dataset=key[0], partition=key[1], method=rows[0]['method'], variant=key[2])
        for metric in ('finalAccuracy', 'primaryAccuracy', 'worstClient', 'worstMeanClassRecall', 'honestFinalAccuracy'):
            row[metric] = float(np.mean([r[metric] for r in rows]))
            row[metric + 'SD'] = float(np.std([r[metric] for r in rows], ddof=1))
        summary.append(row)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Preserve earlier review output; use an empty destination.')
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (('seed_endpoints.csv', seeds), ('summary_endpoints.csv', summary)):
        with (destination / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if figures:
        generate(root, destination / 'main_figures', arms=['cart_full', 'acds_checked', 'sharp_checked'], individual=True)
        generate(root, destination / 'component_review', arms=['cart_full', 'cart_macro', 'cart_ingress', 'cart_commit'])
    result = dict(status='EXPORTED_EVIDENCE_VERIFIED', configurations=len(records),
        checkedFiles=len(checksums), activations=sum(c['nodes']*c['rounds'] for c in plan['configs']),
        independentModels=len(records)*plan['configs'][0]['nodes'],
        maximumClientInferenceDifference=max(v['maximumValidationAccuracyDifference'] for v in verified.values()),
        receiptSha256=sha(root/'independent_validation.json'), inputSha256=inputs,
        reviewerSha256=sha(Path(__file__)), summary=summary,
        requiredProfessorPlotsComplete=False,
        missingEvidence='Matched BASIL runs and the complete clean/attack-only comparison are absent from this export.',
        thesisImagesReplaced=False)
    (destination/'review.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    result = review(args.output, args.destination, figures=not args.no_figures)
    print(json.dumps({k:v for k,v in result.items() if k not in ('summary','inputSha256')}, indent=2))
