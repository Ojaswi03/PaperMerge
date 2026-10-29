"""Line-only figures and endpoints for the matched contribution matrix."""
from pathlib import Path
from reporting.cart_publication_curves import collect, generate, sha

GROUPS = dict(
    comparison=('cart_full', 'acds_checked', 'sharp_checked', 'local_only', 'acds_single'),
    admission_and_commit=('cart_full', 'cart_macro', 'cart_ingress', 'cart_commit'),
    training_and_repair=('cart_full', 'cart_no_prior', 'cart_no_calibration', 'cart_no_repair'))


def report(root):
    import csv
    import json
    import numpy as np
    root = Path(root)
    plan, records, inputs = collect(root)
    destinations = dict(all_methods=None, **GROUPS)
    for name, arms in destinations.items():
        if arms is not None and not any(r['arm'] in arms for r in records):
            continue
        destination = root / 'publication_curves' / name
        manifest = destination / 'learning_curves_manifest.json'
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old['producerSource'] != plan['source'] or old['inputSha256'] != inputs or any(
                    sha(destination / path) != value for path, value in old['outputSha256'].items()):
                raise ValueError('Existing publication figures differ; preserve them before regenerating.')
        else:
            generate(root, destination, arms=arms)
    # One row per independent seed, not one independent replicate per client.
    destination = root / 'publication_curves' / 'seed_endpoints.csv'
    with destination.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'partition', 'method', 'seed', 'finalAccuracy',
            'lastTenRoundAccuracy', 'finalWorstClientAccuracy', 'finalHonestMeanAccuracy'])
        for row in records:
            with np.load(root / row['runId'] / 'metrics.npz', allow_pickle=False) as arrays:
                accuracy = arrays['roundNodeAccuracy']
                honest = arrays['roundHonestMeanAccuracy'][-1]
            writer.writerow([row['dataset'], row['partition'], row['arm'], row['seed'],
                float(accuracy[-1].mean()), float(accuracy[-10:].mean()),
                float(accuracy[-1].min()), float(honest)])
    return len(records)
