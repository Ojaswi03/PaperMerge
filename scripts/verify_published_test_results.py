#!/usr/bin/env python3
"""Read-only recomputation of the eight published official-test summaries."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reporting.official_test_results import load_results
from scripts.render_official_test_figures import verify_recall


def verify_eurosat():
    """Recompute confidence diagnostics without inventing land-cover accuracy."""
    base = ROOT / 'docs/results/official-test-20261110'
    folder = base / 'eurosat-ood-20261110'
    receipt = json.loads((folder / 'results.json').read_text())
    reference = {}
    for name in ('full-official-test-20261110', 'basil-full-test-20261110'):
        source = base / name
        for row in json.loads((source / 'results.json').read_text())['results']:
            if row['dataset'] == 'cifar10':
                reference[row['runId']] = source / row['runId'] / 'predictions.npz'
    labels = ('airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    count = 0
    for row in receipt['results']:
        with np.load(folder / (row['runId'] + '.npz'), allow_pickle=False) as data:
            confidence = data['predictionConfidence']
            predicted = data['predictedCifar10Class']
        with np.load(reference[row['runId']], allow_pickle=False) as data:
            familiar = data['predictionConfidence']
        assert confidence.shape == predicted.shape == (100, 2000)
        expected = [row['eurosatMeanConfidence'], row['cifar10MeanConfidence'],
                    row['eurosatConfidenceAtLeast90']]
        observed = [confidence.mean(), familiar.mean(), (confidence >= .9).mean()]
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(np.bincount(predicted.ravel(), minlength=10),
            [row['cifar10LabelCounts'][label] for label in labels])
        auc = []
        for positive, negative in zip(familiar, confidence):
            ordered = np.sort(negative)
            left = np.searchsorted(ordered, positive, side='left')
            right = np.searchsorted(ordered, positive, side='right')
            auc.append(np.mean((left + right) / (2 * len(ordered))))
        np.testing.assert_allclose(np.mean(auc), row['meanConfidenceAUROC'], rtol=0, atol=1e-12)
        count += predicted.size
    assert count == 800_000
    return count


if __name__ == '__main__':
    rows, predictions = load_results()
    verify_recall(rows)
    print(json.dumps(dict(runs=len(rows), clientImagePredictions=predictions,
        savedAccuracyLossAndRecallVerified=True,
        eurosatDiagnosticPredictionsVerified=verify_eurosat(),
        inferenceRepeated=False), indent=2))
