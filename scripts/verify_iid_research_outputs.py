"""Verify the authorized paired 50-round outputs without running training."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / 'newResults/IID'
NAMES = ('test_01_basil_iid_no_noise_ce', 'test_02_basil_iid_noise_ebm')


def verify(name, noisy):
    directory = ROOT / name
    manifest = json.loads((directory / 'run.json').read_text())
    assert manifest['status'] == 'completed', manifest['status']
    assert manifest['completedRounds'] == manifest['nRounds'] == 50
    assert manifest['researchValid'] is True
    assert manifest['resolvedAttackerIds'] == [0, 1, 5, 7]
    assert manifest['trainingSamplesUsed'] == 50000
    assert manifest['localSampleCounts'] == [5000] * 10
    assert manifest['evaluationSamples'] == 10000
    assert manifest['evaluationClassCounts'] == [1000] * 10
    assert manifest['sigma_e'] == (.01 if noisy else 0.)
    assert manifest['localObjective'] == ('source_ebm' if noisy else 'cross_entropy')
    for source, expected_hash in manifest['implementationSha256'].items():
        assert hashlib.sha256((REPOSITORY / source).read_bytes()).hexdigest() == expected_hash, source
    with (directory / 'round_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 50
    assert [int(row['round']) for row in rows] == list(range(50))
    assert all(np.isfinite(float(value)) for row in rows for value in row.values() if value)
    with np.load(directory / 'partition_indices.npz', allow_pickle=False) as saved:
        indices = [saved[f'node_{i}'] for i in range(10)]
    assert all(len(index) == 5000 for index in indices)
    all_indices = np.concatenate(indices)
    assert np.array_equal(np.sort(all_indices), np.arange(50000))
    digest = hashlib.sha256()
    for index in indices:
        digest.update(np.asarray(index, dtype='<i8').tobytes())
    assert digest.hexdigest() == manifest['iidPartitionHash']
    with gzip.open(directory / 'activation_telemetry.json.gz', 'rt') as handle:
        records = json.load(handle)
    assert len(records) == 500
    by_activation = {(r['round'], r['nodeId']): r for r in records}
    assert len(by_activation) == 500
    transmissions = {}
    steps = 0
    attacked = 0
    for r in records:
        round_id, node = r['round'], r['nodeId']
        assert r['localSampleCount'] == 5000
        assert r['optimizerSteps'] == 50
        assert len(r['epochs']) == 5
        assert [e['epoch'] for e in r['epochs']] == [1, 2, 3, 4, 5]
        assert all(e['samplesSeen'] == 5000 and e['batchSizes'] == [512] * 9 + [392] for e in r['epochs'])
        assert set(r['memorySenderIds']) == {(node - distance) % 10 for distance in range(1, 6)}
        assert len(set(r['memorySenderIds'])) == 5
        candidates = r['candidates']
        assert abs(r['selectedCandidateLoss'] - min(c['receiverLocalLoss'] for c in candidates)) <= 1e-8
        assert r['attackActive'] == (node in [0, 1, 5, 7] and round_id >= 20)
        attacked += r['attackActive']
        assert r['channelNoiseActive'] is noisy
        assert len(r['outgoingLinks']) == 5
        for link in r['outgoingLinks']:
            assert link['coordinateSigma'] == (.01 if noisy else 0.)
            assert link['parameterCount'] == 117706
            assert link['noiseL2Norm'] > 0 if noisy else link['noiseL2Norm'] == 0
            if not noisy:
                assert link['transmittedHash'] == r['outboundAttackHash']
            transmissions[(round_id, node, link['receiverId'])] = link['transmittedHash']
        for step in r['optimizerTelemetry']:
            assert np.isclose(step['ebmCoefficient'], .0001 if noisy else 0., rtol=1e-6)
            assert np.isclose(step['ebmPenalty'], step['ebmCoefficient'] * step['gradientNormSquared'], rtol=1e-6, atol=1e-10)
            if not noisy:
                assert step['ebmCorrectionNorm'] == step['ebmPenalty'] == 0
            steps += 1
        if r['inputSnapshotRound'] >= 0:
            assert r['inputHash'] == transmissions[(r['inputSnapshotRound'], r['inputSourceNode'], node)]
    assert steps == 25000
    assert attacked == 120
    for node, expected in ((0, {i: 10 for i in [5, 6, 7, 8, 9]}), (1, {**{i: 10 for i in [6, 7, 8, 9]}, 0: 11})):
        r = by_activation[(11, node)]
        assert dict(zip(r['memorySenderIds'], r['memorySnapshotRounds'])) == expected
    batch_count = 0
    first_batch_hashes = {}
    data_stream_hash = hashlib.sha256()
    with (directory / 'audit/batch_trace.jsonl').open() as handle:
        for line in handle:
            trace = json.loads(line)
            data_stream_hash.update(json.dumps({k: trace[k] for k in ('round_id', 'node_id', 'epoch', 'batch', 'indicesHash', 'imagesHash', 'labelsHash', 'augmentedImagesHash')}, sort_keys=True).encode())
            key = (trace['round_id'], trace['node_id'])
            r = by_activation[key]
            if trace['epoch'] == 1 and trace['batch'] == 0:
                assert trace['parameterHashBefore'] == r['inputHash']
                first_batch_hashes[key] = {k: trace[k] for k in ('indicesHash', 'imagesHash', 'labelsHash', 'augmentedImagesHash')}
            if trace['epoch'] == 5 and trace['batch'] == 9:
                assert trace['parameterHashAfter'] == r['trainedOutputHash']
            batch_count += 1
    assert batch_count == 25000
    with np.load(directory / 'metrics.npz', allow_pickle=False) as metrics:
        assert metrics['fullTestAccuracy'].shape == (50,)
        assert metrics['fullTestPerClassAccuracy'].shape == (50, 10)
        assert metrics['roundNodeAccuracy'].shape == (50, 10)
        assert metrics['roundNodePerClassAccuracy'].shape == (50, 10, 10)
        assert np.allclose(metrics['fullTestAccuracy'], metrics['roundNodeAccuracy'][:, 9])
        assert np.allclose(metrics['averageAccuracy'], metrics['roundNodeAccuracy'].mean(axis=1))
        assert np.allclose(metrics['worstNodeAccuracy'], metrics['roundNodeAccuracy'].min(axis=1))
        assert np.allclose(metrics['fullTestAccuracy'], [float(r['full_test_accuracy']) for r in rows])
    return manifest, first_batch_hashes, dict(name=name, rounds=50, activations=500, optimizerSteps=steps, transmissions=len(transmissions), attackActiveActivations=attacked, strictHandoffHashChecks=500, fullEpochChecks=2500, round11MemoryMatches=True, dataAugmentationStreamSha256=data_stream_hash.hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-evidence', action='store_true', help='Save verification.json alongside the paired results')
    args = parser.parse_args()
    a, data_a, evidence_a = verify(NAMES[0], False)
    b, data_b, evidence_b = verify(NAMES[1], True)
    for field in ('iidPartitionHash', 'initialModelHash', 'resolvedAttackerIds', 'seedStreams', 'learningRateSchedule', 'attackType', 'attackStartRound'):
        assert a[field] == b[field], field
    assert data_a == data_b, 'Paired data order/augmentation hashes differ'
    assert evidence_a['dataAugmentationStreamSha256'] == evidence_b['dataAugmentationStreamSha256'], 'Paired batch data/augmentation streams differ'
    evidence = {'runs': [evidence_a, evidence_b], 'pairedDataAugmentationActivationChecks': len(data_a), 'pairedBatchDataAugmentationChecks': 25000, 'partitionAndInitializationHashesMatch': True, 'implementationSourceHashesMatch': True}
    encoded = json.dumps(evidence, indent=2) + '\n'
    if args.write_evidence:
        (ROOT / 'comparison/artifact_verification.json').write_text(encoded)
    print(encoded)
