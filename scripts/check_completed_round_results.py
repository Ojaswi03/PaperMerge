#!/usr/bin/env python3
"""Read-only audit of saved Dirichlet 100/500-round results; never train."""
from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def metrics(path):
    import numpy as np

    manifest = load(path/'run.json')
    config = manifest['config']
    rounds = config['nRounds']
    with (path/'round_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert manifest['status'] == 'completed'
    assert manifest['completedRounds'] == rounds
    if config.get('nonIID'):
        assert manifest['completedActivations'] == rounds*10
    assert [int(row['round']) for row in rows] == list(range(rounds))
    with np.load(path/'metrics.npz', allow_pickle=False) as packed:
        for key in packed.files:
            if packed[key].dtype.kind in 'fc':
                assert np.isfinite(packed[key]).all(), key
        for field, key in [('full_test_accuracy', 'fullTestAccuracy'),
                           ('mean_node_accuracy', 'averageAccuracy'),
                           ('worst_node_accuracy', 'worstNodeAccuracy'),
                           ('full_test_loss', 'fullTestLoss')]:
            np.testing.assert_allclose([float(row[field]) for row in rows], packed[key], atol=1e-8, rtol=0)
        np.testing.assert_allclose(packed['averageAccuracy'], packed['roundNodeAccuracy'].mean(axis=1), atol=1e-8, rtol=0)
        np.testing.assert_allclose(packed['worstNodeAccuracy'], packed['roundNodeAccuracy'].min(axis=1), atol=1e-8, rtol=0)
        np.testing.assert_array_equal(packed['fullTestAccuracy'], packed['roundNodeAccuracy'][:, 9])
        # Different float32 reduction orders can differ by a few ULPs.
        # Float64 class averaging avoids introducing another float32 sum.
        np.testing.assert_allclose(packed['fullTestPerClassAccuracy'].astype(np.float64).mean(axis=1),
                                   packed['fullTestAccuracy'], atol=2e-7, rtol=0)
        avg = packed['averageAccuracy'].copy()
        full = packed['fullTestAccuracy'].copy()
        nodes = packed['roundNodeAccuracy'][-1].copy()
        result = dict(status=manifest['status'], rounds=rounds,
            activations=manifest.get('completedActivations'), researchValid=manifest['researchValid'],
            initialBeforeNode0=(float(packed['beforeTrainingAccuracy'][0, 0])
                if 'beforeTrainingAccuracy' in packed else manifest.get('initialGlobalAccuracy')),
            firstRoundMean=float(avg[0]), finalMean=float(avg[-1]),
            bestMean=float(avg.max()), bestMeanRound=int(avg.argmax()),
            finalFull=float(full[-1]), bestFull=float(full.max()), bestFullRound=int(full.argmax()),
            finalWorst=float(packed['worstNodeAccuracy'][-1]), finalLoss=float(packed['fullTestLoss'][-1]),
            nodes=nodes.tolist(), nodeStd=float(nodes.std()), nodeSpread=float(nodes.max()-nodes.min()),
            classes=packed['fullTestPerClassAccuracy'][-1].tolist())
    for key in ('runtimeSeconds', 'partitionHash', 'initialModelHash', 'config', 'implementationSha256',
                'testUsage', 'multiprocessingUsed', 'legacyLambdaIgnored', 'numericalExecution', 'timestamp'):
        result[key] = manifest.get(key)
    result['partitionHash'] = manifest.get('partitionHash', manifest.get('iidPartitionHash'))
    result.update(device=manifest.get('deviceName'), memory=manifest.get('memorySafety'), tracking=manifest.get('liveTracking'))
    for key, array in [('mean', avg), ('full', full)]:
        windows = [(0, 20), (20, 80), (80, 90), (90, 100), (rounds-20, rounds-10), (rounds-10, rounds)]
        result[key+'Windows'] = {f'{start}-{end-1}': float(array[start:end].mean())
            for start, end in windows if 0 <= start < end <= rounds}
        result[key+'AtRounds'] = {str(i): float(array[i]) for i in (19, 20, 21, 29, 49, 99, 199, 299, 399, 499) if i < rounds}
        result[key+'LateSlopePpPerRound'] = float(np.polyfit(np.arange(rounds-20, rounds), 100*array[-20:], 1)[0])
    return result


def main():
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from basil_core.dirichlet_diagnosis import summarize_activations
    from reporting.dirichlet_calibration import verify_final_transmissions

    out = dict(nonIID={}, IID={}, integrity={}, queue=[])
    paths = []
    for horizon in (100, 500):
        for condition in 'ABCD':
            directory = ROOT/f'newResults/nonIID/dirichlet_prior_anchor_{horizon}r_v1/{condition}'
            paths.extend(directory/name for name in ('run.json', 'config.json', 'metrics.npz', 'round_metrics.csv',
                'activation_telemetry.json.gz', 'checkpoint.npz', 'partition_audit.json'))
    for directory in (ROOT/'newResults/IID').glob('*500r_gpu'):
        paths.extend(directory/name for name in ('run.json', 'metrics.npz', 'round_metrics.csv', 'checkpoint.npz')
                     if (directory/name).exists())
    before = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    for horizon in (100, 500):
        out['nonIID'][str(horizon)] = {c: metrics(ROOT/f'newResults/nonIID/dirichlet_prior_anchor_{horizon}r_v1/{c}') for c in 'ABCD'}
    iid_names = dict(A='A_clean_no_attack_ce_500r_gpu', B='B_attack_clean_ce_500r_gpu',
                     C='C_attack_noise_ce_500r_gpu', D='D_attack_noise_ebm_500r_gpu')
    out['IID'] = {c: metrics(ROOT/'newResults/IID'/name) for c, name in iid_names.items()}
    channel = {}
    prefix = {}
    for condition in 'ABCD':
        directory = ROOT/f'newResults/nonIID/dirichlet_prior_anchor_500r_v1/{condition}'
        manifest = load(directory/'run.json')
        config = manifest['config']
        partition = load(directory/'partition_audit.json')
        with gzip.open(directory/'activation_telemetry.json.gz', 'rt') as stream:
            records = json.load(stream)
        contracts = summarize_activations(records, partition, config)
        keys = list(records[0]['optimizerTelemetry'][0])
        steps = np.array([[step[key] for key in keys] for record in records for step in record['optimizerTelemetry']], dtype=np.float64)
        assert np.isfinite(steps).all()
        values = {key: steps[:, i] for i, key in enumerate(keys)}
        coefficient = .0001 if condition == 'D' else 0.
        np.testing.assert_allclose(values['ebmCoefficient'], coefficient, atol=1e-12, rtol=1e-6)
        np.testing.assert_allclose(values['ebmPenalty'], values['ebmCoefficient']*values['gradientNormSquared'], atol=1e-9, rtol=1e-6)
        np.testing.assert_allclose(values['snapshotAnchorPenalty'], .005*values['snapshotDistanceSquared'], atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(values['ebmObjective'], values['baseCrossEntropy']+values['ebmPenalty']+values['snapshotAnchorPenalty'], atol=1e-6, rtol=1e-6)
        attacked = designated = 0
        links, noise_hashes, ages = [], [], []
        for record in records:
            expected = condition != 'A' and record['nodeId'] in [0, 1, 5, 7] and record['round'] >= 20
            assert record['attackActive'] == expected
            assert (record['trainedOutputHash'] != record['outboundAttackHash']) == expected
            assert record['legacyLambdaIgnored'] and len(record['outgoingLinks']) == 5
            assert math.isclose(record['learningRate'], .05/(1+.05*record['round']), rel_tol=1e-12)
            attacked += int(expected)
            designated += int(record['inputSourceNode'] in config['resolvedAttackerIds'])
            if record['inputSnapshotRound'] >= 0:
                ages.append(record['round']-record['inputSnapshotRound'])
            assert [link['receiverId'] for link in record['outgoingLinks']] == [(record['nodeId']+d) % 10 for d in range(1, 6)]
            for link in record['outgoingLinks']:
                assert link['coordinateSigma'] == (.01 if condition in 'CD' else 0.)
                assert bool(link['noiseL2Norm']) == (condition in 'CD')
                links.append(link)
                if condition in 'CD':
                    noise_hashes.append(link['noiseHash'])
        if condition in 'CD':
            channel[condition] = noise_hashes
        archives = all((directory/'source_snapshots'/digest).is_file() and sha(directory/'source_snapshots'/digest) == digest
                       for digest in manifest['implementationSha256'].values())
        assert archives
        result = out['nonIID']['500'][condition]
        result['audit'] = dict(contracts=contracts, finiteStepValues=True, checkedSteps=len(steps),
            attackedActivations=attacked, attackerDesignatedSelections=designated,
            selectionAgeMean=float(np.mean(ages)), directedLinks=len(links), uniqueNoiseHashes=len(set(noise_hashes)),
            archivedSourceHashesValid=archives,
            failureFiles=[str(path.relative_to(directory)) for path in directory.rglob('*failure*') if path.is_file()],
            initialTestClassCounts=load(directory/'audit/initial.json')['testClassCounts'])
        late = np.array([record['round'] >= 480 for record in records for _ in record['optimizerTelemetry']])
        result['steps'] = {key: dict(mean=float(values[key].mean()), max=float(values[key].max()), lateMean=float(values[key][late].mean()))
            for key in ('baseCrossEntropy', 'ordinaryCrossEntropy', 'ebmPenalty', 'baseGradientNorm', 'ebmCorrectionNorm',
                        'ebmCorrectionRatio', 'anchorCorrectionRatio', 'snapshotAnchorPenalty', 'parameterNorm',
                        'snapshotDistanceSquared', 'robustObjectiveGradientNorm')}
        result['noise'] = {key: dict(mean=float(np.mean([link[key] for link in links])), min=float(min(link[key] for link in links)),
            max=float(max(link[key] for link in links)), lateMean=float(np.mean([link[key] for link in links[-1000:]])))
            for key in ('noiseL2Norm', 'relativeNoiseL2', 'modelL2Norm')}
        del records, steps, values, links
        gc.collect()
        with np.load(directory/'metrics.npz') as big, np.load(ROOT/f'newResults/nonIID/dirichlet_prior_anchor_100r_v1/{condition}/metrics.npz') as small:
            prefix[condition] = all(np.array_equal(small[key], big[key][:100])
                for key in ('roundNodeAccuracy', 'fullTestLoss', 'selectedSources', 'selectedSnapshotRounds'))
        print(f'{condition}: saved training/selection/noise contracts verified', file=sys.stderr, flush=True)
    assert channel['C'] == channel['D']
    out['pairedNoisyLinks'] = len(channel['C'])
    out['100rMatches500rPrefix'] = prefix
    out['finalTransmissionAudit'] = verify_final_transmissions(ROOT/'newResults/nonIID/dirichlet_prior_anchor_500r_v1', tuple('ABCD'))
    out['integrity'] = dict(rawFilesChecked=len(before), changed=[name for name, digest in before.items() if sha(ROOT/name) != digest],
        scope='Read-only analysis before/after; not a reconstruction of historical pre-cleanup hashes.')
    out['queue'] = [dict(condition=r['config']['conditionId'], rounds=r['config']['nRounds'], status=r['status'], error=r.get('error'))
                    for r in load(ROOT/'gui/queue_state.json')]
    # Derived analysis only; original manifests/checkpoints remain read-only.
    destination = ROOT/'newResults/nonIID/500round/COMPLETED_RESULTS_AUDIT.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(destination), integrity=out['integrity'], pairedNoisyLinks=out['pairedNoisyLinks'])))


if __name__ == '__main__':
    main()
