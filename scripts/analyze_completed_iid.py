"""Read-only completed-study audit. No TensorFlow, training, or result writes.

Run from the repository root: python scripts/analyze_completed_iid.py
The JSON on stdout is evidence for the post-run Markdown report.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {
    'A': 'A_clean_no_attack_ce_100r_gpu',
    'B': 'B_attack_clean_ce_100r_gpu',
    'C': 'C_attack_noise_ce_100r_gpu',
    'D': 'D_attack_noise_ebm_100r_gpu',
}


def summary(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    if not np.isfinite(a).all():
        raise ValueError('Non-finite diagnostic metric')
    return dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                median=float(np.median(a)), p95=float(np.quantile(a, .95)))


def accuracy_summary(a):
    a = np.asarray(a, dtype=float)
    if a.shape != (100,) or not np.isfinite(a).all():
        raise ValueError('Expected 100 actual finite round measurements')
    return dict(final=float(a[-1]), best=float(a.max()), bestRound=int(a.argmax()),
                round49=float(a[49]), mean0_19=float(a[:20].mean()),
                mean20_79=float(a[20:80].mean()), mean20_99=float(a[20:].mean()),
                mean80_89=float(a[80:90].mean()), mean90_99=float(a[90:].mean()),
                lateChange=float(a[90:].mean()-a[80:90].mean()),
                lateSlope=float(np.polyfit(np.arange(80, 100), a[80:], 1)[0]),
                round19=float(a[19]), round20=float(a[20]), round21=float(a[21]),
                maxPostDropFromRound19=float(a[19]-a[20:].min()))


def parameter_hash(arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.asarray(value, dtype=np.float32)
        h.update(str(a.shape).encode('ascii')); h.update(a.tobytes())
    return h.hexdigest()


def corrupted_selection(record):
    # Sender identity alone is insufficient: source round determines activation.
    return bool(record['selectedSenderByzantine'] and record['inputSnapshotRound'] >= 20)


def audit_condition(letter):
    path = ROOT/'newResults/IID'/FOLDERS[letter]
    manifest = json.loads((path/'run.json').read_text())
    config = json.loads((path/'config.json').read_text())
    assert config == manifest['config']
    assert manifest['status'] == 'completed' and manifest['completedRounds'] == 100
    assert config['executionStartMode'] == 'fresh' and config['executionDevice'] == 'GPU'
    assert config['experimentProtocol'] == 'sequential_basil_iid_v1'
    for key, expected in dict(partitionStrategy='iid', nNodes=10, nRounds=100,
        localEpochs=5, batchSize=512, learningRate=.05, learningRateSchedule='basil_round_decay',
        momentum=0, optimizer='sgd', optimizerStateMode='reset_each_activation',
        basilMemorySize=5, assumedByzantineCount=4, snapshotSelection=True,
        snapshotSelectionRule='lowest_receiver_local_loss', aggregationMode='strict_sequential_handoff',
        weightDecayCoefficient=0, gradientClipNorm=None, snapshotAnchorMu=0,
        snapshotAnchorMuMax=0, monteCarloNoiseSamples=0).items():
        assert config[key] == expected, (letter, key)
    assert config['resolvedAttackerIds'] == ([] if letter == 'A' else [0, 1, 5, 7])
    assert config['localObjective'] == ('source_ebm' if letter == 'D' else 'cross_entropy')
    assert config['useChannelNoise'] == (letter in 'CD')
    assert config['channelNoiseSigmaAbsolute'] == (.01 if letter in 'CD' else 0)
    assert config['channelNoiseSemantics'] == 'paper_absolute_gaussian'
    assert config['attackStartRound'] == 20
    with np.load(path/'metrics.npz', allow_pickle=False) as f:
        metrics = {k: f[k] for k in f.files}
    rows = list(csv.DictReader((path/'round_metrics.csv').open()))
    assert [int(r['round']) for r in rows] == list(range(100))
    for column, key in [('full_test_accuracy', 'fullTestAccuracy'),
        ('mean_node_accuracy', 'averageAccuracy'), ('worst_node_accuracy', 'worstNodeAccuracy'),
        ('full_test_loss', 'fullTestLoss')]:
        np.testing.assert_allclose([float(r[column]) for r in rows], metrics[key], atol=1e-7, rtol=0)
    for c in range(10):
        np.testing.assert_allclose([float(r[f'class_{c}_accuracy']) for r in rows],
                                   metrics['fullTestPerClassAccuracy'][:, c], atol=1e-7, rtol=0)
    np.testing.assert_allclose(metrics['fullTestAccuracy'], metrics['roundNodeAccuracy'][:, 9], atol=1e-7)
    np.testing.assert_allclose(metrics['averageAccuracy'], metrics['roundNodeAccuracy'].mean(axis=1), atol=1e-7)
    np.testing.assert_allclose(metrics['worstNodeAccuracy'], metrics['roundNodeAccuracy'].min(axis=1), atol=1e-7)
    with gzip.open(path/'activation_telemetry.json.gz', 'rt') as f:
        records = json.load(f)
    assert len(records) == 1000
    links, steps, ties, source_distances = {}, [], 0, []
    for i, r in enumerate(records):
        rr, node = divmod(i, 10)
        assert (r['round'], r['nodeId']) == (rr, node)
        senders = [(node-d) % 10 for d in range(5, 0, -1)]
        assert r['memorySenderIds'] == senders
        assert r['memorySnapshotRounds'] == [rr if s < node else rr-1 for s in senders]
        assert [c['senderId'] for c in r['candidates']] == senders
        low = min(c['receiverLocalLoss'] for c in r['candidates'])
        tied = [c for c in r['candidates'] if c['receiverLocalLoss'] <= low+1e-8]
        selected = min(tied, key=lambda c: (node-c['senderId']) % 10)
        assert r['inputSourceNode'] == selected['senderId']
        assert r['inputSnapshotRound'] == selected['snapshotRound']
        ties += len(tied) > 1
        source_distances.append((node-r['inputSourceNode']) % 10)
        if r['inputSnapshotRound'] < 0:
            assert r['inputHash'] == manifest['initialModelHash']
        else:
            assert r['inputHash'] == links[(r['inputSnapshotRound'], r['inputSourceNode'], node)]['transmittedHash']
        assert r['optimizerSteps'] == 50 and len(r['optimizerTelemetry']) == 50
        assert len(r['epochs']) == 5 and r['localSampleCount'] == 5000
        for epoch in r['epochs']:
            assert epoch['samplesSeen'] == 5000 and epoch['batchSizes'] == [512]*9+[392]
        assert r['attackActive'] == (letter != 'A' and node in [0, 1, 5, 7] and rr >= 20)
        if not r['attackActive']:
            assert r['outboundAttackHash'] == r['trainedOutputHash']
        else:
            assert r['outboundAttackHash'] != r['trainedOutputHash']
        assert r['channelNoiseActive'] == (letter in 'CD')
        assert [l['receiverId'] for l in r['outgoingLinks']] == [(node+d) % 10 for d in range(1, 6)]
        for link in r['outgoingLinks']:
            links[rr, node, link['receiverId']] = link
            if letter in 'CD':
                assert link['coordinateSigma'] == .01 and link['noiseL2Norm'] > 0
            else:
                assert link['noiseL2Norm'] == 0 and link['transmittedHash'] == r['outboundAttackHash']
        for step in r['optimizerTelemetry']:
            assert np.isclose(step['ebmCoefficient'], .0001 if letter == 'D' else 0, rtol=1e-6)
            assert np.isclose(step['ebmObjective'], step['baseCrossEntropy']+step['ebmPenalty'], atol=2e-6)
        steps.extend(r['optimizerTelemetry'])
    partition = np.load(path/'partition_indices.npz', allow_pickle=False)
    chunks = [partition[k] for k in partition.files]
    combined = np.concatenate(chunks)
    assert len(chunks) == 10 and all(len(c) == 5000 for c in chunks)
    np.testing.assert_array_equal(np.sort(combined), np.arange(50000))
    labels = np.load(ROOT/'experiments/cache/cifar10/y_train_int32.npy').reshape(-1)
    hist = [np.bincount(labels[c], minlength=10).tolist() for c in chunks]
    assert min(min(row) for row in hist) > 0
    partition_digest = hashlib.sha256(combined.astype(np.int64).tobytes()).hexdigest()
    assert partition_digest == manifest['iidPartitionHash'] == config['expectedPartitionHash']
    assert manifest['initialModelHash'] == config['expectedInitialModelHash']
    for source, digest in manifest['implementationSha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
    assert hashlib.sha256((ROOT/'basil_core/iid_gpu_worker.py').read_bytes()).hexdigest() == manifest['numericalExecution']['gpuWorkerSha256']
    # Verify the repository's actual partition-hash convention, not a new one.
    audit = json.loads((path/'iid_partition_audit.json').read_text())
    assert hist == [n['classCounts'] for n in audit['nodes']]
    with np.load(path/'checkpoint.npz', allow_pickle=False) as f:
        checkpoint = json.loads(str(f['metadata']))
        assert checkpoint['nextActivation'] == 1000 and checkpoint['completedRounds'] == 100
        assert checkpoint['config'] == config
        for digest, count in checkpoint['weights'].items():
            params = [f[f'{digest}_{j}'] for j in range(count)]
            assert parameter_hash(params) == digest and all(np.isfinite(p).all() for p in params)
        assert len(checkpoint['nodeParams']) == 10 and all(len(m) == 5 for m in checkpoint['memories'])
    for node in range(10):
        with np.load(path/f'node_{node}_weights.npz', allow_pickle=False) as f:
            params = [f[k] for k in f.files]
            assert sum(p.size for p in params) == 117706
            assert parameter_hash(params) == records[990+node]['trainedOutputHash']
    data_digest, count, unchanged, previous = hashlib.sha256(), 0, 0, None
    for line in (path/'audit/batch_trace.jsonl').open():
        b = json.loads(line)
        activation, index = divmod(count, 50)
        rr, node = divmod(activation, 10)
        epoch, batch = divmod(index, 10)
        assert (b['round_id'], b['node_id'], b['epoch'], b['batch']) == (rr, node, epoch+1, batch)
        assert b['legacy_lambda'] == 0 and np.isclose(b['coefficient'], .0001 if letter == 'D' else 0, rtol=1e-6)
        assert np.isclose(b['learning_rate'], .05/(1+.05*rr))
        if index == 0:
            assert b['parameterHashBefore'] == records[activation]['inputHash']
        else:
            assert b['parameterHashBefore'] == previous
        previous = b['parameterHashAfter']
        if index == 49:
            assert previous == records[activation]['trainedOutputHash']
        unchanged += b['parameterHashBefore'] == previous
        for name in ['parameterStatsBefore', 'gradients', 'parameterStatsAfter']:
            for tensor in b[name]:
                assert tensor['nan'] == tensor['positiveInf'] == tensor['negativeInf'] == 0
        for key in ['indicesHash', 'imagesHash', 'labelsHash', 'augmentedImagesHash']:
            data_digest.update(b[key].encode())
        count += 1
    assert count == 50000
    confusions = json.loads((path/'audit/confusions.json').read_text())
    assert len(confusions) == 6000
    last = [c for c in confusions if c['round'] == 99 and c['node'] == 9][-1]
    cm = np.asarray(last['confusion'])
    assert cm.shape == (10, 10) and np.all(cm.sum(axis=1) == 1000)
    np.testing.assert_allclose(np.diag(cm)/1000, metrics['fullTestPerClassAccuracy'][-1], atol=1e-7)
    corrupt = [r for r in records if corrupted_selection(r)]
    out = dict(folder=FOLDERS[letter], manifest=manifest, config=config,
        accuracy=accuracy_summary(metrics['fullTestAccuracy']),
        finalMean=float(metrics['averageAccuracy'][-1]), finalWorst=float(metrics['worstNodeAccuracy'][-1]),
        finalLoss=float(metrics['fullTestLoss'][-1]), perClass=metrics['fullTestPerClassAccuracy'][-1].tolist(),
        perNode=metrics['roundNodeAccuracy'][-1].tolist(),
        nodeStd=float(metrics['roundNodeAccuracy'][-1].std()), nodeSpread=float(np.ptp(metrics['roundNodeAccuracy'][-1])),
        designatedSelections=sum(r['selectedSenderByzantine'] for r in records),
        corruptedSelections=len(corrupt), corruptedReceivers=sorted({r['nodeId'] for r in corrupt}),
        corruptedRounds=sorted({r['round'] for r in corrupt}),
        postDesignatedSelections=sum(r['selectedSenderByzantine'] for r in records[200:]),
        selectionCounts=np.bincount([r['inputSourceNode'] for r in records], minlength=10).tolist(),
        selectionDistances=np.bincount(source_distances, minlength=6).tolist(), selectionTies=ties,
        snapshotAges=summary([r['selectedSnapshotAgeRounds'] for r in records if r['selectedSnapshotAgeRounds'] is not None]),
        partitionHistogram=hist, partitionBytesHash=partition_digest, dataAugmentationDigest=data_digest.hexdigest(),
        batches=count, unchangedUpdates=unchanged, transmissions=len(links), checkpointWeightStates=len(checkpoint['weights']),
        predictionHistogram=cm.sum(axis=0).tolist(), confusion=cm.tolist(),
        roundAggregates={k: summary([float(r[k]) for r in rows if r[k]]) for k in rows[0] if k not in ['round']},
        stepAggregates={k: summary([s[k] for s in steps]) for k in ['baseCrossEntropy', 'trainingBatchAccuracy',
            'ebmPenalty', 'baseGradientNorm', 'ebmCorrectionNorm', 'ebmCorrectionRatio', 'robustObjectiveGradientNorm', 'parameterNorm']},
        lateStepAggregates={k: summary([s[k] for r in records[900:] for s in r['optimizerTelemetry']]) for k in
            ['baseCrossEntropy', 'trainingBatchAccuracy', 'ebmPenalty', 'ebmCorrectionRatio']},
        noiseStats={k: summary([l[k] for l in links.values()]) for k in ['noiseL2Norm', 'relativeNoiseL2', 'modelL2Norm']},
        noiseHonest=summary([l['relativeNoiseL2'] for r in records if not r['attackActive'] for l in r['outgoingLinks']]),
        noiseAttacked=summary([l['relativeNoiseL2'] for r in records if r['attackActive'] for l in r['outgoingLinks']]),
        failureFiles=[str(f.relative_to(path)) for f in path.rglob('*failure*')],
        maxCorrectionContext=max((dict(round=r['round'], node=r['nodeId'], **s) for r in records for s in r['optimizerTelemetry']),
                                 key=lambda s: s['ebmCorrectionRatio']))
    for field in ['config', 'finalPerNodeAccuracy', 'finalPerClassAccuracy', 'roundAggregateStatistics']:
        out['manifest'].pop(field, None)
    return out, links


def main():
    results, links = {}, {}
    for letter in FOLDERS:
        results[letter], links[letter] = audit_condition(letter)
    assert len({r['dataAugmentationDigest'] for r in results.values()}) == 1
    assert len({r['partitionBytesHash'] for r in results.values()}) == 1
    assert len({r['manifest']['initialModelHash'] for r in results.values()}) == 1
    for key in links['C']:
        assert links['C'][key]['noiseHash'] == links['D'][key]['noiseHash']
    assert len({l['noiseHash'] for l in links['C'].values()}) == 5000
    print(json.dumps(dict(conditions=results, pairedNoiseHashes=5000,
                         pairedBatchesPerCondition=50000, distinctNoiseHashesPerCondition=5000), indent=2))


if __name__ == '__main__':
    main()
