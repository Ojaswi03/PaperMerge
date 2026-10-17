#!/usr/bin/env python3
"""Read completed Dirichlet telemetry; emit diagnostics to stdout, with zero SGD.

Does not write reports, checkpoints, approvals, queues, or tracking state.
Squared-norm accounting follows selected ancestry, not every delivered link.
These are descriptive measurements, not counterfactual accuracy effects.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unique_match(items, predicate, description):
    matches=[item for item in items if predicate(item)]
    if len(matches)!=1:
        raise ValueError(f'Exactly one {description} required.')
    return matches[0]


def squared_norm(value):
    value=float(value)
    if not math.isfinite(value) or value<0 or not math.isfinite(value*value):
        raise ValueError('Saved parameter norms must be finite, nonnegative and squareable.')
    return value**2


def verify_source_archives(directory, fingerprints, root=ROOT):
    # Archived bytes establish executed provenance. Current-source differences
    # are descriptive here; they remain blockers for operational exact resume.
    for name,digest in fingerprints.items():
        if sha(Path(directory)/'source_snapshots'/digest)!=digest:
            raise ValueError(f'Executed-source archive hash mismatch: {name}')
    return {name:sha(Path(root)/name)==digest for name,digest in fingerprints.items()}


def selected_ancestry(records):
    """Telescope ||theta||² into attack, received-link, and local-training terms.

    Link terms include the realized cross term 2*outbound.dot(noise), and
    selection conditions on that noise. No random-walk independence assumption.
    Norms are saved scalar telemetry; this is not a parameter-distance audit.
    """
    ancestry = []
    for index, record in enumerate(records):
        if (record['round'], record['nodeId']) != divmod(index, 10):
            raise ValueError('Contiguous ordered activations required.')
        chosen = unique_match(record['candidates'],lambda c:
            (c['senderId'],c['snapshotRound']) ==
            (record['inputSourceNode'],record['inputSnapshotRound']), 'selected candidate')
        received_energy = squared_norm(chosen['receivedModelNorm'])
        trained_energy = squared_norm(record['modelL2Norm'])
        parent = record['inputSnapshotRound'] * 10 + record['inputSourceNode']
        if record['inputSnapshotRound'] < 0:
            previous = dict(initialEnergy=received_energy, attackEnergy=0.,
                            channelEnergy=0., trainingEnergy=0., links=0,
                            attackedLinks=0)
            attack = channel = 0.
        else:
            if not 0 <= parent < index:
                raise ValueError('Selected parent must precede its receiver.')
            source = records[parent]
            link = unique_match(source['outgoingLinks'],lambda l:
                l['receiverId']==record['nodeId'], 'selected directed link')
            if link['transmittedHash'] != record['inputHash']:
                raise ValueError('Selected link/input hash mismatch.')
            previous = ancestry[parent]
            outbound_energy = squared_norm(link['modelL2Norm'])
            attack = outbound_energy - squared_norm(source['modelL2Norm'])
            channel = received_energy - outbound_energy
        result = dict(previous)
        result['attackEnergy'] += attack
        result['channelEnergy'] += channel
        result['trainingEnergy'] += trained_energy - received_energy
        result['links'] += int(record['inputSnapshotRound'] >= 0)
        result['attackedLinks'] += int(record['inputSnapshotRound'] >= 0 and records[parent]['attackActive'])
        result['closureResidual'] = (trained_energy -
            sum(result[k] for k in ('initialEnergy', 'attackEnergy', 'channelEnergy', 'trainingEnergy')))
        if not math.isclose(result['closureResidual'], 0., abs_tol=1e-7):
            raise ValueError('Selected ancestry squared-norm accounting failed.')
        ancestry.append(result)
    return ancestry


def analyze(directory):
    sys.path.insert(0, str(ROOT))
    from basil_core.dirichlet_diagnosis import summarize_activations
    from scripts.check_completed_round_results import metrics

    directory = Path(directory)
    names = ('run.json', 'partition_audit.json', 'metrics.npz',
             'round_metrics.csv', 'activation_telemetry.json.gz')
    before = {name: sha(directory / name) for name in names}
    manifest = json.loads((directory / 'run.json').read_text())
    config = manifest['config']
    if (config.get('horizonStudyRevision') != 'dirichlet_prior_fixed_horizons_v1'
            or config.get('nRounds') not in (100,500)):
        raise ValueError('Expected the completed fixed-horizon study family.')
    measured = metrics(directory)
    partition = json.loads((directory / 'partition_audit.json').read_text())
    with gzip.open(directory / 'activation_telemetry.json.gz', 'rt') as stream:
        records = json.load(stream)
    contracts = summarize_activations(records, partition, config)
    ancestry = selected_ancestry(records)
    windows = {}
    for start, stop in ((0, 20), (40, 60), (80, 100), (180, 200),
                        (267, 287), (392, 412), (480, 500)):
        if stop > config['nRounds']:
            continue
        selected = records[start * 10:stop * 10]
        steps = [s for r in selected for s in r['optimizerTelemetry']]
        path_bounds = [r['learningRate'] * sum(s['robustObjectiveGradientNorm']
                      for s in r['optimizerTelemetry']) for r in selected]
        noise_norms = []
        for record in selected:
            if record['inputSnapshotRound'] < 0:
                noise_norms.append(0.)
            else:
                parent = records[record['inputSnapshotRound'] * 10 + record['inputSourceNode']]
                noise_norms.append(next(link['noiseL2Norm'] for link in parent['outgoingLinks']
                                        if link['receiverId'] == record['nodeId']))
        windows[f'{start}-{stop-1}'] = dict(
            meanAccuracyPct=100 * mean(r['globalTestAccuracy'] for r in selected),
            meanTrainedNorm=mean(r['modelL2Norm'] for r in selected),
            meanStepNorm=mean(r['learningRate'] * s['robustObjectiveGradientNorm']
                              for r in selected for s in r['optimizerTelemetry']),
            meanActivationPathLengthUpperBound=mean(path_bounds),
            meanSelectedLinkNoiseNorm=mean(noise_norms),
            fractionPathBoundBelowIncomingNoise=mean(float(bound < noise)
                                                     for bound, noise in zip(path_bounds, noise_norms)),
            **{k: mean(s[k] for s in steps) for k in (
                'baseCrossEntropy', 'ordinaryCrossEntropy', 'baseGradientNorm',
                'robustObjectiveGradientNorm', 'ebmCorrectionRatio', 'anchorCorrectionRatio')})
    by_batch_size = {}
    for size in sorted({s['samples'] for r in records for s in r['optimizerTelemetry']}):
        steps = [s for r in records for s in r['optimizerTelemetry'] if s['samples'] == size]
        by_batch_size[size] = dict(count=len(steps),
            meanEbmCorrectionRatio=mean(s['ebmCorrectionRatio'] for s in steps))
    source_matches = verify_source_archives(directory,manifest['implementationSha256'])
    if before != {name: sha(directory / name) for name in names}:
        raise ValueError('Evidence changed while being read.')
    first = contracts['firstCorruptedSelection']
    before_first = records[(first['round'] - 1) * 10:first['round'] * 10] if first else []
    result = dict(condition=config['conditionId'], rounds=config['nRounds'],
        optimizerUpdatesPerformed=0, sourceHashes=before,
        analysisSourceSha256=sha(Path(__file__)),
        currentExecutedSourceMatches=source_matches, archivedSourcesMatch=True,
        savedContracts=contracts, finalMeanPct=100 * measured['finalMean'],
        bestMeanRound=measured['bestMeanRound'],
        meanPctRoundBeforeFirstCorruptSelection=(100 * mean(r['globalTestAccuracy'] for r in before_first)
                                               if before_first else None),
        windows=windows, byBatchSize=by_batch_size,
        finalNodeSelectedAncestry=ancestry[-10:])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=ROOT / 'newResults/nonIID/dirichlet_prior_anchor_500r_v1')
    args = parser.parse_args()
    results = {}
    for condition in 'ABCD':
        result = analyze(args.source / condition)
        if result['condition']!=condition:
            raise ValueError(f'Condition identity mismatch in {args.source / condition}')
        results[condition] = result
        gc.collect()
        print(f'{condition}: read-only mechanism analysis complete', file=sys.stderr, flush=True)
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
