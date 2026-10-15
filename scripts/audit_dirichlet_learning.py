#!/usr/bin/env python3
"""Inspect completed Dirichlet evidence; never train or execute a queue.

--probe adds read-only CPU forward/CE-gradient checks on a fixed TRAIN probe.
No GPU allocation, test-set loading, optimizer or W&B session is performed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect(source, *, probe=False, executed_source_commit=None):
    from basil_core.dirichlet_diagnosis import summarize_activations, confusion_summary
    from basil_core.dirichlet_study import verify_pairing
    paths = [source / f'{letter}_dirichlet_alpha02_local_anchor_30r_gpu' for letter in 'ABCD']
    manifests = [json.loads((path / 'run.json').read_text()) for path in paths]
    configs = [m['config'] for m in manifests]
    contrasts = verify_pairing(configs)
    if any(m['status'] != 'completed' or m['completedRounds'] != 30 for m in manifests):
        raise ValueError('All four completed 30-round conditions required.')
    hashes = {key: {m[key] for m in manifests} for key in ('partitionHash', 'initialModelHash')}
    if any(len(h) != 1 for h in hashes.values()):
        raise ValueError('Saved pairing hashes differ.')
    if any(m['dataSha256'] != manifests[0]['dataSha256'] for m in manifests):
        raise ValueError('Saved dataset identities differ.')
    before = {str(p.relative_to(ROOT)): file_hash(p) for p in source.rglob('*') if p.is_file()}
    # Only completed study sources and the existing cited source PDFs are read.
    for name in ('001-Basil A Fast and Byzantine-Resilient Approach for Decentralized Training.pdf',
                 '002-Robust Federated Learning with Noisy Communication.pdf',
                 '018-Federated Optimization in Heterogeneous Networks.pdf'):
        path = ROOT / 'papers' / name
        before[str(path.relative_to(ROOT))] = file_hash(path)
    output = dict(optimizerSteps=0, newResearchRunsStarted=0, testSetLoaded=False,
        savedTestMetricsUsage='analysis_only', pairedConfigDifferences=contrasts,
        partitionHash=next(iter(hashes['partitionHash'])),
        initialModelHash=next(iter(hashes['initialModelHash'])), conditions={})
    for path, manifest in zip(paths, manifests):
        cfg = manifest['config']
        # Protect claims about the exact code that generated the evidence.
        mismatches=[name for name,digest in manifest['implementationSha256'].items() if file_hash(ROOT/name)!=digest]
        if mismatches:
            if not executed_source_commit:
                raise ValueError('Executed source differs; supply --executed-source-commit to verify archived sources explicitly.')
            import subprocess
            for name,digest in manifest['implementationSha256'].items():
                raw=subprocess.check_output(['git','show',f'{executed_source_commit}:{name}'],cwd=ROOT)
                if hashlib.sha256(raw).hexdigest()!=digest:
                    raise ValueError('Archived commit does not match the actual executed source hashes.')
            if probe and any(name in mismatches for name in ('basil_core/models.py','basil_core/trainer.py')):
                raise ValueError('Read-only probes require the executed model/loss implementation; do not substitute changed versions.')
            output['executedSourceCommitVerified']=executed_source_commit
            output['executedSourcesMatchWorkingTree']=False
        records = json.loads(gzip.decompress((path / 'activation_telemetry.json.gz').read_bytes()))
        partition = json.loads((path / 'partition_audit.json').read_text())
        if partition['partitionHash'] != manifest['partitionHash']:
            raise ValueError('Partition audit and run manifest disagree.')
        summary = summarize_activations(records, partition, cfg)
        confusions = json.loads((path / 'audit/confusions.json').read_text())
        final = [c for c in confusions if (c['round'], c['node'], c['stage']) == (29, 9, 5)]
        if len(final) != 1:
            raise ValueError('Final evaluated confusion missing / duplicated.')
        summary.update(finalNode9=confusion_summary(final[0]['confusion']),
            finalMeanNodeAccuracy=manifest['finalAverageAccuracy'],
            finalWorstNodeAccuracy=manifest['finalWorstAccuracy'],
            node9NormByRound=[r['modelL2Norm'] for r in records if r['nodeId'] == 9],
            partitionNodes=partition['nodes'])
        output['conditions'][cfg['conditionId']] = summary
    if probe:
        import numpy as np
        from basil_core.iid_runtime import prepare_environment, configure_tensorflow
        prepare_environment('CPU')
        import tensorflow as tf
        configure_tensorflow(tf, 'CPU')
        from basil_core.dirichlet_diagnosis import probe_saved_parameters, array_identity_hash, full_local_training_metrics
        from basil_core.research_protocol import build_model, keyed_rng
        cache = ROOT / 'experiments/cache/cifar10'
        x = np.load(cache / 'x_train_float32.npy', mmap_mode='r', allow_pickle=False)
        y = np.load(cache / 'y_train_int32.npy', mmap_mode='r', allow_pickle=False)
        for array, key in [(x, 'trainImages'), (y, 'trainLabels')]:
            if array_identity_hash(array) != manifests[0]['dataSha256'][key]:
                raise ValueError('TRAIN probe cache differs from executed training data.')
        from basil_core.data.cifar import _dirichletPartition
        local_indices = _dirichletPartition(y, 10, .2, keyed_rng(configs[0]['seed'], 'dirichlet_partition'))
        partition_digest = hashlib.sha256()
        for chunk in local_indices:
            partition_digest.update(np.asarray(chunk, dtype='<i8').tobytes())
        if partition_digest.hexdigest() != output['partitionHash']:
            raise ValueError('Reconstructed TRAIN partition differs.')
        ids = np.concatenate([keyed_rng(2025, 'dirichlet_saved_state_probe', c).choice(
            np.flatnonzero(y == c), 32, replace=False) for c in range(10)])
        output['trainingProbeIndices'] = ids.tolist()
        output['probeDevice'] = 'CPU'
        with tf.device('/CPU:0'):
            model = build_model('basil_paper_cnn', 2025)
            from basil_core.research_protocol import params_hash
            if params_hash(model.get_weights()) != output['initialModelHash']:
                raise ValueError('Canonical probe architecture/initialization differs.')
            for path, cfg in zip(paths, configs):
                probes = {}
                records = json.loads(gzip.decompress((path / 'activation_telemetry.json.gz').read_bytes()))
                for node in range(10):
                    with np.load(path / f'node_{node}_weights.npz', allow_pickle=False) as saved:
                        params = [saved[f'weight_{i}'].copy() for i in range(10)]
                    if params_hash(params) != records[290 + node]['trainedOutputHash']:
                        raise ValueError('Saved final node weights differ from recorded honest training output.')
                    probes[str(node)] = probe_saved_parameters(model, params, np.asarray(x[ids]), np.asarray(y[ids]))
                    probes[str(node)]['fullLocalTrainingMetrics'] = full_local_training_metrics(
                        model, params, x, y, local_indices[node])
                    probes[str(node)]['historicalFirst512Accuracy'] = records[290 + node]['localTrainingAccuracy']
                output['conditions'][cfg['conditionId']]['finalNodeTrainProbes'] = probes
    after = {name: file_hash(ROOT / name) for name in before}
    if before != after:
        raise ValueError('Completed evidence changed during diagnosis.')
    current = {str(p.relative_to(ROOT)) for p in source.rglob('*') if p.is_file()}
    if current != {name for name in before if name.startswith(str(source.relative_to(ROOT)) + '/')}:
        raise ValueError('Completed study file set changed during diagnosis.')
    output.update(sourceHashes=before, sourceFilesUnchanged=True, hashedSourceFileCount=len(before))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true', help='Zero-update CPU TRAIN probe.')
    parser.add_argument('--executed-source-commit',help='Explicitly verify archived code when the working tree has advanced.')
    parser.add_argument('--output', type=Path, default=ROOT / 'gui/worker_state/dirichlet_learning_diagnosis.json')
    args = parser.parse_args()
    source = ROOT / 'newResults/nonIID/dirichlet_alpha02_local_anchor_30r_v1'
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'gui/worker_state').resolve()):
        raise ValueError('Analysis output must remain outside completed evidence, in gui/worker_state.')
    if output.exists():
        raise ValueError('Refusing to overwrite an existing diagnosis; choose a new output filename.')
    result = inspect(source, probe=args.probe,executed_source_commit=args.executed_source_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), optimizerSteps=0, newResearchRunsStarted=0,
        sourceFilesUnchanged=result['sourceFilesUnchanged'], hashedSourceFileCount=result['hashedSourceFileCount'])))


if __name__ == '__main__':
    main()
