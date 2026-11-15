#!/usr/bin/env python3
"""Audit zero attacks and freshly verify/export the 18 noise-only results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core import fedsharp_noise_only as study
from scripts.export_cart_campaign import copy_matching, write_matching, sha


def audit_no_attacks(root, plan):
    import numpy as np
    verified = []
    for view in plan['configs']:
        cfg = study.effective_config(view)
        with np.load(root / view['runId'] / 'checkpoint.npz', allow_pickle=False) as archive:
            metadata = json.loads(archive['metadata'].tobytes())
        records = metadata['telemetry']
        total = cfg['nodes'] * cfg['rounds']
        if metadata['contract']['config'] != cfg or metadata['cursor'] != total or len(records) != total:
            raise ValueError('Incomplete or foreign noise-only checkpoint.')
        for cursor, record in enumerate(records):
            if ((record['round'], record['node'], record['activation']) !=
                    (cursor // cfg['nodes'], cursor % cfg['nodes'], cursor + 1)
                    or record['attackActive'] or record['configuredByzantine']
                    or record['trainedHash'] != record['outboundHash']
                    or record['training']['optimizerSteps'] != cfg['localSteps']):
                raise ValueError('Noise-only activation/attack/SGD evidence differs: ' + view['runId'])
        verified.append(dict(runId=view['runId'], activationsChecked=total,
            optimizerStepsChecked=total * cfg['localSteps'], allOutgoingHashesEqualTrained=True,
            actualAttackers=0, physicalChannelSigma=cfg['sigma']))
    return dict(status='NOISE_ONLY_NO_ATTACKS_VERIFIED', source=plan['source'], verified=verified,
        verifierSha256=sha(Path(__file__)), scope='Recorded full activation chain; not a replay of every SGD update.')


def export(output, destination):
    import numpy as np
    root, dest = Path(output).resolve(), Path(destination).resolve()
    if root == dest or root in dest.parents or dest in root.parents:
        raise ValueError('Separate export destination required.')
    plan = json.loads((root / 'plan.json').read_text())
    if plan['revision'] != study.REVISION or plan.get('continuations'):
        raise ValueError('Expected a fresh noise-only study.')
    study.check_source(plan['source'])
    args = SimpleNamespace(datasets=['mnist', 'cifar10'], partitions=['iid', 'dirichlet'],
        seeds=[2025, 2026, 2027], families=['sharp'], environments=['noise_only'],
        nodes=100, rounds=500, local_steps=5, alpha=.5, sigma=.01, attack_start=20,
        attack_type='hidden', gaussian_std=1.)
    if plan['configs'] != study.selected(args) or plan.get('executionOptimization') != study.runtime_policy():
        raise ValueError('Export requires the exact eighteen-case 500-round matrix.')
    names = ['plan.json', 'campaign.json']
    if json.loads((root / 'campaign.json').read_text())['status'] != 'COMPLETED':
        raise ValueError('Wait for the campaign to complete.')
    checkpoints = {}
    for view in plan['configs']:
        run = view['runId']
        folder = root / run
        cfg = study.effective_config(view)
        row = json.loads((folder / 'run.json').read_text())
        if (row['status'] != 'COMPLETED' or row['config'] != view or row['source'] != plan['source']
                or row['contract']['config'] != cfg or row['completedActivations'] != 50000
                or row['optimizerSteps'] != 250000 or row['completedRounds'] != 500
                or row['data'] != plan['data'][cfg['dataset']]
                or row['partition'] != plan['partitions'][run]):
            raise ValueError('Incomplete/mismatched noise-only run: ' + run)
        with np.load(folder / 'metrics.npz', allow_pickle=False) as archive:
            accuracy = archive['roundNodeAccuracy']
            if accuracy.shape != (500, 100) or not np.isfinite(accuracy).all() or np.any((accuracy < 0) | (accuracy > 1)):
                raise ValueError('Invalid noise-only accuracy array.')
        names += [run + '/' + name for name in ('run.json', 'metrics.npz', 'mechanisms.json',
                                                 'optimization.json', 'round_timings.jsonl')]
        checkpoints[run] = (folder / 'checkpoint.npz').stat()
    for name, digest in plan['source'].items():
        path = root / 'source_snapshots' / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or sha(path) != digest:
            raise ValueError('Frozen producer snapshot differs.')
        names.append('source_snapshots/' + name)
    before = {name: sha(root / name) for name in names}
    write_matching(root / 'noise_only_verification.json', audit_no_attacks(root, plan))
    subprocess.run([sys.executable, str(ROOT / 'scripts/run_fedsharp_noise_only.py'),
                    '--role', 'verify', '--output', str(root)], cwd=ROOT, check=True)
    receipt = json.loads((root / 'independent_validation.json').read_text())
    if (receipt['status'] != 'ALL_COMPARISON_MODELS_VERIFIED' or receipt['source'] != plan['source']
            or receipt['officialTestUsed'] is not False or len(receipt['verified']) != 18
            or {r['runId'] for r in receipt['verified']} != {v['runId'] for v in plan['configs']}):
        raise ValueError('Fresh inference receipt does not cover all eighteen cases.')
    for run, previous in checkpoints.items():
        current = (root / run / 'checkpoint.npz').stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
                previous.st_ino, previous.st_size, previous.st_mtime_ns, previous.st_ctime_ns):
            raise ValueError('Checkpoint changed during verification: ' + run)
    if before != {name: sha(root / name) for name in names}:
        raise ValueError('Measurements changed during verification.')
    names += ['noise_only_verification.json', 'independent_validation.json']
    allowed = set(names) | {'SHA256.json'}
    if dest.exists() and any(str(p.relative_to(dest)) not in allowed for p in dest.rglob('*') if p.is_file()):
        raise ValueError('Unexpected files in export destination.')
    hashes = {}
    for name in names:
        if (root / name).is_symlink():
            raise ValueError('Symlink exports refused.')
        copy_matching(root / name, dest / name)
        hashes[name] = sha(root / name)
        if sha(dest / name) != hashes[name]:
            raise ValueError('Export checksum differs.')
    write_matching(dest / 'SHA256.json', hashes)
    return dict(configurations=18, files=len(hashes), noAttackChainVerified=True,
                freshInferenceVerified=True, destination=str(dest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.destination), indent=2))
