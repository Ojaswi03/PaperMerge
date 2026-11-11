#!/usr/bin/env python3
"""Download and hash-check the published final-500-round client models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(asset, target):
    """Install only a complete verified file, preserving any conflicting file."""
    target = Path(target)
    if target.exists():
        if target.stat().st_size != asset['bytes'] or sha(target) != asset['sha256']:
            raise ValueError(f'Existing file differs; preserved: {target}')
        return 'already verified'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.download-', delete=False) as out:
            temporary = Path(out.name)
            with urlopen(asset['url'], timeout=120) as response:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if temporary.stat().st_size != asset['bytes'] or sha(temporary) != asset['sha256']:
            raise ValueError(f'Download failed size/SHA-256 verification: {target.name}')
        # Hard linking refuses a concurrent installation instead of overwriting it.
        os.link(temporary, target)
        return 'downloaded and verified'
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / '.papermerge/models/final-500r')
    parser.add_argument('--variant', choices=('compact', 'original'), default='compact')
    parser.add_argument('--dataset', choices=('mnist', 'cifar10'))
    parser.add_argument('--run-id', help='Download only this exact published run ID.')
    parser.add_argument('--list', action='store_true', help='Print selected files without downloading.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    selected = [m for m in manifest['models']
                if (not args.dataset or m['dataset'] == args.dataset)
                and (not args.run_id or m['runId'] == args.run_id)]
    if not selected:
        parser.error('No published models match the requested filters.')
    for model in selected:
        kind = 'original_download' if args.variant == 'original' else 'compact_inference'
        candidates = [f for f in model['files'] if f['kind'] == kind]
        asset = candidates[0] if candidates else model['files'][0]
        run = model['runId']
        if Path(run).name != run or run in ('.', '..') or asset['installAs'] not in ('model.npz', 'checkpoint.npz'):
            raise ValueError('Unsafe model manifest path.')
        target = args.destination / run / asset['installAs']
        if args.list:
            print(f"{model['dataset']:7} | {model['method']:12} | {asset['bytes'] / 1024**2:.1f} MiB | {run}")
            print(asset['url'])
        else:
            print(f'{run}: {asset["bytes"] / 1024**2:.1f} MiB', flush=True)
            print(f'  {download(asset, target)}', flush=True)


if __name__ == '__main__':
    main()
