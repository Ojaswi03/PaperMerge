#!/usr/bin/env python3
"""Record or verify saved IID evidence and papers without modifying them."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREES = ('newResults/IID', 'newPlots/IID', 'papers')


def inventory():
    files = {}
    for tree in TREES:
        for path in sorted((ROOT / tree).rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(block)
                files[str(path.relative_to(ROOT))] = dict(sha256=digest.hexdigest(), size=path.stat().st_size)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    files = inventory()
    if args.record:
        if args.manifest.exists():
            raise ValueError('Refusing to overwrite an evidence baseline.')
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(dict(trees=TREES, files=files), indent=2) + '\n')
        print(json.dumps(dict(status='recorded', count=len(files), bytes=sum(f['size'] for f in files.values()))))
    else:
        original = json.loads(args.manifest.read_text())['files']
        changed = [p for p in original.keys() | files.keys() if original.get(p) != files.get(p)]
        print(json.dumps(dict(status='unchanged' if not changed else 'changed', before=len(original),
            after=len(files), matched=len(original)-len(set(changed)&original.keys()), changed=changed)))
        if changed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
