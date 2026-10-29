#!/usr/bin/env python3
"""Copy verified metrics, source snapshots and line figures; omit checkpoints."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(root, destination):
    from reporting.cart_contribution import report
    from reporting.cart_publication_curves import collect, sha
    from scripts.export_cart_campaign import copy_matching, write_matching
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if root == destination or root in destination.parents or destination in root.parents:
        raise ValueError('Separate export directory required.')
    report(root)
    plan, _, _ = collect(root)
    names = ['plan.json', 'campaign.json', 'adaptive_execution.json']
    names += [n for n in ('independent_validation.json', 'official_test_evaluation.json') if (root / n).exists()]
    for view in plan['configs']:
        names += [view['runId'] + '/' + n for n in ('run.json', 'metrics.npz', 'mechanisms.json')]
    names += ['source_snapshots/' + n for n in plan['source']]
    names += [str(p.relative_to(root)) for p in (root / 'publication_curves').rglob('*') if p.is_file()]
    allowed = set(names) | {'SHA256.json'}
    if destination.exists() and any(str(p.relative_to(destination)) not in allowed
            for p in destination.rglob('*') if p.is_file()):
        raise ValueError('Unexpected files in export destination.')
    for name in names:
        origin = root / name
        if origin.is_symlink():
            raise ValueError('Symlink exports refused.')
        copy_matching(origin, destination / name)
    write_matching(destination / 'SHA256.json', {n: sha(root / n) for n in names})
    return len(plan['configs'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    print(f'Exported {export(args.output, args.destination)} completed configurations.')
