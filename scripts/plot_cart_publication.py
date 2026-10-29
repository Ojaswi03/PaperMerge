#!/usr/bin/env python3
"""Write line-only thesis/paper figures, optionally waiting for a running study."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core.cart_campaign_io import read_json


def readiness(root):
    """A campaign banner alone cannot certify every metrics file is committed."""
    plan = read_json(root / 'plan.json')
    completed = 0
    for cfg in plan['configs']:
        folder = root / cfg['runId']
        row = read_json(folder / 'run.json') if (folder / 'run.json').exists() else {}
        if row.get('status') == 'COMPLETED' and (folder / 'metrics.npz').is_file():
            completed += 1
    status = read_json(root / 'campaign.json').get('status') if (root / 'campaign.json').exists() else 'PREPARED'
    return completed, len(plan['configs']), status


def wait_for_completion(root, interval=15):
    previous = None
    while True:
        complete, total, status = readiness(root)
        current = (complete, total, status)
        if current != previous:
            print(f'Figure preparation: {complete}/{total} configurations complete; {status}', flush=True)
            previous = current
        if complete == total:
            return
        if status in ('FAILED', 'INTERRUPTED', 'COMPLETED_WITH_FAILURES', 'COMPLETED_REPORT_FAILED', 'COMPLETED'):
            raise RuntimeError('Some runs are incomplete. Resume training before generating complete-group figures.')
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--wait', action='store_true', help='Wait without using a GPU; render only complete results.')
    parser.add_argument('--tmux', help='Keep the waiting plot job alive after SSH disconnects.')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.destination = (args.destination or args.output / 'publication_curves').resolve()
    if args.destination.exists() and any(args.destination.iterdir()):
        parser.error('Destination is not empty; preserve it and select a new directory.')
    if args.tmux:
        forwarded = sys.argv[1:]
        position = forwarded.index('--tmux')
        forwarded = forwarded[:position] + forwarded[position + 2:]
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), *forwarded])
        command += ' >> ' + shlex.quote(str(args.output / 'publication_plot.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', args.tmux, 'bash', '-c', command], check=True)
        print(f'Started {args.tmux}. Plot log: {args.output / "publication_plot.log"}')
        return 0
    if args.wait:
        wait_for_completion(args.output)
    from reporting.cart_publication_curves import generate
    manifest = generate(args.output, args.destination)
    print(json.dumps(dict(destination=manifest['destination'], configurations=manifest['configurations'],
                          figures=sum(n.endswith('.svg') for n in manifest['outputSha256']),
                          metric=manifest['metric']), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
