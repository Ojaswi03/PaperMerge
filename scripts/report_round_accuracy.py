#!/usr/bin/env python3
"""Refresh organized A/B/C/D overall-accuracy views without touching training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distribution', choices=('IID', 'nonIID', 'both'), default='both')
    parser.add_argument('--rounds', type=int, choices=(30, 100, 500), nargs='+', default=[30, 100, 500])
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from reporting.round_accuracy import build_views
    distributions = ('IID', 'nonIID') if args.distribution == 'both' else (args.distribution,)
    print(json.dumps(build_views(ROOT, distributions, args.rounds), indent=2))


if __name__ == '__main__':
    main()
