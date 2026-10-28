#!/usr/bin/env python3
"""Read-only SHARP monitor with per-run and measured queue ETA.

Can be updated during an active campaign without modifying its frozen runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from basil_core.sharp_monitor import monitor
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--once',action='store_true',help='Print one snapshot and exit.')
    args=parser.parse_args()
    if not args.output.is_dir():parser.error(f'Campaign directory does not exist: {args.output}')
    try:monitor(args.output.resolve(),once=args.once)
    except (OSError,ValueError,KeyError) as error:parser.exit(2,f'Monitor could not read campaign: {error}\n')


if __name__=='__main__':main()
