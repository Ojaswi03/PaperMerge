#!/usr/bin/env python3
"""Read-only two-server monitor with overall speed, elapsed time and finish ETA."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from basil_core.continuation_monitor import render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--interval',type=float,default=3.)
    args=parser.parse_args()
    if args.interval<1:parser.error('--interval must be at least one second.')
    try:
        while True:
            print(('' if args.once else '\033[2J\033[H')+render(args.queue.resolve()),flush=True)
            if args.once:return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:return 0


if __name__=='__main__':raise SystemExit(main())
