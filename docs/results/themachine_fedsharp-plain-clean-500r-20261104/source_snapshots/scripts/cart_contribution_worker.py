#!/usr/bin/env python3
"""Source-frozen contribution worker; one independent configuration/process."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--lease-fd', type=int)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    from basil_core.cart_contribution_study import installed
    from scripts.cart_comparison_worker import run
    with installed():
        run(args.output.resolve(), args.worker, args.lease_fd, args.cpu)
