#!/usr/bin/env python3
"""One complete independent BASIL or SHARP thesis configuration."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',type=int,required=True)
    p.add_argument('--lease-fd',type=int);p.add_argument('--cpu',action='store_true');args=p.parse_args()
    from basil_core.basil_sharp_thesis import installed
    from scripts.cart_comparison_worker import run
    with installed():run(args.output.resolve(),args.worker,args.lease_fd,args.cpu)
