#!/usr/bin/env python3
"""Hash-pinned execution adapter for explicitly transitioned hal9k workers."""
import argparse
import socket
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--lease-fd',type=int,required=True)
    parser.add_argument('--optimization-manifest',type=Path,required=True)
    args=parser.parse_args()
    if socket.gethostname().split('.')[0]!='hal9k':raise ValueError('This rollout is restricted to hal9k.')
    from basil_core.cart_campaign_io import read_json
    from basil_core.basil_sharp_thesis import installed
    from basil_core import fedsharp_runtime_optimization as optimization
    from scripts.cart_comparison_worker import run
    root=args.output.resolve();plan=read_json(root/'plan.json');policy=read_json(args.optimization_manifest)
    if policy.get('host')!='hal9k' or str(root) not in policy['outputs']:raise ValueError('Optimization is not authorized for this output.')
    optimization.validate(policy)
    folder=root/plan['configs'][args.worker]['runId'];folder.mkdir(exist_ok=True)
    with installed(),optimization.installed(root,plan['configs'][args.worker],policy):
        run(root,args.worker,args.lease_fd)


if __name__=='__main__':main()
