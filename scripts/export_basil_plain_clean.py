#!/usr/bin/env python3
"""Audit and freshly verify the twelve matched BASIL clean ring controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import basil_plain_clean as study
from scripts import export_fedsharp_plain_clean as shared
from scripts.export_cart_campaign import write_matching


def export(output,destination):
    root=Path(output).resolve();dest=Path(destination).resolve()
    if root==dest or root in dest.parents or dest in root.parents:raise ValueError('Separate export required.')
    plan=json.loads((root/'plan.json').read_text())
    if plan['revision']!=study.REVISION:raise ValueError('Expected the BASIL clean ring study.')
    study.check_source(plan['source'])
    verified=[shared.check_routing(root,v,plan['source'],study_adapter=study) for v in plan['configs']]
    audit=dict(status='PLAIN_RING_CHAIN_AND_SGD_VERIFIED',algorithm='BASIL',source=plan['source'],verified=verified,
               verifierSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               routingVerifierSha256=hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest())
    write_matching(root/'plain_ring_verification.json',audit)
    return dict(shared.common.export(root,dest,study_adapter=study,verifier_script='run_basil_plain_clean.py',
                extra_files=('plain_ring_verification.json',)),plainRingRoutingVerified=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export(args.output,args.destination),indent=2))
