#!/usr/bin/env python3
"""Gate the unchanged compiled scorer/objectives before a comparison campaign."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(root,gpu,lease_fd):
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core.cart_comparison_study import check_source
    from basil_core.cart_revision_study import source as original_source
    from scripts.verify_cart_revision_gpu import verify as original_verify
    plan=read_json(root/'plan.json');check_source(plan['source'])
    gate=root/'comparison_base_gate';gate.mkdir(exist_ok=True)
    write_json(gate/'plan.json',dict(configs=[c['comparisonCase']['protocolConfig'] for c in plan['configs']],source=original_source()))
    original_verify(gate,gpu,lease_fd)
    receipt=read_json(gate/f'revision_gpu_{gpu}.json')
    prior_configs=[c['comparisonCase']['protocolConfig'] for c in plan['configs'] if c['comparisonCase']['arm'] in ('prior_cart','prior_macro','prior_wide')]
    if prior_configs:
        from scripts.verify_cart_local_prior_gpu import verify as verify_prior
        for dataset in dict.fromkeys(c['dataset'] for c in prior_configs):
            receipt['checks'].extend(verify_prior(next(c for c in prior_configs if c['dataset']==dataset)))
    receipt.update(source=plan['source'],scientificScope='Execution/objective equivalence only; efficacy requires the comparison runs.')
    write_json(root/f'revision_gpu_{gpu}.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--lease-fd',type=int,required=True)
    a=p.parse_args();verify(a.output.resolve(),a.gpu,a.lease_fd)
