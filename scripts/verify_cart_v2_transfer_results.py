#!/usr/bin/env python3
"""Check explicit transfer-adapter provenance, then independently verify on CPU."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');args=p.parse_args();root=args.output.resolve()
    from basil_core.cart_campaign_io import read_json
    from scripts.run_cart_v2_transfer import program_hashes
    execution=read_json(root/'compiled_transfer_execution.json');plan=read_json(root/'plan.json')
    if (execution['programSha256']!=program_hashes() or
        execution['planSha256']!=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()):
        raise ValueError('Compiled-transfer provenance differs from its declared implementation/plan.')
    for cfg in plan['configs']:
        path=root/cfg['runId']/'run.json'
        if not path.exists():continue
        row=read_json(path)
        if row.get('runtimeAdapter',{}).get('programSha256',execution['programSha256'])!=execution['programSha256']:
            raise ValueError('Run declares a different transfer adapter.')
    from scripts.verify_sharp_results import verify
    result=verify(root,require_complete=args.require_complete)
    result=dict(result,runtimeAdapterProgramSha256=execution['programSha256'],
        adapterVerifierSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    from basil_core.cart_campaign_io import write_json
    write_json(root/'compiled_transfer_independent_validation.json',result)
    import json
    print(json.dumps(dict(status=result['status'],verified=result['verified'],pending=len(result['pending'])),indent=2))
