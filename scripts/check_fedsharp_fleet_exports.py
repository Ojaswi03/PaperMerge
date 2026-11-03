#!/usr/bin/env python3
"""Check exact completed-fleet coverage, receipts and export hashes before staging."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core.cart_campaign_io import read_json
from basil_core.round_continuation import sha


def check(queue,exports):
    queue=Path(queue);master=read_json(queue/'continuation_plan.json');state=read_json(queue/'fleet/state.json')
    expected={item['config']['runId']:item['config'] for item in master['items']};seen=set();counts={}
    if any(row['state']!='DONE' for row in state['runs'].values()):raise ValueError('Fleet is not completely DONE.')
    if len(expected)!=150:raise ValueError('Expected the complete 150-run continuation.')
    for root in map(Path,exports):
        manifest=read_json(root/'SHA256.json')
        files={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}-{'SHA256.json'}
        if files!=set(manifest):raise ValueError('Export manifest coverage differs.')
        for name,digest in manifest.items():
            path=root/name
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha(path)!=digest:
                raise ValueError('Export path/checksum mismatch: '+name)
            if path.name.startswith('checkpoint') or path.suffix=='.log':raise ValueError('Export contains runtime data.')
        plan=read_json(root/'plan.json');receipt=read_json(root/'independent_validation.json')
        host=plan['host'].split('.')[0];configs={c['runId']:c for c in plan['configs']}
        verified={r['runId']:r for r in receipt['verified']}
        if (receipt.get('status')!='ALL_COMPARISON_MODELS_VERIFIED' or receipt.get('officialTestUsed') is not False
                or receipt.get('source')!=master['source'] or plan['source']!=master['source']
                or plan['continuationQueueId']!=master['queueId'] or set(configs)!=set(verified)
                or len(verified)!=len(receipt['verified']) or len(configs)!=len(plan['configs'])):
            raise ValueError('Fragment verification/identity differs.')
        for run,cfg in configs.items():
            if run in seen or expected.get(run)!=cfg or state['runs'][run]['owner']!=host:
                raise ValueError('Missing, duplicated or misplaced configuration: '+run)
            for name,digest in verified[run]['resultSha256'].items():
                if sha(root/run/name)!=digest:raise ValueError('Verified result identity differs.')
            seen.add(run)
        counts[host]=counts.get(host,0)+len(configs)
    if seen!=set(expected):raise ValueError('Exports do not cover all 150 configurations.')
    return dict(total=len(seen),hosts=counts,checksumsVerified=True,freshInferenceReceiptsPresent=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--exports',nargs='+',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(check(args.queue,args.exports),indent=2))
