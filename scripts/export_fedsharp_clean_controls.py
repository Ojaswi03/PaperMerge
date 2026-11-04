#!/usr/bin/env python3
"""Freshly verify and export the twelve completed clean controls for Git."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from basil_core import fedsharp_clean_controls as study
from scripts.export_cart_campaign import copy_matching,write_matching,sha


def export(output,destination,*,study_adapter=None,verifier_script='run_fedsharp_clean_controls.py',extra_files=()):
    from reporting.basil_sharp_thesis import collect
    adapter=study if study_adapter is None else study_adapter
    root=Path(output).resolve();dest=Path(destination).resolve()
    if root==dest or root in dest.parents or dest in root.parents:
        raise ValueError('Separate export destination required.')
    with adapter.installed():
        plan,rows,_=collect(root)
    expected={(d,p,s) for d in ('mnist','cifar10') for p in ('iid','dirichlet')
              for s in (2025,2026,2027)}
    if len(rows)!=12 or {(r['dataset'],r['partition'],r['seed']) for r in rows}!=expected:
        raise ValueError('Export requires all twelve clean controls.')
    for view in plan['configs']:
        cfg=adapter.effective_config(view)
        if (cfg['rounds'],cfg['nodes'],cfg['localSteps'])!=(500,100,5):
            raise ValueError('Expected 500 rounds, 100 clients and five local steps.')
    names=['plan.json','campaign.json',*extra_files]
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in extra_files):
        raise ValueError('Invalid extra export path.')
    names += [f"{v['runId']}/{n}" for v in plan['configs']
              for n in ('run.json','metrics.npz','mechanisms.json')]
    names += ['source_snapshots/'+n for n in plan['source']]
    # Capture measurements before inference; publication must use those same bytes.
    before={n:sha(root/n) for n in names}
    checkpoints={v['runId']:(root/v['runId']/'checkpoint.npz').stat()
                 for v in plan['configs']}
    subprocess.run([sys.executable,str(ROOT/'scripts'/verifier_script),
                    '--role','verify','--output',str(root)],cwd=ROOT,check=True)
    receipt=json.loads((root/'independent_validation.json').read_text())
    entries=receipt.get('verified',[])
    if (receipt.get('status')!='ALL_COMPARISON_MODELS_VERIFIED' or
            receipt.get('source')!=plan['source'] or receipt.get('officialTestUsed') is not False or
            len(entries)!=12 or {r['runId'] for r in entries}!={v['runId'] for v in plan['configs']}):
        raise ValueError('Fresh inference receipt does not cover the complete clean study.')
    for run,previous in checkpoints.items():
        current=(root/run/'checkpoint.npz').stat()
        if (previous.st_ino,previous.st_size,previous.st_mtime_ns,previous.st_ctime_ns)!=(
                current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns):
            raise ValueError('Checkpoint changed during verification: '+run)
    if before!={n:sha(root/n) for n in names}:
        raise ValueError('Export inputs changed during verification.')
    names.append('independent_validation.json')
    # Explicit allowlist excludes checkpoints, logs, process state and configs.json.
    allowed=set(names)|{'SHA256.json'}
    if dest.exists() and any(str(p.relative_to(dest)) not in allowed
                             for p in dest.rglob('*') if p.is_file()):
        raise ValueError('Unexpected files in export destination.')
    hashes={n:sha(root/n) for n in names}
    for name in names:
        path=root/name
        if path.is_symlink():raise ValueError('Symlink exports refused.')
        copy_matching(path,dest/name)
        if sha(dest/name)!=hashes[name]:raise ValueError('Export checksum mismatch: '+name)
    write_matching(dest/'SHA256.json',hashes)
    return dict(configurations=12,files=len(hashes),freshInferenceVerified=True,
                destination=str(dest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.output,args.destination),indent=2))
