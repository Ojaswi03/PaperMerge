#!/usr/bin/env python3
"""Export completed CART metrics and figures for Git without model checkpoints.

Existing identical files are reused; conflicting files are never overwritten.
Run records/metrics retain their original provenance. This is metric export,
not independent inference verification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def copy_matching(source,target):
    if target.exists():
        if not target.is_file() or sha(source)!=sha(target):
            raise FileExistsError(f'Conflicting export file: {target}. Use a fresh destination.')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.export-tmp')
    shutil.copy2(source,temporary);temporary.replace(target)


def write_matching(target,value):
    raw=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if target.exists():
        try:matching=json.loads(target.read_text())==value
        except (ValueError,UnicodeError):matching=False
        if not matching:raise FileExistsError(f'Conflicting export file: {target}. Use a fresh destination.')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.export-tmp');temporary.write_text(raw);temporary.replace(target)


def export(output,destination):
    from scripts.report_cart_v2_completed import summarize
    root=Path(output).resolve();dest=Path(destination).resolve()
    if dest==root or root in dest.parents or dest in root.parents:
        raise ValueError('Export destination must be outside the campaign and its parent directories.')
    summary=summarize(root)
    if summary['pendingGroups'] or summary['unpairedCartGroups']:
        raise ValueError('Export requires completed three-seed groups and matched CART controls.')
    plan=json.loads((root/'plan.json').read_text())
    # Validate every record before creating or extending the destination.
    for cfg in plan['configs']:
        if Path(cfg['runId']).name!=cfg['runId']:raise ValueError('Invalid run directory.')
        record=json.loads((root/cfg['runId']/'run.json').read_text())
        if record['status']!='COMPLETED':raise ValueError(f'Run incomplete: {cfg["runId"]}')
        if not (root/cfg['runId']/'metrics.npz').is_file():raise FileNotFoundError('Missing saved metrics.')
    write_matching(dest/'summary.json',summary)
    for name in ('plan.json','campaign.json','managed_status.json','compiled_transfer_execution.json'):
        if (root/name).exists():copy_matching(root/name,dest/name)
    for cfg in plan['configs']:
        for name in ('run.json','metrics.npz'):copy_matching(root/cfg['runId']/name,dest/cfg['runId']/name)
    figures=root/'figures'
    if figures.is_dir():
        for source in sorted(figures.rglob('*')):
            if source.is_file() and source.suffix.lower() in ('.json','.csv','.png','.pdf','.svg'):
                if source.is_symlink():raise ValueError('Symlinks are not accepted in figure exports.')
                copy_matching(source,dest/'figures'/source.relative_to(figures))
    # Never silently publish an unrelated file already in a partial destination.
    allowed_root={'summary.json','plan.json','campaign.json','managed_status.json',
        'compiled_transfer_execution.json','SHA256.json'}
    run_dirs={cfg['runId'] for cfg in plan['configs']}
    files=[p for p in dest.rglob('*') if p.is_file() and p.name!='SHA256.json']
    for path in files:
        rel=path.relative_to(dest)
        valid=(len(rel.parts)==1 and rel.name in allowed_root or
            len(rel.parts)==2 and rel.parts[0] in run_dirs and rel.name in ('run.json','metrics.npz') or
            rel.parts[0]=='figures' and rel.suffix.lower() in ('.json','.csv','.png','.pdf','.svg'))
        if not valid:raise ValueError(f'Unexpected export file: {path}; use a fresh destination.')
    hashes={str(p.relative_to(dest)):sha(p) for p in sorted(files)}
    write_matching(dest/'SHA256.json',hashes)
    return dict(configs=len(plan['configs']),files=len(hashes),bytes=sum(p.stat().st_size for p in files),
        destination=str(dest),independentInferencePerformed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args();row=export(args.output,args.destination)
    print(f'Export ready: {row["configs"]} configurations, {row["files"]} files, {row["bytes"]/1024**2:.1f} MiB.')
    print(row['destination']);print('Checkpoints excluded. Existing matching files reused. Conflicts are preserved.')


if __name__=='__main__':main()
