#!/usr/bin/env python3
"""Export completed revision metrics, source snapshots and plots without checkpoints."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.export_cart_campaign import copy_matching,write_matching,sha


def export(output,destination,figures=None):
    from reporting.cart_revision import collect
    root=Path(output).resolve();dest=Path(destination).resolve()
    if root==dest or root in dest.parents or dest in root.parents:
        raise ValueError('Export destination must be separate from training inputs.')
    plan,_,_,_,_=collect(root)
    paths=[]
    for name in ('plan.json','campaign.json','managed_status.json','revision_execution.json','independent_validation.json'):
        if (root/name).is_file():paths.append((root/name,Path(name)))
    for cfg in plan['configs']:
        for name in ('run.json','metrics.npz'):paths.append((root/cfg['runId']/name,Path(cfg['runId'])/name))
    for name,value in plan['source'].items():
        path=root/'source_snapshots'/name
        if sha(path)!=value:raise ValueError('Source snapshot mismatch: '+name)
        paths.append((path,Path('source_snapshots')/name))
    folder=Path(figures).resolve() if figures else Path(plan.get('figureDestination') or root/'figures')
    manifest=json.loads((folder/'artifacts_manifest.json').read_text())
    if manifest['revision']!=plan['revision'] or manifest['source']!=plan['source']:
        raise ValueError('Figure provenance differs from the revision.')
    for name,value in manifest['outputSha256'].items():
        path=folder/name
        if Path(name).name!=name or sha(path)!=value:raise ValueError('Figure checksum mismatch.')
        paths.append((path,Path('figures')/name))
    paths.append((folder/'artifacts_manifest.json',Path('figures/artifacts_manifest.json')))
    allowed={str(rel) for _,rel in paths}
    if dest.exists() and any(str(p.relative_to(dest)) not in allowed|{'SHA256.json'} for p in dest.rglob('*') if p.is_file()):
        raise ValueError('Unexpected file in export destination; use a fresh destination.')
    for path,rel in paths:
        if path.is_symlink():raise ValueError('Symlink exports are not accepted.')
        copy_matching(path,dest/rel)
    write_matching(dest/'SHA256.json',{str(rel):sha(path) for path,rel in paths})
    return len(plan['configs']),len(paths)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--figures',type=Path)
    a=p.parse_args();configs,files=export(a.output,a.destination,a.figures)
    print(f'Export ready: {configs} configurations, {files} files. Checkpoints excluded.')
    print(a.destination)
