#!/usr/bin/env python3
"""Export comparison metrics, ACDS audits and compact mechanism evidence."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.export_cart_campaign import copy_matching,write_matching,sha


def export(root,destination):
    from reporting.cart_comparison import collect
    root=Path(root).resolve();destination=Path(destination).resolve();plan,_,_,_=collect(root)
    if root==destination or root in destination.parents or destination in root.parents:raise ValueError('Separate export destination required.')
    paths=[]
    for name in ('plan.json','campaign.json','revision_execution.json','independent_validation.json','official_test_evaluation.json'):
        if (root/name).is_file():paths.append((root/name,Path(name)))
    for c in plan['configs']:
        for name in ('run.json','metrics.npz','mechanisms.json'):
            paths.append((root/c['runId']/name,Path(c['runId'])/name))
    for name,value in plan['source'].items():
        path=root/'source_snapshots'/name
        if sha(path)!=value:raise ValueError('Source snapshot differs: '+name)
        paths.append((path,Path('source_snapshots')/name))
    figures=root/'figures';manifest=json.loads((figures/'artifacts_manifest.json').read_text())
    if manifest['source']!=plan['source']:raise ValueError('Figure source differs.')
    for name,value in manifest['outputSha256'].items():
        if Path(name).name!=name or sha(figures/name)!=value:raise ValueError('Figure hash differs.')
        paths.append((figures/name,Path('figures')/name))
    paths.append((figures/'artifacts_manifest.json',Path('figures/artifacts_manifest.json')))
    allowed={str(rel) for _,rel in paths}|{'SHA256.json'}
    if destination.exists() and any(str(p.relative_to(destination)) not in allowed for p in destination.rglob('*') if p.is_file()):
        raise ValueError('Unexpected file in export destination.')
    for src,rel in paths:
        if src.is_symlink():raise ValueError('Symlink exports refused.')
        copy_matching(src,destination/rel)
    write_matching(destination/'SHA256.json',{str(rel):sha(src) for src,rel in paths})
    return len(plan['configs']),len(paths)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();n,k=export(a.output,a.destination);print(f'Exported {n} configurations, {k} files to {a.destination}.')
