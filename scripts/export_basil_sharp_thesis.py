#!/usr/bin/env python3
"""Export matched BASIL/SHARP measurements and four thesis figures for Git."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def export(root,destination):
    from reporting.basil_sharp_thesis import collect,generate,sha
    from scripts.export_cart_campaign import copy_matching,write_matching
    root,destination=Path(root).resolve(),Path(destination).resolve()
    if root==destination or root in destination.parents or destination in root.parents:raise ValueError('Separate export required.')
    plan,rows,_=collect(root);generate(root,root/'thesis_figures')
    names=['plan.json','campaign.json','adaptive_execution.json']
    names += [n for n in ('independent_validation.json',) if (root/n).exists()]
    for c in plan['configs']:names += [c['runId']+'/'+n for n in ('run.json','metrics.npz','mechanisms.json')]
    names += ['source_snapshots/'+n for n in plan['source']]
    names += [str(p.relative_to(root)) for p in (root/'thesis_figures').rglob('*') if p.is_file()]
    if destination.exists() and any(str(p.relative_to(destination)) not in set(names)|{'SHA256.json'}
            for p in destination.rglob('*') if p.is_file()):raise ValueError('Unexpected files in export.')
    for name in names:
        if (root/name).is_symlink():raise ValueError('Symlink exports refused.')
        copy_matching(root/name,destination/name)
    write_matching(destination/'SHA256.json',{n:sha(root/n) for n in names})
    return len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    print(f'Exported {export(a.output,a.destination)} matched BASIL/SHARP configurations.')
