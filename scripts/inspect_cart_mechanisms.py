#!/usr/bin/env python3
"""Read-only plateau, partition and routing audit of existing CART studies."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core.cart_mechanism_audit import inspect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--json',action='store_true')
    a=p.parse_args();report=inspect(a.output)
    if a.json:print(json.dumps(report,indent=2));return
    print('CART metric and mechanism audit | development holdout')
    print('DATA/SPLIT       SEED CONDITION                         FINAL   LAST10  WORST   ENTROPY TELEMETRY')
    for r in report['runs']:
        m=r.get('metrics',{});part=r.get('partitionAudit',{})
        fmt=lambda v:f'{100*v:6.2f}%' if v is not None else '     --'
        print(f'{r["dataset"]+"/"+r["partition"]:<16} {r["seed"]} {(r["arm"] or r["condition"]):<33} '
            f'{fmt(m.get("final"))} {fmt(m.get("primary"))} {fmt(m.get("worstClient"))} '
            f'{part.get("meanNormalizedClassEntropy",0):7.3f} {"available" if r["mechanisms"] is not None else "missing"}')
        if r['mechanisms']:
            for row in r['mechanisms'][-3:]:
                print(f'  R{row["displayRound"]}: routes={row["routes"]}; commit scales={row["commitScales"]}; '
                    f'mean qualified classes={row["meanQualifiedClassCounts"]:.2f}')
    print(report['limitation'])


if __name__=='__main__':main()
