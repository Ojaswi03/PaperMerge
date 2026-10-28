#!/usr/bin/env python3
"""Export three-seed CART v2 line/bar figures, CSVs and source provenance."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True);args=p.parse_args()
    from reporting.cart_v2_campaign import generate
    result=generate(args.output,args.figures)
    print(f'{result["status"]}: {len(result["files"])} artifacts in {args.figures}')
