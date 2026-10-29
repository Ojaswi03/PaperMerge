#!/usr/bin/env python3
"""Plot complete comparison seeds, optionally alongside frozen result exports."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--reference',action='append',type=Path)
    a=p.parse_args()
    from reporting.cart_comparison import generate
    result=generate(a.output,a.destination,a.reference)
    print('Figures: '+result['destination'])
