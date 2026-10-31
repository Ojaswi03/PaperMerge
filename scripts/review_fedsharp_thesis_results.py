#!/usr/bin/env python3
"""Verify all HPC evidence and generate the four renamed publication figures."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    from reporting.fedsharp_publication import review
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--png-only',action='store_true',help='Generate PNG previews without creating SVG files.')
    args=p.parse_args();result=review(args.output,args.destination,formats=('png',) if args.png_only else ('svg','png'))
    print(f'Checked {result["checkedFiles"]} files; {result["configurations"]} configurations; '
        f'{result["activations"]} activations; {result["optimizerSteps"]} optimizer steps.')
    print('Four PNG previews generated; no SVGs created.' if args.png_only else
        'Four SVGs and PNGs generated. Inspect the PNG previews before installing the SVGs.')
