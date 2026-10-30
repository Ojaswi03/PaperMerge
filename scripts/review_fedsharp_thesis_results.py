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
    args=p.parse_args();result=review(args.output,args.destination)
    print(f'Checked {result["checkedFiles"]} files; {result["configurations"]} configurations; '
        f'{result["activations"]} activations; {result["optimizerSteps"]} optimizer steps.')
    print('Four SVGs generated. Inspect the PNG previews before installing the SVGs.')
