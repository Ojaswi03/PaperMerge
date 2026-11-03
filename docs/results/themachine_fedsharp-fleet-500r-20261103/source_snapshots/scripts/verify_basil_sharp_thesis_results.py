#!/usr/bin/env python3
"""Independent final-model inference using the thesis family's exact data contract."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from basil_core.basil_sharp_thesis import installed
    from scripts.verify_cart_comparison_results import verify
    with installed():verify(args.output.resolve())
