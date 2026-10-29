#!/usr/bin/env python3
"""Regenerate revision plots and complete-seed comparisons without retraining."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    from basil_core.cart_campaign_io import read_json
    from reporting.cart_revision import generate
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();plan=read_json(a.output/'plan.json')
    generate(a.output,a.figures,reference=a.reference or plan.get('reference'))
    print('Averages and plots updated: '+str(a.figures))
