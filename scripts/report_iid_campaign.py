#!/usr/bin/env python3
"""Regenerate completed A/B/C/D figures and contrasts without training."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('CPU','GPU'),default='CPU')
    args=parser.parse_args()
    suffix='_gpu' if args.device=='GPU' else ''
    from basil_core.iid_campaign import CONDITIONS
    from reporting.iid_study_plots import generate_run_plots
    from reporting.iid_campaign_plots import report_campaign
    import json
    for item in CONDITIONS.values():
        result=ROOT/'newResults/IID'/(item[1]+suffix)
        if (result/'run.json').exists() and json.loads((result/'run.json').read_text())['status']=='completed':
            generate_run_plots(result,ROOT/'newPlots/IID'/(item[1]+suffix))
    print(json.dumps(report_campaign(device=args.device),indent=2))


if __name__=='__main__':main()
