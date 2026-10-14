#!/usr/bin/env python3
"""Regenerate completed non-IID plots/review from saved evidence, no training."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from reporting.noniid_study_plots import generate_condition_plots, generate_study_report
    configs=[json.loads(path.read_text()) for path in sorted((ROOT/'gui/configs/nonIID/sequential_basil_50r').glob('*.json'))]
    if len(configs)!=4 or [c['conditionId'] for c in configs]!=list('ABCD'):
        raise ValueError('Exactly the saved four 50-round A/B/C/D configurations are required.')
    for cfg in configs:
        if not cfg['resultDirectory'].startswith('newResults/nonIID/') or not cfg['plotDirectory'].startswith('newPlots/nonIID/'):
            raise ValueError('Refusing outputs outside the non-IID study.')
        generate_condition_plots(ROOT/cfg['resultDirectory'],ROOT/cfg['plotDirectory'])
    print(json.dumps(generate_study_report([ROOT/c['resultDirectory'] for c in configs],
        ROOT/'newResults/nonIID/ABCD_50r_comparison',ROOT/'newPlots/nonIID/ABCD_50r_comparison')))


if __name__=='__main__':main()
