#!/usr/bin/env python3
"""Review all four completed confirmation runs; never train or fill missing data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    from basil_core.dirichlet_prior_study import STUDY
    from reporting.dirichlet_calibration import review,generate_plots,verify_final_transmissions
    from scripts.run_noniid_basil_study import save_json
    source=ROOT/'newResults/nonIID'/STUDY
    output=source/'comparison'
    plots=ROOT/'newPlots/nonIID'/STUDY/'comparison'
    if output.exists() or plots.exists():
        raise ValueError('Completed comparison exists; preserve it instead of overwriting.')
    result,histories=review(source,names=tuple('ABCD'))
    result['transmissionAudit']=verify_final_transmissions(source,tuple('ABCD'))
    summaries=result['summaries']
    result['contrastsPp']={label:{metric:100*(summaries[right][metric]-summaries[left][metric])
        for metric in ('finalFullTestAccuracy','finalMeanNodeAccuracy','finalWorstNodeAccuracy')}
        for label,left,right in [('attack_B_minus_A','A','B'),('noise_C_minus_B','B','C'),
            ('ebm_D_minus_C','C','D'),('total_D_minus_A','A','D')]}
    result['interpretation']='Single-seed fixed 30-round confirmation; not 500-round readiness or universal robustness.'
    generate_plots(result,histories,plots)
    save_json(output/'analysis.json',result)
    print(f'Completed measured comparison: {output / "analysis.json"}')


if __name__=='__main__':main()
