#!/usr/bin/env python3
"""One fixed-horizon condition; default is a zero-training dry run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basil_core.execution_profile import profile_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--recover',action='store_true')
    args=parser.parse_args()
    from basil_core.dirichlet_horizon_study import validate_configuration,gui_approved,recovery_plan
    config=json.loads(args.config.read_text())
    validate_configuration(config)
    approved=gui_approved(config)
    plan=recovery_plan(profile_path(ROOT/config['resultDirectory']),config) if approved else dict(mode='blocked',reason='Prerequisite validation required')
    print(json.dumps(dict(trainingStarted=False,executionReady=approved,
        condition=config['conditionId'],rounds=config['nRounds'],recovery=plan,
        wandbEnabled=config['liveTrackingEnabled'])),flush=True)
    if not args.execute:return
    if not approved:raise ValueError('Prerequisite protocol/numerical validation has not passed.')
    from scripts.run_dirichlet_calibration import run_pair
    try:
        run_pair(config['executionDevice'],recover=args.recover,stage='threat',
            output_root=profile_path(ROOT/config['resultDirectory'].rsplit('/',1)[0]),configs_override=[config],gui_events=True)
    except InterruptedError:raise SystemExit(130)
    import os
    if os.environ.get('PAPERMERGE_SKIP_PLOTS')!='1':
        from reporting.noniid_study_plots import generate_condition_plots
        generate_condition_plots(profile_path(ROOT/config['resultDirectory']),profile_path(ROOT/config['plotDirectory']))


if __name__=='__main__':main()
