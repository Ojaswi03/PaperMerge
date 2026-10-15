#!/usr/bin/env python3
"""Inspect one evidence-gated prior confirmation; train only on --execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--recover',action='store_true')
    args=parser.parse_args()
    from basil_core.dirichlet_prior_study import validate_configuration,gui_approved,recovery_plan
    config=json.loads(args.config.read_text())
    validate_configuration(config)
    approved=gui_approved(config)
    plan=recovery_plan(ROOT/config['resultDirectory'],config) if approved else dict(mode='blocked',reason='Readiness gates not approved')
    print(json.dumps(dict(status='configuration_valid',trainingStarted=False,
        executionReady=approved,condition=config['conditionId'],rounds=30,recovery=plan)),flush=True)
    if not args.execute:return
    if not approved:raise ValueError('Bounded learning/threat checks have not approved this confirmation queue.')
    from scripts.run_dirichlet_calibration import run_pair
    try:
        run_pair(config['executionDevice'],recover=args.recover,stage='threat',
            output_root=ROOT/config['resultDirectory'].rsplit('/',1)[0],configs_override=[config],gui_events=True)
    except InterruptedError:raise SystemExit(130)
    from reporting.noniid_study_plots import generate_condition_plots
    # A completed checkpoint may need plots rebuilt after a prior plotting error.
    generate_condition_plots(ROOT/config['resultDirectory'],ROOT/config['plotDirectory'])


if __name__=='__main__':main()
