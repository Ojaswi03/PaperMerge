#!/usr/bin/env python3
"""Require the existing exact method gates and explicit zero-attack controls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(root, gpu, lease_fd):
    from basil_core import fedsharp_noise_only as study
    from basil_core.cart_campaign_io import read_json, write_json
    from scripts.verify_basil_sharp_thesis_gpu import verify as verify_methods
    from basil_core.research_protocol import resolve_attackers
    plan = read_json(root / 'plan.json')
    study.check_source(plan['source'])
    for view in plan['configs']:
        cfg = study.effective_config(view)
        if cfg['actualAttackers'] != 0 or cfg['switches']['attack'] or cfg['sigma'] <= 0:
            raise ValueError('Device gate requires physical noise and zero attackers.')
        if resolve_attackers(cfg['seed'], cfg['actualAttackers'], cfg['nodes']):
            raise ValueError('Noise-only attacker set is not empty.')
    # Includes device-local candidate/SGD checks and the informative 19-draw
    # admission/commit branch, using the same frozen numerical implementation.
    with study.installed():
        verify_methods(root, gpu, lease_fd)
    receipt = read_json(root / f'revision_gpu_{gpu}.json')
    receipt.update(noiseOnly=True, actualAttackers=0, emptyAttackerSetsVerified=True)
    write_json(root / f'revision_gpu_{gpu}.json', receipt)
    return receipt
