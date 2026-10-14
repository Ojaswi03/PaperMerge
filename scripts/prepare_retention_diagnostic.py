#!/usr/bin/env python3
"""Prepare two clean full-data controls; this command never trains or queues."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def diagnostic_configs(rounds=5):
    from basil_core.noniid_study import study_configs
    from basil_core.research_protocol import validate_config

    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds not in {5, 30}:
        raise ValueError('Retention diagnostics support the explicit 5- or 30-round horizon.')
    configs = []
    for label, mu in [('clean_ce_control', 0.0), ('clean_fixed_anchor_control', 0.01)]:
        config = study_configs(mu, device='GPU')[0]
        name = f'{label}_{rounds}r_gpu'
        config.update(nRounds=rounds, researchValid=False,
            purpose='full_data_retention_mechanism_diagnostic',
            controlId=label, runId=name, experimentName=name,
            effectiveLocalObjective='cross_entropy' if mu == 0 else 'snapshot_anchor_only',
            resultDirectory=f'newResults/nonIID/retention_diagnostic/{name}',
            plotDirectory=f'newPlots/nonIID/retention_diagnostic/{name}')
        # The zero-mu anchor path is exactly CE; using the same worker isolates
        # the anchor without introducing a different augmentation/training path.
        validate_config(config)
        configs.append(config)
    excluded = {'controlId', 'runId', 'experimentName', 'effectiveLocalObjective',
                'resultDirectory', 'plotDirectory'}
    scientific_diff = {k for k in configs[0].keys() | configs[1].keys()
                       if configs[0].get(k) != configs[1].get(k)} - excluded
    if scientific_diff != {'snapshotAnchorMu'}:
        raise ValueError(f'Unexpected paired-control differences: {scientific_diff}')
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--rounds', type=int, choices=[5, 30], default=5)
    args = parser.parse_args()
    os.environ['PAPERMERGE_LIVE_TRACKING'] = 'disabled'
    os.environ['WANDB_MODE'] = 'disabled'
    configs = diagnostic_configs(args.rounds)
    folder = ROOT / 'gui/configs/nonIID/retention_diagnostic'
    if args.prepare:
        folder.mkdir(parents=True, exist_ok=True)
        for cfg in configs:
            path = folder / f"{cfg['runId']}.json"
            content = json.dumps(cfg, indent=2, sort_keys=True) + '\n'
            if path.exists() and path.read_text() != content:
                raise ValueError(f'Refusing to replace a different config: {path}')
            path.write_text(content)
    print(json.dumps(dict(status='prepared' if args.prepare else 'validated',
        executionStarted=False, queueModified=False, wandbEnabled=False,
        rounds=args.rounds, localEpochs=5, samplesPerNode=5000,
        scientificDifferences=['snapshotAnchorMu'],
        configs=[str(folder / f"{cfg['runId']}.json") for cfg in configs]), indent=2))


if __name__ == '__main__':
    main()
