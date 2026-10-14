#!/usr/bin/env python3
"""Validate/prepare one separate clean 30-round diagnostic; never start a queue."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    os.environ['PAPERMERGE_LIVE_TRACKING']='disabled'
    os.environ['WANDB_MODE']='disabled'
    from basil_core.class_aware_retention import diagnostic_config
    from basil_core.research_protocol import validate_config
    config=diagnostic_config()
    validate_config(config)
    path=ROOT/'gui/configs/nonIID/class_aware_retention'/f"{config['runId']}.json"
    content=json.dumps(config,indent=2,sort_keys=True)+'\n'
    if args.prepare:
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_text()!=content:
            raise ValueError('Refusing to change an existing diagnostic config.')
        path.write_text(content)
    print(json.dumps(dict(status='prepared' if args.prepare else 'validated',
        configPath=str(path),rounds=30,localEpochs=5,samplesPerNode=5000,
        executionStarted=False,queueModified=False,wandbEnabled=False)))


if __name__=='__main__':main()
