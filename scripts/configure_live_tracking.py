#!/usr/bin/env python3
"""Save local W&B destination settings, without starting a run or uploading data."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    from basil_core.live_tracking import SETTINGS_PATH
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entity')
    parser.add_argument('--project')
    parser.add_argument('--mode',choices=('online','offline'),default='online')
    parser.add_argument('--group',help='Optional shared tracking group for one fresh A/B/C/D attempt')
    parser.add_argument('--disable',action='store_true')
    args=parser.parse_args(argv)
    if not args.disable and (not args.entity or not args.project):
        parser.error('--entity and --project are required; do not pass an API key here.')
    settings=dict(enabled=not args.disable,entity=args.entity,project=args.project,mode=args.mode)
    if args.group:settings['group']=args.group
    SETTINGS_PATH.parent.mkdir(parents=True,exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings,indent=2)+'\n')
    print(json.dumps(dict(settingsPath=str(SETTINGS_PATH),**settings,researchRunsStarted=0)))
    if not args.disable:
        print('Authenticate locally with wandb login. Public visibility is set in W&B project settings, not this script.')


if __name__=='__main__':main()
