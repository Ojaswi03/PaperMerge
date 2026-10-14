#!/usr/bin/env python3
"""Read-only W&B access check. Never create runs or upload metrics."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    from basil_core.live_tracking import read_settings
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    settings=read_settings()
    if not settings.get('enabled'):
        raise ValueError('Live tracking is disabled; configure it first.')
    import wandb
    api=wandb.Api(timeout=15)
    try:
        project=api.project(settings['project'],entity=settings['entity'])
        if not project.id:raise ValueError('Project not found')
    except Exception as error:
        raise ValueError(f"Cannot access {settings['entity']}/{settings['project']}. "
            f"Saved key's default entity: {api.default_entity}. "
            'Run wandb login --relogin locally with the correct account key; do not share the key.') from error
    print(json.dumps(dict(status='accessible',entity=settings['entity'],project=settings['project'],
        workspace=project.url,researchRunsStarted=0,metricsUploaded=0,
        publicVisibility='Verify Public in project settings; API access alone does not prove public viewing.')))


if __name__=='__main__':main()
