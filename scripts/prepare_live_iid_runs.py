#!/usr/bin/env python3
"""Create four empty W&B records for fresh IID runs; zero training/metric points."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from basil_core.live_tracking import LiveTracker,SETTINGS_PATH,read_settings
    from scripts.run_iid_queue import load_prepared_queue,queue_lease,PRESET
    settings=read_settings()
    if not settings.get('enabled') or settings.get('mode')!='online' or not settings.get('group'):
        raise ValueError('Configure online tracking with a fresh attempt group first.')
    with queue_lease(ROOT):
        _,configs,plans=load_prepared_queue(PRESET,root=ROOT)
        if any(plan['mode']!='fresh' for plan in plans):
            raise ValueError('Cloud preparation requires fresh local outputs; do not replace active/resumed tracking.')
        for config in configs:
            tracker=LiveTracker(config,dict(initialModelHash=config['expectedInitialModelHash'],
                iidPartitionHash=config['expectedPartitionHash'],device='GPU'),settings=settings,prepare_only=True)
            if tracker.metadata.get('status')!='active':
                raise ValueError(f"W&B preparation failed for {config['conditionId']}: {tracker.metadata.get('error')}")
            tracker.run.summary['research_rounds_completed']=0
            tracker.run.summary['research_training_started']=False
            tracker.finish('prepared')
            if tracker.metadata.get('status')!='prepared':
                raise ValueError(f"W&B preparation finish failed: {tracker.metadata.get('error')}")
            settings.setdefault('preparedRuns',{})[config['conditionId']]=dict(tracker.metadata)
            temporary=SETTINGS_PATH.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(settings,indent=2)+'\n');temporary.replace(SETTINGS_PATH)
            print(json.dumps(dict(condition=config['conditionId'],status='prepared_idle',
                runId=tracker.metadata['runId'],url=tracker.metadata['url'],group=settings['group'],
                researchRunsStarted=0,metricPoints=0)),flush=True)
    print('Four cloud records prepared; queue idle, no training and no fabricated metrics.',flush=True)


if __name__=='__main__':main()
