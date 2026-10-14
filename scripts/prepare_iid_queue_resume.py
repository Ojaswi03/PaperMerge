#!/usr/bin/env python3
"""Prepare an idle A-first queue without changing configs or research results."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from scripts.run_iid_queue import load_prepared_queue,queue_lease,PRESET
    from gui.services.queue_service import QueueService
    from gui.state.queue_state import QueueStatus
    target=ROOT/'gui/queues/iid_abcd_500r_gpu_resume.json'
    with queue_lease(ROOT):
        state,configs,plans=load_prepared_queue(PRESET,root=ROOT)
        QueueService.describe_recovery(state,ROOT)
        for entry,plan in zip(state.entries,plans):
            entry.status=QueueStatus.COMPLETED if plan['mode']=='completed' else QueueStatus.PENDING
            entry.error=''
        QueueService(target).save(state)
        QueueService(ROOT/'gui/queue_state.json').save(state)
    for config,plan in zip(configs,plans):
        print(json.dumps(dict(condition=config['conditionId'],completedRounds=plan['completedRounds'],
            totalRounds=config['nRounds'],startActivation=plan['startActivation'],mode=plan['mode'])))
    print(f'Idle queue saved: {target.relative_to(ROOT)}. Research runs not started.')


if __name__=='__main__':main()
