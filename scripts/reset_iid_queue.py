#!/usr/bin/env python3
"""Reset ONLY the current 500-round IID outputs; never start training.

Requires --reset. Partial evidence is moved to ignored local recovery storage.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def reset(root, preset):
    from basil_core.iid_campaign import verify_campaign
    from gui.services.queue_service import QueueService
    from gui.state.experiment_state import to_persisted
    from scripts.run_iid_queue import queue_lease
    state=QueueService(preset).load()
    configs=[to_persisted(e.config) for e in state.entries]
    verify_campaign(configs)
    if any(c['nRounds']!=500 or c['executionDevice']!='GPU' for c in configs):
        raise ValueError('Only the current 500-round GPU study can be reset.')
    relative=[c[k] for c in configs for k in ('resultDirectory','plotDirectory')]
    relative += [configs[0][k] for k in ('comparisonResultsDirectory','comparisonPlotsDirectory')]
    targets=[root / path for path in relative]
    for path in targets:
        if path.resolve()!=path.absolute() or (path.exists() and not path.is_dir()):
            raise ValueError(f'Refusing symlink or non-directory reset target: {path}')
        if path.exists() and any(p.is_symlink() for p in path.rglob('*')):
            raise ValueError(f'Refusing symlinked evidence under: {path}')
    settings_path=root/'gui/worker_state/live_tracking.json'
    settings=json.loads(settings_path.read_text()) if settings_path.exists() else {'enabled':False}
    group=f"iid_abcd_500r_v1_fresh_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    with queue_lease(root):
        backup=Path(tempfile.mkdtemp(prefix='iid-reset-',dir=root/'gui/worker_state'))
        hashes={}
        for path in targets:
            if path.exists():
                for file in path.rglob('*'):
                    if file.is_file():hashes[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
        for path in targets:
            if path.exists():
                destination=backup / path.relative_to(root)
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(path),str(destination))
            path.mkdir(parents=True,exist_ok=True)
        if any(hashlib.sha256((backup/name).read_bytes()).hexdigest()!=digest for name,digest in hashes.items()):
            raise RuntimeError('Recovery backup hash verification failed; do not start training.')
        settings['group']=group
        settings.pop('preparedRuns',None)
        settings_path.write_text(json.dumps(settings,indent=2)+'\n')
        QueueService(root/'gui/queue_state.json').save(state)
        QueueService(root/'gui/queues/iid_abcd_500r_gpu_resume.json').save(state)
        record=dict(status='fresh_idle',researchRunsStarted=0,order=list('ABCD'),rounds=500,
            trackingGroup=group,backup=str(backup.relative_to(root)),preservedFiles=len(hashes),sha256=hashes)
        (backup/'reset_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reset',action='store_true',help='Move current partial IID evidence aside and prepare all four fresh runs')
    args=parser.parse_args(argv)
    if not args.reset:
        print('No changes. Pass --reset only after stopping the queue to prepare fresh A/B/C/D runs.')
        return
    evidence=reset(ROOT,ROOT/'gui/queues/iid_abcd_500r_gpu_fresh.json')
    print(json.dumps({k:v for k,v in evidence.items() if k!='sha256'},indent=2))


if __name__=='__main__':main()
