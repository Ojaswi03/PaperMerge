"""Five-second terminal view of the factorial campaign's committed progress."""
from __future__ import annotations

import json
import math
from pathlib import Path
import socket
import time


def snapshot(directory):
    from basil_core.terminal.state import alive
    root=Path(directory)
    plan=json.loads((root/'plan.json').read_text())
    campaign=json.loads((root/'campaign.json').read_text())
    entries=[]
    for cfg in plan['configs']:
        folder=root/cfg['runId'];path=folder/'run.json'
        row=json.loads(path.read_text()) if path.exists() else dict(status='PENDING')
        cursor=row.get('completedActivations',0);total=10*cfg['nRounds']
        elapsed=row.get('elapsedSeconds');eta=None
        if cursor and elapsed is not None and row['status']=='RUNNING':eta=elapsed/cursor*(total-cursor)
        elif row['status']=='COMPLETED':eta=0.
        counts_path=folder/'partition.json'
        counts=json.loads(counts_path.read_text())['counts'] if counts_path.exists() else None
        updates=total_updates=None
        if counts:
            steps=[5*math.ceil(c/cfg['batchSize']) for c in counts]
            total_updates=cfg['nRounds']*sum(steps)
            completed,nodes=divmod(cursor,10)
            updates=completed*sum(steps)+sum(steps[:nodes])
        label=f'{"IID" if cfg["partitionStrategy"]=="iid" else "Dir"} S{cfg["seed"]} sigma={cfg["channelNoiseSigmaAbsolute"]:g} {cfg["conditionId"]}'
        entries.append(dict(name=cfg['runId'],condition=label,status=row['status'],dataset='cifar10',
            nonIID=cfg['partitionStrategy']!='iid',nodes=10,device='GPU',rounds=cfg['nRounds'],
            completedRounds=row.get('completedRounds',0),completedActivations=cursor,totalActivations=total,
            completedOptimizerSteps=updates,totalOptimizerSteps=total_updates,progress=cursor/total,
            gpu=row.get('physicalGpuId'),elapsed_seconds=elapsed,eta_seconds=eta,
            checkpoint=dict(activation=cursor) if cursor else None,
            error=row.get('error'),metrics=dict(latest_node_accuracy=row.get('latestValidationNode'),
                mean_node_accuracy=row.get('validationMean'),worst_node_accuracy=row.get('validationWorst'),
                local_training_loss=row.get('latestTrainCE')),tracking=dict(status='disabled')))
    total=len(entries)
    totals=dict(total=total,completed=sum(e['status']=='COMPLETED' for e in entries),
        pending=sum(e['status']=='PENDING' for e in entries),failed=sum(e['status']=='FAILED' for e in entries),
        blocked=0,progress=sum(e['progress'] for e in entries)/max(total,1))
    status=campaign['status']
    if status=='INTERRUPTED':status='STOPPED'
    if status=='RUNNING' and campaign.get('owner') and alive(campaign['owner']) is False:status='STOPPED'
    if totals['completed']==total:status='COMPLETED'
    eta=0. if status=='COMPLETED' else None
    if not totals['pending']:
        estimates=[e['eta_seconds'] for e in entries if e['status']=='RUNNING']
        if estimates and all(v is not None for v in estimates):eta=max(estimates)
    order={'RUNNING':0,'FAILED':1,'INTERRUPTED':2,'PENDING':3,'COMPLETED':4}
    shown=sorted(entries,key=lambda e:order.get(e['status'],5))
    if status=='RUNNING':shown=shown[:8]
    return dict(status=status,entries=shown,totals=totals,resources=plan['profiles'],
        profile=plan['profiles'][0]['name'],host=campaign.get('host',socket.gethostname()),
        queue=f'TRAIN holdout IID-Dirichlet factorial: {total} selected configs',
        elapsed_seconds=campaign.get('elapsedSeconds'),eta_seconds=eta,activity=[],
        last_event_wall=max((p.stat().st_mtime for p in root.glob('*/run.json')),default=None))


def monitor(root):
    from basil_core.terminal.dashboard import monitor as terminal_monitor
    # Reuse screen fitting, clear-on-refresh, resource sampling and terminal exit.
    return terminal_monitor(root,interval=5.,screen=True,fetch=snapshot)
