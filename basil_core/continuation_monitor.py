"""Operational continuation timing, outside the frozen numerical producer."""
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from basil_core.cart_campaign_io import read_json


def snapshot(queue):
    from basil_core.continuation_campaign import public_snapshot
    queue=Path(queue);state=public_snapshot(queue);master=read_json(queue/'continuation_plan.json')
    inherited=sum(item['sourceCursor'] for item in master['items'])
    elapsed=state['elapsedSeconds']
    if state['status'] in ('COMPLETED','COMPLETED_WITH_FAILURES','COMPLETED_REPORT_FAILED'):
        reports=[read_json(p) for p in (queue/'hosts').glob('*.json')]
        if reports:elapsed=max(0,max(r['updatedAt'] for r in reports)-min(r['startedAt'] for r in reports))
        state['elapsedSeconds']=elapsed
    gained=max(0,state['completedActivations']-inherited)
    speed=gained/elapsed if elapsed and elapsed>0 and gained>0 else None
    blocked=[r for r in state['runs'] if r['status'] not in ('PENDING','RUNNING','STARTING','COMPLETED')]
    remaining=sum(max(0,r['totalActivations']-r['completedActivations']) for r in state['runs']
        if r['status'] in ('PENDING','RUNNING','STARTING'))
    eta=remaining/speed if speed and state['status'] in ('RUNNING','WAITING_FOR_RESOURCES') else None
    if state['completed']==state['total']:eta=0.
    state.update(activationsPerSecond=speed,etaSeconds=eta,etaLowSeconds=eta,etaHighSeconds=eta,
        untimedJobs=len(blocked),etaMode='activation_throughput_excludes_blocked' if blocked else 'activation_throughput')
    return state


def render(queue):
    from basil_core.sharp_monitor import duration
    state=snapshot(queue);speed=state['activationsPerSecond'];eta=state['etaSeconds']
    active=[r for r in state['runs'] if r['status'] in ('RUNNING','STARTING')]
    finish=(datetime.fromisoformat(state['generatedAt'])+timedelta(seconds=eta)).astimezone(ZoneInfo('America/New_York')).strftime('%a %b %d %H:%M:%S %Z') if eta is not None else 'unavailable'
    percent=100*state['completedActivations']/state['totalActivations'] if state['totalActivations'] else 0.
    lines=[f"500-round continuation | {state['status']} | {state['completed']}/{state['total']} complete | failed {state['failed']} | pending {state['pending']} | active {len(active)}",
        f"Progress {percent:.2f}% | {state['completedActivations']}/{state['totalActivations']} activations",
        f"Elapsed {duration(state['elapsedSeconds'])} | Overall speed {f'{speed:.2f} activations/s' if speed else 'measuring'} | Estimated remaining {duration(eta)}",
        'Estimated finish '+finish,
        'GPU workers: '+(' | '.join(f'{key}: {count}' for key,count in sorted(Counter(
            r['phase'].split(':')[1]+' GPU '+str(r['gpu']) for r in active).items())) or 'none')]
    if state['untimedJobs']:lines.append(f"ETA covers running and queued work; {state['untimedJobs']} failed/stopped configurations excluded until requeued.")
    lines.append('SERVER      GPU CLAIM           STATUS          DATA/SPLIT       SEED CONDITION                         ROUND    ACTIVATIONS    ELAPSED  ETA')
    for row in state['runs']:
        _,host,claim,_,attack=row['phase'].split(':',4)
        split='IID' if row['partition']=='iid' else 'nonIID'
        lines.append(f"{host:11} {str(row['gpu']):3} {claim:15} {row['status']:15} {row['dataset']+'/'+split:16} "
            f"{row['seed']} {row['condition']+' '+attack:33} {row['completedRounds']:3}/{row['rounds']:<3} "
            f"{row['completedActivations']:5}/{row['totalActivations']:<5} {duration(row['elapsedSeconds']):8} {duration(row['etaSeconds'])}")
    return '\n'.join(lines)
