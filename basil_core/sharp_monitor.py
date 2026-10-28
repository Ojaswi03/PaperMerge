"""Read-only SHARP progress and measured runtime estimates.

Kept outside the frozen training fingerprint so active workers need no changes.
Unknown dataset/model/condition timings never borrow a different workload's rate.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import textwrap
import time


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):return 'unavailable'
    value=max(0,math.ceil(seconds));days,value=divmod(value,86400)
    hours,value=divmod(value,3600);minutes,seconds=divmod(value,60)
    clock=f'{hours:02}:{minutes:02}:{seconds:02}'
    return f'{days}d {clock}' if days else clock


def timing_key(cfg):
    # Pool timing samples across seeds/noisy sigmas only, never accuracy values.
    return json.dumps({k:v for k,v in cfg.items() if k not in ('seed','sigma','runId')},sort_keys=True)


def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return default


def snapshot(directory):
    root=Path(directory);plan=read_json(root/'plan.json')
    if plan is None:return dict(status='WAITING_FOR_PLAN',output=str(root),entries=[])
    configs=plan['configs']
    if not configs:raise ValueError('The frozen plan contains no experiments.')
    campaign=read_json(root/'campaign.json',dict(status='PREPARED'))
    gpus=[p['gpu'] for p in plan.get('profiles',[])] or [None]
    entries=[];rates=defaultdict(list);local_rates=defaultdict(list)
    for index,cfg in enumerate(configs):
        row=read_json(root/cfg['runId']/'run.json',dict(status='PENDING'))
        total=cfg['nodes']*cfg['rounds'];cursor=row.get('completedActivations',0)
        if type(cursor) is not int or not 0<=cursor<=total:
            raise ValueError(f'Invalid activation counter in {cfg["runId"]}.')
        elapsed=row.get('elapsedSeconds');rate=None
        if (isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed>0
                and cursor>=cfg['nodes'] and row['status'] in ('RUNNING','COMPLETED')):
            rate=elapsed/cursor
            rates[timing_key(cfg)].append(rate)
            local_rates[(timing_key(cfg),gpus[index%len(gpus)])].append(rate)
        eta=0. if row['status']=='COMPLETED' else None
        if row['status']=='RUNNING' and rate is not None:eta=rate*(total-cursor)
        entries.append(dict(config=cfg,status=row['status'],cursor=cursor,total=total,
            rounds=row.get('completedRounds',0),elapsed=elapsed,eta=eta,
            gpu=row.get('physicalGpuId',gpus[index%len(gpus)]),lane=gpus[index%len(gpus)],
            mean=row.get('meanAccuracy'),rate=rate,error=row.get('error')))
    done=sum(r['status']=='COMPLETED' for r in entries)
    failed=sum(r['status']=='FAILED' for r in entries)
    stopped=any(r['status'] in ('FAILED','INTERRUPTED') for r in entries)
    status='COMPLETED' if done==len(entries) else campaign['status']
    remaining={gpu:0. for gpu in gpus};unknown=[]
    for row in entries:
        if row['status']=='COMPLETED':continue
        if row['status']=='RUNNING':estimate=row['eta']
        elif row['status']=='PENDING':
            key=timing_key(row['config'])
            samples=local_rates.get((key,row['lane'])) or rates.get(key)
            estimate=statistics.median(samples)*row['total'] if samples else None
        else:estimate=None
        if estimate is None:
            unknown.append(row['config']['dataset']);remaining[row['lane']]=None
        elif remaining[row['lane']] is not None:remaining[row['lane']]+=estimate
    eta=0. if status=='COMPLETED' else None
    if status=='RUNNING' and not unknown and not stopped:eta=max(remaining.values())
    return dict(status=status,entries=entries,completed=done,failed=failed,total=len(entries),
        pending=sum(r['status']=='PENDING' for r in entries),
        completedActivations=sum(r['cursor'] for r in entries),totalActivations=sum(r['total'] for r in entries),
        elapsed=campaign.get('elapsedSeconds'),eta=eta,remainingByGpu=remaining,
        untimedJobs=len(unknown),untimedDatasets=sorted(set(unknown)),output=str(root))


def render(state,*,width=100,height=30):
    width=max(width,30)
    if state['status']=='WAITING_FOR_PLAN':return f'Waiting for frozen plan in {state["output"]}\n'
    lines=[f'SHARP + CART | {state["status"]} | {state["completed"]}/{state["total"]} complete | '
        f'failed {state["failed"]} | pending {state["pending"]}',
        f'Progress {state["completedActivations"]/state["totalActivations"]:.2%} | '
        f'{state["completedActivations"]}/{state["totalActivations"]} activations',
        f'Elapsed {duration(state["elapsed"])} | Campaign ETA {duration(state["eta"])}']
    if state['untimedJobs'] and state['status'] not in ('COMPLETED','FAILED','INTERRUPTED'):
        lines.append(f'Waiting for timing samples: {state["untimedJobs"]} jobs '
            f'({", ".join(state["untimedDatasets"])})')
    wrap=lambda values:[part for line in values for part in (textwrap.wrap(line,width,break_long_words=True) or [''])]
    lines=wrap(lines);lines.append('')
    active=[r for r in state['entries'] if r['status'] in ('RUNNING','FAILED','INTERRUPTED')]
    recent=[r for r in state['entries'] if r['status']=='COMPLETED'][-4:]
    visible=active+recent;shown=0
    for row in visible:
        cfg=row['config'];gpu=f'G{row["gpu"]}' if row['gpu'] is not None else 'CPU'
        mean=f'{row["mean"]:.2%}' if row['mean'] is not None else '--'
        block=wrap([f'{gpu} {row["status"]} {cfg["dataset"]}/{cfg["partition"]} S{cfg["seed"]} sigma={cfg["sigma"]:g}',
            f'{cfg["condition"]} | R{row["rounds"]}/{cfg["rounds"]} A{row["cursor"]}/{row["total"]}',
            f'mean {mean} | elapsed {duration(row["elapsed"])} | ETA {duration(row["eta"])}'])
        if len(lines)+len(block)+4>height:break
        lines.extend(block);shown+=1
    if shown<len(active):lines.append(f'Showing {shown}/{len(active)} active/stopped jobs; enlarge terminal for more.')
    if state['status']=='COMPLETED':lines.append('ALL EXPERIMENTS COMPLETE.')
    elif state['status'] in ('FAILED','INTERRUPTED'):lines.append('Campaign stopped; inspect worker logs before recovery.')
    else:
        lines.append('ETA uses recorded training speed; setup/JIT and later workloads can change it.')
        lines.append('Refresh 5 seconds | Ctrl+C closes only this monitor')
    return '\n'.join(wrap(lines))+'\n'


def monitor(directory,*,once=False):
    try:
        while True:
            state=snapshot(directory);size=shutil.get_terminal_size(fallback=(100,30))
            if not once and sys.stdout.isatty():print('\033[2J\033[H',end='')
            print(render(state,width=size.columns,height=size.lines),end='',flush=True)
            if once or state['status'] in ('COMPLETED','FAILED','INTERRUPTED'):return
            time.sleep(5)
    except KeyboardInterrupt:print('\nMonitor closed.',flush=True)
