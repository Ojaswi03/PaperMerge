"""Read-only SHARP progress with measured and explicitly labeled prior ETAs.

Kept outside the frozen training fingerprint so active workers need no changes.
Unseen workloads use their own planning prior, never another dataset's speed.
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
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):return 'unavailable'
    value=max(0,math.ceil(seconds));days,value=divmod(value,86400)
    hours,value=divmod(value,3600);minutes,seconds=divmod(value,60)
    clock=f'{hours:02}:{minutes:02}:{seconds:02}'
    return f'{days}d {clock}' if days else clock


def timing_key(cfg):
    # Pool timing samples across seeds/noisy sigmas only, never accuracy values.
    return json.dumps({k:v for k,v in cfg.items() if k not in
        ('seed','sigma','runId','baseRunId','guardedRunId','rounds','attackStart')},sort_keys=True)


def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return default


def snapshot(directory,*,timing_roots=()):
    root=Path(directory);plan=read_json(root/'plan.json')
    if plan is None:return dict(status='WAITING_FOR_PLAN',output=str(root),entries=[])
    configs=plan['configs']
    if not configs:raise ValueError('The frozen plan contains no experiments.')
    campaign=read_json(root/'campaign.json')
    startup=None
    if campaign is None:
        managed=read_json(root/'managed_status.json',{})
        stage=managed.get('status')
        status=stage if stage in ('FAILED','INTERRUPTED') else ('STARTING' if managed else 'PREPARED')
        campaign=dict(status=status,elapsedSeconds=managed.get('wallSeconds'))
        startup=dict(managerStatus=stage,error=managed.get('error'),supervisorPid=managed.get('supervisorPid'))
    continuing=campaign.get('workerFailurePolicy')=='continue'
    queued=set(campaign.get('pendingRunIds',[]));active_ids=set(campaign.get('activeRunIds',[]))
    gpus=[p['gpu'] for p in plan.get('profiles',[])] or [None]
    lanes=gpus if len(set(gpus))==len(gpus) else [f'{gpu}/slot{index}' for index,gpu in enumerate(gpus)]
    concurrency=plan.get('jobsPerGpu',max(gpus.count(gpu) for gpu in gpus))
    entries=[];rates=defaultdict(list);local_rates=defaultdict(list);history=defaultdict(list)
    for location in timing_roots:
        previous=Path(location)
        if previous.resolve()==root.resolve():continue
        old=read_json(previous/'plan.json')
        if not old or old.get('source')!=plan.get('source'):continue
        if old.get('jobsPerGpu',1)!=concurrency:continue
        old_campaign=read_json(previous/'campaign.json',{})
        if old_campaign.get('runtimeAdapter')!=campaign.get('runtimeAdapter'):continue
        if campaign.get('host') and old_campaign.get('host')!=campaign['host']:continue
        if campaign.get('profile') and old_campaign.get('profile')!=campaign['profile']:continue
        for cfg in old['configs']:
            row=read_json(previous/cfg['runId']/'run.json',{})
            elapsed=row.get('elapsedSeconds');cursor=row.get('completedActivations',0)
            adapter=row.get('runtimeAdapter',{})
            cursor-=adapter.get('startedCursor',0)
            if isinstance(elapsed,(int,float)):elapsed-=adapter.get('startedElapsedSeconds',0.)
            if (row.get('status')=='COMPLETED' and isinstance(elapsed,(int,float)) and math.isfinite(elapsed)
                    and elapsed>0 and cursor>=cfg['nodes']):
                history[(timing_key(cfg),row.get('runtimeAdapter',{}).get('name'))].append(elapsed/cursor)
    for index,cfg in enumerate(configs):
        row=read_json(root/cfg['runId']/'run.json',dict(status='PENDING'))
        if continuing and cfg['runId'] in queued:row=dict(row,status='PENDING')
        elif continuing and cfg['runId'] in active_ids and row['status'] in ('FAILED','INTERRUPTED'):
            row=dict(row,status='RUNNING')
        elif continuing and cfg['runId'] in active_ids and row['status']=='PENDING':
            row=dict(row,status='STARTING',phase='WORKER_INITIALIZATION')
        total=cfg['nodes']*cfg['rounds'];cursor=row.get('completedActivations',0)
        if type(cursor) is not int or not 0<=cursor<=total:
            raise ValueError(f'Invalid activation counter in {cfg["runId"]}.')
        elapsed=row.get('elapsedSeconds');rate=None
        adapter=row.get('runtimeAdapter',{})
        backend=adapter.get('name') or (campaign.get('runtimeAdapter') if row['status']!='COMPLETED' else None)
        sampled_cursor=cursor-adapter.get('startedCursor',0)
        sampled_elapsed=elapsed-adapter.get('startedElapsedSeconds',0.) if isinstance(elapsed,(int,float)) else elapsed
        key=(timing_key(cfg),backend)
        if (isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed>0
                and isinstance(sampled_elapsed,(int,float)) and sampled_elapsed>0
                and sampled_cursor>=cfg['nodes'] and row['status'] in ('RUNNING','COMPLETED')):
            rate=sampled_elapsed/sampled_cursor
            rates[key].append(rate)
            local_rates[(key,lanes[index%len(lanes)])].append(rate)
        eta=0. if row['status']=='COMPLETED' else None
        if row['status']=='RUNNING' and rate is not None:eta=rate*(total-cursor)
        entries.append(dict(config=cfg,status=row['status'],cursor=cursor,total=total,
            rounds=row.get('completedRounds',0),elapsed=elapsed,eta=eta,
            gpu=row.get('physicalGpuId',gpus[index%len(gpus)]),lane=lanes[index%len(lanes)],
            mean=row.get('meanAccuracy'),rate=rate,error=row.get('error'),runtimeAdapter=backend,phase=row.get('phase')))
    done=sum(r['status']=='COMPLETED' for r in entries)
    failed=sum(r['status']=='FAILED' for r in entries)
    stopped=any(r['status'] in ('FAILED','INTERRUPTED') for r in entries)
    status='COMPLETED' if done==len(entries) else campaign['status']
    from basil_core.sharp_eta import prior_rate
    remaining={lane:0. for lane in lanes};lower={lane:0. for lane in lanes};upper={lane:0. for lane in lanes}
    unknown=[];history_jobs=0
    for row in entries:
        if row['status']=='COMPLETED':continue
        if continuing and row['status'] in ('FAILED','INTERRUPTED'):
            row['eta']=None;continue
        key=(timing_key(row['config']),row['runtimeAdapter']);samples=local_rates.get((key,row['lane'])) or rates.get(key)
        if row['rate'] is not None:rate=row['rate'];origin='measured';spread=.20
        elif samples:rate=statistics.median(samples);origin='measured';spread=.25
        elif history.get(key):
            rate=statistics.median(history[key]);origin='history';spread=.35;history_jobs+=1
        else:
            # Initial two-worker forecasts include an explicit contention
            # buffer. Live rates replace it; this is not a measured speedup.
            rate=prior_rate(row['config'])*(1+.4*(concurrency-1))
            origin='prior';spread=.5;unknown.append(row['config']['dataset'])
        estimate=rate*(row['total']-row['cursor'])
        row.update(eta=estimate,etaSource=origin)
        remaining[row['lane']]+=estimate;lower[row['lane']]+=estimate*(1-spread);upper[row['lane']]+=estimate*(1+spread)
    eta=0. if status in ('COMPLETED','COMPLETED_WITH_FAILURES') else None
    if status in ('RUNNING','PREPARED','STARTING') and (continuing or not stopped):eta=max(remaining.values())
    return dict(status=status,entries=entries,completed=done,failed=failed,total=len(entries),
        pending=sum(r['status']=='PENDING' for r in entries),
        completedActivations=sum(r['cursor'] for r in entries),totalActivations=sum(r['total'] for r in entries),
        elapsed=campaign.get('elapsedSeconds'),eta=eta,remainingByGpu=remaining,
        untimedJobs=len(unknown),untimedDatasets=sorted(set(unknown)),historyJobs=history_jobs,
        etaLow=max(lower.values()),etaHigh=max(upper.values()),output=str(root),revision=plan.get('revision'),startup=startup)


def render(state,*,width=100,height=30,timezone='America/New_York'):
    width=max(width,30)
    if state['status']=='WAITING_FOR_PLAN':return f'Waiting for frozen plan in {state["output"]}\n'
    methods={r['config'].get('method') for r in state['entries']}
    title='SHARP + CART v2' if 'sharp_cart_v2' in methods else ('SHARP GUARD + CART' if 'sharp_guarded' in methods else 'SHARP + CART')
    if state.get('revision'):title='CART admission revision'
    preparing=state['status'] in ('PREPARED','STARTING')
    timing=(f'Planning duration {duration(state["eta"])} | training has not started' if preparing else
        f'Elapsed {duration(state["elapsed"] or 0)} | Campaign ETA {duration(state["eta"]) if state["eta"] is not None else "not running"}')
    lines=[f'{title} | {state["status"]} | {state["completed"]}/{state["total"]} complete | '
        f'failed {state["failed"]} | pending {state["pending"]}',
        f'Progress {state["completedActivations"]/state["totalActivations"]:.2%} | '
        f'{state["completedActivations"]}/{state["totalActivations"]} activations',
        timing]
    if state.get('startup'):
        startup=state['startup']
        if startup.get('error'):lines.append('Manager error: '+startup['error'])
        elif preparing:lines.append('Waiting for supervisor startup; showing planned GPU assignments below.')
    if state['eta'] is not None and state['status']=='RUNNING':
        finish=(datetime.now(ZoneInfo(timezone))+timedelta(seconds=state['eta'])).strftime('%a %b %d %H:%M %Z')
        lines.append(f'Estimated finish {finish} | planning range {duration(state["etaLow"])} to {duration(state["etaHigh"])}')
    if state['untimedJobs'] and state['status'] not in ('COMPLETED','COMPLETED_WITH_FAILURES','FAILED','INTERRUPTED'):
        lines.append(f'ETA uses workload priors for {state["untimedJobs"]} jobs; replaces them with measurements as runs progress.')
    elif state.get('historyJobs',0):lines.append(f'ETA uses matching prior campaign timings for {state["historyJobs"]} jobs.')
    if any(r['status']=='RUNNING' and r['cursor']==r['total'] for r in state['entries']):
        lines.append('Training activations finished for a worker; final evaluation/checkpoint writes can still be running.')
    wrap=lambda values:[part for line in values for part in (textwrap.wrap(line,width,break_long_words=True) or [''])]
    lines=wrap(lines);lines.append('')
    active=[r for r in state['entries'] if r['status'] in ('RUNNING','STARTING')]+[
        r for r in state['entries'] if r['status'] in ('FAILED','INTERRUPTED')]
    recent=[r for r in state['entries'] if r['status']=='COMPLETED'][-4:]
    pending=[r for r in state['entries'] if r['status']=='PENDING']
    visible=active+recent if active or recent else pending;shown=0
    for row in visible:
        cfg=row['config'];gpu=f'G{row["gpu"]}' if row['gpu'] is not None else 'CPU'
        mean=f'{row["mean"]:.2%}' if row['mean'] is not None else '--'
        block=wrap([f'{gpu} {row["status"]} {cfg["dataset"]}/{cfg["partition"]} S{cfg["seed"]} sigma={cfg["sigma"]:g}',
            f'{cfg["condition"]} | R{row["rounds"]}/{cfg["rounds"]} A{row["cursor"]}/{row["total"]}',
            f'mean {mean} | elapsed {duration(row["elapsed"] or 0)} | ETA {duration(row["eta"])}'])
        if len(lines)+len(block)+4>height:break
        lines.extend(block);shown+=1
    if shown<len(active):lines.append(f'Showing {shown}/{len(active)} active/stopped jobs; enlarge terminal for more.')
    elif not active and not recent and shown<len(pending):lines.append(f'Showing {shown}/{len(pending)} planned jobs; enlarge terminal for more.')
    if state['status']=='COMPLETED':lines.append('ALL EXPERIMENTS COMPLETE.')
    elif state['status']=='COMPLETED_WITH_FAILURES':lines.append('Queue finished with failures; completed results are preserved. Retry failed runs for final figures.')
    elif state['status'] in ('FAILED','INTERRUPTED'):lines.append('Campaign stopped; inspect worker logs before recovery.')
    else:
        lines.append('ETA is a forecast; setup/JIT and workload variation can change it.')
        lines.append('Refresh 5 seconds | Ctrl+C closes only this monitor')
    return '\n'.join(wrap(lines))+'\n'


def monitor(directory,*,once=False,timing_roots=(),timezone='America/New_York'):
    try:
        while True:
            state=snapshot(directory,timing_roots=timing_roots);size=shutil.get_terminal_size(fallback=(100,30))
            if not once and sys.stdout.isatty():print('\033[2J\033[H',end='')
            print(render(state,width=size.columns,height=size.lines,timezone=timezone),end='',flush=True)
            if once or state['status'] in ('COMPLETED','COMPLETED_WITH_FAILURES','FAILED','INTERRUPTED','COMPLETED_REPORT_FAILED'):return
            time.sleep(5)
    except KeyboardInterrupt:print('\nMonitor closed.',flush=True)
