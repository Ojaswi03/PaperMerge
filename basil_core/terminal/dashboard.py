"""Read-only SSH dashboard; refreshing never controls the training process."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import time

from basil_core.execution_profile import gpu_inventory, gpu_processes
from basil_core.terminal.state import inspect_state, safe_text

DEFAULT_INTERVAL=5.
TERMINAL_STATES={'COMPLETED','FAILED','STOPPED','BLOCKED'}


def system_info():
    result=dict(host=socket.gethostname(),cpu_percent=None,ram_used_mib=None,ram_total_mib=None,gpus=[],gpu_processes=None)
    try:
        import psutil
        value=psutil.cpu_percent(interval=None)
        if getattr(system_info,'cpu_ready',False):result['cpu_percent']=value
        system_info.cpu_ready=True
        memory=psutil.virtual_memory()
        result.update(ram_used_mib=(memory.total-memory.available)/1024**2,ram_total_mib=memory.total/1024**2)
    except ImportError:
        try:
            values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
            result.update(ram_used_mib=(values['MemTotal']-values['MemAvailable'])/1024,ram_total_mib=values['MemTotal']/1024)
        except (OSError,ValueError,KeyError):pass
    try:result['gpus']=gpu_inventory();result['gpu_processes']=gpu_processes()
    except RuntimeError as error:result['gpu_error']=str(error)
    return result


def duration(value):
    if value is None:return 'unavailable'
    hours,remainder=divmod(int(max(0,value)),3600);minutes,seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def percent(value):
    return 'unavailable' if value is None else f'{100*value:.2f}%'


def bar(progress):
    width=32;filled=int(max(0,min(1,progress))*width)
    return '['+'#'*filled+'-'*(width-filled)+']'


def completion_message(snapshot):
    status=snapshot.get('status','IDLE')
    entries=snapshot.get('entries',[])
    completed=sum(e['status']=='COMPLETED' for e in entries)
    if status=='COMPLETED':
        if not entries:return 'QUEUE COMPLETED (experiment details unavailable)'
        if completed!=len(entries):return 'QUEUE COMPLETED (entry status mismatch; inspect saved state)'
        return f'ALL EXPERIMENTS COMPLETE ({completed}/{len(entries)})'
    return f'QUEUE {status}: {completed}/{len(entries)} experiments completed'


def render(snapshot, system, interval=DEFAULT_INTERVAL, now=None):
    now=time.time() if now is None else now
    active=[e for e in snapshot.get('entries',[]) if e['status'] in ('RUNNING','STOPPING')]
    totals=snapshot.get('totals',{})
    lines=['PAPERMERGE - TERMINAL EXECUTION MONITOR',
        f"Host: {system.get('host',socket.gethostname())}  Profile: {snapshot.get('profile','unavailable')}",
        'Selected physical GPUs: '+', '.join(str(p['gpu']) for p in snapshot.get('resources',[])),
        '',f"QUEUE: {Path(snapshot.get('queue','unavailable')).name}  State: {snapshot.get('status','IDLE')}",
        f"Experiments: {totals.get('completed',0)}/{totals.get('total',0)} completed; "
        f"{totals.get('pending',0)} pending; {totals.get('failed',0)} failed; {totals.get('blocked',0)} blocked",
        f"Overall {percent(totals.get('progress'))}  Elapsed {duration(snapshot.get('elapsed_seconds'))}  "
        f"ETA {duration(snapshot.get('eta_seconds'))}"]
    if snapshot.get('status') in TERMINAL_STATES:lines.insert(1,completion_message(snapshot))
    for entry in active:
        metric=entry.get('metrics',{})
        lines+=['',f"EXPERIMENT: {entry['name']}  Condition: {entry.get('condition','unavailable')}",
            f"{entry.get('dataset')} / {'non-IID' if entry.get('nonIID') else 'IID'} / {entry['nodes']} nodes / {entry['device']}",
            f"Rounds completed: {entry['completedRounds']}/{entry['rounds']}  Activations: {entry['completedActivations']}/{entry['totalActivations']}",
            'Optimizer batches: '+(f"{entry['completedOptimizerSteps']}/{entry['totalOptimizerSteps']}" if entry.get('totalOptimizerSteps') is not None else 'unavailable'),
            f"{bar(entry['progress'])} {percent(entry['progress'])}",
            f"Elapsed {duration(entry.get('elapsed_seconds'))}  ETA {duration(entry.get('eta_seconds'))}",
            'Checkpoint: '+(f"activation {entry['checkpoint']['activation']} (committed)" if entry.get('checkpoint') else 'not yet reported'),
            f"Latest node accuracy: {percent(metric.get('latest_node_accuracy'))}  Mean: {percent(metric.get('mean_node_accuracy'))}  Worst: {percent(metric.get('worst_node_accuracy'))}",
            f"Local training loss: {metric.get('local_training_loss','unavailable')}  Node-9 test CE: {metric.get('full_test_loss','unavailable')}",
            f"Tracking: {entry.get('tracking',{}).get('status','unavailable')} "
            f"{entry.get('tracking',{}).get('mode','')} {entry.get('tracking',{}).get('runId','')}"]
    lines+=['','SYSTEM',f"CPU: {system.get('cpu_percent') if system.get('cpu_percent') is not None else 'unavailable'}%  "
            f"RAM MiB used/total: {system.get('ram_used_mib') if system.get('ram_used_mib') is not None else 'unavailable'}/{system.get('ram_total_mib') if system.get('ram_total_mib') is not None else 'unavailable'}"]
    selected={p['gpu'] for p in snapshot.get('resources',[])}
    for gpu in system.get('gpus',[]):
        if selected and gpu['index'] not in selected:continue
        lines.append(f"GPU {gpu['index']} {gpu['name']}: {gpu['memory.used']}/{gpu['memory.total']} MiB used, "
            f"{gpu['memory.free']} MiB free; utilization {gpu.get('utilization.gpu','unavailable')}%; "
            f"temperature {gpu.get('temperature.gpu','unavailable')} C")
    if system.get('gpu_error'):lines.append(system['gpu_error'])
    if system.get('gpu_processes') is None:lines.append('GPU processes: unavailable')
    else:
        for proc in system['gpu_processes'][:8]:
            lines.append(f"GPU process {proc['pid']}: {proc['process']} ({proc['memory_mib']} MiB, {proc['gpu_uuid']})")
        if not system['gpu_processes']:lines.append('GPU compute processes: none reported')
    lines+=['','QUEUE ITEMS']
    for entry in snapshot.get('entries',[]):
        lines.append(f"{entry['condition'] or entry['name']}  {entry['status']:10}  {percent(entry.get('progress'))}  "
                     f"{entry['completedRounds']}/{entry['rounds']} rounds")
        if entry.get('error'):lines.append('  '+entry['error'].splitlines()[-1])
        elif entry.get('problems'):lines.append('  '+entry['problems'][0])
        if entry['status']=='COMPLETED':
            metric=entry.get('metrics',{})
            lines.append(f"  Mean: {percent(metric.get('mean_node_accuracy'))}  "
                         f"Worst: {percent(metric.get('worst_node_accuracy'))}  "
                         f"Latest node: {percent(metric.get('latest_node_accuracy'))}")
    lines+=['','RECENT ACTIVITY']
    lines.extend(f"{item['time']} - {item['message']}" for item in snapshot.get('activity',[])[-6:])
    if snapshot.get('error'):lines.append('Error: '+snapshot['error'])
    age=max(0.,now-snapshot['last_event_wall']) if snapshot.get('last_event_wall') else None
    lines.append(f"Refresh: {interval:g} seconds; last event age: {duration(age)}; counters change only on worker events.")
    return safe_text('\n'.join(lines))


def screen_size(stream):
    try:return os.get_terminal_size(stream.fileno())
    except (AttributeError,OSError,ValueError):return shutil.get_terminal_size((100,30))


def render_screen(snapshot,system,interval,columns,rows):
    """Fit one current frame; a compact queue table replaces tall detail blocks."""
    width=max(1,columns-1);height=max(1,rows-1)
    full=render(snapshot,system,interval).splitlines()
    if len(full)<=height and all(len(line)<=width for line in full):return '\n'.join(full)
    entries=snapshot.get('entries',[]);totals=snapshot.get('totals',{})
    gpus=','.join(str(p['gpu']) for p in snapshot.get('resources',[]))
    lines=[f"PAPERMERGE | {snapshot.get('status','IDLE')} | {Path(snapshot.get('queue','unavailable')).name}",
        f"Host {snapshot.get('host',system.get('host','unavailable'))} | {snapshot.get('profile','unavailable')} | GPUs {gpus}",
        f"Queue {percent(totals.get('progress'))} | done {totals.get('completed',0)}/{totals.get('total',len(entries))} "
        f"pending {totals.get('pending',0)} failed {totals.get('failed',0)} blocked {totals.get('blocked',0)}",
        f"Elapsed {duration(snapshot.get('elapsed_seconds'))} | ETA {duration(snapshot.get('eta_seconds'))}"]
    if snapshot.get('status') in TERMINAL_STATES:lines[0]=completion_message(snapshot)
    for entry in entries:
        label=f"{entry.get('condition') or entry['name']} {entry['rounds']}r"
        checkpoint=entry.get('checkpoint')
        cp=checkpoint.get('activation','?') if checkpoint else '--'
        lines.append(f"{label} G{entry.get('gpu','?')} {entry['status']} {percent(entry.get('progress'))} "
            f"R{entry['completedRounds']}/{entry['rounds']} A{entry['completedActivations']}/{entry['totalActivations']} "
            f"U{entry.get('completedOptimizerSteps','?')}/{entry.get('totalOptimizerSteps','?')} CP{cp} "
            f"t{duration(entry.get('elapsed_seconds'))}")
        if entry['status'] in ('RUNNING','STOPPING','COMPLETED'):
            metric=entry.get('metrics',{})
            def loss(key):
                value=metric.get(key)
                return 'unavailable' if value is None else f'{value:.4g}'
            lines.append(f" ETA {duration(entry.get('eta_seconds'))} mean {percent(metric.get('mean_node_accuracy'))} "
                f"worst {percent(metric.get('worst_node_accuracy'))} node {percent(metric.get('latest_node_accuracy'))} "
                f"trainCE {loss('local_training_loss')} testCE {loss('full_test_loss')}")
        if entry.get('error'):lines.append('ERROR: '+entry['error'].splitlines()[-1])
    cpu=system.get('cpu_percent');used=system.get('ram_used_mib');total=system.get('ram_total_mib')
    ram='unavailable' if used is None or total is None else f'{used/1024:.1f}/{total/1024:.1f} GiB'
    lines.append(f"CPU {'unavailable' if cpu is None else str(cpu)+'%'} | RAM {ram}")
    selected={p['gpu'] for p in snapshot.get('resources',[])}
    for gpu in system.get('gpus',[]):
        if selected and gpu['index'] not in selected:continue
        processes=system.get('gpu_processes') or []
        pids=','.join(str(p['pid']) for p in processes if p['gpu_uuid']==gpu['uuid']) or '--'
        lines.append(f"GPU{gpu['index']} {gpu['memory.used']}/{gpu['memory.total']}MiB "
            f"free {gpu['memory.free']} util {gpu.get('utilization.gpu','?')}% "
            f"temp {gpu.get('temperature.gpu','?')}C PID {pids}")
    if system.get('gpu_error'):lines.append(system['gpu_error'])
    if snapshot.get('activity'):
        item=snapshot['activity'][-1];message=item['message']
        for entry in entries:message=message.replace(entry['name'],str(entry.get('condition') or entry['name']))
        lines.append(f"Last {item['time'][11:19]}: {message}")
    if snapshot.get('error'):lines.append('ERROR: '+snapshot['error'])
    age=max(0.,time.time()-snapshot['last_event_wall']) if snapshot.get('last_event_wall') else None
    footer=f'Refresh {interval:g}s | event age {duration(age)} | CP=committed activation | Ctrl+C exits monitor'
    if len(lines)>=height:lines=lines[:max(0,height-2)]+['Compact view; use status for full details.']
    lines.append(footer)
    def clip(line):
        line=safe_text(line).replace('\n',' ').replace('\t',' ')
        return line if len(line)<=width else line[:max(0,width-3)]+'.'*min(3,width)
    return '\n'.join(clip(line) for line in lines[:height])


def monitor(directory, interval=DEFAULT_INTERVAL, *, once=False, plain=False, screen=False, stream=None, wait=time.sleep,
            clock=time.monotonic, fetch=inspect_state, resources=system_info):
    if not math.isfinite(interval) or interval<=0:raise ValueError('Refresh interval must be finite and positive.')
    if plain and screen:raise ValueError('Choose plain output or screen rendering, not both.')
    stream=stream or sys.stdout
    interactive=(screen or stream.isatty() and os.environ.get('TERM')!='dumb') and not plain
    pool=ThreadPoolExecutor(max_workers=1) if resources is system_info and not once else None
    pending=None;latest=dict(host=socket.gethostname(),gpu_error='Resource discovery pending')
    final=None;closed=False
    try:
        if interactive:stream.write('\x1b[?1049h\x1b[?25l');stream.flush()
        while True:
            start=clock()
            snapshot=fetch(directory)
            terminal=snapshot.get('status') in TERMINAL_STATES
            if terminal:
                if pending is not None and pending.done():latest=pending.result();pending=None
            elif pool:
                if pending is not None and pending.done():latest=pending.result();pending=None
                if pending is None:pending=pool.submit(resources)
            else:latest=resources()
            if interactive:
                size=screen_size(stream)
                text=render_screen(snapshot,latest,interval,size.columns,size.lines)
                stream.write('\x1b[H\x1b[2J'+text)
            else:stream.write(render(snapshot,latest,interval)+'\n')
            stream.flush()
            if terminal:final=snapshot;return snapshot['status']
            if once:final=snapshot;return
            wait(max(0.,interval-(clock()-start)))
    except KeyboardInterrupt:
        closed=True
    finally:
        if pool:pool.shutdown(wait=False,cancel_futures=True)
        if interactive:stream.write('\x1b[?25h\x1b[?1049l')
        if final is not None and interactive:stream.write(render(final,latest,interval)+'\n')
        if closed:stream.write('\nMonitor closed; queue execution is independent.\n')
        stream.flush()


def follow_logs(directory, *, follow=False, lines=30, stream=None, wait=time.sleep, path=None):
    if lines<1:raise ValueError('Log line count must be positive.')
    from collections import deque
    stream=stream or sys.stdout;path=Path(path) if path is not None else Path(directory)/'queue.log'
    if not path.exists():raise ValueError('No execution log exists in this state directory.')
    source=path.open()
    try:
        stream.writelines(deque(source,maxlen=lines));stream.flush()
        inode=os.fstat(source.fileno()).st_ino
        while follow:
            text=source.read()
            if text:stream.write(text);stream.flush()
            try:
                if path.stat().st_ino!=inode:
                    source.close();source=path.open();inode=os.fstat(source.fileno()).st_ino
            except FileNotFoundError:pass
            wait(.2)
    except KeyboardInterrupt:pass
    finally:source.close()
