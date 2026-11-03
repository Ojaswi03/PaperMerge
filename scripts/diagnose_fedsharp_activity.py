#!/usr/bin/env python3
"""Read-only fleet lease/progress diagnosis; never attach to or signal workers."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core.cart_campaign_io import read_json
from basil_core import live_gpu_scheduler as live


def counters(identity):
    if not live.same_process(identity):return None
    proc=Path('/proc')/str(identity['pid'])
    try:
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        try:io={k:int(v) for k,v in (line.split(':',1) for line in (proc/'io').read_text().splitlines())}
        except OSError:io={}
        return dict(cpuTicks=int(stat[11])+int(stat[12]),readBytes=io.get('read_bytes'),writeBytes=io.get('write_bytes'))
    except (OSError,ValueError,IndexError):return None


def lease_fds(identity,path):
    """Check kernel fdinfo for this exact live identity and GPU-lock inode."""
    if not live.same_process(identity):return dict(status='process exited or identity changed',descriptors=[])
    result=[]
    try:
        expected=path.stat();proc=Path('/proc')/str(identity['pid'])
        for fd in (proc/'fd').iterdir():
            try:
                actual=fd.stat()
                if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):continue
                info=(proc/'fdinfo'/fd.name).read_text()
                if any(line.startswith('lock:') and 'FLOCK' in line and 'WRITE' in line for line in info.splitlines()):
                    result.append(int(fd.name))
            except OSError:continue
    except OSError as error:return dict(status=type(error).__name__,descriptors=[])
    if not live.same_process(identity):return dict(status='process exited or identity changed',descriptors=[])
    return dict(status='confirmed holder' if result else 'no matching held lease found',descriptors=result)


def reports(queue):
    current=read_json(queue/'fleet/state.json');values={}
    for host in current['hosts']:
        path=queue/'fleet/reports'/f'{host}.json'
        if path.exists():values[host]=read_json(path)
    return current,values


def inspect(queue,seconds=10):
    queue=Path(queue);current,first=reports(queue);local=socket.gethostname().split('.')[0]
    identities={run:entry['process'] for run,entry in current['runs'].items()
        if entry['owner']==local and entry['state']=='RUNNING' and entry.get('process',{}).get('start')}
    initial={run:counters(identity) for run,identity in identities.items()}
    started=time.monotonic()
    if seconds:time.sleep(seconds)
    elapsed=max(time.monotonic()-started,1e-9);current,last=reports(queue);result={}
    for host,report in last.items():
        old=first.get(host,{});old_rows=dict(zip(old.get('runIds',[]),old.get('snapshot',{}).get('runs',[])))
        rows=dict(zip(report['runIds'],report['snapshot']['runs']));gained=0
        for run,row in rows.items():
            if run in old_rows:gained+=max(0,row['completedActivations']-old_rows[run]['completedActivations'])
        details=[];profiles={}
        if host==local:
            plan=read_json(Path(current['hosts'][host]['fleet'])/'plan.json')
            profiles={p['gpu']:p for p in plan['profiles']}
        for run,entry in current['runs'].items():
            if entry['owner']!=host or entry['state']!='RUNNING':continue
            row=rows.get(run,{})
            legacy=entry['location']['output']==current['hosts'][host]['original']
            item=dict(runId=run,pid=entry.get('process',{}).get('pid'),gpu=entry['boundGpu'],
                legacyWorker=legacy,condition=row.get('condition'),round=row.get('completedRounds'),
                etaSeconds=row.get('etaSeconds'),phase=row.get('phase'),
                lease=dict(status='identity unavailable' if host==local else 'remote host, not inspected'))
            identity=identities.get(run)
            if host==local and identity:
                profile=profiles[entry['boundGpu']]
                path=Path('/tmp')/f'papermerge-{os.getuid()}-{profile["gpu_uuid"]}.lock'
                item['lease']=lease_fds(identity,path)
                before=initial.get(run);after=counters(identity)
                if before and after and seconds:
                    item['cpuPercent']=100*max(0,after['cpuTicks']-before['cpuTicks'])/os.sysconf('SC_CLK_TCK')/elapsed
                    if all(values.get(key) is not None for values in (before,after) for key in ('readBytes','writeBytes')):
                        item.update(readMiBPerSecond=max(0,after['readBytes']-before['readBytes'])/2**20/elapsed,
                            writeMiBPerSecond=max(0,after['writeBytes']-before['writeBytes'])/2**20/elapsed)
            details.append(item)
        result[host]=dict(reportAgeSeconds=max(0,time.time()-report['updatedAt']),
            schedulerBlocked=report.get('blocked',{}),reportedActivationsPerSecond=gained/elapsed if seconds else None,
            workers=details)
    return dict(localHost=local,sampleSeconds=elapsed,trainingModified=False,workerSignalsSent=False,hosts=result)


def render(result):
    from basil_core.sharp_monitor import duration
    lines=['Read-only activity check; workers were not signalled.','Reported activation rates reflect batched status updates.']
    for host,report in result['hosts'].items():
        rate=report['reportedActivationsPerSecond']
        lines.append(f"\n{host}: report age {report['reportAgeSeconds']:.1f}s; sample speed {f'{rate:.2f} activations/s' if rate is not None else 'not sampled'}")
        for gpu,decision in report['schedulerBlocked'].items():
            lines.append(f"  GPU {gpu}: {decision.get('reason',str(decision)) if isinstance(decision,dict) else decision}")
        for worker in sorted(report['workers'],key=lambda w:(w['gpu'],w['pid'] or 0)):
            cpu=f"{worker['cpuPercent']:.1f}%" if 'cpuPercent' in worker else 'remote/not sampled'
            kind='adopted' if worker['legacyWorker'] else 'fleet'
            lines.append(f"  GPU {worker['gpu']} PID {worker['pid']} {kind} round {worker['round']} ETA {duration(worker['etaSeconds'])} CPU {cpu} | {worker['condition']}")
            lines.append('    GPU lease: '+worker['lease']['status'])
            if 'readMiBPerSecond' in worker:
                lines.append(f"    Disk I/O: read {worker['readMiBPerSecond']:.2f} MiB/s; write {worker['writeMiBPerSecond']:.2f} MiB/s")
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    if not 0<=args.seconds<=30:parser.error('--seconds must be between 0 and 30.')
    result=inspect(args.queue.resolve(),args.seconds)
    print(json.dumps(result,indent=2) if args.json else render(result),flush=True)


if __name__=='__main__':main()
