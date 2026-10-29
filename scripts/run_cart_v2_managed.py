#!/usr/bin/env python3
"""Run/watch CART with bounded GPU slots, ETA, automatic charts and ntfy alerts.

The training source is unchanged. This supervisor owns only scheduling and
postprocessing; no notification secret is stored in experiment manifests.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def export_figures(root,destination):
    plan=json.loads((root/'plan.json').read_text())
    if plan.get('revision'):
        from reporting import cart_revision as charts
    else:
        from reporting import cart_v2_campaign as charts
    manifest=destination/'artifacts_manifest.json'
    if not manifest.exists():
        return charts.generate(root,destination,reference=plan.get('reference')) if plan.get('revision') else charts.generate(root,destination)
    saved=json.loads(manifest.read_text())
    if saved['reportingProgramSha256']!=hashlib.sha256(Path(charts.__file__).read_bytes()).hexdigest():
        raise ValueError('Reporting code changed; use --figures with a fresh directory.')
    for name,value in saved['outputSha256'].items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=value:raise ValueError('Saved figure artifact changed.')
    for run,value in saved['inputMetricSha256'].items():
        path=Path(run) if Path(run).is_absolute() else root/run/'metrics.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=value:raise ValueError('Input metrics changed after plotting.')
    return saved


def run(args,forwarded):
    from basil_core.ntfy_notifications import publish
    from basil_core.cart_notification_text import event as notice
    from basil_core.sharp_monitor import snapshot,duration
    from basil_core.cart_campaign_io import write_json,read_json,retry_io,transient_io
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    figures=args.figures.resolve() if args.figures else root/'figures'
    status=dict(status='STARTING',ownerPid=__import__('os').getpid(),host=socket.gethostname(),
        supervisorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),notifications=[])
    process=None;stopped=False;started=time.monotonic();notified_failures=set()
    def save():
        from datetime import datetime,timezone
        status.update(wallSeconds=time.monotonic()-started,updatedUtc=datetime.now(timezone.utc).isoformat())
        write_json(root/'managed_status.json',status)
    def notify(title,message):
        if not args.ntfy_topic:return
        receipt=publish(args.ntfy_topic,title,message,server=args.ntfy_server,token_file=args.ntfy_token_file)
        status['notifications'].append(dict(title=title,**receipt));save()
        print('ntfy: '+('accepted by server' if receipt['deliveredToServer'] else 'DELIVERY FAILED; see managed_status.json'),flush=True)
    def stop(*_):
        nonlocal stopped
        stopped=True
        if process is not None and process.poll() is None:process.send_signal(signal.SIGINT)
    with (root/'managed.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        save()
        try:
            if not args.watch and not args.postprocess_only:
                launcher={'standard':'run_cart_v2_parallel.py','scaled':'run_cart_v2_scaled.py',
                    'resilient':'run_cart_v2_resilient.py','transfer':'run_cart_v2_transfer.py','admission':'run_cart_revision_study.py'}[getattr(args,'scheduler','resilient')]
                command=[sys.executable,str(ROOT/'scripts'/launcher),'--execute','--output',str(root),*forwarded]
                with (root/'supervisor.log').open('a') as log:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    status['supervisorPid']=process.pid;save()
                    while process.poll() is None:
                        try:
                            if (root/'plan.json').exists():
                                live=retry_io(snapshot,root,timing_roots=args.timing_from)
                                stage='STARTING' if live['status'] in ('PREPARED','STARTING') else 'RUNNING'
                                status.update(status=stage,etaSeconds=live['eta'],etaLow=live['etaLow'],etaHigh=live['etaHigh'],
                                    completed=live['completed'],total=live['total']);save()
                                timing='startup planning duration' if stage=='STARTING' else 'training ETA'
                                print(f'{stage}: {live["completed"]}/{live["total"]} done | {timing} {duration(live["eta"])}',flush=True)
                                try:current=read_json(root/'campaign.json')
                                except FileNotFoundError:current={}
                                for failure in current.get('failures',[]):
                                    identity=failure['runId']
                                    if identity in notified_failures:continue
                                    notify(*notice('worker_failed',root))
                                    notified_failures.add(identity)
                        except OSError as error:
                            if not transient_io(error):raise
                            # Telemetry is not a training dependency. A repeated
                            # monitoring read failure must not cancel healthy peers.
                            print(f'Monitor filesystem error; training continues: {error}',file=sys.stderr,flush=True)
                        time.sleep(5)
                if stopped:raise InterruptedError('Managed campaign stopped; committed checkpoints preserved.')
                outcome=read_json(root/'campaign.json') if (root/'campaign.json').exists() else {}
                if process.returncode==3 and outcome.get('status')=='COMPLETED_REPORT_FAILED':
                    status.update(status='COMPLETED_REPORT_FAILED',trainingComplete=True);save()
                    notify(*notice('report_failed',root));return 3
                if process.returncode and not (process.returncode==2 and outcome.get('status')=='COMPLETED_WITH_FAILURES'):
                    raise RuntimeError('Training supervisor failed; inspect supervisor.log and worker logs.')
            elif args.watch:
                while True:
                    if stopped:raise InterruptedError('Notification watcher stopped; training is independent.')
                    live=retry_io(snapshot,root,timing_roots=args.timing_from)
                    if live['status'] in ('COMPLETED','COMPLETED_WITH_FAILURES','FAILED','INTERRUPTED'):break
                    status.update(status='WATCHING',etaSeconds=live.get('eta'));save();time.sleep(5)
            campaign=read_json(root/'campaign.json')
            if campaign['status']=='COMPLETED_WITH_FAILURES':
                status.update(status='COMPLETED_WITH_FAILURES',completed=campaign['completed'],
                    total=campaign['total'],failed=campaign['failed'],failures=campaign['failures'],
                    figuresDeferred='Final figures require complete matched three-seed groups; retry failed runs first.')
                save()
                notify(*notice('partial',root,completed=campaign['completed'],total=campaign['total'],failed=campaign['failed']))
                print(f'COMPLETED_WITH_FAILURES: {campaign["completed"]}/{campaign["total"]}; failed {campaign["failed"]}.',flush=True)
                return 2
            if campaign['status']!='COMPLETED':raise RuntimeError('Campaign is not complete: '+campaign['status'])
            status['status']='GENERATING_FIGURES';save()
            manifest=export_figures(root,figures)
            status.update(status='COMPLETED',figures=str(figures),artifacts=len(manifest['files']));save()
            plan=json.loads((root/'plan.json').read_text());count=len(plan['configs'])
            first=next(iter(plan['configs']),{})
            notify(*notice('complete',root,completed=count,total=count,nodes=first.get('nodes'),rounds=first.get('rounds'),elapsed=duration(status['wallSeconds'])))
            print(f'COMPLETE: figures and averages in {figures}',flush=True)
            return 0
        except BaseException as error:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT);process.wait()
            status.update(status='INTERRUPTED' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'FAILED',
                error=str(error));save()
            notify(*notice('interrupted' if status['status']=='INTERRUPTED' else 'failed',root))
            print(status['status']+': '+str(error),file=sys.stderr,flush=True)
            return 130 if status['status']=='INTERRUPTED' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--figures',type=Path)
    p.add_argument('--scheduler',choices=('standard','scaled','resilient','transfer','admission'),default='resilient')
    p.add_argument('--ntfy-topic');p.add_argument('--ntfy-server',default='https://ntfy.sh')
    p.add_argument('--ntfy-token-file',type=Path);p.add_argument('--timing-from',nargs='*',type=Path,default=[])
    p.add_argument('--tmux');action=p.add_mutually_exclusive_group()
    action.add_argument('--execute',action='store_true');action.add_argument('--watch',action='store_true')
    action.add_argument('--postprocess-only',action='store_true');action.add_argument('--test-notification',action='store_true')
    args,forwarded=p.parse_known_args()
    if args.ntfy_topic:
        from basil_core.ntfy_notifications import validate_destination
        validate_destination(args.ntfy_topic,args.ntfy_server,args.ntfy_token_file)
    if args.test_notification:
        if not args.ntfy_topic:p.error('--ntfy-topic is required for the test.')
        from basil_core.ntfy_notifications import publish
        from basil_core.cart_notification_text import event as notice
        title,message=notice('test','CART')
        receipt=publish(args.ntfy_topic,title,message,
            server=args.ntfy_server,token_file=args.ntfy_token_file)
        print(json.dumps(receipt));return 0 if receipt['deliveredToServer'] else 2
    if not args.output:p.error('--output is required.')
    if args.tmux:
        if subprocess.run(['tmux','has-session','-t',args.tmux],capture_output=True).returncode==0:p.error('tmux session already exists.')
        argv=list(sys.argv[1:]);at=argv.index('--tmux');del argv[at:at+2]
        command=shlex.join([sys.executable,str(Path(__file__).resolve()),*argv])
        subprocess.run(['tmux','new-session','-d','-s',args.tmux,command],check=True)
        print(f'Started {args.tmux}; training, figures and notifications continue after SSH disconnect.')
        return 0
    return run(args,forwarded)


if __name__=='__main__':raise SystemExit(main())
