#!/usr/bin/env python3
"""Publish sanitized campaign progress to the read-only Chrome dashboard.

No training controls, checkpoints, paths, usernames or logs are uploaded.
Temporary update credentials stay in memory; persistent keys use a private file.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SITE='https://papermerge-cart-live.foggycoati69.chatgpt.site'
VIEW_SITE='https://ojaswi03.github.io/PaperMerge/'


def view_url(saved):
    base=VIEW_SITE if saved['site']==SITE else saved['site']+'/'
    return base+'?feed='+saved['feedId']


def origin(value):
    url=urlsplit(value)
    if url.scheme!='https' or not url.netloc or url.username or url.password or url.query or url.fragment or url.path not in ('','/'):
        raise ValueError('Use the HTTPS dashboard origin without a path or credentials.')
    return value.rstrip('/')


def request_json(url,method,payload=None,token=None,lease_seconds=None):
    headers={'Content-Type':'application/json','User-Agent':'PaperMerge-Monitor/1.0'}
    if token:headers['Authorization']='Bearer '+token
    if lease_seconds is not None:headers['X-Monitor-Lease-Seconds']=str(lease_seconds)
    raw=json.dumps(payload,allow_nan=False).encode() if payload is not None else None
    with urlopen(Request(url,data=raw,headers=headers,method=method),timeout=15) as response:return json.load(response)


def new_connection(root,site,temporary=False):
    from basil_core.cart_public_monitor import public_snapshot
    result=request_json(site+'/api/feeds','POST',public_snapshot(root),lease_seconds=45 if temporary else None)
    import re
    if not re.fullmatch(r'[a-f0-9-]{36}',result.get('feedId','')) or not re.fullmatch(r'[a-f0-9]{64}',result.get('writeToken','')):
        raise ValueError('Unexpected dashboard setup response.')
    result=dict(site=site,feedId=result['feedId'],writeToken=result['writeToken'])
    result['viewUrl']=view_url(result)
    return result


def connection(root,site):
    from basil_core.cart_public_monitor import public_snapshot
    path=root/'browser_share.json'
    with (root/'browser_share_setup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['site']!=site:raise ValueError('This campaign already has a different dashboard connection.')
            saved['viewUrl']=view_url(saved)
            os.chmod(path,0o600);return saved
        result=new_connection(root,site)
        descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as stream:json.dump(result,stream)
        return result


def publish_loop(args,saved):
    from basil_core.cart_public_monitor import public_snapshot
    from basil_core.stable_monitor import publisher_lock,selected_run
    temporary=getattr(args,'temporary',False)
    stable=getattr(args,'stable',False)
    with publisher_lock(args):
        notified=False;last_terminal=None;last_selection=None
        while True:
            tick=time.monotonic()
            try:
                root=args.output
                if stable:
                    selected=selected_run(args.state_dir)
                    if selected is None:return
                    root=Path(selected['output'])
                    if selected['selection']!=last_selection:
                        notified=False;last_terminal=None;last_selection=selected['selection']
                data=public_snapshot(root)
                terminal=data['status'] in ('COMPLETED','COMPLETED_WITH_FAILURES','COMPLETED_REPORT_FAILED','FAILED','INTERRUPTED')
                if stable and terminal and last_terminal==data['status'] and not args.once:
                    time.sleep(max(0.,args.interval-(time.monotonic()-tick)));continue
                request_json(saved['site']+'/api/feeds/'+saved['feedId'],'PUT',data,saved['writeToken'])
                if args.ntfy_topic and not notified:
                    from basil_core.ntfy_notifications import publish
                    receipt=publish(args.ntfy_topic,'CART browser monitor is live','Share this read-only link with your professor:\n\n'+saved['viewUrl']+'\n\nUpdates automatically. Training is unchanged.')
                    notified=receipt['deliveredToServer']
                last_terminal=data['status'] if terminal else None
                if args.once or (not stable and not temporary and terminal):return
            except (OSError,ValueError) as error:
                if temporary and getattr(error,'code',None) in (404,410):
                    print('Temporary sharing session has ended; training is unaffected.',flush=True);return
                print('Browser update delayed ('+type(error).__name__+'); training is unaffected.',file=sys.stderr,flush=True)
                if args.once:raise
            time.sleep(max(0.,args.interval-(time.monotonic()-tick)))


def temporary_session(args):
    saved=new_connection(args.output,args.site,temporary=True)
    previous={}
    def interrupt(*_):raise KeyboardInterrupt
    try:
        for name in ('SIGTERM','SIGHUP'):
            if hasattr(signal,name):
                sig=getattr(signal,name);previous[sig]=signal.signal(sig,interrupt)
        print('Temporary monitor link:\n'+saved['viewUrl']+'\nKeep this command running. Ctrl+C disables this link; training continues.',flush=True)
        publish_loop(args,saved)
    finally:
        try:
            request_json(saved['site']+'/api/feeds/'+saved['feedId'],'DELETE',token=saved['writeToken'])
            print('Temporary monitor link disabled.',flush=True)
        except (OSError,ValueError):
            print('Immediate removal could not be confirmed. The temporary feed expires within 45 seconds of its last update.',file=sys.stderr,flush=True)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    p.add_argument('--site',default=SITE);p.add_argument('--interval',type=int,default=1)
    p.add_argument('--once',action='store_true');p.add_argument('--print-link',action='store_true')
    p.add_argument('--tmux');p.add_argument('--ntfy-topic')
    from basil_core import stable_monitor
    p.add_argument('--state-dir',type=Path,default=stable_monitor.STATE_ROOT)
    p.add_argument('--background',action='store_true',help='Keep one permanent publisher running after SSH disconnects.')
    p.add_argument('--stop-old-monitors',action='store_true',help='Stop old browser/terminal monitors on this host; keep training and the permanent publisher.')
    p.add_argument('--serve-stable',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--publisher-lock-fd',type=int,help=argparse.SUPPRESS)
    p.add_argument('--temporary',action='store_true',help='Foreground link; Ctrl+C deletes its feed, with a 45-second expiry fallback.')
    a=p.parse_args();a.output=a.output.resolve() if a.output else None;a.state_dir=a.state_dir.resolve();a.stable=not a.temporary
    if a.interval<1:p.error('--interval must be at least 1 second.')
    if a.once and a.tmux:p.error('Use --once or --tmux, not both.')
    if a.background and (a.temporary or a.tmux or a.once or a.print_link):p.error('--background is for permanent sharing only.')
    if a.temporary and (a.tmux or a.once or a.print_link):p.error('--temporary runs in the foreground; do not combine it with --tmux, --once or --print-link.')
    if a.temporary and a.interval>15:p.error('Temporary feeds require an interval of 1 to 15 seconds.')
    if a.stop_old_monitors:
        result=stable_monitor.stop_old_monitors(ROOT,a.state_dir)
        for row in result['stopped']:print(f'Sent stop to old {row["kind"]} PID {row["pid"]}.')
        if not result['stopped']:print('No old monitor processes found on this server.')
        print(f'Permanent publishers kept: {len(result["permanentPublishersKept"])}. Training was not targeted.')
        return
    a.site=origin(a.site)
    if a.print_link:
        path=a.state_dir/'connection.json'
        if not path.exists() and a.output:path=a.output/'browser_share.json'
        if not path.exists():p.error('Start the permanent publisher once before requesting its link.')
        print(view_url(json.loads(path.read_text())));return
    if a.output is None:p.error('--output is required to select a run.')
    if not (a.output/'plan.json').is_file():p.error('Output must contain an existing campaign plan.')
    if a.temporary:return temporary_session(a)
    if a.tmux and subprocess.run(['tmux','has-session','-t',a.tmux],capture_output=True).returncode==0:
        p.error('The browser publisher tmux session already exists; use --print-link to retrieve its URL.')
    print('Connecting the read-only browser monitor.',flush=True)
    saved=stable_monitor.connection(a.output,a.site,a.state_dir,new_connection,view_url)
    if not a.serve_stable:
        stable_monitor.retire_legacy_publisher(a.output,Path(__file__))
        stable_monitor.select_run(a.state_dir,a.output)
    if a.background:
        started=stable_monitor.launch_background(a,Path(__file__).resolve())
        print('Permanent browser publisher started.' if started else 'Permanent monitor now follows this run.',flush=True)
        print('Share this consistent link:\n'+saved['viewUrl'],flush=True)
        return
    if a.tmux:
        forwarded=sys.argv[1:];at=forwarded.index('--tmux');forwarded=forwarded[:at]+forwarded[at+2:]
        command=shlex.join([sys.executable,str(Path(__file__).resolve()),*forwarded])
        command+=' > '+shlex.quote(str(a.output/'browser_monitor.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',a.tmux,'bash','-c',command],check=True)
        print('Browser publisher started; it continues after SSH disconnect.')
        print('Share this link:\n'+saved['viewUrl']);return
    print('Share this link:\n'+saved['viewUrl'],flush=True)
    publish_loop(a,saved)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nBrowser publisher closed; training continues.')
