#!/usr/bin/env python3
"""Publish sanitized campaign progress to the read-only Chrome dashboard.

No training controls, checkpoints, paths, usernames or logs are uploaded.
The update credential stays in a private file inside the ignored campaign output.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
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


def request_json(url,method,payload=None,token=None):
    headers={'Content-Type':'application/json','User-Agent':'PaperMerge-Monitor/1.0'}
    if token:headers['Authorization']='Bearer '+token
    raw=json.dumps(payload,allow_nan=False).encode() if payload is not None else None
    with urlopen(Request(url,data=raw,headers=headers,method=method),timeout=15) as response:return json.load(response)


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
        result=request_json(site+'/api/feeds','POST',public_snapshot(root))
        import re
        if not re.fullmatch(r'[a-f0-9-]{36}',result.get('feedId','')) or not re.fullmatch(r'[a-f0-9]{64}',result.get('writeToken','')):
            raise ValueError('Unexpected dashboard setup response.')
        result=dict(site=site,feedId=result['feedId'],writeToken=result['writeToken'])
        result['viewUrl']=view_url(result)
        descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as stream:json.dump(result,stream)
        return result


def publish_loop(args,saved):
    from basil_core.cart_public_monitor import public_snapshot
    with (args.output/'browser_publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        notified=False
        while True:
            try:
                data=public_snapshot(args.output)
                request_json(saved['site']+'/api/feeds/'+saved['feedId'],'PUT',data,saved['writeToken'])
                print(f'Browser progress updated: {data["completed"]}/{data["total"]} complete.',flush=True)
                if args.ntfy_topic and not notified:
                    from basil_core.ntfy_notifications import publish
                    receipt=publish(args.ntfy_topic,'CART browser monitor is live','Share this read-only link with your professor:\n\n'+saved['viewUrl']+'\n\nUpdates automatically. Training is unchanged.')
                    notified=receipt['deliveredToServer']
                if args.once or data['status'] in ('COMPLETED','COMPLETED_WITH_FAILURES','COMPLETED_REPORT_FAILED','FAILED','INTERRUPTED'):return
            except (OSError,ValueError) as error:
                print('Browser update delayed ('+type(error).__name__+'); training is unaffected.',file=sys.stderr,flush=True)
                if args.once:raise
            time.sleep(args.interval)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--site',default=SITE);p.add_argument('--interval',type=int,default=10)
    p.add_argument('--once',action='store_true');p.add_argument('--print-link',action='store_true')
    p.add_argument('--tmux');p.add_argument('--ntfy-topic');a=p.parse_args();a.output=a.output.resolve()
    if a.interval<5:p.error('--interval must be at least 5 seconds.')
    if a.once and a.tmux:p.error('Use --once or --tmux, not both.')
    a.site=origin(a.site)
    if not (a.output/'plan.json').is_file():p.error('Output must contain an existing campaign plan.')
    if a.tmux and subprocess.run(['tmux','has-session','-t',a.tmux],capture_output=True).returncode==0:
        p.error('The browser publisher tmux session already exists; use --print-link to retrieve its URL.')
    if a.print_link:
        print(view_url(json.loads((a.output/'browser_share.json').read_text())));return
    saved=connection(a.output,a.site)
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
