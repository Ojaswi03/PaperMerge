#!/usr/bin/env python3
"""Enable atomic cross-server claiming while preserving active continuation workers."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import fleet_runner as runner
from basil_core.cart_campaign_io import read_json,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--retry-failed-imports',action='store_true',help='Verify and retry unadvanced failed imports once during initial host activation.')
    parser.add_argument('--ntfy-topic',default='papermerge-ojaswi')
    parser.add_argument('--serve',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--retry-request',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--inherited-gpu-lease',action='append',default=[],help=argparse.SUPPRESS)
    parser.add_argument('--handover-id',help=argparse.SUPPRESS)
    args=parser.parse_args();queue=args.queue.resolve()
    if args.background and not args.execute:parser.error('--background requires --execute.')
    if args.serve:
        if not args.execute or args.background:parser.error('Invalid fleet child invocation.')
        if bool(args.inherited_gpu_lease)!=bool(args.handover_id):parser.error('Incomplete lease handoff invocation.')
        return runner.run(queue,args.retry_request,args.ntfy_topic,args.inherited_gpu_lease,args.handover_id)
    if args.inherited_gpu_lease or args.handover_id:parser.error('Lease handoff is for the verified child invocation only.')
    if not args.execute:
        from basil_core.continuation_campaign import host_name
        from basil_core.live_gpu_scheduler import verified,discover
        host=host_name();original=Path(read_json(queue/'host_bindings'/host/'owner.json')['output'])
        plan,hardware,assigned,digest=verified(original)
        print(json.dumps(dict(host=host,existingWorkers=len(discover(original,len(plan['configs']))),
            verifiedGpus=list(hardware),originalPlanSha256=digest,trainingModified=False,
            action='Join both servers with --execute --background; only untouched jobs enter the shared pool.'),indent=2))
        return 0
    original,output,legacy,plan,master=runner.prepare(queue)
    with (output/'agent.lock').open('a') as guard:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    retry=None
    if args.retry_failed_imports:
        already=(original/'fleet_owner.json').exists()
        failures=[]
        for cfg in ([] if already else legacy['configs']):
            run=cfg['runId'];folder=original/run;path=folder/'run.json'
            if not path.exists():continue
            row=read_json(path)
            if (row.get('status')=='FAILED' and row.get('completedActivations')==legacy['continuations'][run]['sourceCursor']
                    and not (folder/'checkpoint.npz').exists() and not (folder/'continuation.json').exists()
                    and not (original/'operational_adapters/continuation_retries'/run).exists()):failures.append(cfg)
        if failures:
            from scripts.retry_fedsharp_500r import prepare
            actual,request=prepare(queue,original,run_ids=[cfg['runId'] for cfg in failures])
            retry=output/'import-retry-request.json';write_json(retry,request)
            from basil_core.continuation_recovery import validate
            from basil_core import live_gpu_scheduler as live
            validate(original,actual,request['planSha256'],retry,live.discover(original,len(actual['configs'])))
            print(f'Verified {len(failures)} failed round-100 imports for one retry.',flush=True)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--queue',str(queue),'--execute','--serve',
        '--ntfy-topic',args.ntfy_topic]
    if retry:command.extend(['--retry-request',str(retry)])
    if args.background:
        with (output/'fleet_agent.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        print(f'Fleet agent launched: PID {process.pid}. Log: {output / "fleet_agent.log"}',flush=True)
        print('Existing training workers keep running. Start this agent on both servers.',flush=True)
        return 0
    return subprocess.run(command,cwd=ROOT).returncode


if __name__=='__main__':raise SystemExit(main())
