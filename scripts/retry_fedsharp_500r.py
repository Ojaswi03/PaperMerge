#!/usr/bin/env python3
"""Verify failed boundary imports, requeue once, and adopt healthy workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import live_gpu_scheduler as live
from basil_core.cart_campaign_io import read_json,write_json
from basil_core.continuation_recovery import validate
from basil_core.round_continuation import sha
from scripts.diagnose_fedsharp_500r import diagnose


def prepare(queue,output,run_ids=None):
    plan,hardware,assigned,digest=live.verified(output)
    if Path(plan['sharedQueue']).resolve()!=queue:raise ValueError('Host plan belongs to a different shared queue.')
    report=diagnose(queue,check_inputs=True)
    if run_ids is not None:
        report['failures']=[row for row in report['failures'] if row['runId'] in set(run_ids)]
        if {row['runId'] for row in report['failures']}!=set(run_ids):
            raise ValueError('Requested failures changed during verification.')
    if not report['producerUnchanged'] or not report['failures']:raise ValueError('No verified failed boundary imports to retry.')
    for dataset in report['datasets'].values():
        if not all(read['verified'] for read in dataset['reads']):raise ValueError('Dataset fingerprints still fail; no retry queued.')
    for failure in report['failures']:
        for name,check in failure['checkpoints'].items():
            if name=='local' and not Path(check['path']).exists():continue
            if not all(check.get(key) is True for key in ('matchesFrozenSeed','metadataReadable',
                    'originalContractMatches','boundaryMatches','stableChecksum')):
                raise ValueError('Checkpoint validation still fails; no retry queued: '+failure['runId'])
    request=dict(id=uuid.uuid4().hex,output=str(output),planSha256=digest,verifiedAt=time.time(),
        datasetChecksPassed=True,diagnostics=report,runs=[dict(runId=r['runId'],
        failedRowSha256=sha(output/r['runId']/'run.json')) for r in report['failures']])
    return plan,request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--request',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--ntfy-topic',default='papermerge-ojaswi')
    args=parser.parse_args();queue=args.queue.resolve()
    if args.background and not args.execute:parser.error('--background requires --execute.')
    if args.request and (args.background or not args.execute):parser.error('Invalid recovery child invocation.')
    from basil_core.continuation_campaign import host_name
    binding=read_json(queue/'host_bindings'/host_name()/'owner.json')
    output=Path(binding['output']).resolve()
    if args.output and args.output.resolve()!=output:raise ValueError('Output differs from the exclusive server binding.')
    if args.request:
        from scripts.run_fedsharp_500r import publish_updates
        from basil_core.continuation_recovery import stop_live_dispatch
        plan=read_json(output/'plan.json')
        options=SimpleNamespace(output=output,maximum=plan['maximumWorkersPerGpu'],
            poll_seconds=3,ntfy_topic=args.ntfy_topic,retry_request=args.request.resolve())
        # The existing shared dispatcher owns live_scheduler.lock for its whole
        # lifetime. Cooperatively stop dispatch first, then adopt all survivors.
        with (output/'continuation_retry.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            stop_live_dispatch(output,plan,live.sha(output/'plan.json'),options.retry_request)
            with publish_updates(queue,output):return live.execute(options)
    plan,request=prepare(queue,output)
    folder=output/'operational_adapters/retry_requests'
    # Validate against real active process identities before starting any handover.
    # A temporary request is only an operational record; plans and seeds are immutable.
    folder.mkdir(parents=True,exist_ok=True);path=folder/(request['id']+'.json')
    write_json(path,request)
    validate(output,plan,request['planSha256'],path,live.discover(output,len(plan['configs'])))
    print(json.dumps(dict(verifiedRetries=[r['runId'] for r in request['runs']],
        trainingProducerUnchanged=True,healthyWorkersWillBeAdopted=True,request=str(path)),indent=2),flush=True)
    if not args.execute:
        print('Verification only. Add --execute --background to requeue once.');return 0
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--queue',str(queue),
        '--output',str(output),'--request',str(path),'--execute','--ntfy-topic',args.ntfy_topic]
    if args.background:
        with (output/'boundary_retry.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        print(f'Recovery dispatcher PID {child.pid}. Log: {output / "boundary_retry.log"}')
        return 0
    return subprocess.run(command,cwd=ROOT).returncode


if __name__=='__main__':raise SystemExit(main())
