"""Audited one-time boundary retries under the exclusive live supervisor lock."""
import json
from pathlib import Path
import os
import select
import signal
import time

from basil_core.cart_campaign_io import read_json,write_json
from basil_core.round_continuation import sha


def validate(root,plan,digest,request_path,active):
    root=Path(root);request=read_json(request_path)
    if request['planSha256']!=digest or Path(request['output']).resolve()!=root:
        raise ValueError('Retry request does not identify this frozen host plan.')
    if not 0<=time.time()-request['verifiedAt']<=1800:
        raise ValueError('Retry input verification is older than 30 minutes.')
    if not request['datasetChecksPassed']:raise ValueError('Retry dataset checks did not pass.')
    configs={cfg['runId']:(i,cfg) for i,cfg in enumerate(plan['configs'])}
    if not request['runs'] or len({r['runId'] for r in request['runs']})!=len(request['runs']):
        raise ValueError('Empty or duplicated retry request.')
    for item in request['runs']:
        run=item['runId'];index,cfg=configs[run];folder=root/run
        parent=plan['continuations'][run];row=read_json(folder/'run.json')
        if index in active:raise ValueError('Retry target already has a live worker.')
        if row.get('status')!='FAILED' or row.get('completedActivations')!=parent['sourceCursor']:
            raise ValueError('Retry only supports failed, unadvanced boundary imports.')
        if sha(folder/'run.json')!=item['failedRowSha256']:
            raise ValueError('Failed row changed after retry verification.')
        if (folder/'checkpoint.npz').exists() or (folder/'continuation.json').exists():
            raise ValueError('Retry target already has a forked checkpoint; inspect it separately.')
        if (root/'operational_adapters/continuation_retries'/run).exists():
            raise ValueError('This configuration already used its one verified boundary retry.')
        seed=Path(plan['sharedQueue'])/parent['seedRelativePath']
        if sha(seed)!=parent['seedSha256']:raise ValueError('Retry seed changed after verification.')
    return request


def requeue(root,plan,digest,request_path,active):
    """Caller owns supervisor.lock and has retired the old dispatcher."""
    root=Path(root);request=validate(root,plan,digest,request_path,active)
    base=root/'operational_adapters/continuation_retries';base.mkdir(parents=True,exist_ok=True)
    # Preserve every original failure before changing any row.
    for item in request['runs']:
        run=item['runId'];folder=base/run;folder.mkdir()
        (folder/'failed-run.json').write_bytes((root/run/'run.json').read_bytes())
        write_json(folder/'request.json',request)
    for item in request['runs']:
        path=root/item['runId']/'run.json';row=read_json(path)
        row.update(status='PENDING',phase='CONTINUATION_PENDING')
        row.pop('error',None);row.pop('workerExitCode',None)
        row['boundaryRetryRequest']=request['id']
        write_json(path,row)
    return [item['runId'] for item in request['runs']]


def stop_live_dispatch(root,plan,digest,request_path):
    """Ask a verified shared dispatcher to exit; its handler leaves workers alive."""
    from basil_core import live_gpu_scheduler as live
    owner=live.controller(root)
    if owner is None:return
    script,_=live.script_command(owner)
    if script.name not in ('run_fedsharp_500r.py','retry_fedsharp_500r.py','rebalance_basil_sharp_live.py'):
        raise ValueError('This controller does not support a cooperative continuation retry.')
    campaign=read_json(root/'campaign.json')
    if campaign.get('scheduler',{}).get('dispatch')!='shared_queue_verified_gpus':
        raise ValueError('Only the verified live dispatcher supports worker-preserving cooperative stop.')
    found=live.discover(root,len(plan['configs']))
    validate(root,plan,digest,request_path,found)
    if owner['pid'] in {worker['pid'] for worker in found.values()}:
        raise ValueError('Dispatcher and worker identities overlap.')
    descriptor=os.pidfd_open(owner['pid'])
    try:
        if not live.same_process(owner):raise ProcessLookupError('Dispatcher identity changed.')
        # The shared scheduler handles SIGTERM by stopping dispatch and returning
        # without signalling its children. Legacy supervisors are refused above.
        signal.pidfd_send_signal(descriptor,signal.SIGTERM)
        if not select.select([descriptor],[],[],20)[0]:
            raise RuntimeError('Dispatcher has not exited yet; healthy workers were not targeted.')
    finally:os.close(descriptor)
