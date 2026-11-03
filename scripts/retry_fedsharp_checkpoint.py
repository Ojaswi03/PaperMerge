#!/usr/bin/env python3
"""One audited local retry after SIGSEGV, from a strictly validated checkpoint."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core import fleet_queue as fleet,live_gpu_scheduler as live
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic,host_name
from basil_core.round_continuation import sha


def retry(queue,run,execute=False):
    import numpy as np
    from basil_core import basil_sharp_thesis as study
    from basil_core.sharp_protocol import load_state,vector_hash
    host=host_name();queue=Path(queue).resolve();snapshot=fleet.state(queue)
    entry=copy.deepcopy(snapshot['runs'][run])
    if entry['owner']!=host or entry['state']!='FAILED':raise ValueError('Only this host\'s failed claim can be retried.')
    root=Path(entry['location']['output']);plan=read_json(root/'plan.json');study.check_source(plan['source'])
    cfg=plan['configs'][entry['location']['index']]
    if cfg['runId']!=run:raise ValueError('Claim differs from the frozen execution plan.')
    folder=root/run;row=read_json(folder/'run.json');row_sha=sha(folder/'run.json')
    if row.get('status')!='FAILED' or 'exited -11 without a completed result' not in row.get('error',''):
        raise ValueError('This recovery is only for the reported SIGSEGV failure.')
    if entry.get('process') and live.same_process(entry['process']):raise ValueError('Target process is still alive.')
    if entry['location']['index'] in live.discover(root,len(plan['configs'])):raise ValueError('Target already has a worker.')
    path=folder/'checkpoint.npz';digest=sha(path)
    with np.load(path,allow_pickle=False) as archive:
        dimension=archive['nodeParams'].shape[1]
        meta=json.loads(archive['metadata'].tobytes())
    if meta['contract']!=row.get('contract') or meta['contract']['source']!=plan['source']:
        raise ValueError('Checkpoint contract differs from the failed worker.')
    optimized=meta.get('executionOptimization')
    if optimized:
        from basil_core.fedsharp_runtime_optimization import validate
        policy=read_json(Path(snapshot['hosts'][host]['fleet'])/'optimization_policy.json');validate(policy)
        if optimized.get('source')!=policy['source'] or optimized.get('revision')!=policy['revision']:
            raise ValueError('Checkpoint optimization differs from the active policy.')
    state=load_state(path,meta['contract'],dimension);effective=study.effective_config(cfg)
    cursor=state['cursor'];n=effective['nodes']
    if cursor<plan['continuations'][run]['sourceCursor'] or cursor>=n*effective['rounds']:
        raise ValueError('Checkpoint is not an incomplete continuation.')
    for i,event in enumerate(state['telemetry']):
        if (event['round'],event['node'],event['activation'])!=(i//n,i%n,i+1) or event['training']['optimizerSteps']!=effective['localSteps']:
            raise ValueError('Checkpoint activation history is invalid.')
    for node in range(n):
        index=cursor-1-((cursor-1-node)%n);event=state['telemetry'][index]
        if event['trainedHash']!=vector_hash(state['nodeParams'][node]) or event['outboundHash']!=vector_hash(state['outbound'][node]):
            raise ValueError('Saved model hashes differ from committed activation history.')
    item=plan['continuations'][run];metrics=ROOT/item['sourceArchive']/item['sourceRunId']/'metrics.npz'
    if sha(metrics)!=item['sourceMetricsSha256']:raise ValueError('Original metric prefix changed.')
    with np.load(metrics,allow_pickle=False) as original:
        for key,field in (('roundNodeAccuracy','nodeAccuracy'),('roundNodePerClassAccuracy','perClass'),('roundHonestMeanAccuracy','honestMeanAccuracy')):
            np.testing.assert_array_equal(original[key],np.asarray([r[field] for r in state['roundMetrics'][:item['sourceRounds']]]))
    if sha(path)!=digest or sha(folder/'run.json')!=row_sha:raise ValueError('Failed files changed during verification.')
    checked_stat=path.stat()
    checkpoint_identity=(checked_stat.st_ino,checked_stat.st_size,checked_stat.st_mtime_ns)
    result=dict(runId=run,checkpointVerified=True,resumeActivations=cursor,
        replayActivations=max(0,row.get('completedActivations',cursor)-cursor),
        checkpointSha256=digest,workerSignalsSent=False,execute=execute)
    if not execute:return result
    audit=root/'operational_adapters/checkpoint_retries'/run
    host_root=Path(snapshot['hosts'][host]['fleet'])
    with (host_root/'optimization_transition.lock').open('a') as maintenance:
        fcntl.flock(maintenance,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with fleet.locked(queue) as (current,master):
            current_stat=path.stat()
            if current['runs'][run]!=entry or sha(folder/'run.json')!=row_sha or (current_stat.st_ino,current_stat.st_size,current_stat.st_mtime_ns)!=checkpoint_identity:
                raise ValueError('Target changed before requeue.')
            if audit.exists():raise ValueError('This run already used its audited checkpoint retry; inspect before repeating.')
            claim=read_json(queue/'fleet/claims'/(run+'.json'))
            if claim['host']!=host or claim['token']!=entry['claim'] or claim['location']!=entry['location'] or claim['gpu']!=entry['boundGpu']:
                raise ValueError('Exclusive claim differs; no retry applied.')
            audit.mkdir(parents=True)
            (audit/'failed-run.json').write_bytes((folder/'run.json').read_bytes())
            os.link(path,audit/'checkpoint.npz');atomic(audit/'receipt.json',result)
            restored=copy.deepcopy(row);restored.update(status='INTERRUPTED',phase='VERIFIED_CHECKPOINT_RETRY',
                completedActivations=cursor,completedRounds=len(state['roundMetrics']),elapsedSeconds=state['elapsedSeconds'])
            restored.pop('error',None);restored.pop('workerExitCode',None)
            atomic(folder/'run.json',restored)
            current['runs'][run].update(state='WAITING',eligibleHosts=[host],public=fleet.public_row(master['items'][entry['index']],restored))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    print(json.dumps(retry(args.queue,args.run_id,args.execute),indent=2))


if __name__=='__main__':main()
