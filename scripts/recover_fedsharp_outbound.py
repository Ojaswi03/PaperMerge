#!/usr/bin/env python3
"""Recover only derived outgoing models whose original recorded hashes match."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basil_core.cart_campaign_io import read_json
from basil_core.continuation_campaign import atomic,host_name
from basil_core.round_continuation import sha
from scripts.check_fedsharp_checkpoint_integrity import inspect


def rebuild(cfg,params,metadata,worker):
    import numpy as np
    from basil_core.sharp_protocol import attacked_outbound,vector_hash
    from basil_core.research_protocol import resolve_attackers
    n=cfg['nodes'];cursor=cfg['rounds']*n
    if metadata['cursor']!=cursor or len(metadata['roundMetrics'])!=cfg['rounds'] or len(metadata['telemetry'])!=cursor:
        raise ValueError('Recovery requires a complete final checkpoint.')
    if params.shape!=(n,worker.dimension) or params.dtype!=np.float32 or not np.isfinite(params).all():
        raise ValueError('Trained models are invalid.')
    for i,event in enumerate(metadata['telemetry']):
        if (event['round'],event['node'],event['activation'])!=(i//n,i%n,i+1) or event['training']['optimizerSteps']!=cfg['localSteps']:
            raise ValueError('Activation/SGD history differs.')
    attackers=resolve_attackers(cfg['seed'],cfg['actualAttackers'],n);outgoing=[]
    for node,trained in enumerate(params):
        event=metadata['telemetry'][cursor-n+node]
        if vector_hash(trained)!=event['trainedHash']:raise ValueError('Trained model differs from its recorded hash.')
        value=(attacked_outbound(cfg,worker,trained,cfg['rounds']-1,node)
            if node in attackers and cfg['rounds']-1>=cfg['attackStart'] else trained.copy())
        if vector_hash(value)!=event['outboundHash']:
            raise ValueError(f'Reconstructed outgoing model {node} does not match the recorded hash; no recovery applied.')
        outgoing.append(value)
    return np.stack(outgoing)


def candidate(original,target,outgoing):
    import numpy as np
    hashes={}
    with zipfile.ZipFile(original) as source,zipfile.ZipFile(target,'w',allowZip64=True) as destination:
        destination.comment=source.comment
        for member in source.infolist():
            with destination.open(copy.copy(member),'w',force_zip64=True) as output:
                if member.filename=='outbound.npy':
                    np.lib.format.write_array(output,outgoing,allow_pickle=False)
                else:
                    value=hashlib.sha256()
                    with source.open(member) as stream:
                        for block in iter(lambda:stream.read(1024*1024),b''):
                            value.update(block);output.write(block)
                    hashes[member.filename]=value.hexdigest()
    with target.open('rb') as stream:os.fsync(stream.fileno())
    with zipfile.ZipFile(target) as restored:
        if restored.testzip() is not None:raise ValueError('Recovered candidate failed CRC validation.')
        for name,expected in hashes.items():
            value=hashlib.sha256()
            with restored.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
            if value.hexdigest()!=expected:raise ValueError('An intact checkpoint member changed.')
    return hashes


def recover(queue,run,expected_sha,execute=False):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()
    from basil_core import basil_sharp_thesis as study,fleet_queue as fleet,live_gpu_scheduler as live
    from basil_core.sharp_protocol import load_state,vector_hash
    from basil_core.sharp_worker import SharpWorker
    queue=Path(queue).resolve();current=fleet.state(queue);entry=current['runs'][run]
    if entry['owner']!=host_name() or entry['state']!='DONE':raise ValueError('Only this host\'s completed run can be recovered.')
    root=Path(entry['location']['output']);plan=read_json(root/'plan.json');study.check_source(plan['source'])
    view=plan['configs'][entry['location']['index']]
    if view['runId']!=run:raise ValueError('Claim/configuration differs.')
    if entry['location']['index'] in live.discover(root,len(plan['configs'])):raise ValueError('A training worker still owns this run.')
    folder=root/run;checkpoint=folder/'checkpoint.npz';row=read_json(folder/'run.json')
    row_sha=sha(folder/'run.json');metrics_sha=sha(folder/'metrics.npz')
    if row['status']!='COMPLETED' or row['config']!=view or row['source']!=plan['source']:
        raise ValueError('Final run record differs from the frozen plan.')
    if sha(checkpoint)!=expected_sha:raise ValueError('Checkpoint differs from the explicitly identified damaged archive.')
    integrity=inspect(checkpoint)
    if integrity['errors']!=['outbound.npy'] or not integrity.get('fileUnchanged'):
        raise ValueError('Recovery is restricted to an isolated outgoing-array CRC failure.')
    with np.load(checkpoint,allow_pickle=False) as archive:
        params=archive['nodeParams'].copy();metadata=json.loads(archive['metadata'].tobytes())
    cfg=study.effective_config(view)
    if metadata['contract']!=row['contract'] or metadata['contract']['config']!=cfg:
        raise ValueError('Checkpoint contract differs from the final run.')
    worker=SharpWorker(cfg,'CPU')
    if vector_hash(worker.export())!=row['initialHash']:raise ValueError('Model initialization differs.')
    outgoing=rebuild(cfg,params,metadata,worker)
    with np.load(folder/'metrics.npz',allow_pickle=False) as metrics:
        for key,field in (('roundNodeAccuracy','nodeAccuracy'),('roundNodePerClassAccuracy','perClass'),('roundHonestMeanAccuracy','honestMeanAccuracy')):
            np.testing.assert_array_equal(metrics[key],np.asarray([r[field] for r in metadata['roundMetrics']]))
    operational=root/'operational_adapters';operational.mkdir(exist_ok=True)
    with (operational/'outbound_recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='outbound-recovery-',dir=operational) as name:
            target=Path(name)/'checkpoint.npz';members=candidate(checkpoint,target,outgoing)
            state=load_state(target,metadata['contract'],worker.dimension)
            np.testing.assert_array_equal(state['nodeParams'],params);np.testing.assert_array_equal(state['outbound'],outgoing)
            if state['telemetry']!=metadata['telemetry'] or state['roundMetrics']!=metadata['roundMetrics']:
                raise ValueError('Historical records changed.')
            receipt=dict(schema=1,status='PREPARED',runId=run,recipe='deterministic_outbound_from_trained_models',
                originalCheckpointSha256=expected_sha,repairedCheckpointSha256=sha(target),
                cursor=state['cursor'],trainedModelHashesMatched=cfg['nodes'],outboundHashesMatched=cfg['nodes'],
                preservedMemberSha256=members,source=plan['source'],repairScriptSha256=sha(Path(__file__)),
                metricsSha256=metrics_sha,runRecordSha256=row_sha,optimizerUpdatesAdded=0,
                interpretation='Outgoing array reconstructed from retained trained models and seeded attack; every vector matches its recorded final hash. Fresh final inference remains required.')
            if not execute:return receipt
            if sha(checkpoint)!=expected_sha or sha(folder/'run.json')!=row_sha or sha(folder/'metrics.npz')!=metrics_sha or fleet.state(queue)['runs'][run]!=entry:
                raise ValueError('Source files/claim changed during recovery.')
            audit=operational/'outbound_recovery'/run
            if audit.exists():raise ValueError('A recovery attempt already exists; inspect before repeating.')
            audit.mkdir(parents=True);os.link(checkpoint,audit/'original-checkpoint.npz')
            atomic(audit/'receipt.json',receipt);atomic(folder/'checkpoint_recovery.json',receipt)
            snapshot=root/'recovery_source_snapshots'/receipt['repairScriptSha256']/'scripts/recover_fedsharp_outbound.py'
            snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(__file__),snapshot)
            os.replace(target,checkpoint)
            receipt['status']='RECOVERED'
            atomic(folder/'checkpoint_recovery.json',receipt);atomic(audit/'receipt.json',receipt)
            return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True,type=Path);parser.add_argument('--run-id',required=True)
    parser.add_argument('--expected-sha256',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();result=recover(args.queue,args.run_id,args.expected_sha256,args.execute)
    print(json.dumps({key:result[key] for key in ('status','runId','cursor','trainedModelHashesMatched',
        'outboundHashesMatched','originalCheckpointSha256','repairedCheckpointSha256','optimizerUpdatesAdded')},indent=2))
