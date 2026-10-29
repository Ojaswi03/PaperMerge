#!/usr/bin/env python3
"""Probe a frozen CART window without updates, new training, or model selection.

All receivers see the same saved round boundary, not a replay of the next
sequential round. Pre-link outbound states can already contain Byzantine
corruption and inherited channel damage. They are not clean model oracles.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(args):
    if args.device=='GPU':
        from scripts.cart_v2_slot_worker import adopt_lease
        adopt_lease(args.lease_fd)
        from scripts.run_cart_v2_study import setup_gpu
        tf=setup_gpu()
    else:
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    import numpy as np
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core.cart_comparison_study import effective_config,partitions_for
    from basil_core.cart_batched_scoring import BatchedAdmittedWorker,select_input
    from basil_core.cart_class_skill import class_skill
    from basil_core.cart_peer_audit import decompose,cohort,summarize
    from basil_core.sharp_data import training_data,digest
    from basil_core.sharp_cart import probes
    from basil_core.sharp_protocol import load_state,link_vector,vector_hash
    from basil_core.research_protocol import keyed_rng,resolve_attackers
    plan=read_json(args.output/'plan.json')
    if len(plan['configs'])!=1:raise ValueError('Select a one-case saved experiment for this bounded audit.')
    view=plan['configs'][0];folder=args.output/view['runId'];row=read_json(folder/'run.json')
    if row['status']!='COMPLETED' or row['config']!=view or row['source']!=plan['source']:
        raise ValueError('Completed producer-matched run required.')
    needed=('basil_core/cart_comparison_study.py','basil_core/sharp_worker.py','basil_core/models.py',
        'basil_core/sharp_data.py','basil_core/sharp_protocol.py','basil_core/sharp_cart.py',
        'basil_core/cart_batched_scoring.py','basil_core/cart_class_skill.py','basil_core/sharp_math.py')
    for name in needed:
        if sha(ROOT/name)!=plan['source'][name]:raise ValueError('Inference/protocol source changed: '+name)
    for name,value in plan['source'].items():
        if sha(args.output/'source_snapshots'/name)!=value:raise ValueError('Frozen producer bytes changed: '+name)
    cfg=effective_config(view);worker=BatchedAdmittedWorker(cfg,args.device)
    if vector_hash(worker.export())!=row['initialHash']:raise ValueError('Model initialization changed.')
    x,y=training_data(cfg['dataset']);chunks,held,audit=partitions_for(view,y)
    identity=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
    if identity!=row['data'] or audit!=row['partition']:raise ValueError('Private data contract changed.')
    checkpoint=folder/'checkpoint.npz';before=sha(checkpoint)
    state=load_state(checkpoint,row['contract'],worker.dimension)
    receivers=list(range(0,cfg['nodes'],args.stride));started=time.monotonic();records=[]
    provenance=dict(producerRunId=view['runId'],producerSource=plan['source'],checkpointSha256=before,
        scriptSha256=sha(__file__),auditModuleSha256=sha(ROOT/'basil_core/cart_peer_audit.py'),
        receiverIds=receivers,frozenCursor=state['cursor'],tensorflow=tf.__version__,numpy=np.__version__,
        device=args.device,interpretation='Frozen boundary counterfactual; pre-link states retain attack and historical noise; no training or evaluation holdout used.')
    args.destination.mkdir(parents=True,exist_ok=True)
    if (args.destination/'audit.json').exists():raise FileExistsError('Preserve completed audit; choose a new destination.')
    write_json(args.destination/'plan.json',provenance)
    attackers=resolve_attackers(cfg['seed'],cfg['actualAttackers'],cfg['nodes'])
    for node in receivers:
        local=chunks[node];lx=x[local];ly=y[local]
        fit,check=probes(ly,cfg['seed'],node,cfg['cartProbePerClass'])
        if np.intersect1d(local[np.concatenate((fit,check))],held).size:raise ValueError('Probe touches reporting holdout.')
        classes=np.intersect1d(np.unique(ly[fit]),np.unique(ly[check]))
        if len(classes)<2:
            records.append(dict(node=node,reason='fewer than two supported classes'));continue
        fit=fit[np.isin(ly[fit],classes)];check=check[np.isin(ly[check],classes)]
        idx=np.concatenate((fit,check));split=len(fit)
        def evidence(flat):
            logits=worker.outputs(flat,lx[idx])
            return class_skill(logits[:split],ly[fit],classes),class_skill(logits[split:],ly[check],classes)
        own=state['nodeParams'][node];of,ok=evidence(own);target=np.minimum(of,ok)
        qualified=np.asarray([sum(ly[fit]==c)>=8 and sum(ly[check]==c)>=8 for c in classes])&(target>=.2)
        batch=keyed_rng(cfg['seed'],'sharp_batch',state['cursor']//cfg['nodes'],node,0).choice(len(ly),min(cfg['batchSize'],len(ly)),replace=False)
        pool,base=select_input(cfg,state,worker,lx,ly,node,state['cursor']//cfg['nodes'],batch)
        senders=[(node-d)%cfg['nodes'] for d in range(1,cfg['window']+1) if state['senderRounds'][(node-d)%cfg['nodes']]>=0]
        passed=[s for j,s in enumerate(senders) if not base.get('screenApplied') or base['peakRatios'][j]<=base['peakThreshold']]
        near=set(sorted(passed,key=lambda s:(-int(state['senderRounds'][s]),(node-s)%cfg['nodes']))[:8])
        peers=[]
        for sender in senders:
            pre=state['outbound'][sender];received=link_vector(pre,cfg['seed'],int(state['senderRounds'][sender]),sender,node,cfg['sigma'])
            pf,pk=evidence(pre);rf,rk=evidence(received)
            peers.append(dict(sender=sender,senderRound=int(state['senderRounds'][sender]),screenPassed=sender in passed,
                nearEight=sender in near,configuredByzantine=sender in attackers,
                preEvidence=(pf,pk),receivedEvidence=(rf,rk)))
        allowance=np.zeros(len(classes));draws=0;pool_f,pool_k=evidence(pool)
        informative=min(of.mean(),ok.mean())>.02
        near_values=[(p['receivedEvidence']) for p in peers if p['nearEight']]+[(pool_f,pool_k)]
        if informative and qualified.any() and cfg['sigma'] and any(np.any((target-np.minimum(f,k))[qualified]>0) for f,k in near_values):
            for draw in range(cfg['calibrationDraws']):
                noise=keyed_rng(cfg['seed'],'quality_route_channel',node,state['cursor'],draw).normal(0,cfg['sigma'],size=own.shape).astype(np.float32)
                f,k=evidence(own+noise);allowance=np.maximum(allowance,np.maximum(target-np.minimum(f,k),0));draws+=1
        own_score=decompose(of,ok,target,qualified,allowance)
        for peer in peers:
            for key,source in [('received','receivedEvidence'),('preLink','preEvidence')]:
                f,k=peer.pop(source);peer[key]=decompose(f,k,target,qualified,allowance)
                peer[key]['classGainAgainstOwned']=(np.minimum(f,k)-target).tolist()
        groups={mode:{name:cohort(peers,own_score,mode,screened=screened,near=recent)
            for name,screened,recent in [('nearEight',True,True),('screenedWindow',True,False),('wholeWindow',False,False)]}
            for mode in ('received','preLink')}
        records.append(dict(node=node,supportedClasses=classes.tolist(),qualifiedClasses=classes[qualified].tolist(),
            informative=bool(informative),channelAllowance=allowance.tolist(),calibrationDraws=draws,
            own=own_score,pool=decompose(pool_f,pool_k,target,qualified,allowance),peers=peers,cohorts=groups))
        write_json(args.destination/'progress.json',dict(status='RUNNING',completedReceivers=len(records),totalReceivers=len(receivers),elapsedSeconds=time.monotonic()-started))
        print(f'Receiver {node}: near-8 margin={groups["received"]["nearEight"]["bestScoreMargin"]}; full screened margin={groups["received"]["screenedWindow"]["bestScoreMargin"]}',flush=True)
    if sha(checkpoint)!=before:raise ValueError('Input checkpoint changed during read-only audit.')
    result=dict(provenance=provenance,summary=summarize(records),receivers=records,elapsedSeconds=time.monotonic()-started,
        status='COMPLETED',checkpointUnchanged=True,optimizerStepsAdded=0)
    write_json(args.destination/'audit.json',result)
    write_json(args.destination/'progress.json',dict(status='COMPLETED',completedReceivers=len(records),totalReceivers=len(receivers),elapsedSeconds=result['elapsedSeconds']))
    print(json.dumps(result['summary'],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--stride',type=int,default=5)
    p.add_argument('--device',choices=('CPU','GPU'),default='GPU');p.add_argument('--worker',action='store_true')
    p.add_argument('--lease-fd',type=int);args=p.parse_args();args.output=args.output.resolve();args.destination=args.destination.resolve()
    if args.stride<1:p.error('Positive receiver stride required.')
    if args.output==args.destination or args.output in args.destination.parents:p.error('Separate audit destination required.')
    if args.worker or args.device=='CPU':run(args);return
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    from scripts.run_cart_v2_scaled import acquire_leases,available_ram_mib
    from scripts.run_cart_v2_study import worker_environment
    inventory=gpu_inventory();profile=ExecutionProfile(ram_mib=8192).bind(inventory)
    if inventory[0]['memory.free']<8192 or available_ram_mib()<12288:raise RuntimeError('Insufficient local diagnostic resources.')
    leases=acquire_leases([profile])
    try:
        fd=leases[profile.gpu].fileno()
        subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker','--lease-fd',str(fd)],
            cwd=ROOT,env=worker_environment(profile),pass_fds=(fd,),check=True)
    finally:
        for lease in leases.values():lease.close()


if __name__=='__main__':main()
