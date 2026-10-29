"""General N-node SHARP ring with immutable sender versions and lazy link noise.

Each edge is reconstructed from its sender's stored outbound version plus a
keyed independent noise draw. Evaluating it twice never redraws the channel.
Only the sender's latest version is needed because its fixed successor set is
updated atomically on broadcast. No peer models are trusted via sender labels.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')

from basil_core.sharp_math import averaging_count,peak_screen,ranked_indices,hop_coefficient


def vector_hash(values):
    return hashlib.sha256(np.asarray(values,dtype=np.float32).tobytes()).hexdigest()


def link_vector(base,seed,round_id,sender,receiver,sigma):
    from basil_core.research_protocol import keyed_rng
    if not sigma:return np.asarray(base,np.float32).copy()
    noise=keyed_rng(seed,'sharp_link',round_id,sender,receiver).normal(0,sigma,size=base.shape).astype(np.float32)
    return np.asarray(base,np.float32)+noise


def initial_state(cfg,initial):
    n=cfg['nodes']
    return dict(cursor=0,nodeParams=np.repeat(initial[None],n,axis=0),
        outbound=np.repeat(initial[None],n,axis=0),senderRounds=np.full(n,-1,np.int32),
        references={},telemetry=[],roundMetrics=[],elapsedSeconds=0.)


def save_state(path,state,contract):
    path=Path(path);temporary=path.with_suffix('.tmp.npz')
    n,d=state['nodeParams'].shape
    references=np.zeros((n,d),np.float32) if state['references'] else np.empty((0,d),np.float32)
    reference_meta={}
    for node,row in state['references'].items():
        references[int(node)]=row['params']
        reference_meta[str(node)]={k:v for k,v in row.items() if k!='params'}
    metadata=dict(contract=contract,cursor=state['cursor'],references=reference_meta,
        telemetry=state['telemetry'],roundMetrics=state['roundMetrics'],elapsedSeconds=state['elapsedSeconds'])
    raw=json.dumps(metadata,sort_keys=True,allow_nan=False).encode()
    with temporary.open('wb') as handle:
        np.savez(handle,nodeParams=state['nodeParams'],outbound=state['outbound'],
                 senderRounds=state['senderRounds'],referenceParams=references,metadata=np.frombuffer(raw,np.uint8))
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def load_state(path,contract,dimension):
    with np.load(path,allow_pickle=False) as source:
        meta=json.loads(source['metadata'].tobytes())
        if meta['contract']!=contract:raise ValueError('Checkpoint source/data/config/runtime identity differs.')
        n=contract['config']['nodes'];rounds=contract['config']['rounds'];cursor=meta['cursor']
        if type(cursor) is not int or not 0<=cursor<=n*rounds or len(meta['telemetry'])!=cursor:
            raise ValueError('Invalid checkpoint activation cursor.')
        state=dict(cursor=cursor,nodeParams=source['nodeParams'].copy(),outbound=source['outbound'].copy(),
            senderRounds=source['senderRounds'].copy(),references={},telemetry=meta['telemetry'],
            roundMetrics=meta['roundMetrics'],elapsedSeconds=meta['elapsedSeconds'])
        for key in ('nodeParams','outbound'):
            if state[key].shape!=(n,dimension) or not np.isfinite(state[key]).all():
                raise ValueError('Checkpoint model shape or values invalid.')
        if state['senderRounds'].shape!=(n,):raise ValueError('Checkpoint sender state invalid.')
        expected=np.asarray([max(-1,(cursor-1-j)//n) for j in range(n)],np.int32)
        if not np.array_equal(state['senderRounds'],expected):raise ValueError('Sender versions disagree with the checkpoint cursor.')
        if len(state['roundMetrics']) not in ({cursor//n,cursor//n-1} if cursor and cursor%n==0 else {cursor//n}):
            raise ValueError('Checkpoint round metrics disagree with the cursor.')
        for node,row in meta['references'].items():
            if not node.isdigit() or not 0<=int(node)<n:raise ValueError('Invalid CART receiver ID.')
            if (type(row.get('activation')) is not int or not 0<=row['activation']<cursor
                    or row['activation']%n!=int(node)):
                raise ValueError('Reference activation disagrees with its receiver/cursor.')
            classes=np.asarray(row.get('classes',[]));losses=np.asarray(row.get('losses',[]));allowance=np.asarray(row.get('allowance',[]))
            if (not len(classes) or len(np.unique(classes))!=len(classes) or np.any(classes<0)
                    or np.any(classes>=contract['config']['classes']) or losses.shape!=classes.shape
                    or allowance.shape!=classes.shape or not np.isfinite(losses).all()
                    or not np.isfinite(allowance).all() or np.any(losses<0) or np.any(allowance<0)):
                raise ValueError('Invalid stored CART class envelope.')
            params=source['referenceParams'][int(node)].copy()
            if params.shape!=(dimension,) or not np.isfinite(params).all():raise ValueError('Invalid CART reference.')
            state['references'][node]=dict(row,params=params)
    return state


def select_input(cfg,state,worker,x,y,receiver,round_id,first_batch):
    n=cfg['nodes'];size=cfg['window'];flags=cfg['switches']
    senders=[(receiver-distance)%n for distance in range(1,size+1)
             if state['senderRounds'][(receiver-distance)%n]>=0]
    if not senders:
        return state['nodeParams'][receiver].copy(),dict(selectedSenders=[],selectedCount=1,
            availableCopies=0,bootstrap=True,screenAllFailFallback=False,screenApplied=False,ranking=[])
    if not flags['byzantine_mitigation']:
        # No defense means strict immediate-predecessor handoff, never ranking.
        chosen=next((s for s in senders if s==(receiver-1)%n),senders[0])
        value=link_vector(state['outbound'][chosen],cfg['seed'],int(state['senderRounds'][chosen]),chosen,receiver,cfg['sigma'])
        return value,dict(selectedSenders=[chosen],selectedCount=1,availableCopies=len(senders),bootstrap=False,
            screenAllFailFallback=False,screenApplied=False,ranking=[])
    vectors=np.stack([link_vector(state['outbound'][s],cfg['seed'],int(state['senderRounds'][s]),s,receiver,cfg['sigma']) for s in senders])
    passed,screen=peak_screen(vectors,full_window=len(senders)==size,enabled=cfg['method']=='sharp')
    losses=[]
    for j in passed:
        try:loss=worker.probe(vectors[j],x[first_batch],y[first_batch])
        except (FloatingPointError,ValueError):loss=float('inf')
        losses.append(loss)
    wanted=averaging_count(len(senders),cfg['byzantineBudget']) if cfg['method']=='sharp' else 1
    selected=ranked_indices(losses,[senders[j] for j in passed],
        [state['senderRounds'][senders[j]] for j in passed],receiver,n,wanted)
    indices=passed[selected]
    # Float64 accumulator avoids overflow in summation; transmitted result is float32.
    combined=vectors[indices].astype(np.float64).mean(axis=0).astype(np.float32)
    evidence=dict(screen,selectedSenders=[senders[j] for j in indices],selectedCount=len(indices),
        availableCopies=len(senders),bootstrap=False,targetCount=wanted,
        ranking=[dict(sender=senders[j],loss=float(loss) if np.isfinite(loss) else None)
                 for j,loss in zip(passed,losses)])
    return combined,evidence


def evaluate(worker,params,x,y,classes):
    logits=worker.outputs(params,x);pred=logits.argmax(axis=1)
    counts=np.bincount(y,minlength=classes)
    correct=np.bincount(y[pred==y],minlength=classes)
    per_class=np.divide(correct,counts,out=np.zeros(classes,np.float64),where=counts>0)
    return float(np.mean(pred==y)),per_class


def run(cfg,worker,x,y,partitions,vx,vy,*,state=None,on_activation=None,on_round=None,
        checkpoint=None,should_stop=None,max_activations=None):
    from basil_core.research_protocol import keyed_rng,resolve_attackers
    from basil_core.attacks import applyAttack
    from basil_core.sharp_study import validate_config
    validate_config(cfg);n=cfg['nodes'];initial=worker.export()
    current=initial_state(cfg,initial) if state is None else state
    attackers=resolve_attackers(cfg['seed'],cfg['actualAttackers'],n)
    started=time.monotonic();prior=current['elapsedSeconds'];committed=0
    def finish_round(r):
        accuracies=[];per_class=[]
        for params in current['nodeParams']:
            acc,pc=evaluate(worker,params,vx,vy,cfg['classes']);accuracies.append(acc);per_class.append(pc.tolist())
        metrics=dict(round=r,nodeAccuracy=accuracies,perClass=per_class,
            meanAccuracy=float(np.mean(accuracies)),worstNodeAccuracy=float(np.min(accuracies)),
            honestMeanAccuracy=float(np.mean([a for j,a in enumerate(accuracies) if j not in attackers])),
            worstClassMean=float(np.asarray(per_class).mean(axis=0).min()))
        current['roundMetrics'].append(metrics)
        if on_round:on_round(metrics,current)
        if checkpoint:checkpoint(current)
    try:
        # An interrupted evaluation can follow a fully committed last activation.
        # Recover that round's metrics before beginning any subsequent update.
        if current['cursor'] and current['cursor']%n==0 and len(current['roundMetrics'])<current['cursor']//n:
            finish_round(current['cursor']//n-1)
        while current['cursor']<cfg['rounds']*n:
            cursor=current['cursor'];r,node=divmod(cursor,n)
            if should_stop and should_stop():raise InterruptedError('Stop requested at activation boundary.')
            local=partitions[node];lx=x[local];ly=y[local]
            batch=keyed_rng(cfg['seed'],'sharp_batch',r,node,0).choice(len(ly),min(cfg['batchSize'],len(ly)),replace=False)
            incoming,selection=select_input(cfg,current,worker,lx,ly,node,r,batch)
            eta=cfg['learningRate']/(1+cfg['learningRate']*r)
            q=selection['selectedCount'] if cfg['hopDenominator']=='actual_selected' else cfg['byzantineBudget']
            coefficient=(hop_coefficient(cfg['sigma'],int(q),eta,cfg['curvature'])
                if cfg['method']=='sharp' else cfg['sigma']**2) if cfg['switches']['noise_mitigation'] else 0.
            trained,training=worker.train(incoming,lx,ly,r,node,coefficient,batch,should_stop)
            cart=None;memory=None
            if cfg['cart']:
                from basil_core.sharp_cart import transport
                trained,memory,cart=transport(worker,trained,current['references'].get(str(node)),lx,ly,node,cursor)
            attack_active=node in attackers and r>=cfg['attackStart']
            outgoing=trained.copy()
            if attack_active:
                if cfg['attackType']=='hidden':
                    attacked=applyAttack(worker.tensors(trained),'hidden',rng=keyed_rng(cfg['seed'],'sharp_attack',r,node))
                    outgoing=np.concatenate([v.ravel() for v in attacked]).astype(np.float32)
                else:
                    coordinate=int(keyed_rng(cfg['seed'],'sharp_spike',r,node).integers(len(outgoing)))
                    outgoing[coordinate]+=np.float32(20*max(float(np.sqrt(np.mean(trained.astype(np.float64)**2))),.01)*np.sqrt(len(outgoing)))
            record=dict(round=r,node=node,activation=cursor+1,selection=selection,training=training,
                inputHash=vector_hash(incoming),trainedHash=vector_hash(trained),outboundHash=vector_hash(outgoing),
                configuredByzantine=node in attackers,attackActive=attack_active,
                selectedConfiguredAttackers=sum(s in attackers for s in selection['selectedSenders']),
                cart=cart,localSamples=len(local))
            # Commit the complete activation together; reference admission is transactional.
            current['nodeParams'][node]=trained;current['outbound'][node]=outgoing;current['senderRounds'][node]=r
            if memory is not None:current['references'][str(node)]=memory
            current['telemetry'].append(record);current['cursor']=cursor+1;committed+=1
            current['elapsedSeconds']=prior+time.monotonic()-started
            if on_activation:on_activation(record,current)
            if current['cursor']%n==0:
                finish_round(r)
            if max_activations is not None and committed>=max_activations:
                if checkpoint:checkpoint(current)
                return current
        return current
    except BaseException:
        current['elapsedSeconds']=prior+time.monotonic()-started
        if checkpoint:checkpoint(current)
        raise
