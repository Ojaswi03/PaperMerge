"""Versioned survivor-aware development correction to SHARP selection.

Original v1 remains unchanged. Screening cannot certify any unattacked survivor
when at most b copies remain; then use full-window single-copy loss selection.
Otherwise average no more than the surviving pool's p-b lower bound, capped at b.
This is not a general Byzantine guarantee: local loss can still be deceived.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.sharp_math import peak_screen,ranked_indices,hop_coefficient
from basil_core.sharp_protocol import initial_state,evaluate,vector_hash,link_vector

REVISION='sharp_survivor_guard_20261027_v1'


def guarded_configuration(base):
    from basil_core.sharp_study import validate_config as validate_base
    validate_base(base)
    if base['method']!='sharp':raise ValueError('This correction is a separate SHARP family.')
    cfg=dict(base);cfg.pop('runId')
    cfg.update(baseRunId=base['runId'],baseRevision=base['revision'],revision=REVISION,
        method='sharp_guarded',survivorGuard='full_window_single_when_p_le_b_else_min_b_p_minus_b')
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
    cfg['runId']=f'{base["dataset"]}-{base["partition"]}-{base["seed"]}-{base["condition"]}-s{base["sigma"]:g}-guarded-{digest}'
    return cfg


def validate_config(cfg):
    base=dict(cfg)
    base['runId']=base.pop('baseRunId');base['revision']=base.pop('baseRevision')
    base.pop('survivorGuard');base['method']='sharp'
    if guarded_configuration(base)!=cfg:raise ValueError('Guarded configuration differs from its declared revision.')


def fingerprint():
    from scripts.run_sharp_study import fingerprint as base_fingerprint,ROOT
    result=base_fingerprint()
    for name in ('basil_core/sharp_guarded_protocol.py','scripts/run_guarded_sharp_case.py'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def check_source(expected):
    if fingerprint()!=expected:raise ValueError('Guarded development source changed; use a new frozen output.')


def select_input(cfg,state,worker,x,y,receiver,round_id,first_batch):
    if not cfg['switches']['byzantine_mitigation']:
        from basil_core.sharp_protocol import select_input as base_select
        return base_select(cfg,state,worker,x,y,receiver,round_id,first_batch)
    n=cfg['nodes'];size=cfg['window'];b=cfg['byzantineBudget']
    senders=[(receiver-distance)%n for distance in range(1,size+1)
        if state['senderRounds'][(receiver-distance)%n]>=0]
    if not senders:
        return state['nodeParams'][receiver].copy(),dict(selectedSenders=[],selectedCount=1,
            availableCopies=0,bootstrap=True,screenAllFailFallback=False,screenApplied=False,ranking=[])
    vectors=np.stack([link_vector(state['outbound'][s],cfg['seed'],int(state['senderRounds'][s]),
        s,receiver,cfg['sigma']) for s in senders])
    passed,screen=peak_screen(vectors,full_window=len(senders)==size)
    # The raw pool is empty if the v1 helper applied its all-fail keep-all rule.
    raw=np.empty(0,np.int64) if screen['screenAllFailFallback'] else passed
    fallback=len(raw)<=b
    candidates=np.arange(len(senders)) if fallback else raw
    wanted=1 if fallback else min(b,len(raw)-b)
    losses=[]
    for j in candidates:
        try:loss=worker.probe(vectors[j],x[first_batch],y[first_batch])
        except (FloatingPointError,ValueError):loss=float('inf')
        losses.append(loss)
    chosen=ranked_indices(losses,[senders[j] for j in candidates],
        [state['senderRounds'][senders[j]] for j in candidates],receiver,n,wanted)
    indices=candidates[chosen]
    mean=vectors[indices].astype(np.float64).mean(axis=0).astype(np.float32)
    evidence=dict(screen,selectedSenders=[senders[j] for j in indices],selectedCount=len(indices),
        availableCopies=len(senders),bootstrap=False,targetCount=wanted,
        rawSurvivorCount=len(raw),survivorUnattackedLowerBound=max(0,len(raw)-b),
        screenGuardFallback=fallback,
        ranking=[dict(sender=senders[j],loss=float(loss) if np.isfinite(loss) else None)
            for j,loss in zip(candidates,losses)])
    return mean,evidence


def run(cfg,worker,x,y,partitions,vx,vy,*,state=None,on_activation=None,on_round=None,
        checkpoint=None,should_stop=None,max_activations=None):
    from basil_core.research_protocol import keyed_rng,resolve_attackers
    from basil_core.attacks import applyAttack
    from basil_core.sharp_guarded_protocol import validate_config
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
                if cfg['method']=='sharp_guarded' else cfg['sigma']**2) if cfg['switches']['noise_mitigation'] else 0.
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

