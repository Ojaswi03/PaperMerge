"""Separately versioned correctness-qualified CART on the guarded SHARP ring."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.sharp_protocol import initial_state,evaluate,vector_hash
from basil_core.sharp_guarded_protocol import select_input
from basil_core.sharp_math import hop_coefficient
REVISION='sharp_cart_correctness_20261027_v2'


def configuration(base):
    from basil_core.sharp_guarded_protocol import validate_config as validate_guard
    validate_guard(base)
    cfg=dict(base);cfg.pop('runId')
    cfg.update(guardedRunId=base['runId'],guardedRevision=base['revision'],
        revision=REVISION,method='sharp_cart_v2',cartMode='correctness_qualified_disagreement_transport_v2')
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
    cfg['runId']=f'{base["dataset"]}-{base["partition"]}-{base["seed"]}-{base["condition"]}-s{base["sigma"]:g}-cartv2-{digest}'
    return cfg


def validate_config(cfg):
    base=dict(cfg);base['runId']=base.pop('guardedRunId');base['revision']=base.pop('guardedRevision')
    base['method']='sharp_guarded';base['cartMode']='coherent_classifier_residual_transport'
    if configuration(base)!=cfg:raise ValueError('CART v2 configuration differs from its declared revision.')


def fingerprint():
    from basil_core.sharp_guarded_protocol import fingerprint as guarded_fingerprint
    root=Path(__file__).resolve().parents[1];result=guarded_fingerprint()
    for name in ('basil_core/sharp_cart_v2.py','basil_core/sharp_cart_v2_protocol.py','scripts/run_cart_v2_study.py'):
        result[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    return result


def check_source(expected):
    if fingerprint()!=expected:raise ValueError('CART v2 source changed; use a fresh frozen output.')


def run(cfg,worker,x,y,partitions,vx,vy,*,state=None,on_activation=None,on_round=None,
        checkpoint=None,should_stop=None,max_activations=None):
    from basil_core.research_protocol import keyed_rng,resolve_attackers
    from basil_core.attacks import applyAttack
    from basil_core.sharp_cart_v2_protocol import validate_config
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
                if cfg['method']=='sharp_cart_v2' else cfg['sigma']**2) if cfg['switches']['noise_mitigation'] else 0.
            trained,training=worker.train(incoming,lx,ly,r,node,coefficient,batch,should_stop)
            cart=None;memory=None
            if cfg['cart']:
                from basil_core.sharp_cart_v2 import transport
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
