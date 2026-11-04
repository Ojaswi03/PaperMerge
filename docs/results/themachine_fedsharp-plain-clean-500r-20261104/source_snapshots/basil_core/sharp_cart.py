"""Receiver-owned CART classifier transport, generalized to 100 classes/nodes.

This is a post-SHARP ablation. No registry, official TEST data or peer data enter
the class gates, fitting, checking or reference admission.
"""
from __future__ import annotations

import hashlib
from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')

from basil_core.sharp_math import class_ce,ridge_transport


def probes(labels,seed,node,limit=16):
    from basil_core.research_protocol import keyed_rng
    fit=[];check=[]
    for c in np.unique(labels):
        members=np.flatnonzero(labels==c)
        order=keyed_rng(seed,'sharp_cart_probe',node,int(c)).permutation(members)
        amount=min(limit,max(1,len(order)//2))
        fit.extend(order[:amount]);check.extend(order[amount:amount+limit])
    return np.asarray(fit,np.int64),np.asarray(check,np.int64)


def transport(worker,proposal,previous,x,y,node,activation):
    from basil_core.research_protocol import keyed_rng
    cfg=worker.cfg;fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass'])
    fy=y[fit];classes=np.unique(fy);p=worker.outputs(proposal,x[fit]);_,lp=class_ce(p,fy)
    backbone_end=worker.dimension-sum(worker.sizes[-2:])
    before_hash=hashlib.sha256(proposal[:backbone_end].tobytes()).hexdigest()
    final=proposal.copy()
    evidence=dict(scale=0.,classGates=[0.]*cfg['classes'],fitProbes=len(fit),checkProbes=len(check),
        fitCheckDisjoint=not bool(np.intersect1d(fit,check).size),referenceUsed=previous is not None,
        backboneBeforeHash=before_hash,backboneAfterHash=before_hash,checkBefore=None,checkAfter=None)
    if previous is not None and len(check):
        r=worker.outputs(previous['params'],x[fit]);_,lr=class_ce(r,fy)
        if not np.allclose(lr,previous['losses'],rtol=1e-5,atol=1e-6):
            raise ValueError('Stored reference/probe losses changed.')
        gates=np.zeros(cfg['classes']);gap=np.maximum(lp-lr,0.)
        allowance=np.asarray(previous['allowance'])
        for j,c in enumerate(classes):
            if np.any(y[check]==c) and np.mean(r.argmax(axis=1)[fy==c]==c)>1/cfg['classes']:
                gates[c]=gap[j]/(gap[j]+allowance[j]+1e-6)
        h=worker.outputs(proposal,x[fit],features=True)
        correction,ridge=ridge_transport(h,p,r,fy,gates,cfg['cartRidgeFraction'])
        hc=worker.outputs(proposal,x[check],features=True)
        cp=worker.outputs(proposal,x[check]);baseline=float(class_ce(cp,y[check])[1].mean())
        best=baseline;scale=0.;trials=[]
        augmented=np.column_stack((hc,np.ones(len(hc))))
        for a in (.25,.5,.75,1.):
            loss=float(class_ce(cp+a*(augmented@correction),y[check])[1].mean())
            trials.append(dict(scale=a,macroCE=loss))
            if loss<best-1e-4:best=loss;scale=a
        weights=worker.tensors(proposal)
        weights[-2]=weights[-2]+scale*correction[:-1];weights[-1]=weights[-1]+scale*correction[-1]
        final=np.concatenate([v.ravel() for v in weights]).astype(np.float32)
        actual=float(class_ce(worker.outputs(final,x[check]),y[check])[1].mean())
        if actual>baseline+1e-6:final=proposal.copy();scale=0.;actual=baseline
        evidence.update(ridge,scale=scale,classGates=gates.tolist(),checkBefore=baseline,checkAfter=actual,trials=trials)
    if not np.array_equal(final[:backbone_end],proposal[:backbone_end]):
        raise AssertionError('CART correction modified backbone parameters.')
    final_logits=worker.outputs(final,x[fit]);_,losses=class_ce(final_logits,fy)
    admitted=bool(previous is None or losses.mean()<np.mean(previous['losses'])-1e-4)
    memory=previous
    if admitted:
        scale=np.maximum(losses,.1);scores=[]
        for draw in range(cfg['calibrationDraws']):
            if not cfg['sigma']:
                scores.append(0.)
                continue
            noise=keyed_rng(cfg['seed'],'sharp_cart_channel',node,activation,draw).normal(0,cfg['sigma'],size=final.shape).astype(np.float32)
            noised=worker.outputs(final+noise,x[fit]);_,noised_losses=class_ce(noised,fy)
            scores.append(float(np.maximum((noised_losses-losses)/scale,0.).max()))
        memory=dict(params=final.copy(),losses=losses.tolist(),allowance=(max(scores)*scale).tolist(),
            classes=classes.tolist(),activation=activation,calibrationScores=scores)
    evidence.update(referenceAdmitted=admitted,activeGateClasses=np.flatnonzero(np.asarray(evidence['classGates'])>0).tolist())
    worker.load(final)
    return final,memory,evidence
