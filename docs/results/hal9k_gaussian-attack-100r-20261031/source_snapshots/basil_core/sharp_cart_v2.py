"""Correctness-qualified CART: receiver-owned, disagreement-only head repair.

Both fitting and checking use local TRAIN examples. A fixed check probe can
reject local damage; it supplies no population or Byzantine guarantee.
"""
from __future__ import annotations

import hashlib
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.sharp_cart import probes
from basil_core.sharp_math import class_ce


def correct_counts(logits,labels,classes):
    y=np.asarray(labels);pred=np.asarray(logits).argmax(axis=1)
    return np.bincount(y[pred==y],minlength=classes)


def fit_correction(features,proposal,reference,labels,gates,fraction):
    """Balanced ridge fit: nonzero targets only for correct-to-wrong examples."""
    h=np.asarray(features,np.float64);p=np.asarray(proposal,np.float64);r=np.asarray(reference,np.float64)
    y=np.asarray(labels);g=np.asarray(gates,np.float64)
    class_ce(p,y);class_ce(r,y)
    if (p.shape!=r.shape or h.ndim!=2 or len(h)!=len(y) or g.shape!=(p.shape[1],)
            or not np.isfinite(h).all() or not np.isfinite(g).all() or np.any(g<0)
            or np.any(g>1) or not np.isfinite(fraction) or fraction<=0):
        raise ValueError('Invalid correctness-qualified transport arrays.')
    mask=(r.argmax(axis=1)==y)&(p.argmax(axis=1)!=y)&(g[y]>0)
    residual=(r-r.mean(axis=1,keepdims=True))-(p-p.mean(axis=1,keepdims=True))
    target=(mask*g[y])[:,None]*residual
    a=np.column_stack((h,np.ones(len(h))));counts=np.bincount(y,minlength=p.shape[1])
    weights=1./(np.count_nonzero(counts)*counts[y])
    gram=a.T@(weights[:,None]*a);penalty=float(fraction*np.trace(gram)/len(gram))
    correction=np.linalg.solve(gram+penalty*np.eye(len(gram)),a.T@(weights[:,None]*target))
    initial=float(np.sum(weights[:,None]*target**2))
    fitted=float(np.sum(weights[:,None]*(a@correction-target)**2)+penalty*np.sum(correction**2))
    if not np.isfinite(correction).all() or fitted>initial+1e-8*max(1.,initial):
        raise FloatingPointError('Invalid CART v2 ridge solution.')
    return correction,dict(ridgePenalty=penalty,zeroObjective=initial,fittedObjective=fitted,
        repairedFitExamples=int(mask.sum()),referenceMistakeTargets=0)


def transport(worker,proposal,previous,x,y,node,activation):
    from basil_core.research_protocol import keyed_rng
    cfg=worker.cfg;fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass'])
    fy=y[fit];cy=y[check];k=cfg['classes'];classes=np.unique(fy)
    p=worker.outputs(proposal,x[fit]);_,lp=class_ce(p,fy)
    backbone_end=worker.dimension-sum(worker.sizes[-2:]);before=hashlib.sha256(proposal[:backbone_end].tobytes()).hexdigest()
    final=proposal.copy();gates=np.zeros(k)
    evidence=dict(mode='correctness_qualified_disagreement_transport_v2',scale=0.,classGates=gates.tolist(),
        fitProbes=len(fit),checkProbes=len(check),fitCheckDisjoint=not bool(np.intersect1d(fit,check).size),
        referenceUsed=previous is not None,backboneBeforeHash=before,backboneAfterHash=before,
        checkBefore=None,checkAfter=None,checkCorrectBefore=None,checkCorrectAfter=None,
        repairedFitExamples=0,referenceMistakeTargets=0,trials=[])
    cp=worker.outputs(proposal,x[check]) if len(check) else None
    if previous is not None and len(check):
        r=worker.outputs(previous['params'],x[fit]);_,lr=class_ce(r,fy)
        if not np.allclose(lr,previous['losses'],rtol=1e-5,atol=1e-6):
            raise ValueError('Stored CART v2 reference losses changed.')
        rc=worker.outputs(previous['params'],x[check])
        pc=correct_counts(p,fy,k);rr=correct_counts(r,fy,k)
        cc=correct_counts(cp,cy,k);cr=correct_counts(rc,cy,k)
        excess=np.maximum(lp-lr-np.asarray(previous['allowance']),0.)
        for j,c in enumerate(classes):
            if np.any(cy==c) and rr[c]>pc[c] and cr[c]>cc[c]:
                gates[c]=excess[j]/(excess[j]+previous['allowance'][j]+1e-6)
        baseline=float(class_ce(cp,cy)[1].mean());best=baseline;scale=0.;best_gain=0
        evidence.update(checkBefore=baseline,checkAfter=baseline,checkCorrectBefore=cc.tolist(),checkCorrectAfter=cc.tolist())
        if np.any(gates>0):
            h=worker.outputs(proposal,x[fit],features=True)
            correction,ridge=fit_correction(h,p,r,fy,gates,cfg['cartRidgeFraction'])
            hc=worker.outputs(proposal,x[check],features=True);a=np.column_stack((hc,np.ones(len(hc))))
            for step in (.25,.5,.75,1.):
                logits=cp+step*(a@correction);loss=float(class_ce(logits,cy)[1].mean())
                counts=correct_counts(logits,cy,k);gain=int((counts-cc).sum())
                eligible=bool(np.all(counts>=cc) and loss<baseline-1e-4)
                evidence['trials'].append(dict(scale=step,macroCE=loss,correctCounts=counts.tolist(),eligible=eligible))
                if eligible and (gain>best_gain or gain==best_gain and loss<best):
                    best=loss;best_gain=gain;scale=step
            weights=worker.tensors(proposal)
            weights[-2]=weights[-2]+scale*correction[:-1];weights[-1]=weights[-1]+scale*correction[-1]
            candidate=np.concatenate([v.ravel() for v in weights]).astype(np.float32)
            actual=worker.outputs(candidate,x[check]);loss=float(class_ce(actual,cy)[1].mean())
            counts=correct_counts(actual,cy,k)
            if scale and np.all(counts>=cc) and loss<baseline-1e-4:
                final=candidate;evidence.update(scale=scale,checkAfter=loss,checkCorrectAfter=counts.tolist())
            evidence.update(ridge)
        evidence.update(classGates=gates.tolist(),fitReferenceCorrect=rr.tolist(),fitProposalCorrect=pc.tolist(),
            checkReferenceCorrect=cr.tolist())
    if not np.array_equal(final[:backbone_end],proposal[:backbone_end]):
        raise AssertionError('CART v2 changed backbone parameters.')
    final_logits=worker.outputs(final,x[fit]);_,losses=class_ce(final_logits,fy)
    admitted=False;memory=previous
    if len(check):
        final_check=worker.outputs(final,x[check]);_,check_losses=class_ce(final_check,cy)
        counts=correct_counts(final_check,cy,k)
        admitted=bool(previous is None or (np.all(counts>=np.asarray(previous['checkCorrect']))
            and check_losses.mean()<np.mean(previous['checkLosses'])-1e-4))
        if admitted:
            denominator=np.maximum(losses,.1);scores=[]
            for draw in range(cfg['calibrationDraws']):
                if not cfg['sigma']:scores.append(0.);continue
                noise=keyed_rng(cfg['seed'],'sharp_cart_channel',node,activation,draw).normal(0,cfg['sigma'],size=final.shape).astype(np.float32)
                _,noised=class_ce(worker.outputs(final+noise,x[fit]),fy)
                scores.append(float(np.maximum((noised-losses)/denominator,0.).max()))
            memory=dict(params=final.copy(),losses=losses.tolist(),allowance=(max(scores)*denominator).tolist(),
                classes=classes.tolist(),activation=activation,calibrationScores=scores,
                checkLosses=check_losses.tolist(),checkCorrect=counts.tolist())
    evidence.update(referenceAdmitted=admitted,activeGateClasses=np.flatnonzero(gates>0).tolist())
    worker.load(final)
    return final,memory,evidence
