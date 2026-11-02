"""Experimental quality-verified commit of a coherent local SGD model update.

The same SGD steps are computed. A checked full-parameter delta is committed
at the largest allowed scale; reference admission is transactional with it.
"""
import hashlib
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.cart_prior_neutral_routing import selector as route_selector
from basil_core.cart_quality_routing_candidate import utility
from basil_core.cart_class_skill import class_skill as class_f1
from basil_core.sharp_cart import probes


def selector(original,mode,*,peer_budget=8,guard_fallback=False):
    choose=route_selector(original,mode,peer_budget=peer_budget,guard_fallback=guard_fallback)
    def select(cfg,state,worker,x,y,node,round_id,batch):
        chosen,row=choose(cfg,state,worker,x,y,node,round_id,batch)
        record=row['continuity'];worker.quality_context=None
        if record.get('qualityScoreBoundApplied'):
            fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass']);classes=np.asarray(record['supportedClasses'])
            fit=fit[np.isin(y[fit],classes)];check=check[np.isin(y[check],classes)]
            worker.quality_context=dict(ingress=chosen.copy(),indices=np.concatenate((fit,check)),split=len(fit),
                classes=classes,target=np.asarray(record['classTargets']),qualified=np.isin(classes,record['qualifiedClasses']),
                allowance=np.asarray(record['channelF1Allowance']),penalty=mode=='class',baseline=record['selectedScore'])
        return chosen,row
    return select


def commit(worker,proposal,context,x,y):
    def score(flat):
        logits=worker.outputs(flat,x[context['indices']]);n=context['split'];labels=y[context['indices']]
        return utility(class_f1(logits[:n],labels[:n],context['classes']),class_f1(logits[n:],labels[n:],context['classes']),
            context['target'],context['qualified'],context['allowance'],penalty=context['penalty'])
    ingress=context['ingress'];baseline=score(ingress)
    if abs(baseline-context['baseline'])>1e-12:raise ValueError('Ingress class-quality evidence changed.')
    trials=[]
    for scale in (1.,.5,.25,.125,0.):
        candidate=proposal.copy() if scale==1. else (ingress.copy() if not scale else (ingress+np.float32(scale)*(proposal-ingress)).astype(np.float32))
        value=score(candidate);accepted=value>=baseline-.01-1e-12
        trials.append(dict(scale=scale,score=value,accepted=accepted))
        if accepted:
            return candidate,dict(applied=True,scale=scale,baselineScore=baseline,committedScore=value,
                scoreAllowance=.01,trials=trials,optimizerStepsAdded=0)
    raise AssertionError('Ingress fallback must satisfy its own quality bound.')


def transport(original):
    def apply(worker,proposal,previous,x,y,node,activation):
        corrected,memory,evidence=original(worker,proposal,previous,x,y,node,activation)
        context=getattr(worker,'quality_context',None)
        if context is None:return corrected,memory,dict(evidence,qualityCommit=dict(applied=False,reason='receiver recognition not established'))
        chosen,record=commit(worker,corrected,context,x,y)
        end=worker.dimension-sum(worker.sizes[-2:])
        evidence=dict(evidence,qualityCommit=record,
            baseCartBackboneBeforeHash=evidence.get('backboneBeforeHash'),baseCartBackboneAfterHash=evidence.get('backboneAfterHash'),
            backboneBeforeHash=hashlib.sha256(context['ingress'][:end].tobytes()).hexdigest(),
            backboneAfterHash=hashlib.sha256(chosen[:end].tobytes()).hexdigest(),
            interpretation='Full local update commit after the original classifier repair; no classifier-only backbone invariant is claimed.')
        # A scaled model is not the model whose reference was evaluated above.
        # Keep the previous reference rather than admitting inconsistent metadata.
        if record['scale']<1.:
            memory=previous;evidence.update(referenceAdmitted=False)
        worker.load(chosen)
        return chosen,memory,evidence
    return apply
