"""Experimental CART commitment using continuous class-balanced probe loss.

Incoming admission remains the original class-MCC rule. Only the complete-update
gate changes: both private probes must have no larger macro class CE than the
admitted initialization. This is an empirical reused-probe check, not a bound on
unseen examples. No extra SGD, examples, or evaluation data are introduced.
"""
import hashlib
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')


def balanced_loss(logits, labels, classes):
    z=np.asarray(logits,np.float64);labels=np.asarray(labels)
    if z.ndim!=2 or len(z)!=len(labels) or not len(labels):
        raise ValueError('Nonempty aligned probe logits and labels required.')
    classes=np.asarray(classes)
    if not len(classes) or len(np.unique(classes))!=len(classes):
        raise ValueError('Distinct supported classes required.')
    if any(not np.any(labels==c) for c in classes):
        raise ValueError('Every supported class requires probe examples.')
    if not np.isfinite(z).all():return float('inf')
    shifted=z-z.max(axis=1,keepdims=True)
    losses=np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)),labels]
    return float(np.mean([losses[labels==c].mean() for c in classes]))


def commit(worker, proposal, context, x, y):
    indices=context['indices'];split=context['split'];labels=y[indices]
    def scores(flat):
        logits=worker.outputs(flat,x[indices])
        return np.asarray([balanced_loss(logits[:split],labels[:split],context['classes']),
                           balanced_loss(logits[split:],labels[split:],context['classes'])])
    ingress=context['ingress'];baseline=scores(ingress)
    if not np.isfinite(baseline).all():raise ValueError('Finite admitted probe losses required.')
    trials=[]
    for scale in (1.,.5,.25,.125,0.):
        candidate=proposal.copy() if scale==1. else (ingress.copy() if not scale else
            (ingress+np.float32(scale)*(proposal-ingress)).astype(np.float32))
        value=scores(candidate);accepted=bool(np.isfinite(value).all() and np.all(value<=baseline+1e-10))
        trials.append(dict(scale=scale,probeLosses=value.tolist() if np.isfinite(value).all() else None,accepted=accepted))
        if accepted:
            return candidate,dict(applied=True,scale=scale,criterion='both_private_probe_macro_class_cross_entropy',
                baselineProbeLosses=baseline.tolist(),committedProbeLosses=value.tolist(),
                baselineScore=-float(baseline.max()),committedScore=-float(value.max()),
                scoreAllowance=1e-10,trials=trials,optimizerStepsAdded=0)
    raise AssertionError('Unchanged ingress must satisfy its own finite probe loss bound.')


def transport(original):
    def apply(worker,proposal,previous,x,y,node,activation):
        corrected,memory,evidence=original(worker,proposal,previous,x,y,node,activation)
        context=getattr(worker,'quality_context',None)
        if context is None:
            return corrected,memory,dict(evidence,qualityCommit=dict(applied=False,reason='receiver recognition not established'))
        chosen,record=commit(worker,corrected,context,x,y)
        end=worker.dimension-sum(worker.sizes[-2:])
        evidence=dict(evidence,qualityCommit=record,
            baseCartBackboneBeforeHash=evidence.get('backboneBeforeHash'),
            baseCartBackboneAfterHash=evidence.get('backboneAfterHash'),
            backboneBeforeHash=hashlib.sha256(context['ingress'][:end].tobytes()).hexdigest(),
            backboneAfterHash=hashlib.sha256(chosen[:end].tobytes()).hexdigest(),
            interpretation='Full-update check using balanced CE on both reused private probes; no population retention guarantee.')
        if record['scale']<1.:
            memory=previous;evidence['referenceAdmitted']=False
        worker.load(chosen)
        return chosen,memory,evidence
    return apply
