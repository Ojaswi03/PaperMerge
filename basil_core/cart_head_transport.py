"""Class-aware residual transport into a linear classifier.

Ridge regression and logit distillation are established techniques. This
primitive makes no novelty or population-accuracy claim. Its zero correction
and a separate local probe check are essential safeguards.
"""
from __future__ import annotations


def fit_residual(features,proposal_logits,reference_logits,labels,gates,*,ridge_fraction=.01):
    import numpy as np
    h=np.asarray(features,dtype=np.float64);p=np.asarray(proposal_logits,dtype=np.float64)
    r=np.asarray(reference_logits,dtype=np.float64);y=np.asarray(labels);g=np.asarray(gates,dtype=np.float64)
    if (h.ndim!=2 or p.shape!=(len(h),10) or r.shape!=p.shape or y.shape!=(len(h),)
            or not len(h) or not np.issubdtype(y.dtype,np.integer) or np.any(y<0) or np.any(y>=10)
            or g.shape!=(10,) or np.any(g<0) or np.any(g>1)
            or not all(np.isfinite(v).all() for v in (h,p,r,g))
            or not np.isfinite(ridge_fraction) or ridge_fraction<=0):
        raise ValueError('Finite aligned features/logits, local labels and bounded class gates required.')
    a=np.column_stack((h,np.ones(len(h))))
    counts=np.bincount(y,minlength=10);classes=np.flatnonzero(counts)
    weights=1./(len(classes)*counts[y])
    # Softmax is invariant to a common logit shift: transfer centered logits.
    residual=r-r.mean(axis=1,keepdims=True)-(p-p.mean(axis=1,keepdims=True))
    target=g[y,None]*residual
    gram=a.T@(weights[:,None]*a)
    ridge=float(ridge_fraction*np.trace(gram)/len(gram))
    rhs=a.T@(weights[:,None]*target)
    solution=np.linalg.solve(gram+ridge*np.eye(len(gram)),rhs)
    zero=float(np.sum(weights[:,None]*target**2))
    fitted=float(np.sum(weights[:,None]*(a@solution-target)**2)+ridge*np.sum(solution**2))
    if not np.isfinite(solution).all() or fitted>zero+1e-8*max(1.,zero):
        raise ValueError('Residual transport failed its regularized least-squares check.')
    return solution,dict(ridge=ridge,zeroObjective=zero,fittedObjective=fitted,
        activeClasses=classes[g[classes]>0].tolist(),correctionNorm=float(np.linalg.norm(solution)))


def macro_ce(logits,labels):
    import numpy as np
    from basil_core.cart_feasible_memory import class_losses
    return float(np.mean(class_losses(logits,np.asarray(labels))[1]))


def choose_scale(logits,features,labels,correction,*,tolerance=1e-4):
    import numpy as np
    a=np.column_stack((np.asarray(features,dtype=np.float64),np.ones(len(features))))
    base=macro_ce(logits,labels);best=base;scale=0.;trials=[]
    for alpha in (.25,.5,.75,1.):
        loss=macro_ce(np.asarray(logits)+alpha*(a@correction),labels)
        trials.append(dict(scale=alpha,meanClassCE=loss))
        if loss<best-tolerance:best=loss;scale=alpha
    return scale,dict(checkBefore=base,checkAfter=best,trials=trials)
