"""Research candidate: coherent receiver reference and measured class envelopes.

This module supplies numerical primitives, not a validated training method or
novelty claim. GEM is prior work for memory-loss constraints. The finite-sample
rank argument is a standard exchangeability fact, not an original theorem.
"""
from __future__ import annotations

import math

REVISION='cart_coherent_channel_memory_v1'
CALIBRATION_DRAWS=19
SCALE_FLOOR=.1
FEASIBILITY_TOLERANCE=1e-6
BRIDGE_SCALES=(1.,.5,.25,.125,.0625,.03125,.015625,0.)


def class_losses(logits,labels):
    """Raw CE means on the receiver's observed classes only; no invented labels."""
    import numpy as np
    z=np.asarray(logits,dtype=np.float64)
    y=np.asarray(labels)
    if (z.ndim!=2 or z.shape[1]!=10 or y.shape!=(len(z),) or not len(y)
            or not np.issubdtype(y.dtype,np.integer) or np.any(y<0) or np.any(y>=10)
            or not np.isfinite(z).all()):
        raise ValueError('Finite ten-class logits and supported TRAIN labels required.')
    shifted=z-z.max(axis=1,keepdims=True)
    ce=np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(y)),y]
    classes=np.unique(y)
    return classes,np.asarray([ce[y==c].mean() for c in classes])


def calibrate_envelope(reference_losses,noised_losses):
    """Joint class budget from 19 independent simulated channel corruptions.

    With reference/probes fixed before calibration, 19 exchangeable corruption
    scores plus a fresh score yield >=19/20 marginal coverage by their maximum.
    This applies to one fresh corruption of THAT reference under THAT channel.
    It says nothing about Byzantine models, generalization, adaptive model
    selection, repeated simultaneous coverage, or unknown/shifted channels.
    """
    import numpy as np
    reference=np.asarray(reference_losses,dtype=np.float64)
    samples=np.asarray(noised_losses,dtype=np.float64)
    if (reference.ndim!=1 or not 1<=len(reference)<=10 or samples.shape!=(CALIBRATION_DRAWS,len(reference))
            or not np.isfinite(reference).all() or not np.isfinite(samples).all()
            or np.any(reference<0) or np.any(samples<0)):
        raise ValueError('Nonnegative class losses and exactly 19 independent draw rows required.')
    scales=np.maximum(reference,SCALE_FLOOR)
    scores=np.maximum((samples-reference)/scales,0.).max(axis=1)
    gamma=float(scores.max())
    return dict(referenceLosses=reference.tolist(),classScales=scales.tolist(),
        jointScores=scores.tolist(),gamma=gamma,bounds=(reference+gamma*scales).tolist(),
        calibrationDraws=CALIBRATION_DRAWS,marginalReferenceChannelCoverage=CALIBRATION_DRAWS/(CALIBRATION_DRAWS+1))


def within_envelope(losses,bounds,*,tolerance=FEASIBILITY_TOLERANCE):
    import numpy as np
    current=np.asarray(losses,dtype=np.float64)
    limits=np.asarray(bounds,dtype=np.float64)
    if (current.ndim!=1 or not len(current) or current.shape!=limits.shape
            or not np.isfinite(current).all() or not np.isfinite(limits).all()
            or np.any(current<0) or np.any(limits<0) or not math.isfinite(tolerance) or tolerance<0):
        raise ValueError('Finite matching class losses/bounds and nonnegative tolerance required.')
    return bool(np.all(current<=limits+tolerance))


def restore_feasible(anchor,proposal,bounds,evaluate,*,scales=BRIDGE_SCALES):
    """Keep the largest measured-feasible point in a finite segment grid.

    Feasibility is checked on actual cached-class CE, not a linearization.
    The caller supplies a feasible anchor: a coherent stored reference at
    reception, or the previous accepted weights for an optimizer proposal.
    No convexity of the neural-network loss is assumed. This is not an exact
    Euclidean projection or a guarantee of the best feasible model.
    """
    import numpy as np
    a=[np.asarray(v,dtype=np.float32) for v in anchor]
    p=[np.asarray(v,dtype=np.float32) for v in proposal]
    if (not a or len(a)!=len(p) or any(x.shape!=y.shape for x,y in zip(a,p))
            or any(not np.isfinite(v).all() for v in a+p)):
        raise ValueError('Matching finite anchor/proposal parameter tensors required.')
    scales=tuple(scales)
    if (not scales or scales[-1]!=0. or scales[0]!=1. or
            any(not math.isfinite(v) or not 0<=v<=1 for v in scales) or
            any(x<=y for x,y in zip(scales,scales[1:]))):
        raise ValueError('Use a decreasing finite grid from one through zero.')
    trials=[]
    for scale in scales:
        # Preserve endpoints exactly; 0*(proposal-anchor) can overflow for large
        # finite inputs and would invalidate the feasibility witness.
        candidate=([v.copy() for v in p] if scale==1 else [v.copy() for v in a] if scale==0
            else [(x.astype(np.float64)+scale*(y.astype(np.float64)-x)).astype(np.float32) for x,y in zip(a,p)])
        losses=np.asarray(evaluate(candidate),dtype=np.float64)
        accepted=within_envelope(losses,bounds)
        trials.append(dict(scale=scale,classLosses=losses.tolist(),accepted=accepted))
        if accepted:return candidate,dict(scale=scale,trials=trials,classLosses=losses.tolist())
    raise ValueError('The supplied anchor is not feasible; no update may be committed.')


def coherent_admission(previous_losses,candidate_losses,*,improvement=1e-4,bounds=None):
    """One whole reference replaces another only on supported-class dominance.

    A single witness avoids stitching incompatible class minima from different
    historical models. Dominance is a local empirical admission rule; it does
    not certify accuracy, unbiased generalization, or absence of poisoning.
    """
    import numpy as np
    before=np.asarray(previous_losses,dtype=np.float64)
    after=np.asarray(candidate_losses,dtype=np.float64)
    if (before.ndim!=1 or before.shape!=after.shape or not len(before)
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or np.any(before<0) or np.any(after<0) or not math.isfinite(improvement) or improvement<=0):
        raise ValueError('Finite supported-class losses and positive improvement required.')
    feasible=np.all(after<=before) if bounds is None else within_envelope(after,bounds)
    return bool(feasible and after.mean()<before.mean()-improvement)
