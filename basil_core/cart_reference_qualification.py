"""Empirical breadth gate for admitting class-retention anchors.

An early classifier that predicts one label can be locally confident without
being useful for balanced learning. This warm-up/admission rule is not a
statistical significance test, a population certificate or a novelty theorem.
It only concerns classes for which the receiver has its own probe examples.
"""
from __future__ import annotations


def qualified_error_reference(class_errors):
    """Require above-chance recall for a majority of supported classes.

The model always has ten output classes, so uniform-guess recall is 1/10,
including when the receiver observes fewer labels. Require at least two
classes: a single supported label cannot establish cross-class breadth.
    The equal-class mean recall must also exceed uniform-guess recall.
"""
    import numpy as np
    errors=np.asarray(class_errors,dtype=np.float64)
    if (errors.ndim!=1 or not 1<=len(errors)<=10 or not np.isfinite(errors).all()
            or np.any(errors<0) or np.any(errors>1)):
        raise ValueError('Finite supported-class error rates in [0,1] required.')
    required=max(2,(len(errors)+1)//2)
    return bool(errors.mean()<.9 and np.count_nonzero(errors<.9)>=required)
