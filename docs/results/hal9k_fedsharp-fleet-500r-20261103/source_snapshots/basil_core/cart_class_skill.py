"""Standard one-vs-rest MCC, used as chance-adjusted class recognition.

This statistic is prior work, not a novel metric. A constant class prediction
has zero skill even when its class recall is 100 percent. Degenerate classes
with no observable disagreement opportunity have zero evidence of skill.
"""
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')


def class_skill(logits,labels,classes,*,metric='mcc'):
    if metric not in ('mcc','kappa'):raise ValueError('Unknown standard class metric.')
    prediction=np.asarray(logits).argmax(axis=1);y=np.asarray(labels);n=len(y)
    if not n:raise ValueError('Class skill needs nonempty private probes.')
    values=[]
    for c in classes:
        positive=y==c;predicted=prediction==c
        rho=float(np.mean(positive));nu=float(np.mean(predicted));joint=float(np.mean(positive&predicted))
        if metric=='mcc':
            denominator=float(np.sqrt(rho*(1-rho)*nu*(1-nu)));numerator=joint-rho*nu
        else:
            denominator=rho+nu-2*rho*nu;numerator=2*(joint-rho*nu)
        values.append(numerator/denominator if denominator>1e-12 else 0.)
    return np.asarray(values,np.float64)
