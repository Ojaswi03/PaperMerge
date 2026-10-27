"""Numerical prototype: project a completed activation displacement.

Memory-gradient halfspace projection is established in GEM/GEC. This primitive
is not a novelty claim. Class CE gradients select a surrogate feasible cone;
actual nonlinear class-risk checks are still necessary before model commit.
"""
from __future__ import annotations


def project_displacement(displacement,class_gradients,*,tolerance=1e-8,max_sweeps=4000):
    """Approximately minimize ||v-displacement||²/2 subject to G v <= 0.

Rows are positively normalized without changing the cone. Cyclic dual
coordinate descent solves the small PSD Gram problem, never a d-by-d system.
Zero is always feasible. A failed numerical solve returns zero explicitly,
not an unverified approximate projection. The caller must measure the actual
nonlinear risk of its proposed model; this cone supplies no risk certificate.
"""
    import numpy as np
    d=np.asarray(displacement,dtype=np.float64)
    gradients=np.asarray(class_gradients,dtype=np.float64)
    if (d.ndim!=1 or not len(d) or gradients.ndim!=2 or gradients.shape[1]!=len(d)
            or not np.isfinite(d).all() or not np.isfinite(gradients).all()
            or not np.isfinite(tolerance) or tolerance<=0 or type(max_sweeps) is not int or max_sweeps<1):
        raise ValueError('Finite displacement, matching gradient rows and positive solver controls required.')
    norms=np.linalg.norm(gradients,axis=1);rows=gradients[norms>1e-12]/norms[norms>1e-12,None]
    scale=max(1.,float(np.linalg.norm(d)))
    if not len(rows):
        return d.copy(),dict(status='no_nonzero_constraints',sweeps=0,primalViolation=0.,complementarity=0.,rows=0)
    gram=rows@rows.T;a=rows@d;dual=np.zeros(len(rows));residual=a.copy()
    for sweep in range(1,max_sweeps+1):
        for i in range(len(rows)):
            value=max(0.,dual[i]+residual[i]/gram[i,i]);change=value-dual[i]
            dual[i]=value;residual-=change*gram[:,i]
        violation=float(np.maximum(residual,0.).max())
        complementarity=float(np.abs(dual*residual).max())
        if violation<=tolerance*scale and complementarity<=tolerance*scale**2:
            projected=d-rows.T@dual
            actual_violation=float(np.maximum(rows@projected,0.).max())
            if np.isfinite(projected).all() and actual_violation<=2*tolerance*scale:
                return projected,dict(status='projected',sweeps=sweep,primalViolation=actual_violation,
                    complementarity=complementarity,rows=len(rows),correctionNorm=float(np.linalg.norm(projected-d)))
    return np.zeros_like(d),dict(status='fallback_zero',sweeps=max_sweeps,primalViolation=0.,
        complementarity=None,rows=len(rows),correctionNorm=float(np.linalg.norm(d)))
