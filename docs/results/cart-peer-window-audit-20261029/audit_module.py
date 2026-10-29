"""Read-only CART score decomposition and frozen-window diagnostic summaries."""
import numpy as np


def decompose(fit,check,target,qualified,allowance):
    f,k,t,q,a=[np.asarray(v) for v in (fit,check,target,qualified,allowance)]
    if not len(f) or any(v.shape!=f.shape for v in (k,t,q,a)):
        raise ValueError('Aligned nonempty class evidence required.')
    if not all(np.isfinite(v).all() for v in (f,k,t,a)) or np.any(a<0):
        raise ValueError('Finite class evidence and nonnegative allowance required.')
    macro=min(float(f.mean()),float(k.mean()))
    penalty=float((np.maximum(t-np.minimum(f,k)-a,0)*q).mean())
    return dict(macro=macro,penalty=penalty,score=macro-penalty,
                fit=f.tolist(),check=k.tolist())


def cohort(peers,own,mode,*,screened=False,near=False):
    rows=[p for p in peers if (not screened or p['screenPassed']) and (not near or p['nearEight'])]
    if not rows:return dict(candidates=0,bestScoreMargin=None,bestMacroMargin=None)
    best=max(rows,key=lambda p:p[mode]['score'])
    best_macro=max(rows,key=lambda p:p[mode]['macro'])
    return dict(candidates=len(rows),bestSender=best['sender'],
        bestScoreMargin=best[mode]['score']-own['score'],
        bestMacroMargin=best_macro[mode]['macro']-own['macro'],
        bestCandidateMacro=best[mode]['macro'],bestCandidatePenalty=best[mode]['penalty'])


def summarize(receivers):
    result={}
    for mode in ('received','preLink'):
        for name in ('nearEight','screenedWindow','wholeWindow'):
            values=[r['cohorts'][mode][name] for r in receivers if r.get('cohorts')]
            supported=[v for v in values if v['bestScoreMargin'] is not None]
            score=[v['bestScoreMargin'] for v in supported]
            macro=[v['bestMacroMargin'] for v in supported]
            result[mode+'_'+name]=dict(receivers=len(supported),
                peerBeatsOwned=sum(v>0 for v in score),peerWithinOriginalBand=sum(v>=-.01 for v in score),
                macroPeerBeatsOwned=sum(v>0 for v in macro),
                scoreMarginQuantiles=np.quantile(score,[0,.25,.5,.75,1]).tolist() if score else [])
    return result
