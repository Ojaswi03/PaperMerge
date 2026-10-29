"""Experimental discrimination-qualified class admission plus a collapse guard.

A constant class prediction is not learned class discrimination. The scalar
collapse guard is an explicit ablation so its effect cannot be claimed as the
benefit of class-aware semantic admission.
"""
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.sharp_cart import probes
from basil_core.sharp_math import class_ce
from basil_core.cart_discriminative_continuity_candidate import selector as discriminative_selector


def balanced_accuracy(logits,labels):
    return float(np.mean([np.mean(logits.argmax(axis=1)[labels==c]==c) for c in np.unique(labels)]))


def collapse_check(worker,incoming,own,x,y,node):
    cfg=worker.cfg;fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass']);evidence=[]
    for indices in (fit,check):
        if not len(indices):return False,[]
        pi=worker.outputs(incoming,x[indices]);oi=worker.outputs(own,x[indices]);classes=np.unique(y[indices])
        pa=balanced_accuracy(pi,y[indices]);oa=balanced_accuracy(oi,y[indices]);chance=1/len(classes)
        pc=float(class_ce(pi,y[indices])[1].mean());oc=float(class_ce(oi,y[indices])[1].mean())
        reject=bool(len(classes)>1 and oa>=chance+.05 and oa-pa>.05 and pc-oc>.1)
        evidence.append(dict(ownBalancedAccuracy=oa,incomingBalancedAccuracy=pa,chanceReference=chance,
            ownMacroCE=oc,incomingMacroCE=pc,reject=reject))
    return bool(all(row['reject'] for row in evidence)),evidence


def selector(original,mode):
    if mode not in ('class','guard_only'):raise ValueError('Unknown discrimination ablation.')
    if mode=='guard_only':
        from basil_core.cart_qualified_continuity_candidate import selector as fresh_selector
        choose=fresh_selector(original,'always')
    else:
        # Factory substitution is scoped to constructing one process-local
        # closure. No global remains changed during training or in other workers.
        import basil_core.cart_uncertainty_admission_candidate as uncertainty
        previous=uncertainty.qualified_selector
        try:
            uncertainty.qualified_selector=discriminative_selector
            choose=uncertainty.selector(original,'capped')
        finally:uncertainty.qualified_selector=previous
    def select(cfg,state,worker,x,y,node,round_id,batch):
        chosen,row=choose(cfg,state,worker,x,y,node,round_id,batch);record=dict(row['continuity'])
        if state['senderRounds'][node]>=0 and record['actualRoute']!='own':
            own=state['nodeParams'][node];reject,evidence=collapse_check(worker,chosen,own,x,y,node)
            record['balancedCollapseCheck']=evidence
            if reject:
                chosen=own.copy();record.update(route='own',actualRoute='own',reason='catastrophic balanced-probe regression')
                row.update(selectedSenders=[],selectedCount=1)
        record['discriminationMode']=mode
        row=dict(row,continuity=record);worker.admitted_route=record['actualRoute']
        worker.admitted_has_channel=worker.admitted_route!='own' and row.get('availableCopies',0)>0
        worker.continuity_anchor=chosen.copy()
        return chosen,row
    return select
