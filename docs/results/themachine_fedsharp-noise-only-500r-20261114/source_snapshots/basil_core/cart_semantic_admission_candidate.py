"""Experimental pre-SGD semantic admission on a coherent incoming backbone.

Attempt receiver-owned classifier transport before falling back to the carrier.
Ridge fitting is prior work; the repair/admission interaction needs ablation.
"""
import hashlib
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.cart_qualified_continuity_candidate import selector as qualified_selector
from basil_core.sharp_cart import probes
from basil_core.sharp_math import class_ce,ridge_transport
from basil_core.sharp_cart_v2 import correct_counts


def repair(worker,peer,own,x,y,node,record):
    cfg=worker.cfg;fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass'])
    learned=np.asarray(record['learnedClasses']);classes=cfg['classes']
    pf=worker.outputs(peer,x[fit]);pc=worker.outputs(peer,x[check]);teacher=worker.outputs(own,x[fit])
    h=worker.outputs(peer,x[fit],features=True)
    own_features=worker.outputs(own,x[fit],features=True)
    incoming_energy=float(np.var(h.astype(np.float64),axis=0).sum())
    own_energy=float(np.var(own_features.astype(np.float64),axis=0).sum())
    activity=dict(incomingFeatureEnergy=incoming_energy,carrierFeatureEnergy=own_energy,
        minimumRelativeFeatureEnergy=.1)
    if incoming_energy<max(1e-12,.1*own_energy):
        return None,dict(attempted=True,accepted=False,reason='incoming feature activity collapsed',**activity)
    gates=np.zeros(classes);gates[learned]=1.
    targets=teacher.copy();targets[teacher.argmax(axis=1)!=y[fit]]=pf[teacher.argmax(axis=1)!=y[fit]]
    correction,ridge=ridge_transport(h,pf,targets,y[fit],gates,cfg['cartRidgeFraction'])
    mask_f=np.isin(y[fit],learned);mask_c=np.isin(y[check],learned)
    own_fit=np.asarray(record['ownFit']);own_check=np.asarray(record['ownCheck'])
    fb=np.asarray(record['fitBudget']);cb=np.asarray(record['checkBudget'])
    before=correct_counts(pc,y[check],classes)
    protect_new=np.ones(classes,bool);protect_new[learned]=False
    macro_limit=max(float(class_ce(pc,y[check])[1].mean()),
        float(class_ce(worker.outputs(own,x[check]),y[check])[1].mean()))
    end=worker.dimension-sum(worker.sizes[-2:]);backbone=hashlib.sha256(peer[:end].tobytes()).hexdigest()
    best=None;trials=[]
    for scale in (.25,.5,.75,1.):
        weights=[v.copy() for v in worker.tensors(peer)]
        weights[-2]=weights[-2]+scale*correction[:-1];weights[-1]=weights[-1]+scale*correction[-1]
        candidate=np.concatenate([v.ravel() for v in weights]).astype(np.float32)
        # Check the actual rounded model, not only a linear-logit prediction.
        zf=worker.outputs(candidate,x[fit]);zc=worker.outputs(candidate,x[check])
        lf=class_ce(zf[mask_f],y[fit][mask_f])[1];lc=class_ce(zc[mask_c],y[check][mask_c])[1]
        counts=correct_counts(zc,y[check],classes);macro=float(class_ce(zc,y[check])[1].mean())
        accepted=bool(np.all(lf<=own_fit+fb+1e-12) and np.all(lc<=own_check+cb+1e-12)
            and np.all(counts[protect_new]>=before[protect_new]-1) and macro<=macro_limit)
        trials.append(dict(scale=scale,accepted=accepted,macroCheck=macro))
        if accepted and (best is None or macro<best[0]):best=(macro,candidate,lf,lc,scale)
    evidence=dict(attempted=True,accepted=best is not None,trials=trials,learnedClasses=learned.tolist(),
        backboneSha256=backbone,unqualifiedCheckCorrectCountAllowance=1,**activity,**ridge)
    if best is None:return None,evidence
    _,candidate,lf,lc,scale=best
    if not np.array_equal(candidate[:end],peer[:end]):raise AssertionError('Semantic admission changed incoming backbone.')
    evidence.update(scale=scale,protectedFit=lf.tolist(),protectedCheck=lc.tolist(),
        parentRoles=['incoming fresh backbone and classifier','receiver-owned class teacher'])
    return candidate,evidence


def selector(original,mode):
    choose=qualified_selector(original,mode)
    def select(cfg,state,worker,x,y,node,round_id,batch):
        chosen,row=choose(cfg,state,worker,x,y,node,round_id,batch)
        record=dict(row['continuity'])
        if record['actualRoute']=='own' and row.get('poolRankingSkipped') and record.get('learnedClasses'):
            peer=worker.continuity_anchor.copy();own=state['nodeParams'][node]
            candidate,evidence=repair(worker,peer,own,x,y,node,record)
            record['semanticAdmission']=evidence
            if candidate is not None:
                chosen=candidate
                record.update(originalPeerFit=record['peerFit'],originalPeerCheck=record['peerCheck'],
                    peerFit=evidence['protectedFit'],peerCheck=evidence['protectedCheck'],route='peer',actualRoute='repaired_fresh')
                row.update(selectedSenders=[(node-1)%cfg['nodes']],selectedCount=1)
        row=dict(row,continuity=record)
        worker.admitted_route=record['actualRoute']
        worker.admitted_has_channel=worker.admitted_route!='own' and row.get('availableCopies',0)>0
        worker.continuity_anchor=chosen.copy()
        return chosen,row
    return select
