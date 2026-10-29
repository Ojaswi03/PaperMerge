"""Experimental finite-probe uncertainty allowance for semantic admission.

The standard-error margin is descriptive and model-adaptive: it is not a
confidence interval or generalization guarantee. Capped/uncapped are ablations.
"""
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.cart_qualified_continuity_candidate import selector as qualified_selector
from basil_core.cart_semantic_admission_candidate import repair
from basil_core.sharp_cart import probes


def sampling_margin(logits,labels,classes,*,capped=True):
    z=np.asarray(logits,np.float64);z=z-z.max(axis=1,keepdims=True)
    loss=np.log(np.exp(z).sum(axis=1))-z[np.arange(len(labels)),labels]
    margins=[]
    for c in classes:
        values=loss[labels==c]
        if len(values)<2:raise ValueError('Sampling margin needs multiple class examples.')
        allowance=2*float(np.std(values,ddof=1))/np.sqrt(len(values))
        if capped:allowance=min(allowance,.1*float(values.mean()))
        margins.append(allowance)
    return np.asarray(margins)


def selector(original,mode):
    if mode not in ('capped','uncapped'):raise ValueError('Unknown sampling-margin ablation.')
    choose=qualified_selector(original,'qualified')
    def select(cfg,state,worker,x,y,node,round_id,batch):
        chosen,row=choose(cfg,state,worker,x,y,node,round_id,batch);record=dict(row['continuity'])
        if record['actualRoute']=='own' and row.get('poolRankingSkipped') and record.get('learnedClasses'):
            peer=worker.continuity_anchor.copy();own=state['nodeParams'][node]
            fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass']);classes=np.asarray(record['learnedClasses'])
            sf=sampling_margin(worker.outputs(own,x[fit]),y[fit],classes,capped=mode=='capped')
            sc=sampling_margin(worker.outputs(own,x[check]),y[check],classes,capped=mode=='capped')
            fb=np.asarray(record['fitBudget'])+sf;cb=np.asarray(record['checkBudget'])+sc
            record.update(samplingFitMargin=sf.tolist(),samplingCheckMargin=sc.tolist(),samplingMarginMode=mode,
                fitBudget=fb.tolist(),checkBudget=cb.tolist())
            if np.all(np.asarray(record['peerFit'])<=np.asarray(record['ownFit'])+fb+1e-12) and np.all(np.asarray(record['peerCheck'])<=np.asarray(record['ownCheck'])+cb+1e-12):
                chosen=peer;record.update(route='peer',actualRoute='fresh',reason='qualified class loss within channel and sampling allowance')
            else:
                candidate,evidence=repair(worker,peer,own,x,y,node,record);record['semanticAdmission']=evidence
                if candidate is not None:
                    chosen=candidate
                    record.update(originalPeerFit=record['peerFit'],originalPeerCheck=record['peerCheck'],
                        peerFit=evidence['protectedFit'],peerCheck=evidence['protectedCheck'],route='peer',actualRoute='repaired_fresh')
            if record['route']=='peer':row.update(selectedSenders=[(node-1)%cfg['nodes']],selectedCount=1)
        row=dict(row,continuity=record)
        worker.admitted_route=record['actualRoute']
        worker.admitted_has_channel=worker.admitted_route!='own' and row.get('availableCopies',0)>0
        worker.continuity_anchor=chosen.copy()
        return chosen,row
    return select
