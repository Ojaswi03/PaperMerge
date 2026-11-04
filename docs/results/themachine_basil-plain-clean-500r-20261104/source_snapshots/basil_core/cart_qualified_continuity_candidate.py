"""Experimental learned-class protection on a screened fresh-hop handoff.

Protection is conditional on finite private probe evidence. Unlearned classes
are not made immutable. No adversarial or population guarantee is claimed.
"""
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.sharp_cart import probes
from basil_core.sharp_math import class_ce,peak_screen
from basil_core.sharp_protocol import link_vector


def qualified_classes(logits_fit,y_fit,logits_check,y_check,classes):
    supported=np.intersect1d(np.unique(y_fit),np.unique(y_check));qualified=[]
    for c in supported:
        okay=True
        for z,y in ((logits_fit,y_fit),(logits_check,y_check)):
            mask=y==c
            if (mask.sum()<8 or np.mean(z[mask].argmax(axis=1)==c)<.4 or
                    class_ce(z[mask],y[mask])[1][0]>.8*np.log(classes)):okay=False
        if okay:qualified.append(int(c))
    return qualified


def selector(original,mode):
    if mode not in ('always','qualified'):raise ValueError('Unknown qualified-continuity mode.')
    def select(cfg,state,worker,x,y,node,round_id,batch):
        pred=(node-1)%cfg['nodes'];senders=[(node-d)%cfg['nodes'] for d in range(1,cfg['window']+1)
            if state['senderRounds'][(node-d)%cfg['nodes']]>=0]
        has_fresh=pred in senders
        # This architecture uses the pool only as fallback. Do not score all
        # peer models when a passed fresh copy makes that work unnecessary.
        if has_fresh:
            vectors=np.stack([link_vector(state['outbound'][sender],cfg['seed'],
                int(state['senderRounds'][sender]),sender,node,cfg['sigma']) for sender in senders])
            _,screen=peak_screen(vectors,full_window=len(senders)==cfg['window'])
            index=senders.index(pred)
            has_fresh=screen['peakRatios'][index]<=screen['peakThreshold']
        if has_fresh:
            peer=vectors[index].copy()
            evidence=dict(screen,selectedSenders=[pred],selectedCount=1,availableCopies=len(senders),
                bootstrap=False,ranking=[],poolRankingSkipped=True)
        else:
            peer,evidence=original(cfg,state,worker,x,y,node,round_id,batch)
        own=state['nodeParams'][node] if state['senderRounds'][node]>=0 else None
        chosen=peer.copy();record=dict(mode=mode,route='peer',actualRoute='fresh' if has_fresh else 'pool',
            learnedClasses=[],qualifiedInitializationBound=False,calibrationDraws=0,
            reason='bootstrap or unconditional fresh-hop ablation')
        if mode=='qualified' and own is not None:
            fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass'])
            if len(fit) and len(check):
                of=worker.outputs(own,x[fit]);oc=worker.outputs(own,x[check])
                classes=qualified_classes(of,y[fit],oc,y[check],cfg['classes']);record['learnedClasses']=classes
                record['reason']='no learned class; allow fresh learning'
                if classes:
                    fit=fit[np.isin(y[fit],classes)];check=check[np.isin(y[check],classes)]
                    def losses(flat,indices):return class_ce(worker.outputs(flat,x[indices]),y[indices])[1]
                    of=losses(own,fit);oc=losses(own,check);pf=losses(peer,fit);pc=losses(peer,check)
                    fb=np.zeros(len(classes));cb=np.zeros(len(classes))
                    if cfg['sigma'] and (np.any(pf>of) or np.any(pc>oc)):
                        from basil_core.research_protocol import keyed_rng
                        for draw in range(cfg['calibrationDraws']):
                            noise=keyed_rng(cfg['seed'],'qualified_continuity_channel',node,state['cursor'],draw).normal(0,cfg['sigma'],size=own.shape).astype(np.float32)
                            fb=np.maximum(fb,np.maximum(losses(own+noise,fit)-of,0.))
                            cb=np.maximum(cb,np.maximum(losses(own+noise,check)-oc,0.))
                        record['calibrationDraws']=cfg['calibrationDraws']
                    accept=bool(np.all(pf<=of+fb+1e-12) and np.all(pc<=oc+cb+1e-12))
                    if not accept:chosen=own.copy();record.update(route='own',actualRoute='own')
                    record.update(qualifiedInitializationBound=True,reason='protect learned-class probe losses',
                        peerFit=pf.tolist(),ownFit=of.tolist(),peerCheck=pc.tolist(),ownCheck=oc.tolist(),
                        fitBudget=fb.tolist(),checkBudget=cb.tolist(),fitCheckDisjoint=not bool(np.intersect1d(fit,check).size))
        worker.continuity_anchor=peer.copy()
        result=dict(evidence,continuity=record,diagnosticInputProvenance='Actual route and sender list replace peer-pool selection when fresh/own is used.')
        if record['actualRoute']=='fresh':result.update(selectedSenders=[pred],selectedCount=1)
        if record['actualRoute']=='own':result.update(selectedSenders=[],selectedCount=1)
        return chosen,result
    return select
