"""Experimental receiver-evidence activation for transactional CART quality.

Fresh peer learning remains open until the receiver itself has useful class
recognition. Peer winners alone do not activate a retention gate during cold start.
"""
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.sharp_cart import probes
from basil_core.sharp_protocol import link_vector
from basil_core.cart_quality_routing_candidate import utility
from basil_core.cart_class_skill import class_skill as class_f1


def selector(original,mode):
    if mode not in ('class','macro'):raise ValueError('Unknown quality-routing ablation.')
    def select(cfg,state,worker,x,y,node,round_id,batch):
        pool,base=original(cfg,state,worker,x,y,node,round_id,batch)
        own=state['nodeParams'][node] if state['senderRounds'][node]>=0 else None
        fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass'])
        classes=np.intersect1d(np.unique(y[fit]),np.unique(y[check]))
        if len(classes)<2:
            worker.admitted_route='pool';worker.admitted_has_channel=base.get('availableCopies',0)>0
            worker.continuity_anchor=pool.copy()
            return pool,dict(base,continuity=dict(actualRoute='pool',route='peer',reason='insufficient class discrimination support',softRetention=True))
        fit=fit[np.isin(y[fit],classes)];check=check[np.isin(y[check],classes)]
        indices=np.concatenate((fit,check));split=len(fit)
        def evidence(flat):
            logits=worker.outputs(flat,x[indices])
            return (class_f1(logits[:split],y[fit],classes),class_f1(logits[split:],y[check],classes))
        candidates=[dict(kind='pool',params=pool,senders=base['selectedSenders'],q=base['selectedCount'],priority=cfg['nodes'])]
        senders=[(node-d)%cfg['nodes'] for d in range(1,cfg['window']+1) if state['senderRounds'][(node-d)%cfg['nodes']]>=0]
        passed=[s for j,s in enumerate(senders) if not base.get('screenApplied') or base['peakRatios'][j]<=base['peakThreshold']]
        passed.sort(key=lambda s:(-int(state['senderRounds'][s]),(node-s)%cfg['nodes']))
        for sender in passed[:8]:
            value=link_vector(state['outbound'][sender],cfg['seed'],int(state['senderRounds'][sender]),sender,node,cfg['sigma'])
            candidates.append(dict(kind='neighbor',params=value,senders=[sender],q=1,priority=(node-sender)%cfg['nodes']))
        if own is not None:candidates.append(dict(kind='own',params=own.copy(),senders=[],q=1,priority=cfg['nodes']+1))
        for candidate in candidates:candidate['fit'],candidate['check']=evidence(candidate['params'])
        if own is not None:
            own_row=candidates[-1];target=np.minimum(own_row['fit'],own_row['check'])
            qualified=np.asarray([np.sum(y[fit]==c)>=8 and np.sum(y[check]==c)>=8 for c in classes])&(target>=.2)
        else:target=np.zeros(len(classes));qualified=np.zeros(len(classes),bool)
        reference=0.
        informative=bool(own is not None and min(float(own_row['fit'].mean()),float(own_row['check'].mean()))>reference+.02)
        allowance=np.zeros(len(classes));draws=0
        if informative and mode=='class' and own is not None and qualified.any() and cfg['sigma'] and any(
                np.any((target-np.minimum(c['fit'],c['check']))[qualified]>.0) for c in candidates):
            from basil_core.research_protocol import keyed_rng
            for draw in range(cfg['calibrationDraws']):
                noise=keyed_rng(cfg['seed'],'quality_route_channel',node,state['cursor'],draw).normal(0,cfg['sigma'],size=own.shape).astype(np.float32)
                f,k=evidence(own+noise);allowance=np.maximum(allowance,np.maximum(target-np.minimum(f,k),0.));draws+=1
        for c in candidates:c['score']=utility(c['fit'],c['check'],target,qualified,allowance,penalty=mode=='class')
        best=max(c['score'] for c in candidates)
        chosen=min((c for c in candidates if not informative or c['score']>=best-.01),key=lambda c:(c['priority'],c['kind']))
        if informative and own is not None and chosen['score']<own_row['score']-.01-1e-12:raise AssertionError('Quality band violated carrier score.')
        record=dict(mode=mode,route='own' if chosen['kind']=='own' else 'peer',actualRoute=chosen['kind'],softRetention=True,
            supportedClasses=classes.tolist(),qualifiedClasses=classes[qualified].tolist(),classTargets=target.tolist(),
            channelF1Allowance=allowance.tolist(),calibrationDraws=draws,nearPeerBudget=8,scoreBand=.01,
            qualityScoreBoundApplied=informative,nullSkillReference=reference,qualityMeasure='one-vs-rest Matthews correlation coefficient',recognitionEvidenceSource='receiver carrier on both probes',
            selectedScore=chosen['score'],carrierScore=own_row['score'] if own is not None else None,
            candidates=[dict(kind=c['kind'],senders=c['senders'],score=c['score'],fitSkill=c['fit'].tolist(),checkSkill=c['check'].tolist()) for c in candidates])
        worker.admitted_route=chosen['kind'];worker.admitted_has_channel=chosen['kind']!='own' and base.get('availableCopies',0)>0
        worker.continuity_anchor=chosen['params'].copy()
        return chosen['params'].copy(),dict(base,selectedSenders=chosen['senders'],selectedCount=chosen['q'],continuity=record,
            diagnosticInputProvenance='Actual coherent model source and class-quality admission recorded; pooled proposal remains available.')
    return select
