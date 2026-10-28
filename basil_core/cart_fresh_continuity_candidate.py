"""Separate experiment: compare a screened fresh hop, peer pool and local carrier.

No sender identity is trusted. The newest predecessor is only a candidate and
must meet the same private loss gate. Not an established Byzantine defense.
"""
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.cart_continuity_candidate import route
from basil_core.sharp_cart import probes
from basil_core.sharp_math import class_ce
from basil_core.sharp_protocol import link_vector


def selector(original,mode):
    if mode not in ('macro','class_noise'):raise ValueError('Unknown fresh continuity mode.')
    def select(cfg,state,worker,x,y,node,round_id,batch):
        pool,evidence=original(cfg,state,worker,x,y,node,round_id,batch)
        own=state['nodeParams'][node] if state['senderRounds'][node]>=0 else None
        candidates=[('pool',pool,None)]
        predecessor=(node-1)%cfg['nodes']
        senders=[(node-d)%cfg['nodes'] for d in range(1,cfg['window']+1)
            if state['senderRounds'][(node-d)%cfg['nodes']]>=0]
        if predecessor in senders:
            index=senders.index(predecessor)
            passed=(not evidence.get('screenApplied') or
                evidence['peakRatios'][index]<=evidence['peakThreshold'])
            if passed:
                fresh=link_vector(state['outbound'][predecessor],cfg['seed'],
                    int(state['senderRounds'][predecessor]),predecessor,node,cfg['sigma'])
                candidates.append(('fresh',fresh,predecessor))
        fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass'])
        indices=np.unique(np.concatenate((fit,check)))
        def score(flat):return float(class_ce(worker.outputs(flat,x[indices]),y[indices])[1].mean())
        choices=[]
        for name,peer,sender in candidates:
            chosen,record=route(worker,peer,own,x,y,node,state['cursor'],mode=mode)
            identity=name if record['route']=='peer' else 'own'
            choices.append((score(chosen),name,identity,chosen,peer,sender,record))
        _,candidate,identity,chosen,anchor,sender,record=min(choices,key=lambda v:(v[0],v[1]))
        worker.continuity_anchor=anchor.copy()
        result=dict(evidence,continuity=dict(record,candidate=candidate,actualRoute=identity,
            candidateScores=[dict(candidate=v[1],actualRoute=v[2],score=v[0]) for v in choices]),
            diagnosticInputProvenance='Fresh/own/pool route recorded explicitly; peer-pool ranking retained as diagnostic context.')
        if identity=='fresh':result.update(selectedSenders=[sender],selectedCount=1)
        if identity=='own':result.update(selectedSenders=[],selectedCount=1)
        return chosen,result
    return select
