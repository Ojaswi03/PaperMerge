"""Experimental admitted-model anchor and route-aware link-noise regularizer.

A rejected peer must not remain the proximal anchor. A local carrier adds no
new channel draw at this activation; no claim is made about historical noise.
"""
from basil_core.cart_continuity_candidate import ContinuityWorker
from basil_core.cart_qualified_continuity_candidate import selector as qualified_selector


class AdmittedWorker(ContinuityWorker):
    def train(self,flat,x,y,round_id,node,coefficient,first_batch,should_stop=None):
        self.continuity_anchor=flat.copy()
        own=getattr(self,'admitted_route',None)=='own'
        local=own or not getattr(self,'admitted_has_channel',True)
        effective=0. if local else coefficient
        final,row=super().train(flat,x,y,round_id,node,effective,first_batch,should_stop)
        row.update(continuityAnchor='complete admitted initialization; rejected peer is not an anchor',
            peerPoolRlaCoefficient=coefficient,linkRlaSuppressedForLocalCarrier=own,linkRlaSuppressedForLocalInitialization=local,
            noiseInterpretation='No new link perturbation on a receiver-owned carrier; historical noise is not removed.')
        return final,row


def selector(original,mode):
    select=qualified_selector(original,mode)
    def choose(cfg,state,worker,x,y,node,round_id,batch):
        chosen,row=select(cfg,state,worker,x,y,node,round_id,batch)
        worker.admitted_route=row['continuity']['actualRoute']
        worker.admitted_has_channel=worker.admitted_route!='own' and row.get('availableCopies',0)>0
        worker.continuity_anchor=chosen.copy()
        return chosen,row
    return choose
