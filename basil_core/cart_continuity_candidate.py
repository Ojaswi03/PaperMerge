"""Experimental coherent-model continuity gate; not established CART novelty.

The finite-probe guarantee concerns only the selected initialization. It does
not guarantee generalization, Byzantine resilience or non-forgetting after SGD.
"""
from __future__ import annotations
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.sharp_compiled_transfer import CompiledTransferWorker
from basil_core.sharp_cart import probes
from basil_core.sharp_math import class_ce


def eligible(peer_fit,own_fit,peer_check,own_check,fit_budget,check_budget,*,tolerance=1e-4):
    arrays=[np.asarray(v,np.float64) for v in (peer_fit,own_fit,peer_check,own_check,fit_budget,check_budget)]
    p,o,pc,oc,b,bc=arrays
    if (not len(p) or any(v.shape!=p.shape or not np.isfinite(v).all() for v in arrays)
            or np.any(b<0) or np.any(bc<0)):
        raise ValueError('Aligned finite supported-class losses and nonnegative budgets required.')
    return bool(p.mean()<o.mean()-tolerance and pc.mean()<oc.mean()-tolerance
        and np.all(p<=o+b+1e-12) and np.all(pc<=oc+bc+1e-12))


def route(worker,peer,own,x,y,node,activation,*,mode='class_noise'):
    if mode not in ('own','macro','class_noise'):raise ValueError('Unknown continuity ablation.')
    evidence=dict(mode=mode,route='peer',reason='bootstrap',calibrationDraws=0,
        fitCheckDisjoint=True,empiricalInitializationBound=False)
    if own is None:return peer.copy(),evidence
    if mode=='own':return own.copy(),dict(evidence,route='own',reason='unconditional carrier ablation')
    cfg=worker.cfg;fit,check=probes(y,cfg['seed'],node,cfg['cartProbePerClass'])
    supported=np.intersect1d(np.unique(y[fit]),np.unique(y[check]));evidence['supportedClasses']=supported.tolist()
    evidence['fitCheckDisjoint']=not bool(np.intersect1d(fit,check).size)
    if not len(supported):return peer.copy(),dict(evidence,reason='no class supported on both probes')
    fit=fit[np.isin(y[fit],supported)];check=check[np.isin(y[check],supported)]
    def losses(parameters,indices):return class_ce(worker.outputs(parameters,x[indices]),y[indices])[1]
    pf=losses(peer,fit);of=losses(own,fit);pc=losses(peer,check);oc=losses(own,check)
    evidence.update(peerFit=pf.tolist(),ownFit=of.tolist(),peerCheck=pc.tolist(),ownCheck=oc.tolist())
    macro=bool(pf.mean()<of.mean()-1e-4 and pc.mean()<oc.mean()-1e-4)
    if mode=='macro':return (peer.copy() if macro else own.copy()),dict(evidence,
        route='peer' if macro else 'own',reason='macro CE gate ablation')
    fb=np.zeros(len(supported));cb=np.zeros(len(supported))
    # Rejecting an unhelpful peer needs no channel simulation. Calibration is
    # only needed when a macro-improving peer harms at least one supported class.
    if macro and (np.any(pf>of) or np.any(pc>oc)) and cfg['sigma']:
        from basil_core.research_protocol import keyed_rng
        for draw in range(cfg['calibrationDraws']):
            noise=keyed_rng(cfg['seed'],'continuity_channel',node,activation,draw).normal(0,cfg['sigma'],size=own.shape).astype(np.float32)
            fb=np.maximum(fb,np.maximum(losses(own+noise,fit)-of,0.))
            cb=np.maximum(cb,np.maximum(losses(own+noise,check)-oc,0.))
        evidence['calibrationDraws']=cfg['calibrationDraws']
    accept=eligible(pf,of,pc,oc,fb,cb)
    chosen_fit=pf if accept else of;chosen_check=pc if accept else oc
    assert np.all(chosen_fit<=of+fb+1e-12) and np.all(chosen_check<=oc+cb+1e-12)
    assert chosen_fit.mean()<=of.mean()+1e-12 and chosen_check.mean()<=oc.mean()+1e-12
    evidence.update(route='peer' if accept else 'own',reason='class-constrained initialization',
        fitBudget=fb.tolist(),checkBudget=cb.tolist(),empiricalInitializationBound=True)
    return (peer.copy() if accept else own.copy()),evidence


class ContinuityWorker(CompiledTransferWorker):
    """Same five SGD steps and original received-model proximal anchor."""
    def train(self,flat,x,y,round_id,node,coefficient,first_batch,should_stop=None):
        import tensorflow as tf
        from basil_core.research_protocol import keyed_rng
        incoming=getattr(self,'continuity_anchor',flat)
        self.load(flat);anchor=tuple(tf.convert_to_tensor(v) for v in self.tensors(incoming))
        eta=self.cfg['learningRate']/(1+self.cfg['learningRate']*round_id);values=[]
        for step in range(self.cfg['localSteps']):
            if should_stop and should_stop():raise InterruptedError('Stop requested between local steps; activation discarded.')
            indices=first_batch if step==0 else keyed_rng(self.cfg['seed'],'sharp_batch',round_id,node,step).choice(
                len(y),size=min(self.cfg['batchSize'],len(y)),replace=False)
            result=self.step(tf.convert_to_tensor(x[indices]),tf.convert_to_tensor(y[indices],tf.int32),anchor,
                tf.constant(eta,tf.float32),tf.constant(coefficient,tf.float32),tf.constant(self.cfg['proximalMu'],tf.float32))
            values.append(dict(zip(('ce','gradientSquared','objective','proximalPenalty','updateGradientNorm'),map(float,result))))
        final=self.export()
        if not np.isfinite(final).all():raise FloatingPointError('Non-finite continuity candidate.')
        return final,dict(optimizerSteps=len(values),learningRate=eta,rlaCoefficient=coefficient,
            proximalMu=self.cfg['proximalMu'],steps=values,localTrainingLoss=float(np.mean([v['ce'] for v in values])),
            continuityAnchor='original received model; coefficient remains the declared SHARP regularizer, not a variance guarantee for the carrier')


def selector(original,mode):
    def select(cfg,state,worker,x,y,node,round_id,batch):
        peer,evidence=original(cfg,state,worker,x,y,node,round_id,batch)
        own=state['nodeParams'][node] if state['senderRounds'][node]>=0 else None
        chosen,record=route(worker,peer,own,x,y,node,state['cursor'],mode=mode)
        worker.continuity_anchor=peer.copy()
        return chosen,dict(evidence,continuity=record,
            diagnosticInputProvenance='carrier or peer as recorded; original sender list describes peer pool only')
    return select
