"""Exactness-gated execution adapter: batch sequential model inference calls.

CE remains the original NumPy float64 calculation. Candidate order, screen,
ranking/ties and mean are unchanged. Hardware-specific verification is required.
"""
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')
from basil_core.cart_admitted_continuity_candidate import AdmittedWorker
from basil_core.sharp_math import peak_screen,ranked_indices
from basil_core.sharp_protocol import link_vector


class BatchedAdmittedWorker(AdmittedWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        import tensorflow as tf
        @tf.function(reduce_retracing=True)
        def logits(flats,images):
            return tf.map_fn(lambda flat:self._forward(flat,images),flats,
                fn_output_signature=tf.TensorSpec((None,self.cfg['classes']),tf.float32),parallel_iterations=1)
        self._candidate_logits=logits
    def score_many(self,flats,x,y):
        values=np.asarray(flats,np.float32)
        if values.ndim!=2 or values.shape[1]!=self.dimension or not np.isfinite(values).all():raise ValueError('Finite aligned model copies required.')
        # Guarded SHARP scoring uses <=80 examples: exactly one original batch.
        if len(x)>128:raise ValueError('Batched original scoring is limited to one original inference batch.')
        logits=self._candidate_logits(values,x).numpy()
        shifted=logits.astype(np.float64)-logits.max(axis=2,keepdims=True)
        losses=np.log(np.exp(shifted).sum(axis=2))-shifted[:,np.arange(len(y)),y]
        scores=losses.mean(axis=1)
        return np.where(np.isfinite(scores),scores,np.inf)


def select_input(cfg,state,worker,x,y,receiver,round_id,first_batch):
    if not cfg['switches']['byzantine_mitigation']:
        from basil_core.sharp_protocol import select_input as ordinary
        return ordinary(cfg,state,worker,x,y,receiver,round_id,first_batch)
    n=cfg['nodes'];size=cfg['window'];b=cfg['byzantineBudget']
    senders=[(receiver-distance)%n for distance in range(1,size+1) if state['senderRounds'][(receiver-distance)%n]>=0]
    if not senders:
        return state['nodeParams'][receiver].copy(),dict(selectedSenders=[],selectedCount=1,
            availableCopies=0,bootstrap=True,screenAllFailFallback=False,screenApplied=False,ranking=[])
    vectors=np.stack([link_vector(state['outbound'][s],cfg['seed'],int(state['senderRounds'][s]),s,receiver,cfg['sigma']) for s in senders])
    passed,screen=peak_screen(vectors,full_window=len(senders)==size)
    raw=np.empty(0,np.int64) if screen['screenAllFailFallback'] else passed
    fallback=len(raw)<=b;candidates=np.arange(len(senders)) if fallback else raw
    wanted=1 if fallback else min(b,len(raw)-b)
    losses=worker.score_many(vectors[candidates],x[first_batch],y[first_batch])
    chosen=ranked_indices(losses,[senders[j] for j in candidates],
        [state['senderRounds'][senders[j]] for j in candidates],receiver,n,wanted)
    indices=candidates[chosen];mean=vectors[indices].astype(np.float64).mean(axis=0).astype(np.float32)
    evidence=dict(screen,selectedSenders=[senders[j] for j in indices],selectedCount=len(indices),
        availableCopies=len(senders),bootstrap=False,targetCount=wanted,rawSurvivorCount=len(raw),
        survivorUnattackedLowerBound=max(0,len(raw)-b),screenGuardFallback=fallback,
        ranking=[dict(sender=senders[j],loss=float(loss) if np.isfinite(loss) else None) for j,loss in zip(candidates,losses)])
    return mean,evidence
