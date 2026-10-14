"""Single-process simulation of independently recomputed TRAIN helper replies.

All nine other logical replicas can read committed TRAIN evidence. This is an
explicit raw-training-data replication change, NOT the old exclusive one-class
information model or a privacy guarantee. No test data, SGD or queue is used.
"""
from __future__ import annotations

import hashlib

from .replicated_verification import (
    ComputeRequest,ComputeResult,CertificateCollector,generate_identity,sign_result,digest_json,
)


class TrainingReference:
    """Immutable shared backing in the simulator, logically replicated evidence."""
    def __init__(self,images,labels,*,source_split,synthetic=False):
        import numpy as np
        if source_split!='train' or type(synthetic) is not bool:
            raise ValueError('Only explicitly TRAIN-derived evidence is allowed.')
        x,y=np.asarray(images),np.asarray(labels)
        if (x.dtype!=np.float32 or y.dtype!=np.int32 or x.shape!=(len(y),32,32,3) or
                y.ndim!=1 or len(y)<10 or not np.isfinite(x).all() or
                np.any(x<0) or np.any(x>1) or np.any(y<0) or np.any(y>9)):
            raise ValueError('Normalized float32 CIFAR images and int32 labels required.')
        counts=np.bincount(y,minlength=10)
        if (not synthetic and (len(y)!=50000 or not np.array_equal(counts,[5000]*10))) or np.any(counts==0):
            raise ValueError('Complete 50000-example TRAIN corpus required; never substitute TEST.')
        # Immutable bytes prevent one simulated recipient from changing backing
        # arrays in place. They do not isolate arbitrary hostile Python code.
        self.images=np.frombuffer(x.astype('<f4',copy=False).tobytes(),dtype='<f4').reshape(x.shape)
        self.labels=np.frombuffer(y.astype('<i4',copy=False).tobytes(),dtype='<i4')
        h=hashlib.sha256(digest_json(dict(split='train',synthetic=synthetic,shape=x.shape)).encode())
        h.update(self.images.data);h.update(self.labels.data)
        self.root=h.hexdigest()
        self.synthetic=synthetic


class TrainingReplica:
    """One logical identity; recompute first, then sign its own complete result."""
    def __init__(self,identity,model,reference):
        if type(identity) is not int or not 0<=identity<10:
            raise ValueError('Replica ID 0..9 required.')
        self.identity,self.model,self.reference=identity,getattr(model,'model',model),reference
        self._private,self.public_key=generate_identity()
        self.evaluations=0

    def compute(self,request,params):
        import numpy as np
        import tensorflow as tf
        from .research_protocol import params_hash
        from .research_audit import augment_batch
        if self.identity not in request.committee or request.data_root!=self.reference.root:
            raise ValueError('Wrong committee or committed TRAIN reference.')
        if params_hash(params)!=request.model_hash:
            raise ValueError('Wrong request parameter state.')
        ids=np.asarray(request.indices,np.int64)
        if np.any(ids>=len(self.reference.labels)) or np.any(self.reference.labels[ids]!=request.owner):
            raise ValueError('Request indices do not belong to the specified TRAIN class.')
        if self.model.non_trainable_variables:
            raise ValueError('Only the approved stateless computation is supported.')
        weights=self.model.trainable_variables
        if len(weights)!=len(params) or any(tuple(w.shape)!=p.shape for w,p in zip(weights,params)):
            raise ValueError('Wrong model tensor layout.')
        for w,p in zip(weights,params):w.assign(p)
        images=self.reference.images[ids]
        if request.kind=='gradient':
            with tf.device('/CPU:0'):
                images=augment_batch(images,np.asarray(request.augmentation_key,np.int32))
        labels=tf.convert_to_tensor(self.reference.labels[ids])
        with tf.GradientTape() as tape:
            logits=self.model(images,training=request.kind=='gradient')
            tf.debugging.assert_all_finite(logits,'Non-finite replica logits')
            ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
        gradients=tape.gradient(ce,weights) if request.kind=='gradient' else ()
        if any(g is None for g in gradients):
            raise ValueError('Disconnected replicated gradient.')
        if params_hash([w.numpy() for w in weights])!=request.model_hash:
            raise ValueError('Replica evaluation modified model parameters.')
        self.evaluations+=1
        return ComputeResult.from_values(float(ce),[g.numpy() for g in gradients])

    def reply(self,request,params):
        return sign_result(self._private,self.identity,request,self.compute(request,params))


class ReplicatedTrainingService:
    """Certificate consumer: knows public keys, never the attacker-ID set.

Each endpoint is called separately. No copying one computed result into nine
votes. Endpoints may return poisoned/stale/forged replies or raise TimeoutError.
The local diagnostic has no actual RPC timeouts or process isolation.
"""
    def __init__(self,replicas):
        self.replicas=dict(replicas)
        if set(self.replicas)!=set(range(10)) or any(r.identity!=i for i,r in self.replicas.items()):
            raise ValueError('Ten distinct configured replica endpoints required.')
        self.public_keys={i:r.public_key for i,r in self.replicas.items()}
        self.calls=0

    def evaluate(self,request,params):
        import numpy as np
        from .research_protocol import params_hash
        # Seal caller-owned parameters so responses cannot mutate later requests.
        arrays=[np.asarray(p) for p in params]
        if not arrays or any(p.dtype!=np.float32 or not np.isfinite(p).all() for p in arrays):
            raise ValueError('Finite float32 model required.')
        sealed=tuple(np.frombuffer(p.tobytes(),dtype=np.float32).reshape(p.shape) for p in arrays)
        if params_hash(sealed)!=request.model_hash:
            raise ValueError('Requested model hash mismatch.')
        collector=CertificateCollector(request,self.public_keys)
        unavailable=[]
        for identity in request.committee:
            self.calls+=1
            try:
                vote=self.replicas[identity].reply(request,sealed)
            except (TimeoutError,ValueError,FloatingPointError) as error:
                unavailable.append(dict(replica=identity,errorType=type(error).__name__,reason=str(error)))
                continue
            # Bind response to its authenticated/configured endpoint too.
            if getattr(vote,'signer',None)!=identity:
                collector.rejected.append('endpoint_identity_mismatch');continue
            collector.add(vote)
        result,metadata=collector.certificate()
        if request.kind=='gradient' and (result.shapes!=tuple(p.shape for p in sealed)):
            raise ValueError('Certified gradient tensor layout differs from requested model.')
        metadata.update(endpointCalls=len(request.committee),unavailableReplicas=unavailable,
            logicalSimulation=True,physicalRPC=False,privateDataReplication=True,
            fullResearchReady=False)
        return result,metadata

    def gradient_response(self,request,params):
        from .cart_owner_gradients import OwnerGradientResponse
        if request.kind!='gradient':raise ValueError('Gradient request required.')
        result,proof=self.evaluate(request,params)
        response=OwnerGradientResponse(request.owner,request.receiver,request.round_id,request.step,
            request.model_hash,len(request.indices),result.cross_entropy,result.arrays())
        return response,proof


def verified_balanced_selection(service,candidates,*,reference,run_id,receiver,round_id,probes):
    """Choose one snapshot using certified per-owner TRAIN CE, not claimed CE."""
    import math
    from .research_protocol import params_hash
    owners=tuple(sorted(probes))
    expected=tuple(range(receiver+1)) if round_id==0 else tuple(range(10))
    if owners!=expected:
        raise ValueError('Exact available-owner TRAIN evidence required.')
    candidates=tuple(candidates)
    if len(candidates)!=5 or {s.sender_id for s in candidates}!={(receiver-d)%10 for d in range(1,6)}:
        raise ValueError('Exactly five distinct predecessor snapshots required.')
    records=[]
    for s in candidates:
        if s.round_id!=(round_id if s.sender_id<receiver else round_id-1):
            raise ValueError('Wrong rolling snapshot round.')
        losses,certificates={},{}
        for owner in owners:
            request=ComputeRequest(run_id,receiver,owner,round_id,0,'selection_ce',params_hash(s.params),
                                   reference.root,tuple(int(i) for i in probes[owner]))
            result,proof=service.evaluate(request,s.params)
            losses[owner]=result.cross_entropy;certificates[owner]=proof
        records.append(dict(senderId=s.sender_id,snapshotRound=s.round_id,parameterHash=params_hash(s.params),
            ownerLosses=losses,balancedTrainingLoss=math.fsum(losses.values())/len(losses),
            receiverLocalLoss=losses[receiver],certificates=certificates))
    best=min(r['balancedTrainingLoss'] for r in records)
    chosen=min((i for i,r in enumerate(records) if r['balancedTrainingLoss']<=best+1e-8),
               key=lambda i:(receiver-candidates[i].sender_id)%10)
    return candidates[chosen],records
