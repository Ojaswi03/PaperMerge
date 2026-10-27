"""Isolated experimental CART worker; not wired into a production protocol.

It replaces a class-target mosaic with a whole receiver-owned reference and
checks measured class envelopes at reception and optimizer boundaries.
"""
from __future__ import annotations

import copy

from basil_core.factorial_worker import PlainWorker
from basil_core.cart_feasible_memory import (REVISION,CALIBRATION_DRAWS,
    calibrate_envelope,class_losses,coherent_admission,restore_feasible,within_envelope)


class GuardedOptimizer:
    """Run ordinary SGD, then measure a feasible step on the same local probes."""
    def __init__(self,inner,worker):
        self.inner=inner;self.worker=worker

    def __getattr__(self,name):return getattr(self.inner,name)

    def apply_gradients(self,pairs):
        import tensorflow as tf
        worker=self.worker
        before=[tf.identity(v) for v in worker.model.trainable_variables] if worker.active_bounds is not None else None
        result=self.inner.apply_gradients(pairs)
        evidence=dict(scale=1.,trials=0,classLosses=None)
        if before is not None:
            losses=worker.measure_current()
            evidence['classLosses']=losses.tolist()
            evidence['trials']=1
            if not within_envelope(losses,worker.active_bounds):
                proposal=worker.export()
                final,restored=restore_feasible([v.numpy() for v in before],proposal,
                    worker.active_bounds,worker.measure_params)
                worker.load(final)
                scale=restored['scale']
                slots=self.inner.momentums
                if len(slots)!=len(before):raise ValueError('Unexpected SGD momentum schema.')
                for slot in slots:slot.assign(scale*slot)
                evidence=dict(scale=scale,trials=1+len(restored['trials']),
                    classLosses=restored['classLosses'])
            if not within_envelope(evidence['classLosses'],worker.active_bounds):
                raise ValueError('Infeasible update cannot be committed.')
        worker.step_guards.append(evidence)
        return result


class FeasibleCartWorker(PlainWorker):
    """Receiver-local reference; no trusted sender scores or TEST feedback."""
    def __init__(self,model,*,seed=2025,momentum=.9,weight_decay=.0005,refresh_policy='dominance'):
        import tensorflow as tf
        if model.non_trainable_variables:
            raise ValueError('The coherent reference requires a model without untracked non-trainable state.')
        super().__init__(model)
        if refresh_policy not in ('dominance','envelope'):raise ValueError('Unknown reference refresh policy.')
        self.refresh_policy=refresh_policy
        self.seed=seed;self.beta=1.;self.diagnostic_revision=REVISION
        self.memory=dict(revision=REVISION,seed=seed,nodes={})
        self.factorial_config=dict(weightDecayCoefficient=weight_decay)
        self.optimizer_factory=lambda rate:GuardedOptimizer(tf.keras.optimizers.SGD(
            learning_rate=rate,momentum=momentum),self)
        self.active_bounds=None;self.step_guards=[]
        self.probe_x=None;self.probe_y=None;self.classes=None

    def export_local_memory(self):return copy.deepcopy(self.memory)

    def restore_local_memory(self,state):
        validate_memory(state,seed=self.seed)
        self.memory=copy.deepcopy(state)

    def measure_current(self):
        import tensorflow as tf
        import numpy as np
        classes,losses=class_losses(self.predict(tf.convert_to_tensor(self.probe_x)).numpy(),self.probe_y)
        if not np.array_equal(classes,self.classes):raise ValueError('Probe support changed.')
        return losses

    def measure_params(self,params):
        self.load(params)
        return self.measure_current()

    def prepare(self,node,x,y):
        import numpy as np
        from basil_core.cart_local_memory import _digest,PROBE_PER_CLASS
        from basil_core.research_protocol import keyed_rng
        labels=np.asarray(y)
        if (type(node) is not int or not 0<=node<10 or labels.ndim!=1 or not len(labels)
                or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>=10)
                or len(x)!=len(labels) or not np.isfinite(x).all()):
            raise ValueError('Receiver-local finite TRAIN examples required.')
        counts=np.bincount(labels,minlength=10)
        classes=np.flatnonzero(counts)
        indices=np.concatenate([keyed_rng(self.seed,'cart_memory_probe',node,int(c)).choice(
            np.flatnonzero(labels==c),min(PROBE_PER_CLASS,int(counts[c])),replace=False) for c in classes])
        identity=dict(imageHash=_digest(x),labelHash=_digest(labels),counts=counts.tolist(),
            probeIndices=indices.tolist(),classes=classes.tolist())
        old=self.memory['nodes'].get(str(node))
        if old and old['identity']!=identity:raise ValueError('Receiver TRAIN/probe identity changed.')
        self.probe_x=np.asarray(x[indices]);self.probe_y=labels[indices];self.classes=classes
        return identity

    def admit(self,node,params,losses,identity,activation,sigma):
        import numpy as np
        from basil_core.research_protocol import apply_channel_noise,keyed_rng,params_hash
        noisy_losses=[]
        if sigma:
            for draw in range(CALIBRATION_DRAWS):
                corrupted,_=apply_channel_noise(params,semantics='paper_absolute_gaussian',sigma=sigma,
                    rng=keyed_rng(self.seed,'cart_reference_channel',node,activation,draw))
                noisy_losses.append(self.measure_params(corrupted))
        else:noisy_losses=[np.asarray(losses).copy() for _ in range(CALIBRATION_DRAWS)]
        self.load(params)
        envelope=calibrate_envelope(losses,noisy_losses)
        self.memory['nodes'][str(node)]=dict(identity=identity,referenceParams=[p.copy() for p in params],
            referenceHash=params_hash(params),activation=activation,sigma=float(sigma),envelope=envelope)

    def train(self,params,x,y,**kwargs):
        import numpy as np
        from basil_core.research_protocol import params_hash
        if kwargs['seed']!=self.seed:raise ValueError('CART seed/config mismatch.')
        if kwargs.get('semantics')!='paper_absolute_gaussian':raise ValueError('This candidate models the explicit absolute Gaussian channel only.')
        old=self.export_local_memory()
        node=kwargs['node_id'];activation=kwargs['round_id']*10+node;sigma=float(kwargs['sigma'])
        identity=self.prepare(node,x,y)
        before=self.memory['nodes'].get(str(node))
        self.step_guards=[];self.active_bounds=None
        input_evidence=dict(scale=1.,trials=[])
        try:
            incoming=params
            if before:
                if before['sigma']!=sigma:raise ValueError('Channel changed without independent recalibration.')
                self.active_bounds=before['envelope']['bounds']
                # The coherent reference is a measured feasible witness.
                reference_losses=self.measure_params(before['referenceParams'])
                if not within_envelope(reference_losses,self.active_bounds):
                    raise ValueError('The coherent reference is no longer feasible.')
                incoming,input_evidence=restore_feasible(before['referenceParams'],params,
                    self.active_bounds,self.measure_params)
            trained,record=super().train(incoming,x,y,**kwargs)
            self.load(trained);losses=self.measure_current()
            admitted=before is None or coherent_admission(before['envelope']['referenceLosses'],losses,
                bounds=before['envelope']['bounds'] if self.refresh_policy=='envelope' else None)
            if admitted:self.admit(node,trained,losses,identity,activation,sigma)
            after=self.memory['nodes'][str(node)]
            if len(self.step_guards)!=record['steps']:raise ValueError('Missing optimizer guard evidence.')
            for step,evidence in zip(record['optimizerTelemetry'],self.step_guards):
                step.update(cartAcceptedStepScale=evidence['scale'],cartProbeEvaluations=evidence['trials'])
            record.update(localClassCounts=np.bincount(y,minlength=10).tolist(),
                cartMemory=dict(revision=REVISION,beta=1.,probeCount=len(self.probe_y),
                    classes=self.classes.tolist(),referenceBefore=before['referenceHash'] if before else None,
                    referenceAfter=after['referenceHash'],referenceAdmitted=admitted,
                    inputReferenceHash=params_hash(incoming),inputBridge=input_evidence,
                    activeBounds=self.active_bounds,referenceLosses=after['envelope']['referenceLosses'],
                    nextBounds=after['envelope']['bounds'],channelSigma=sigma,
                    calibrationDraws=CALIBRATION_DRAWS if admitted and sigma else 0,
                    guardEnabled=before is not None,stepGuards=self.step_guards))
            return trained,record
        except BaseException:
            self.memory=old
            raise
        finally:
            self.active_bounds=None


def validate_memory(state,*,seed,cursor=None):
    """Validate full in-memory references; checkpoint packing is separate."""
    import numpy as np
    from basil_core.research_protocol import params_hash
    if (not isinstance(state,dict) or set(state)!={'revision','seed','nodes'} or
            state['revision']!=REVISION or state['seed']!=seed or not isinstance(state['nodes'],dict)):
        raise ValueError('Invalid coherent CART state schema.')
    if cursor is not None:
        if type(cursor) is not int or cursor<0 or set(state['nodes'])!={str(i) for i in range(min(cursor,10))}:
            raise ValueError('Coherent memory does not match completed receiver visits.')
    for key,node in state['nodes'].items():
        if key not in {str(i) for i in range(10)}:raise ValueError('Invalid receiver.')
        if (type(node['activation']) is not int or node['activation']<0 or node['activation']%10!=int(key)
                or cursor is not None and node['activation']>=cursor):
            raise ValueError('Invalid or future reference admission.')
        params=node['referenceParams']
        if not params or any(not np.isfinite(v).all() for v in params) or params_hash(params)!=node['referenceHash']:
            raise ValueError('Corrupt coherent reference parameters.')
        identity=node['identity'];counts=np.asarray(identity['counts'])
        classes=np.asarray(identity['classes'])
        if (counts.shape!=(10,) or not np.issubdtype(counts.dtype,np.integer)
                or not np.issubdtype(classes.dtype,np.integer) or np.any(counts<0) or counts.sum()<=0 or
                not np.array_equal(classes,np.flatnonzero(counts))):raise ValueError('Invented or missing class support.')
        probes=np.asarray(identity['probeIndices'])
        if (probes.ndim!=1 or not np.issubdtype(probes.dtype,np.integer)
                or len(probes)!=int(np.minimum(counts,16).sum()) or len(np.unique(probes))!=len(probes)
                or np.any(probes<0) or np.any(probes>=counts.sum())):
            raise ValueError('Invalid receiver-local probe membership.')
        if not np.isfinite(node['sigma']) or node['sigma']<0:raise ValueError('Invalid channel sigma.')
        envelope=node['envelope']
        reference=np.asarray(envelope['referenceLosses'],dtype=float)
        scales=np.maximum(reference,.1);scores=np.asarray(envelope['jointScores'],dtype=float)
        if (reference.shape!=(len(classes),) or not np.isfinite(reference).all() or np.any(reference<0)
                or scores.shape!=(19,) or not np.isfinite(scores).all() or np.any(scores<0)
                or envelope['calibrationDraws']!=19 or envelope['marginalReferenceChannelCoverage']!=.95
                or envelope['gamma']!=float(scores.max())
                or not np.array_equal(envelope['classScales'],scales)
                or not np.array_equal(envelope['bounds'],reference+envelope['gamma']*scales)):
            raise ValueError('Invalid coherent class envelope.')
    return state


def pack_memory(state,*,seed,cursor,pack_params):
    """Store model tensors through the checkpoint's shared parameter table."""
    import hashlib
    import json
    validate_memory(state,seed=seed,cursor=cursor)
    encoded=copy.deepcopy(state)
    for entry in encoded['nodes'].values():
        digest=pack_params(entry['referenceParams'])
        if digest!=entry['referenceHash']:raise ValueError('Reference hash changed during checkpoint packing.')
        entry['referenceParams']=digest
    raw=json.dumps(encoded,sort_keys=True,separators=(',',':'),allow_nan=False)
    return encoded,hashlib.sha256(raw.encode()).hexdigest()


def unpack_memory(encoded,digest,*,seed,cursor,unpack_params):
    import hashlib
    import json
    raw=json.dumps(encoded,sort_keys=True,separators=(',',':'),allow_nan=False)
    if hashlib.sha256(raw.encode()).hexdigest()!=digest:raise ValueError('Coherent memory metadata hash differs.')
    state=copy.deepcopy(encoded)
    for entry in state['nodes'].values():entry['referenceParams']=unpack_params(entry['referenceParams'])
    return validate_memory(state,seed=seed,cursor=cursor)
