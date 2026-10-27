"""Opt-in classifier-only class-memory transport after ordinary local SGD.

The backbone stays at the completed proposal. No hard per-class risk envelope
is enforced; the measured invariant is non-increasing macro CE on a disjoint
local selection probe. Both probe sets are TRAIN data also seen by SGD.
"""
from __future__ import annotations

from basil_core.cart_feasible_worker import FeasibleCartWorker
from basil_core.factorial_worker import PlainWorker


class HeadTransportWorker(FeasibleCartWorker):
    def __init__(self,model,**kwargs):
        import tensorflow as tf
        super().__init__(model,**kwargs)
        dense=model.layers[-1]
        if (not isinstance(dense,tf.keras.layers.Dense) or dense.units!=10 or not dense.use_bias
                or dense.get_config()['activation']!='linear'
                or any(a is not b for a,b in zip(model.trainable_variables[-2:],dense.trainable_variables))):
            raise ValueError('Transport requires a final ten-logit linear classifier.')
        self.features=tf.keras.Model(model.inputs[0],model.layers[-2].output)
        self.optimizer_factory=lambda rate:tf.keras.optimizers.SGD(rate,momentum=kwargs.get('momentum',.9))

    def prepare(self,node,x,y):
        import numpy as np
        from basil_core.cart_local_memory import _digest
        from basil_core.research_protocol import keyed_rng
        labels=np.asarray(y);counts=np.bincount(labels,minlength=10);classes=np.flatnonzero(counts)
        if counts.shape!=(10,) or labels.ndim!=1 or not len(labels) or len(x)!=len(labels):
            raise ValueError('Receiver-local ten-class TRAIN data required.')
        fit=[];check=[]
        for c in classes:
            members=np.flatnonzero(labels==c)
            selected=keyed_rng(self.seed,'cart_memory_probe',node,int(c)).choice(members,min(16,len(members)),replace=False)
            fit.extend(selected.tolist())
            remaining=np.setdiff1d(members,selected)
            check.extend(keyed_rng(self.seed,'cart_transport_check',node,int(c)).choice(
                remaining,min(16,len(remaining)),replace=False).tolist())
        identity=dict(imageHash=_digest(x),labelHash=_digest(labels),counts=counts.tolist(),
            probeIndices=fit,checkProbeIndices=check,classes=classes.tolist())
        before=self.memory['nodes'].get(str(node))
        if before and before['identity']!=identity:raise ValueError('Local transport probes changed.')
        self.probe_x=np.asarray(x[fit]);self.probe_y=labels[fit];self.classes=classes
        self.check_x=np.asarray(x[check]);self.check_y=labels[check]
        return identity

    def restore_local_memory(self,state):
        for entry in state['nodes'].values():
            identity=entry['identity'];fit=set(identity['probeIndices']);check=identity['checkProbeIndices']
            if (fit.intersection(check) or len(set(check))!=len(check)
                    or any(type(i) is not int or not 0<=i<sum(identity['counts']) for i in check)):
                raise ValueError('Transport fitting and selection probes must be disjoint local samples.')
        super().restore_local_memory(state)

    def transport(self,before,proposal):
        import numpy as np
        from basil_core.cart_head_transport import fit_residual,choose_scale,macro_ce
        from basil_core.cart_feasible_memory import class_losses
        from basil_core.research_protocol import params_hash
        backbone=params_hash(proposal[:-2])
        evidence=dict(scale=0.,referenceUsed=before is not None,backboneBeforeHash=backbone,
            backboneAfterHash=backbone,fitProbeCount=len(self.probe_y),checkProbeCount=len(self.check_y),
            checkBefore=None,checkAfter=None,activeClasses=[],classGates=[0.]*10)
        if before is None or not len(self.check_y):return proposal,evidence
        self.load(proposal)
        current=self.predict(self.probe_x).numpy();check_logits=self.predict(self.check_x).numpy()
        h=self.features(self.probe_x,training=False).numpy();hc=self.features(self.check_x,training=False).numpy()
        self.load(before['referenceParams']);reference=self.predict(self.probe_x).numpy()
        _,lp=class_losses(current,self.probe_y);_,lr=class_losses(reference,self.probe_y)
        np.testing.assert_allclose(lr,before['envelope']['referenceLosses'],rtol=1e-5,atol=1e-6)
        allowance=np.maximum(np.asarray(before['envelope']['bounds'])-lr,0.)
        gap=np.maximum(lp-lr,0.)
        gates=np.zeros(10);predicted=reference.argmax(axis=1)
        for j,c in enumerate(self.classes):
            supported=np.any(self.check_y==c)
            knows=(predicted[self.probe_y==c]==c).mean()>.1
            if supported and knows:gates[c]=gap[j]/(gap[j]+allowance[j]+1e-6)
        correction,fit=fit_residual(h,current,reference,self.probe_y,gates)
        scale,selection=choose_scale(check_logits,hc,self.check_y,correction)
        final=[v.copy() for v in proposal]
        final[-2]=(np.asarray(final[-2],dtype=np.float64)+scale*correction[:-1]).astype(np.float32)
        final[-1]=(np.asarray(final[-1],dtype=np.float64)+scale*correction[-1]).astype(np.float32)
        self.load(final);actual=macro_ce(self.predict(self.check_x).numpy(),self.check_y)
        if actual>selection['checkBefore']+1e-6:
            final=proposal;scale=0.;self.load(final);actual=selection['checkBefore']
        evidence.update({**fit,**selection,'scale':scale,'checkAfter':actual,'classGates':gates.tolist(),
            'backboneAfterHash':params_hash(final[:-2]),'fitSelectionDisjoint':True})
        return final,evidence

    def train(self,params,x,y,**kwargs):
        import numpy as np
        from basil_core.cart_feasible_memory import REVISION,CALIBRATION_DRAWS
        from basil_core.research_protocol import params_hash,evaluate_params
        if kwargs['seed']!=self.seed or kwargs.get('semantics')!='paper_absolute_gaussian':
            raise ValueError('Transport requires its declared seed and absolute Gaussian channel.')
        node=kwargs['node_id'];activation=kwargs['round_id']*10+node;sigma=float(kwargs['sigma'])
        old=self.export_local_memory();identity=self.prepare(node,x,y);before=self.memory['nodes'].get(str(node))
        try:
            if before and before['sigma']!=sigma:raise ValueError('Reference channel changed.')
            proposal,record=PlainWorker.train(self,params,x,y,**kwargs)
            trained,evidence=self.transport(before,proposal)
            self.load(trained);losses=self.measure_current()
            admitted=before is None or float(np.mean(losses))<float(np.mean(before['envelope']['referenceLosses']))-1e-4
            if admitted:self.admit(node,trained,losses,identity,activation,sigma)
            after=self.memory['nodes'][str(node)]
            record['proposalTrainingAccuracy']=record['accuracy']
            record['accuracy'],_=evaluate_params(self,trained,x,y,batch_size=kwargs['batch_size'])
            record['cartMemory']=dict(revision=REVISION,beta=1.,guardMode='classifier_transport',
                guardEnabled=False,stepGuards=[],transport=evidence,
                referenceBefore=before['referenceHash'] if before else None,referenceAfter=after['referenceHash'],
                referenceAdmitted=admitted,inputReferenceHash=params_hash(params),proposalOutputHash=params_hash(proposal),
                committedOutputHash=params_hash(trained),referenceLosses=after['envelope']['referenceLosses'],
                nextBounds=after['envelope']['bounds'],channelSigma=sigma,
                calibrationDraws=CALIBRATION_DRAWS if admitted and sigma else 0)
            return trained,record
        except BaseException:
            self.memory=old
            raise
