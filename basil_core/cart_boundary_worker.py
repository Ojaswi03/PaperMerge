"""Diagnostic: enforce class budgets at communication commit boundaries.

Local SGD may explore outside the envelope. Only its final committed output is
restored, so this variant makes no per-optimizer-step retention claim. It shares
the ordinary SGD/EBM implementation and the coherent channel calibration.
"""
from __future__ import annotations

from basil_core.cart_feasible_worker import FeasibleCartWorker
from basil_core.factorial_worker import PlainWorker


class BoundaryCartWorker(FeasibleCartWorker):
    def __init__(self,model,*,seed=2025,momentum=.9,weight_decay=.0005,
                 restore_input=True,risk_metric='cross_entropy',require_breadth=False):
        import tensorflow as tf
        if type(restore_input) is not bool or risk_metric not in ('cross_entropy','classification_error'):
            raise ValueError('Explicit reception policy and supported class risk metric required.')
        super().__init__(model,seed=seed,momentum=momentum,weight_decay=weight_decay,refresh_policy='envelope')
        self.restore_input=restore_input;self.risk_metric=risk_metric
        if type(require_breadth) is not bool or require_breadth and risk_metric!='classification_error':
            raise ValueError('Breadth qualification is defined for supported-class error rates.')
        self.require_breadth=require_breadth
        self.optimizer_factory=lambda rate:tf.keras.optimizers.SGD(learning_rate=rate,momentum=momentum)

    def mean_probe_ce(self):
        from basil_core.cart_feasible_memory import class_losses
        return float(class_losses(self.predict(self.probe_x).numpy(),self.probe_y)[1].mean())

    def measure_current(self):
        if self.risk_metric=='cross_entropy':return super().measure_current()
        import numpy as np
        from basil_core.cart_feasible_memory import class_losses
        logits=self.predict(self.probe_x).numpy()
        classes,_=class_losses(logits,self.probe_y)
        if not np.array_equal(classes,self.classes):raise ValueError('Class support changed.')
        predicted=logits.argmax(axis=1)
        return np.asarray([(predicted[self.probe_y==c]!=c).mean() for c in classes],dtype=np.float64)

    def admit(self,node,params,losses,identity,activation,sigma):
        from basil_core.cart_reference_qualification import qualified_error_reference
        self.load(params);mean_ce=self.mean_probe_ce()
        super().admit(node,params,losses,identity,activation,sigma)
        self.memory['nodes'][str(node)].update(riskMetric=self.risk_metric,referenceMeanCE=mean_ce,
            referenceQualified=qualified_error_reference(losses) if self.require_breadth else None)

    def restore_local_memory(self,state):
        import math
        from basil_core.cart_reference_qualification import qualified_error_reference
        for entry in state['nodes'].values():
            if (entry.get('riskMetric')!=self.risk_metric or not math.isfinite(entry['referenceMeanCE'])
                    or entry['referenceMeanCE']<0):raise ValueError('Boundary risk state differs from its declared metric.')
            if self.require_breadth and entry.get('referenceQualified')!=qualified_error_reference(entry['envelope']['referenceLosses']):
                raise ValueError('Reference qualification differs from measured class errors.')
        super().restore_local_memory(state)

    def admission(self,before,losses):
        import numpy as np
        from basil_core.cart_feasible_memory import coherent_admission,within_envelope
        from basil_core.cart_reference_qualification import qualified_error_reference
        if before is None:return True
        bounds=before['envelope']['bounds'];old=before['envelope']['referenceLosses']
        if self.risk_metric=='cross_entropy':return coherent_admission(old,losses,bounds=bounds)
        difference=float(np.mean(losses)-np.mean(old))
        improvement=difference < -1e-4 or abs(difference)<=1e-12 and self.mean_probe_ce()<before['referenceMeanCE']-1e-4
        if self.require_breadth:
            old_qualified=qualified_error_reference(old);new_qualified=qualified_error_reference(losses)
            if not old_qualified:return improvement or new_qualified and difference<=1e-12
            if not new_qualified:return False
        return within_envelope(losses,bounds) and improvement

    def repair_proposal(self,anchor,proposal,bounds):
        from basil_core.cart_feasible_memory import restore_feasible
        return restore_feasible(anchor,proposal,bounds,self.measure_params)

    def train(self,params,x,y,**kwargs):
        from basil_core.cart_feasible_memory import REVISION,CALIBRATION_DRAWS,restore_feasible,within_envelope
        from basil_core.research_protocol import params_hash,evaluate_params
        if kwargs['seed']!=self.seed or kwargs.get('semantics')!='paper_absolute_gaussian':
            raise ValueError('Boundary CART needs the declared seed and absolute Gaussian channel.')
        node=kwargs['node_id'];activation=kwargs['round_id']*10+node;sigma=float(kwargs['sigma'])
        old=self.export_local_memory();identity=self.prepare(node,x,y)
        before=self.memory['nodes'].get(str(node))
        from basil_core.cart_reference_qualification import qualified_error_reference
        protected=before is not None and (not self.require_breadth or qualified_error_reference(before['envelope']['referenceLosses']))
        incoming=params;input_evidence=dict(scale=1.,trials=[])
        self.active_bounds=None
        try:
            if before and before['sigma']!=sigma:raise ValueError('Channel changed without recalibration.')
            if protected:
                self.active_bounds=before['envelope']['bounds']
                if not within_envelope(self.measure_params(before['referenceParams']),self.active_bounds):
                    raise ValueError('The coherent reference is not feasible.')
                if self.restore_input:
                    incoming,input_evidence=restore_feasible(before['referenceParams'],params,self.active_bounds,self.measure_params)
            proposal,record=PlainWorker.train(self,incoming,x,y,**kwargs)
            trained=proposal;output_evidence=dict(scale=1.,trials=[],classLosses=self.measure_params(proposal).tolist())
            if protected:
                # Input restoration established a feasible activation start.
                # No convexity or feasibility of the intervening SGD path is assumed.
                anchor=incoming if self.restore_input else before['referenceParams']
                trained,output_evidence=self.repair_proposal(anchor,proposal,self.active_bounds)
            self.load(trained);losses=self.measure_current()
            admitted=self.admission(before,losses)
            if admitted:self.admit(node,trained,losses,identity,activation,sigma)
            after=self.memory['nodes'][str(node)]
            record['proposalTrainingAccuracy']=record['accuracy']
            record['accuracy'],_=evaluate_params(self,trained,x,y,batch_size=kwargs['batch_size'])
            record['cartMemory']=dict(revision=REVISION,beta=1.,guardMode='activation_boundary',
                riskMetric=self.risk_metric,inputRestorationEnabled=self.restore_input,
                referenceQualified=after.get('referenceQualified'),
                referenceWasQualified=bool(protected) if self.require_breadth else None,
                activeReferenceRisks=before['envelope']['referenceLosses'] if before else None,
                probeCount=len(self.probe_y),classes=self.classes.tolist(),
                referenceBefore=before['referenceHash'] if before else None,
                referenceAfter=after['referenceHash'],referenceAdmitted=admitted,
                inputReferenceHash=params_hash(incoming),inputBridge=input_evidence,
                proposalOutputHash=params_hash(proposal),committedOutputHash=params_hash(trained),outputBridge=output_evidence,
                activeBounds=self.active_bounds,referenceLosses=after['envelope']['referenceLosses'],
                nextBounds=after['envelope']['bounds'],channelSigma=sigma,
                calibrationDraws=CALIBRATION_DRAWS if admitted and sigma else 0,
                guardEnabled=protected,stepGuards=[])
            return trained,record
        except BaseException:
            self.memory=old
            raise
        finally:self.active_bounds=None
