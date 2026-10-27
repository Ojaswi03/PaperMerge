"""Activation-repair prototype, using established gradient projection.

GEM/GEC supply the halfspace projection idea. Here the proposed direction is
the whole completed local activation, and the gradients concern only classes
whose measured risk budget is exceeded. Actual risk checks remain essential.
"""
from __future__ import annotations

from basil_core.cart_boundary_worker import BoundaryCartWorker


class ProjectedCommitWorker(BoundaryCartWorker):
    def __init__(self,model,**kwargs):
        super().__init__(model,restore_input=False,risk_metric='classification_error',require_breadth=True,**kwargs)

    def repair_proposal(self,anchor,proposal,bounds):
        import numpy as np
        import tensorflow as tf
        from basil_core.cart_activation_projection import project_displacement
        from basil_core.cart_feasible_memory import restore_feasible,within_envelope
        self.load(proposal);risks=self.measure_current()
        if within_envelope(risks,bounds):
            return [v.copy() for v in proposal],dict(scale=1.,trials=[],classLosses=risks.tolist(),
                projection=dict(status='not_needed',family='unchanged',proposalFidelity=1.))
        scalar,scalar_evidence=restore_feasible(anchor,proposal,bounds,self.measure_params)
        unsafe=self.classes[risks>np.asarray(bounds)+1e-6]
        rows=[];row_metadata=[]
        for endpoint,params in (('reference',anchor),('proposal',proposal)):
            self.load(params)
            for c in unsafe:
                mask=self.probe_y==c;images=tf.convert_to_tensor(self.probe_x[mask]);labels=tf.convert_to_tensor(self.probe_y[mask],tf.int32)
                with tf.GradientTape() as tape:
                    logits=self.model(images,training=False)
                    ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
                gradients=tape.gradient(ce,self.model.trainable_variables)
                if any(g is None for g in gradients):raise ValueError('Disconnected class-CE gradient.')
                row=np.concatenate([np.asarray(g.numpy(),dtype=np.float64).reshape(-1) for g in gradients])
                rows.append(row);row_metadata.append(dict(endpoint=endpoint,classId=int(c),gradientNorm=float(np.linalg.norm(row))))
        displacement=np.concatenate([(np.asarray(p,dtype=np.float64)-a).reshape(-1) for a,p in zip(anchor,proposal)])
        direction,solver=project_displacement(displacement,np.asarray(rows))
        candidate=[];cursor=0
        for a in anchor:
            size=a.size;candidate.append((np.asarray(a,dtype=np.float64)+direction[cursor:cursor+size].reshape(a.shape)).astype(np.float32));cursor+=size
        projected,projected_evidence=restore_feasible(anchor,candidate,bounds,self.measure_params)
        def distance(params):
            return float(sum(np.sum((np.asarray(a,dtype=np.float64)-b)**2) for a,b in zip(params,proposal)))
        scalar_distance=distance(scalar);projected_distance=distance(projected)
        use_projected=projected_distance<scalar_distance
        final,evidence=(projected,projected_evidence) if use_projected else (scalar,scalar_evidence)
        evidence=dict(evidence)
        selected_distance=min(scalar_distance,projected_distance)
        norm=float(np.linalg.norm(displacement))
        evidence['projection']=dict(solver=solver,family='projected' if use_projected else 'scalar',
            unsafeClasses=unsafe.tolist(),gradientRows=row_metadata,
            scalarSquaredDistance=scalar_distance,projectedSquaredDistance=projected_distance,
            selectedSquaredDistance=selected_distance,scalarScale=scalar_evidence['scale'],
            proposalFidelity=1.-selected_distance**.5/norm if norm else 1.)
        self.load(final)
        if not within_envelope(self.measure_current(),bounds):raise ValueError('Projected commit is not actually risk-feasible.')
        return final,evidence
