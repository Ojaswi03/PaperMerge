import numpy as np
from basil_core.cart_semantic_admission_candidate import repair


class Worker:
    cfg=dict(seed=2025,cartProbePerClass=16,classes=2,cartRidgeFraction=.01)
    sizes=[1,4,2];dimension=7
    def tensors(self,flat):return [flat[:1],flat[1:5].reshape(2,2),flat[5:]]
    def outputs(self,flat,x,features=False):
        h=np.asarray(x)*flat[0]
        return h if features else h@flat[1:5].reshape(2,2)+flat[5:]


def test_semantic_admission_repairs_head_on_the_incoming_feature_basis():
    worker=Worker();y=np.repeat(np.arange(2,dtype=int),32);x=np.eye(2)[y]
    own=np.asarray([1.,3.,0.,0.,3.,0.,0.]);peer=np.asarray([2.,.2,0.,0.,2.,0.,0.])
    from basil_core.sharp_cart import probes
    from basil_core.sharp_math import class_ce
    fit,check=probes(y,2025,0,16)
    losses=lambda flat,indices:class_ce(worker.outputs(flat,x[indices])[y[indices]==0],y[indices][y[indices]==0])[1].tolist()
    row=dict(learnedClasses=[0],ownFit=losses(own,fit),ownCheck=losses(own,check),fitBudget=[.01],checkBudget=[.01])
    fixed,evidence=repair(worker,peer,own,x,y,0,row)
    assert fixed is not None and evidence['accepted']
    np.testing.assert_array_equal(fixed[:1],peer[:1])
    assert losses(fixed,check)[0]<=row['ownCheck'][0]+.01


def test_bias_only_repair_cannot_admit_collapsed_incoming_features():
    worker=Worker();y=np.repeat(np.arange(2,dtype=int),32);x=np.eye(2)[y]
    own=np.asarray([1.,3.,0.,0.,3.,0.,0.]);peer=np.zeros(7)
    row=dict(learnedClasses=[0],ownFit=[.05],ownCheck=[.05],fitBudget=[.01],checkBudget=[.01])
    fixed,evidence=repair(worker,peer,own,x,y,0,row)
    assert fixed is None and not evidence['accepted'] and 'collapsed' in evidence['reason']
