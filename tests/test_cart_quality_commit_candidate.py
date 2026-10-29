import numpy as np
from basil_core.cart_quality_commit_candidate import commit,transport
from basil_core.cart_quality_routing_candidate import class_f1,utility


class Worker:
    def outputs(self,flat,x):return x*flat


def test_commit_scales_the_computed_update_without_extra_gradient_steps():
    worker=Worker();x=np.eye(2).repeat(4,axis=0);y=np.repeat(np.arange(2),4)
    ingress=np.asarray([1.,1.]);proposal=np.asarray([-2.,-2.]);classes=np.arange(2)
    context=dict(ingress=ingress,indices=np.arange(8),split=4,classes=classes,target=np.zeros(2),
        qualified=np.zeros(2,bool),allowance=np.zeros(2),penalty=True)
    logits=worker.outputs(ingress,x)
    context['baseline']=utility(class_f1(logits[:4],y[:4],classes),class_f1(logits[4:],y[4:],classes),
        context['target'],context['qualified'],context['allowance'])
    chosen,row=commit(worker,proposal,context,x,y)
    assert row['scale']<1 and row['committedScore']>=row['baselineScore']-.01
    assert row['optimizerStepsAdded']==0
