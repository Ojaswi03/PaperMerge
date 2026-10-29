import numpy as np
from basil_core.cart_discriminative_continuity_candidate import qualified_classes
from basil_core.cart_discriminative_admission_candidate import collapse_check,balanced_accuracy


def test_constant_majority_prediction_cannot_be_class_memory():
    y=np.repeat(np.arange(10),16);z=np.zeros((len(y),10));z[:,0]=4.
    from basil_core.cart_qualified_continuity_candidate import qualified_classes as recall_only
    assert recall_only(z,y,z,y,10)==[0]
    assert qualified_classes(z,y,z,y,10)==[]
    assert balanced_accuracy(z,y)==.1
    z=np.eye(10)[y]*4.
    assert qualified_classes(z,y,z,y,10)==list(range(10))


def test_discrimination_needs_counterexamples():
    y=np.zeros(32,dtype=int);z=np.zeros((32,10));z[:,0]=4.
    assert qualified_classes(z,y,z,y,10)==[]


def test_balanced_guard_rejects_collapse_but_does_not_freeze_a_biased_teacher():
    from tests.test_cart_continuity_candidate import Worker
    worker=Worker();y=np.repeat(np.arange(2),32);x=y[:,None]
    reject,evidence=collapse_check(worker,np.asarray([-.1,-.1]),np.asarray([3.,3.]),x,y,0)
    assert reject and len(evidence)==2
    reject,_=collapse_check(worker,np.asarray([3.,3.]),np.asarray([3.,-.1]),x,y,0)
    assert not reject
