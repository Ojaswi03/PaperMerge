import numpy as np
import pytest
from basil_core.cart_uncertainty_admission_candidate import sampling_margin


def test_margin_is_zero_for_consistent_class_loss_and_does_not_depend_on_peer():
    labels=np.repeat(np.arange(2),16);z=np.eye(2)[labels]*3
    np.testing.assert_allclose(sampling_margin(z,labels,[0,1]),0,atol=1e-12)


def test_capped_margin_has_explicit_relative_limit():
    labels=np.zeros(16,dtype=int);z=np.zeros((16,2));z[0,0]=-40.
    uncapped=sampling_margin(z,labels,[0],capped=False)[0]
    capped=sampling_margin(z,labels,[0])[0]
    losses=np.logaddexp(0,-z[:,0])
    assert capped==pytest.approx(.1*losses.mean()) and uncapped>capped


def test_margin_shrinks_with_repeated_example_distribution():
    labels=np.zeros(8,dtype=int);z=np.zeros((8,2));z[::2,0]=1
    first=sampling_margin(z,labels,[0],capped=False)[0]
    repeated=sampling_margin(np.tile(z,(4,1)),np.tile(labels,4),[0],capped=False)[0]
    assert repeated<first
