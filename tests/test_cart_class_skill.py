import itertools
import numpy as np
import pytest
from basil_core.cart_class_skill import class_skill


def test_constant_class_prediction_has_zero_skill_under_balanced_and_skewed_data():
    for labels in (np.repeat(np.arange(10),16),np.asarray([0]*90+[1]*10)):
        z=np.zeros((len(labels),10));z[:,0]=4.
        np.testing.assert_array_equal(class_skill(z,labels,np.unique(labels)),np.zeros(len(np.unique(labels))))


def test_skill_matches_binary_cohen_kappa_and_perfect_discrimination():
    y=np.asarray([0,0,0,1,1,1]);prediction=np.asarray([0,0,1,1,1,1]);z=np.eye(2)[prediction]
    observed=np.mean(y==prediction);chance=sum(np.mean(y==c)*np.mean(prediction==c) for c in range(2))
    np.testing.assert_allclose(class_skill(z,y,[0,1],metric='kappa'),(observed-chance)/(1-chance))
    np.testing.assert_array_equal(class_skill(np.eye(2)[y],y,[0,1]),np.ones(2))


def test_random_label_permutations_have_zero_average_skill_for_fixed_marginals():
    labels=np.asarray([0,0,0,1,1,1]);prediction=np.asarray([0,0,1,1,1,1]);z=np.eye(2)[prediction]
    values=[class_skill(z,np.asarray(p),[0,1]) for p in set(itertools.permutations(labels))]
    np.testing.assert_allclose(np.mean(values,axis=0),0.,atol=1e-15)


def test_mcc_matches_the_standard_four_cell_confusion_formula():
    y=np.asarray([0,0,0,1,1,1]);prediction=np.asarray([0,0,1,1,1,1]);z=np.eye(2)[prediction]
    expected=(2*3-0*1)/np.sqrt((2+0)*(2+1)*(3+0)*(3+1))
    np.testing.assert_allclose(class_skill(z,y,[0,1]),expected)
