from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')
import pytest

from basil_core.sharp_data import prepare_partition
from basil_core.sharp_study import configuration


@pytest.mark.parametrize('dataset,classes',[('mnist',10),('cifar10',10),('cifar100',100)])
@pytest.mark.parametrize('partition',['iid','dirichlet'])
def test_train_holdout_partition_is_complete_disjoint_and_repeatable(dataset,classes,partition):
    y=np.repeat(np.arange(classes,dtype=np.int32),200)
    cfg=configuration(dataset,partition,2025,'clean',0.,nodes=100,rounds=5)
    chunks,holdout,audit=prepare_partition(y,cfg)
    again,h2,a2=prepare_partition(y,cfg)
    assert len(chunks)==100 and min(map(len,chunks))>=2
    assert sum(map(len,chunks))+1000==len(y) and len(holdout)==1000
    assert not set(np.concatenate(chunks)).intersection(holdout)
    assert audit==a2
    for a,b in zip(chunks,again):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(holdout,h2)
    assert len(audit['classCounts'][0])==classes


def test_paired_configs_have_identical_data_and_heldout_is_not_reintroduced():
    y=np.repeat(np.arange(10,dtype=np.int32),300)
    a=configuration('mnist','dirichlet',2025,'attack_noise_defense_rla',.01,nodes=10,train_per_class=100)
    b=configuration('mnist','dirichlet',2025,'attack_noise_defense_rla_cart',.01,nodes=10,train_per_class=100)
    assert prepare_partition(y,a)[2]==prepare_partition(y,b)[2]
    c=configuration('mnist','dirichlet',2026,'clean',0.,nodes=10,train_per_class=100)
    assert prepare_partition(y,a)[2]['holdoutHash']==prepare_partition(y,c)[2]['holdoutHash']


def test_too_small_subset_fails_before_training():
    y=np.repeat(np.arange(10,dtype=np.int32),120)
    cfg=configuration('mnist','iid',2025,'clean',0.,nodes=100,train_per_class=2)
    with pytest.raises(ValueError):prepare_partition(y,cfg)
