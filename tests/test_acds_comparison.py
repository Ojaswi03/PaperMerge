from types import SimpleNamespace
import numpy as np
import pytest
from basil_core.acds import cyclic_training_share
from basil_core.cart_comparison_study import selected,effective_config


def arguments(**changes):
    defaults=dict(noise_levels=1,noise_values=None,share_fraction=.05,acds_batches=5,
        arms=['cart','acds'],environments=['joint'],datasets=['cifar10'],partitions=['iid','dirichlet'],
        seeds=[2025,2026,2027],nodes=100,rounds=100,local_steps=5,model='paper',alpha=.5,
        attack_start=20,train_per_class=None,validation_per_class=None)
    return SimpleNamespace(**(defaults|changes))


def test_acds_circulation_delivers_every_shared_sample_without_leaking_holdout():
    chunks=[np.arange(0,20),np.arange(20,60),np.arange(60,90)]
    y=np.arange(100)%10
    before=np.random.get_state();expanded,a=cyclic_training_share(chunks,y,fraction=.25,batches=7)
    after=np.random.get_state()
    assert all(np.array_equal(x,z) for x,z in zip(before,after))
    assert a['sharedCounts']==[5,10,7] and a['sharedUniqueSamples']==22
    assert a['expandedCounts']==[37,52,45]
    shared=set.intersection(*(set(e) for e in expanded))
    assert len(shared)==22
    for original,result in zip(chunks,expanded):
        assert set(original).issubset(result) and len(result)==len(set(result))
        assert not set(range(90,100)).intersection(result)
    assert not a['cryptographicAnonymityImplemented']
    second,b=cyclic_training_share(chunks,y,fraction=.25,batches=7)
    assert a==b and all(np.array_equal(x,z) for x,z in zip(expanded,second))


def test_zero_sharing_is_identity_and_sensitive_eligibility_is_enforced():
    chunks=[np.arange(10),np.arange(10,20)];y=np.arange(24)%2
    result,a=cyclic_training_share(chunks,y,fraction=0)
    assert all(np.array_equal(x,z) for x,z in zip(chunks,result)) and a['sharedUniqueSamples']==0
    with pytest.raises(ValueError,match='Insufficient'):
        cyclic_training_share(chunks,y,fraction=.5,non_sensitive=[0,10])
    with pytest.raises(ValueError,match='fitting'):
        cyclic_training_share(chunks,y,non_sensitive=[0,23])
    with pytest.raises(ValueError,match='disjoint'):
        cyclic_training_share([np.arange(10),np.arange(9,20)],y)


def test_matrix_is_complete_unique_and_keeps_controls_separate():
    _,configs=selected(arguments())
    assert len(configs)==12 and len({c['runId'] for c in configs})==12
    by_seed={}
    for c in configs:
        cfg=effective_config(c);case=c['comparisonCase']
        assert cfg['sigma']==.01 and cfg['actualAttackers']==33 and cfg['localSteps']==5
        assert cfg['cart']==(case['arm']=='cart')
        assert case['shareFraction']==(.05 if case['arm']=='acds' else 0)
        by_seed.setdefault((cfg['partition'],cfg['seed']),[]).append(cfg)
    for pair in by_seed.values():
        assert pair[0]['learningRate']==pair[1]['learningRate'] and pair[0]['nodes']==pair[1]['nodes']


def test_environment_isolation_keeps_defenses_but_removes_only_declared_impairments():
    _,cases=selected(arguments(arms=['cart'],environments=['joint','clean','noise_only','attack_only'],
        partitions=['iid'],seeds=[2025],noise_levels=2))
    assert len(cases)==6
    for case in cases:
        cfg=effective_config(case);env=case['comparisonCase']['environment']
        assert cfg['switches']['byzantine_mitigation'] and cfg['switches']['noise_mitigation']
        assert bool(cfg['actualAttackers'])==(env in ('joint','attack_only'))
        assert bool(cfg['sigma'])==(env in ('joint','noise_only'))
    cases[0]['comparisonCase']['effectiveAttackers']=0
    with pytest.raises(ValueError,match='attacker'):effective_config(cases[0])


def test_partition_expansion_changes_only_acds_and_preserves_original_holdout():
    from basil_core.cart_comparison_study import partitions_for
    _,cases=selected(arguments(nodes=4,seeds=[2025],partitions=['dirichlet'],train_per_class=10,validation_per_class=2))
    y=np.repeat(np.arange(10),130)
    ordinary,held,a=partitions_for(cases[0],y);shared,other,b=partitions_for(cases[1],y)
    assert np.array_equal(held,other) and a['partitionHash']==b['partitionHash']
    assert 'acds' not in a and b['acds']['expandedCounts']==list(map(len,shared))
    assert all(set(p).issubset(q) for p,q in zip(ordinary,shared))
    assert not np.intersect1d(np.concatenate(shared),held).size
