"""A small classifier residual must earn its commit on separate local probes."""
import pytest


def test_balanced_ridge_solution_satisfies_normal_equations_and_zero_bound():
    import numpy as np
    from basil_core.cart_head_transport import fit_residual
    rng=np.random.default_rng(71);h=rng.normal(size=(31,4));p=rng.normal(size=(31,10));r=rng.normal(size=(31,10))
    y=np.arange(31,dtype=np.int32)%3;g=np.linspace(0,1,10)
    delta,evidence=fit_residual(h,p,r,y,g)
    a=np.column_stack((h,np.ones(len(h))));counts=np.bincount(y,minlength=10);w=1/(3*counts[y])
    target=g[y,None]*((r-r.mean(axis=1,keepdims=True))-(p-p.mean(axis=1,keepdims=True)))
    np.testing.assert_allclose(a.T@(w[:,None]*(a@delta-target))+evidence['ridge']*delta,0.,atol=1e-12)
    assert evidence['fittedObjective']<=evidence['zeroObjective']
    unchanged,_=fit_residual(h,p,r,y,np.zeros(10))
    np.testing.assert_array_equal(unchanged,np.zeros_like(unchanged))
    shifted,_=fit_residual(h,p+10,r-20,y,g)
    np.testing.assert_allclose(shifted,delta,atol=1e-12)


def test_local_selection_can_reject_harmful_and_accept_helpful_corrections():
    import numpy as np
    from basil_core.cart_head_transport import choose_scale
    h=np.eye(3);y=np.arange(3,dtype=np.int32);logits=np.zeros((3,10));delta=np.zeros((4,10))
    for c in range(3):delta[c,c]=2.
    scale,e=choose_scale(logits,h,y,delta)
    assert scale==1. and e['checkAfter']<e['checkBefore']
    scale,e=choose_scale(logits,h,y,-delta)
    assert scale==0. and e['checkAfter']==e['checkBefore']


def test_actual_transport_keeps_backbone_and_uses_disjoint_local_selection():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_head_worker import HeadTransportWorker
    from basil_core.research_protocol import params_hash
    inputs=tf.keras.Input((32,32,3));features=tf.keras.layers.Conv2D(3,1,use_bias=False)(inputs)
    features=tf.keras.layers.GlobalAveragePooling2D()(features)
    model=tf.keras.Model(inputs,tf.keras.layers.Dense(10)(features))
    worker=HeadTransportWorker(model)
    reference=worker.export();reference[0]=np.eye(3,dtype=np.float32).reshape(1,1,3,3)
    reference[-2]=np.zeros((3,10),np.float32);reference[-1]=np.zeros(10,np.float32)
    for c in range(3):reference[-2][c,c]=3.
    y=np.repeat(np.arange(3,dtype=np.int32),40);x=np.zeros((120,32,32,3),np.float32)
    for i,c in enumerate(y):x[i,:,:,c]=1.
    identity=worker.prepare(0,x,y)
    assert not set(identity['probeIndices']).intersection(identity['checkProbeIndices'])
    worker.load(reference);worker.admit(0,reference,worker.measure_current(),identity,0,0.)
    proposal=[v.copy() for v in reference];proposal[-2][0,0]=-3.;proposal[-2][0,3]=3.
    final,e=worker.transport(worker.memory['nodes']['0'],proposal)
    assert e['scale']>0 and e['checkAfter']<e['checkBefore']
    assert e['classGates'][0]>0 and e['activeClasses']==[0]
    assert params_hash(final[:-2])==params_hash(proposal[:-2])
    worker.restore_local_memory(worker.export_local_memory())
    with pytest.raises(ValueError):
        bad=worker.export_local_memory();bad['nodes']['0']['identity']['checkProbeIndices'][0]=identity['probeIndices'][0]
        worker.restore_local_memory(bad)


def test_mini_is_an_explicit_five_round_paired_development_stage():
    from basil_core.cart_coherent_study import configuration,validate_config
    for variant in ('none','coherent_head_transport'):
        cfg=configuration('dirichlet',2025,.01,variant,stage='mini',training_objective='local_prior',compiled_optimizer=True)
        validate_config(cfg)
        assert cfg['nRounds']==5 and cfg['attackStartRound']==1 and not cfg['officialTestUsed']
