"""Certificates require repeated computation, not agreement on owner claims."""
from dataclasses import replace
import subprocess
import sys

import pytest


@pytest.fixture
def fixture():
    import numpy as np
    from basil_core.replicated_verification import ComputeRequest,ComputeResult,generate_identity
    keys={i:generate_identity() for i in range(10)}
    request=ComputeRequest('run-a',0,0,1,0,'gradient','a'*64,'b'*64,(0,1),(7,8))
    good=ComputeResult.from_values(1.,[np.array([1.,2.],np.float32)])
    bad=ComputeResult.from_values(0.,[np.array([-999.,999.],np.float32)])
    return request,keys,good,bad


def collector(fixture):
    from basil_core.replicated_verification import CertificateCollector
    request,keys,*_=fixture
    return CertificateCollector(request,{i:k[1] for i,k in keys.items()})


@pytest.mark.parametrize('fault',['poison','withhold','forged','stale','wrong_kind','equivocate'])
def test_four_faulty_replicas_do_not_replace_five_honest_results(fixture,fault):
    from basil_core.replicated_verification import sign_result,ComputeResult
    request,keys,good,bad=fixture;c=collector(fixture)
    for i in request.committee:
        result=bad if i<=4 else good
        if i<=4 and fault=='withhold':continue
        vote=sign_result(keys[i][0],i,request,result)
        if i<=4 and fault=='forged':vote=replace(vote,signature=b'0'*64)
        if i<=4 and fault=='stale':vote=sign_result(keys[i][0],i,replace(request,run_id='old'),good)
        if i<=4 and fault=='wrong_kind':vote=sign_result(keys[i][0],i,request,ComputeResult.from_values(0.))
        c.add(vote)
        if i<=4 and fault=='equivocate':c.add(sign_result(keys[i][0],i,request,good))
    result,proof=c.certificate()
    assert result.digest==good.digest and proof['matchingSigners']==[5,6,7,8,9]


def test_duplicates_and_impersonation_cannot_build_quorum(fixture):
    from basil_core.replicated_verification import sign_result
    request,keys,good,_=fixture;c=collector(fixture)
    for i in range(1,5):
        v=sign_result(keys[i][0],i,request,good)
        assert c.add(v)
        for _ in range(9):assert not c.add(v)
    assert not c.add(sign_result(keys[1][0],5,request,good))
    assert not c.add(sign_result(keys[0][0],0,request,good))  # Owner cannot vote.
    with pytest.raises(ValueError,match='certificate'):c.certificate()


def test_payload_tampering_and_replay_fail(fixture):
    from basil_core.replicated_verification import sign_result
    request,keys,good,bad=fixture;c=collector(fixture)
    signed=sign_result(keys[1][0],1,request,good)
    assert not c.add(replace(signed,result=bad))
    for kwargs in ({'model_hash':'c'*64},{'data_root':'d'*64},{'indices':(2,3)},
                   {'augmentation_key':(8,9)},{'step':1},{'round_id':2},{'receiver':1}):
        assert not c.add(sign_result(keys[1][0],1,replace(request,**kwargs),good))


def test_honest_disagreement_fails_closed_without_tolerance_or_averaging(fixture):
    from basil_core.replicated_verification import sign_result
    request,keys,good,_=fixture;c=collector(fixture)
    for i in request.committee:c.add(sign_result(keys[i][0],i,request,replace(good,cross_entropy=1.+i*1e-8)))
    with pytest.raises(ValueError,match='certificate'):c.certificate()


def test_fault_bound_is_not_silently_extended_to_five(fixture):
    from basil_core.replicated_verification import sign_result
    request,keys,good,bad=fixture;c=collector(fixture)
    for i in request.committee:c.add(sign_result(keys[i][0],i,request,bad if i<=5 else good))
    assert c.certificate()[0].digest==bad.digest  # Outside f<=4, no claimed guarantee.


def test_immutable_payload_and_nonfinite_rejection(fixture):
    import numpy as np
    from basil_core.replicated_verification import ComputeResult
    good=fixture[2]
    with pytest.raises(ValueError):good.arrays()[0][0]=9
    with pytest.raises(ValueError):good.arrays()[0].flags.writeable=True
    for value in (float('nan'),float('inf')):
        with pytest.raises(ValueError):ComputeResult.from_values(value)
        with pytest.raises(ValueError):ComputeResult.from_values(1.,[np.array([value],np.float32)])


def test_requests_reject_future_classes_and_augmented_selection(fixture):
    r=fixture[0]
    with pytest.raises(ValueError,match='Future'):replace(r,round_id=0,owner=1)
    with pytest.raises(ValueError,match='unaugmented'):replace(r,kind='selection_ce')
    with pytest.raises(ValueError):replace(r,indices=(0,0))


def test_new_modules_import_lazily():
    code=('import basil_core.replicated_verification,basil_core.replicated_training_service;'
          'import sys;assert not ({"numpy","tensorflow","cryptography","wandb","matplotlib"}&sys.modules.keys())')
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.fixture
def service_fixture():
    import numpy as np
    import tensorflow as tf
    from basil_core.replicated_training_service import TrainingReference,TrainingReplica,ReplicatedTrainingService
    from basil_core.replicated_verification import ComputeRequest
    from basil_core.research_protocol import params_hash
    images=np.random.default_rng(123).uniform(size=(20,32,32,3)).astype(np.float32)
    labels=np.repeat(np.arange(10,dtype=np.int32),2)
    ref=TrainingReference(images,labels,source_split='train',synthetic=True)
    def model():
        return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
             tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.Constant(.1))])
    replicas={i:TrainingReplica(i,model(),ref) for i in range(10)}
    service=ReplicatedTrainingService(replicas)
    params=[w.numpy() for w in replicas[0].model.trainable_variables]
    request=ComputeRequest('synthetic',0,2,1,0,'gradient',params_hash(params),ref.root,(4,5),(7,8))
    return service,replicas,ref,params,request


@pytest.mark.parametrize('fault',['poison','withhold'])
def test_real_independent_recomputation_tolerates_four_faulty_endpoints(service_fixture,fault):
    import numpy as np
    from basil_core.replicated_verification import ComputeResult,sign_result
    service,replicas,ref,params,request=service_fixture
    expected=replicas[3].compute(request,params)
    replicas[3].evaluations=0
    for i in (0,1,5,7):
        r=replicas[i]
        if fault=='withhold':
            def missing(*_):raise TimeoutError('simulated unavailable replica')
            r.reply=missing
        else:
            def poisoned(req,p,r=r):
                honest=r.compute(req,p)
                bad=ComputeResult.from_values(0.,[np.full_like(g,-100.) for g in honest.arrays()])
                return sign_result(r._private,r.identity,req,bad)
            r.reply=poisoned
    result,proof=service.evaluate(request,params)
    assert result.digest==expected.digest and len(proof['matchingSigners'])==5
    assert service.calls==9
    assert sum(r.evaluations for r in replicas.values())==(5 if fault=='withhold' else 9)
    assert not proof['physicalRPC'] and not proof['fullResearchReady']


def test_reference_rejects_test_data_and_is_immutable(service_fixture):
    import numpy as np
    from basil_core.replicated_training_service import TrainingReference
    _,_,ref,_,_=service_fixture
    with pytest.raises(ValueError):ref.images[0,0,0,0]=1.
    with pytest.raises(ValueError):ref.labels.flags.writeable=True
    with pytest.raises(ValueError,match='TRAIN'):TrainingReference(ref.images,ref.labels,source_split='test',synthetic=True)
    with pytest.raises(ValueError,match='50000'):TrainingReference(ref.images,ref.labels,source_split='train')


def test_certified_selection_ignores_four_forged_owner_scores(service_fixture):
    import numpy as np
    from basil_core.research_protocol import Snapshot,params_hash
    from basil_core.replicated_verification import ComputeResult,sign_result,ComputeRequest
    from basil_core.replicated_training_service import verified_balanced_selection
    service,replicas,ref,params,_=service_fixture
    candidates=[]
    for sender in range(5,10):
        p=[a.copy() for a in params];p[-1][0]=0. if sender==8 else float(sender)
        candidates.append(Snapshot(sender,0,p))
    for i in (0,1,5,7):
        r=replicas[i]
        def lying(req,p,r=r):
            honest=r.compute(req,p)
            bad=ComputeResult.from_values(0. if float(p[-1][0])==9. else 100.)
            return sign_result(r._private,r.identity,req,bad)
        r.reply=lying
    selected,records=verified_balanced_selection(service,candidates,reference=ref,run_id='synthetic',
        receiver=0,round_id=1,probes={c:(2*c,2*c+1) for c in range(10)})
    assert selected.sender_id==8
    assert service.calls==5*10*9
    assert all(len(proof['matchingSigners'])>=5 for r in records for proof in r['certificates'].values())
    assert all(params_hash(s.params)==r['parameterHash'] for s,r in zip(candidates,records))


@pytest.mark.parametrize('fault',['none','poison','withhold'])
def test_certified_gradients_preserve_full_objective_and_ignore_lambda(service_fixture,fault):
    """Verification changes provenance, not the balanced CE/EBM/anchor formula."""
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_ebm import owner_objective_gradients
    from basil_core.research_audit import augment_batch
    from basil_core.replicated_verification import ComputeResult,sign_result
    service,replicas,ref,params,request=service_fixture
    for i in (() if fault=='none' else (0,1,5,7)):
        replica=replicas[i]
        if fault=='withhold':
            def unavailable(*_):raise TimeoutError('injected withheld gradient')
            replica.reply=unavailable
        else:
            def poisoned(req,p,r=replica):
                honest=r.compute(req,p)
                bad=ComputeResult.from_values(999.,[np.full_like(g,-100.) for g in honest.arrays()])
                return sign_result(r._private,r.identity,req,bad)
            replica.reply=poisoned
    helpers=[]
    for owner in range(1,10):
        r=replace(request,owner=owner,indices=(2*owner,2*owner+1))
        response,proof=service.gradient_response(r,params)
        assert len(proof['matchingSigners'])>=5
        helpers.append(response)
    model=replicas[0].model
    for w,p in zip(model.trainable_variables,params):w.assign(p)
    anchor=[p.copy()-.01 for p in params]
    saved=[p.copy() for p in anchor]
    x=augment_batch(ref.images[:2],np.array([7,8],np.int32))
    y=ref.labels[:2]
    calls=[]
    for lam in (1.,999999.):
        calls.append(owner_objective_gradients(model,x,y,anchor,helpers,
            receiver=0,round_id=1,optimizer_step=0,mu=.01,sigma_e=.01,
            ebm=True,legacy_lambda=lam))
    a,values,meta=calls[0]
    for g,h in zip(a,calls[1][0]):np.testing.assert_array_equal(g,h)
    np.testing.assert_array_equal([float(v) for v in values],[float(v) for v in calls[1][1]])
    assert meta['legacyLambdaIgnored']
    # Independent, single connected autodiff objective; no helper aggregation.
    with tf.GradientTape() as outer:
        losses=[]
        for owner in range(10):
            xx=augment_batch(ref.images[2*owner:2*owner+2],np.array([7,8],np.int32))
            yy=ref.labels[2*owner:2*owner+2]
            with tf.GradientTape() as inner:
                ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=yy,logits=model(xx,training=True)))
            if owner==0:
                local_grads=inner.gradient(ce,model.trainable_variables)
                penalty=.0001*tf.add_n([tf.reduce_sum(g*g) for g in local_grads])
            losses.append(ce)
        distance=tf.add_n([tf.reduce_sum((w-r)**2) for w,r in zip(model.trainable_variables,anchor)])
        total=tf.add_n(losses)/10+penalty+.005*distance
    expected=outer.gradient(total,model.trainable_variables)
    np.testing.assert_allclose(float(values[2]),float(total),rtol=1e-6,atol=1e-7)
    for g,h,p in zip(a,expected,params):
        np.testing.assert_allclose(g,h,rtol=1e-5,atol=1e-7)
        np.testing.assert_allclose(p-.05*g.numpy(),p-.05*h.numpy(),rtol=1e-6,atol=1e-7)
    for before,after in zip(saved,anchor):np.testing.assert_array_equal(before,after)
    for p,w in zip(params,model.trainable_variables):np.testing.assert_array_equal(p,w)
    assert service.calls==81  # Nine helpers, independently recomputed nine times each.
