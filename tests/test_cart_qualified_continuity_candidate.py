import numpy as np
import pytest
from basil_core.cart_qualified_continuity_candidate import qualified_classes,selector
from tests.test_cart_fresh_continuity_candidate import setup
from tests.test_cart_continuity_candidate import Worker


def test_uniform_model_is_not_knowledge_to_protect():
    y=np.repeat(np.arange(10),16);z=np.zeros((len(y),10))
    assert qualified_classes(z,y,z,y,10)==[]
    z[np.arange(len(y)),y]=2.
    assert qualified_classes(z,y,z,y,10)==list(range(10))
    assert qualified_classes(z[:7],y[:7],z,y,10)==[]


def test_qualification_requires_both_probes():
    y=np.repeat(np.arange(2),16);z=3*np.eye(2)[y]
    assert qualified_classes(z,y,np.zeros_like(z),y,2)==[]


def test_bootstrap_allows_fresh_learning_without_a_macro_veto():
    cfg,state,x,y,evidence=setup();cfg['classes']=2;worker=Worker()
    value,row=selector(lambda *_:(np.ones(2),evidence),'qualified')(cfg,state,worker,x,y,1,1,None)
    np.testing.assert_array_equal(value,state['outbound'][0])
    assert row['continuity']['learnedClasses']==[] and row['continuity']['actualRoute']=='fresh'


def test_learned_class_damage_restores_local_carrier():
    cfg,state,_,_,evidence=setup();cfg['classes']=2;worker=Worker();worker.cfg=dict(worker.cfg,cartProbePerClass=16)
    cfg['cartProbePerClass']=16
    state['nodeParams'][1]=[3.,3.];state['outbound'][0]=[.1,.1]
    y=np.repeat(np.arange(2),32);x=y[:,None]
    out,row=selector(lambda *_:(np.ones(2),evidence),'qualified')(cfg,state,worker,x,y,1,1,None)
    np.testing.assert_array_equal(out,state['nodeParams'][1])
    assert row['continuity']['qualifiedInitializationBound'] and row['selectedSenders']==[]


def test_passed_fresh_route_skips_unused_pool_loss_ranking():
    cfg,state,x,y,evidence=setup();cfg['classes']=2;worker=Worker()
    def unused(*_):raise AssertionError('Unused peer pool should not be scored.')
    _,row=selector(unused,'always')(cfg,state,worker,x,y,1,1,None)
    assert row['poolRankingSkipped'] and row['selectedCount']==1


@pytest.mark.parametrize('pipeline',['qualified','admitted','semantic','prior_neutral_class','prior_neutral_macro','prior_neutral_no_commit'])
def test_real_protocol_fresh_process_checkpoint_recovery_preserves_protected_training(tmp_path,monkeypatch,pipeline):
    import tensorflow as tf
    import basil_core.sharp_cart_v2_protocol as protocol
    from basil_core.sharp_study import configuration as base
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.cart_continuity_candidate import ContinuityWorker
    from basil_core.sharp_protocol import save_state,load_state
    worker_class=ContinuityWorker;choose=selector
    if pipeline in ('admitted','semantic') or pipeline.startswith('prior_neutral_'):
        from basil_core.cart_admitted_continuity_candidate import AdmittedWorker
        worker_class=AdmittedWorker
        if pipeline=='admitted':
            from basil_core.cart_admitted_continuity_candidate import selector as choose
        elif pipeline=='semantic':
            from basil_core.cart_semantic_admission_candidate import selector as choose
        else:
            from basil_core.cart_prior_neutral_commit import selector as choose
            if pipeline!='prior_neutral_no_commit':
                import basil_core.sharp_cart_v2 as cart
                from basil_core.cart_prior_neutral_commit import transport
                monkeypatch.setattr(cart,'transport',transport(cart.transport))
    cfg=protocol.configuration(guarded_configuration(base('mnist','iid',2025,
        'attack_noise_defense_rla_cart',.01,nodes=4,rounds=3,attack_start=1)))
    def make():
        inputs=tf.keras.Input((2,));features=tf.keras.layers.Dense(2,use_bias=False)(inputs)
        model=tf.keras.Model(inputs,tf.keras.layers.Dense(10)(features))
        model.layers[-2].kernel.assign(np.eye(2,dtype=np.float32))
        head=np.zeros((2,10),np.float32);head[0,0]=4.;head[1,1]=4.
        model.layers[-1].kernel.assign(head);model.layers[-1].bias.assign(np.zeros(10,np.float32))
        return worker_class(cfg,model=model)
    y=np.tile(np.repeat(np.arange(2,dtype=np.int32),32),4);x=np.eye(2,dtype=np.float32)[y]
    partitions=[np.arange(i*64,(i+1)*64) for i in range(4)]
    mode=('macro' if pipeline=='prior_neutral_macro' else 'class') if pipeline.startswith('prior_neutral_') else 'qualified'
    monkeypatch.setattr(protocol,'select_input',choose(protocol.select_input,mode))
    full=protocol.run(cfg,make(),x,y,partitions,x,y)
    contract=dict(config=cfg,intervention='qualified-continuity test')
    path=tmp_path/'checkpoint.npz';first=make()
    from scripts.cart_revision_worker import compact_record
    callback=(lambda record,_:compact_record(record)) if pipeline.startswith('prior_neutral_') else None
    protocol.run(cfg,first,x,y,partitions,x,y,max_activations=5,checkpoint=lambda s:save_state(path,s,contract),on_activation=callback)
    fresh=make();restored=load_state(path,contract,fresh.dimension)
    resumed=protocol.run(cfg,fresh,x,y,partitions,x,y,state=restored,on_activation=callback)
    np.testing.assert_array_equal(full['nodeParams'],resumed['nodeParams'])
    np.testing.assert_array_equal(full['outbound'],resumed['outbound'])
    if callback:
        for record in full['telemetry']:compact_record(record)
    assert full['telemetry']==resumed['telemetry'] and full['roundMetrics']==resumed['roundMetrics']
    key='qualityScoreBoundApplied' if callback else 'qualifiedInitializationBound'
    assert any(r['selection']['continuity'][key] for r in resumed['telemetry'])
    assert any(r['attackActive'] for r in resumed['telemetry'])
