import json

import pytest
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')


def test_all_288_configs_are_explicitly_versioned():
    from basil_core.sharp_study import matrix
    from basil_core.sharp_guarded_protocol import guarded_configuration,validate_config as old_validate
    from basil_core.sharp_cart_v2_protocol import configuration,validate_config
    configs=[configuration(guarded_configuration(c)) for c in matrix(datasets=('mnist','cifar10'))]
    assert len(configs)==288 and len({c['runId'] for c in configs})==288
    for cfg in configs:
        validate_config(cfg)
        with pytest.raises(ValueError):old_validate(cfg)


def test_reference_mistakes_cannot_be_transport_targets():
    from basil_core.sharp_cart_v2 import fit_correction
    h=np.eye(4);y=np.array([0,1,0,1]);p=np.array([[0,2],[2,0],[0,2],[2,0]],float)
    bad=p.copy();good=np.array([[2,0],[0,2],[2,0],[0,2]],float)
    zero,e=fit_correction(h,p,bad,y,np.ones(2),.01)
    assert not np.any(zero) and e['repairedFitExamples']==0
    correction,e=fit_correction(h,p,good,y,np.ones(2),.01)
    assert e['repairedFitExamples']==4 and e['referenceMistakeTargets']==0
    assert e['fittedObjective']<e['zeroObjective'] and np.any(correction)


def test_real_head_repair_is_correctness_qualified_and_preserves_backbone():
    import tensorflow as tf
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_study import configuration
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration as v2
    from basil_core.sharp_cart_v2 import transport,probes,correct_counts
    from basil_core.sharp_math import class_ce
    cfg=v2(guarded_configuration(configuration('mnist','iid',2025,'attack_noise_defense_rla_cart',.01,nodes=4,rounds=2)))
    inputs=tf.keras.Input((10,));features=tf.keras.layers.Dense(10,use_bias=False)(inputs)
    model=tf.keras.Model(inputs,tf.keras.layers.Dense(10)(features))
    model.layers[-2].kernel.assign(np.eye(10));model.layers[-1].kernel.assign(4*np.eye(10));model.layers[-1].bias.assign(np.zeros(10))
    worker=SharpWorker(cfg,model=model);reference=worker.export()
    y=np.repeat(np.arange(10,dtype=np.int32),8);x=np.eye(10,dtype=np.float32)[y]
    fit,check=probes(y,2025,0);_,losses=class_ce(worker.outputs(reference,x[fit]),y[fit])
    _,cl=class_ce(worker.outputs(reference,x[check]),y[check])
    memory=dict(params=reference,losses=losses.tolist(),allowance=[0.]*10,classes=list(range(10)),
        activation=0,checkCorrect=correct_counts(worker.outputs(reference,x[check]),y[check],10).tolist(),checkLosses=cl.tolist())
    params=worker.tensors(reference);params[-2]=np.roll(params[-2],1,axis=1)
    damaged=np.concatenate([a.ravel() for a in params]).astype(np.float32)
    repaired,_,e=transport(worker,damaged,memory,x,y,0,4)
    assert e['scale']>0 and e['repairedFitExamples']==len(fit)
    assert np.all(np.array(e['checkCorrectAfter'])>=e['checkCorrectBefore'])
    assert e['checkAfter']<e['checkBefore'] and e['backboneBeforeHash']==e['backboneAfterHash']
    assert np.mean(worker.outputs(repaired,x).argmax(1)==y)>np.mean(worker.outputs(damaged,x).argmax(1)==y)
    json.dumps(e,allow_nan=False)


def test_v2_partial_round_recovery_is_exact_and_no_cart_matches_guard(tmp_path):
    from tests.test_sharp_guarded_protocol import SmallWorker
    from basil_core.sharp_study import configuration
    from basil_core.sharp_guarded_protocol import guarded_configuration,run as old_run
    from basil_core.sharp_cart_v2_protocol import configuration as v2,run
    from basil_core.sharp_protocol import save_state,load_state
    base=guarded_configuration(configuration('mnist','iid',2025,'attack_noise_defense_rla',.01,nodes=4,rounds=3,attack_start=1))
    cfg=v2(base);worker=SmallWorker();x=np.zeros((40,1),np.float32);y=np.arange(40,dtype=np.int32)%10
    chunks=np.array_split(np.arange(40),4)
    whole=run(cfg,worker,x,y,chunks,x[:10],y[:10]);old=old_run(base,worker,x,y,chunks,x[:10],y[:10])
    np.testing.assert_array_equal(whole['nodeParams'],old['nodeParams'])
    partial=run(cfg,worker,x,y,chunks,x[:10],y[:10],max_activations=7)
    path=tmp_path/'checkpoint.npz';contract={'config':cfg};save_state(path,partial,contract)
    resumed=run(cfg,worker,x,y,chunks,x[:10],y[:10],state=load_state(path,contract,worker.dimension))
    np.testing.assert_array_equal(resumed['nodeParams'],whole['nodeParams'])
    assert resumed['telemetry']==whole['telemetry'] and resumed['roundMetrics']==whole['roundMetrics']


def test_all_288_logical_cases_apply_attack_noise_and_mitigation_switches(monkeypatch):
    from tests.test_sharp_guarded_protocol import SmallWorker
    from basil_core.sharp_study import matrix
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration,run
    import basil_core.sharp_cart_v2 as cart
    calls=[]
    def correction(worker,proposal,previous,x,y,node,activation):
        calls.append((node,activation));return proposal,None,{'scale':0.}
    monkeypatch.setattr(cart,'transport',correction)
    x=np.zeros((80,1),np.float32);y=np.arange(80,dtype=np.int32)%10;chunks=np.array_split(np.arange(80),4)
    for base in matrix(datasets=('mnist','cifar10'),nodes=4,rounds=2,attack_start=0):
        cfg=configuration(guarded_configuration(base));before=len(calls)
        result=run(cfg,SmallWorker(),x,y,chunks,x[:10],y[:10])
        assert result['cursor']==8 and len(calls)-before==(8 if cfg['cart'] else 0)
        flags=cfg['switches']
        for row in result['telemetry']:
            assert (row['outboundHash']!=row['trainedHash'])==row['attackActive']
            assert (row['training']['rlaCoefficient']>0)==flags['noise_mitigation']
            selection=row['selection']
            if flags['byzantine_mitigation'] and selection['availableCopies']:
                assert 'screenGuardFallback' in selection
            elif not flags['byzantine_mitigation']:
                assert selection['selectedCount']==1
        assert sum(row['attackActive'] for row in result['telemetry'])==(2 if flags['attack'] else 0)


def test_v2_paired_report_retains_negative_effects(tmp_path,monkeypatch):
    from tests.test_guarded_cart_pilot_report import fixture
    import basil_core.sharp_cart_v2_protocol as v2
    from scripts.report_cart_v2_pilot import summarize
    roots=fixture(tmp_path,monkeypatch);monkeypatch.setattr(v2,'check_source',lambda _:None)
    for root in roots:
        plan=json.loads((root/'plan.json').read_text());old=plan['configs'][0];cfg=v2.configuration(old)
        folder=root/old['runId'];new=root/cfg['runId'];folder.rename(new)
        run=json.loads((new/'run.json').read_text());run['config']=cfg;(new/'run.json').write_text(json.dumps(run))
        plan['configs']=[cfg];(root/'plan.json').write_text(json.dumps(plan))
        checked=json.loads((root/'independent_validation.json').read_text());checked['rows'][0]['runId']=cfg['runId']
        (root/'independent_validation.json').write_text(json.dumps(checked))
    result=summarize(roots)
    assert result['pairs'][0]['effects']['final']==pytest.approx(-.1)
