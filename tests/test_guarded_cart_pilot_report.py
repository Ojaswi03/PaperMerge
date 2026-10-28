import json

import pytest


def fixture(tmp_path,monkeypatch):
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    from basil_core.sharp_study import configuration
    import basil_core.sharp_guarded_protocol as guarded
    monkeypatch.setattr(guarded,'check_source',lambda _:None)
    roots=[]
    for cart in (False,True):
        condition='attack_noise_defense_rla'+('_cart' if cart else '')
        cfg=guarded.guarded_configuration(configuration('mnist','iid',2025,condition,.01,nodes=4,rounds=10,attack_start=2))
        root=tmp_path/condition;folder=root/cfg['runId'];folder.mkdir(parents=True);roots.append(root)
        (root/'plan.json').write_text(json.dumps(dict(source={},configs=[cfg])))
        (root/'independent_validation.json').write_text(json.dumps(dict(status='ALL_SELECTED_RUNS_VERIFIED',verified=1,rows=[{'runId':cfg['runId']}])))
        (folder/'run.json').write_text(json.dumps(dict(status='COMPLETED',config=cfg,source={},
            initialHash='same',partition={'partitionHash':'same'},data={'same':True},runtime={'device':'CPU'})))
        np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((10,4),.7 if cart else .8))
        history=[dict(inputHash='same',trainedHash='same',cart={'scale':1} if cart and i>=4 else None) for i in range(40)]
        (folder/'activation_telemetry.json').write_text(json.dumps(history))
    return roots


def test_single_seed_negative_effect_is_retained_without_seed_aggregation(tmp_path,monkeypatch):
    from scripts.report_guarded_sharp_cart_pilot import summarize
    result=summarize(fixture(tmp_path,monkeypatch))
    assert result['pairs'][0]['effects']['primary']==pytest.approx(-.1)
    assert result['pairs'][0]['nonzeroCartCorrections']==36
    assert not result['officialTestUsed']


def test_report_requires_a_control_and_checks_the_pre_memory_prefix(tmp_path,monkeypatch):
    from scripts.report_guarded_sharp_cart_pilot import summarize
    roots=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='exactly one'):summarize(roots[:1])
    cfg=json.loads((roots[1]/'plan.json').read_text())['configs'][0]
    path=roots[1]/cfg['runId']/'activation_telemetry.json'
    records=json.loads(path.read_text());records[0]['trainedHash']='different';path.write_text(json.dumps(records))
    with pytest.raises(ValueError,match='historical CART memory'):summarize(roots)
