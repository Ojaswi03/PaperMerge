import hashlib
import json

import pytest


def fixture(tmp_path,monkeypatch):
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    from basil_core.sharp_study import configuration
    from basil_core.sharp_guarded_protocol import guarded_configuration
    import basil_core.sharp_cart_v2_protocol as v2
    monkeypatch.setattr(v2,'check_source',lambda _:None)
    configs=[];partitions={};data={'mnist':{'trainingOnly':True}}
    for sigma in (.005,.01):
        for seed in (2025,2026,2027):
            for cart in (False,True):
                cfg=v2.configuration(guarded_configuration(configuration('mnist','iid',seed,
                    'attack_noise_defense_rla'+('_cart' if cart else ''),sigma,nodes=4,rounds=10,attack_start=2)))
                configs.append(cfg);folder=tmp_path/cfg['runId'];folder.mkdir()
                partition={'partitionHash':f'partition-{seed}'};partitions[cfg['runId']]=partition
                metadata=dict(status='COMPLETED',config=cfg,source={},data=data['mnist'],partition=partition,
                    completedActivations=40,initialHash=f'init-{seed}',runtime={'device':'/GPU:0','profile':{'gpu':int(cart)}})
                (folder/'run.json').write_text(json.dumps(metadata))
                value=.8+.01*(seed-2025)-(.02 if cart else 0)
                np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((10,4),value))
    (tmp_path/'plan.json').write_text(json.dumps(dict(configs=configs,source={},data=data,partitions=partitions)))
    return configs


def test_report_keeps_sigmas_separate_and_never_mutates_campaign(tmp_path,monkeypatch):
    from scripts.report_cart_v2_completed import summarize
    fixture(tmp_path,monkeypatch)
    hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    before=hashes();result=summarize(tmp_path)
    assert hashes()==before and len(result['groups'])==4 and len(result['pairedCartEffects'])==2
    assert not result['independentInferencePerformed']
    assert all(p['effects']['final']['mean']==pytest.approx(-.02) for p in result['pairedCartEffects'])
    assert {p['sigma'] for p in result['pairedCartEffects']}=={.005,.01}


def test_pending_seed_prevents_group_average_and_unpaired_cart_claim(tmp_path,monkeypatch):
    from scripts.report_cart_v2_completed import summarize
    configs=fixture(tmp_path,monkeypatch)
    cfg=next(c for c in configs if c['sigma']==.01 and c['seed']==2027 and not c['cart'])
    path=tmp_path/cfg['runId']/'run.json';run=json.loads(path.read_text());run['status']='RUNNING';path.write_text(json.dumps(run))
    result=summarize(tmp_path,sigma=.01)
    assert len(result['groups'])==1 and len(result['pendingGroups'])==1
    assert result['pairedCartEffects']==[] and len(result['unpairedCartGroups'])==1


def test_mismatched_initialization_rejected(tmp_path,monkeypatch):
    from scripts.report_cart_v2_completed import summarize
    configs=fixture(tmp_path,monkeypatch);cfg=next(c for c in configs if c['cart'])
    path=tmp_path/cfg['runId']/'run.json';run=json.loads(path.read_text());run['initialHash']='different';path.write_text(json.dumps(run))
    with pytest.raises(ValueError,match='not paired'):summarize(tmp_path)
