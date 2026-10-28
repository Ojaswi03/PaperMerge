import json

import pytest


def fixtures(tmp_path,monkeypatch,*,wrong_mu=False):
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    from basil_core.sharp_study import configuration
    from basil_core.sharp_guarded_protocol import guarded_configuration
    import basil_core.sharp_guarded_protocol as guarded
    monkeypatch.setattr(guarded,'check_source',lambda _:None)
    roots=[]
    for sigma in (.005,.02):
        for seed in (2025,2026,2027):
            root=tmp_path/f'{sigma}-{seed}';root.mkdir();configs=[]
            for condition in ('attack_noise_defense_rla','attack_noise_defense_rla_cart'):
                cfg=guarded_configuration(configuration('mnist','iid',seed,condition,sigma,nodes=4,
                    rounds=2,attack_start=1,proximal_mu=0 if wrong_mu and seed==2026 else 1.))
                configs.append(cfg);folder=root/cfg['runId'];folder.mkdir()
                (folder/'run.json').write_text(json.dumps(dict(status='COMPLETED',config=cfg,
                    initialHash=f'init-{seed}',partition={'partitionHash':f'partition-{seed}'},
                    data={'images':'same'},runtime={'device':'/CPU:0'})))
                value=.5+.01*(seed-2025)+(.03 if condition.endswith('_cart') else 0)
                np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((2,4),value),
                    roundNodePerClassAccuracy=np.full((2,4,10),value))
            (root/'plan.json').write_text(json.dumps(dict(source={'fixture':'same'},configs=configs)))
            (root/'independent_validation.json').write_text(json.dumps(dict(
                status='ALL_SELECTED_RUNS_VERIFIED',verified=2,rows=[{'runId':c['runId']} for c in configs])))
            roots.append(root)
    return roots


def test_guarded_seed_report_handles_derived_base_ids_and_keeps_noise_separate(tmp_path,monkeypatch):
    from scripts.report_guarded_sharp_seed_study import summarize
    result=summarize(fixtures(tmp_path,monkeypatch))
    assert result['status']=='COMPLETE_VERIFIED_GUARDED_SEED_STUDY'
    assert len(result['groups'])==4 and len(result['pairedCartEffects'])==2
    assert {g['sigma'] for g in result['groups']}=={.005,.02}
    assert all(p['effects']['primary']['mean']==pytest.approx(.03) for p in result['pairedCartEffects'])


def test_guarded_seed_report_rejects_unmatched_proximal_settings(tmp_path,monkeypatch):
    from scripts.report_guarded_sharp_seed_study import summarize
    with pytest.raises(ValueError,match='Seed configurations'):
        summarize(fixtures(tmp_path,monkeypatch,wrong_mu=True))
