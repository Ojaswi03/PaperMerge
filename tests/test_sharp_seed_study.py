import json


def campaigns(tmp_path,monkeypatch,*,wrong_mu=False):
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    from basil_core.sharp_study import configuration
    import scripts.run_sharp_study as runner
    monkeypatch.setattr(runner,'check_source',lambda _:None)
    result=[]
    for sigma,base in ((.005,.5),(.02,.3)):
        for seed in (2025,2026,2027):
            root=tmp_path/f'{sigma}-{seed}';root.mkdir();configs=[]
            for condition in ('attack_noise_defense_rla','attack_noise_defense_rla_cart'):
                cfg=configuration('mnist','dirichlet',seed,condition,sigma,nodes=4,rounds=2,attack_start=1,
                    proximal_mu=0. if wrong_mu and seed==2026 else 1.)
                configs.append(cfg);folder=root/cfg['runId'];folder.mkdir()
                row=dict(status='COMPLETED',config=cfg,initialHash=f'initial-{seed}',
                    partition=dict(partitionHash=f'partition-{seed}'),data={'images':'same'},
                    runtime={'device':'/CPU:0','tensorflow':'test'})
                (folder/'run.json').write_text(json.dumps(row))
                value=base+.01*(seed-2025)+(.03 if condition.endswith('_cart') else 0)
                np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((2,4),value),
                    roundNodePerClassAccuracy=np.full((2,4,10),value))
            (root/'plan.json').write_text(json.dumps(dict(source={'test':'same'},configs=configs)))
            (root/'independent_validation.json').write_text(json.dumps(dict(status='ALL_SELECTED_RUNS_VERIFIED',
                verified=2,rows=[{'runId':c['runId']} for c in configs])))
            result.append(root)
    return result


def test_verified_seed_report_keeps_noise_levels_and_paired_effects_separate(tmp_path,monkeypatch):
    from scripts.report_sharp_seed_study import summarize
    report=summarize(campaigns(tmp_path,monkeypatch))
    assert len(report['rows'])==12 and len(report['groups'])==4 and len(report['pairedCartEffects'])==2
    for effect in report['pairedCartEffects']:
        assert abs(effect['effects']['primary']['mean']-.03)<1e-12
        assert effect['effects']['primary']['positiveSeeds']==3
    assert {g['sigma'] for g in report['groups']}=={.005,.02}


def test_seed_report_rejects_different_proximal_settings_across_seeds(tmp_path,monkeypatch):
    import pytest
    from scripts.report_sharp_seed_study import summarize
    with pytest.raises(ValueError,match='Seed configurations'):
        summarize(campaigns(tmp_path,monkeypatch,wrong_mu=True))


def test_seed_report_rejects_counting_a_rerun_twice(tmp_path,monkeypatch):
    import pytest
    from scripts.report_sharp_seed_study import summarize
    roots=campaigns(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Duplicate'):
        summarize([*roots,roots[0]])
