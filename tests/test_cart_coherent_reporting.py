"""Paired effects must retain objective, seed and noise strata."""
import json
import pytest


def fixture_runs(root,monkeypatch,*,missing_control=False):
    import scripts.report_factorial_study as verified
    from basil_core.cart_coherent_study import configuration
    rows=[]
    for objective in ('raw_ce','local_prior'):
        for sigma in (.005,.01):
            for seed in (2025,2026,2027):
                for variant in ('none','budgeted'):
                    if missing_control and objective=='local_prior' and seed==2025 and sigma==.005 and variant=='none':continue
                    cfg=configuration('dirichlet',seed,sigma,variant,training_objective=objective)
                    value=.2+.1*(seed-2025)+sigma+(.1 if objective=='local_prior' else 0.)
                    if variant=='budgeted':value+=.01 if objective=='local_prior' else -.02
                    folder=root/cfg['runId'];folder.mkdir()
                    (folder/'run.json').write_text(json.dumps(dict(elapsedSeconds=100)))
                    rows.append(dict(config=cfg,curve=[value]*3,finalMean=value))
    monkeypatch.setattr(verified,'report',lambda *args,**kwargs:dict(runs=rows))


def test_pilot_aggregation_never_mixes_noise_or_training_objectives(tmp_path,monkeypatch):
    from scripts.report_cart_coherent_pilot import report
    fixture_runs(tmp_path,monkeypatch)
    output=report(tmp_path)
    groups=output['threeSeedGroups']
    assert len(groups)==4 and output['officialTestUsed'] is False
    for row in groups:
        difference=1. if row['objective']=='local_prior' else -2.
        assert row['pairedFinalDeltaMeanPercentagePoints']==pytest.approx(difference)
        assert row['pairedFinalDeltaSampleStdPercentagePoints']==pytest.approx(0.,abs=1e-12)
        assert row['variantFinalMean']-row['controlFinalMean']==pytest.approx(difference/100)
        assert row['variantFinalSampleStd']==pytest.approx(.1)
        assert row['positiveSeeds']==(3 if difference>0 else 0)


def test_another_objectives_control_cannot_replace_missing_matched_control(tmp_path,monkeypatch):
    from scripts.report_cart_coherent_pilot import report
    fixture_runs(tmp_path,monkeypatch,missing_control=True)
    with pytest.raises(ValueError,match='control is missing'):report(tmp_path)
    subset=report(tmp_path,completed_only=True)
    assert len(subset['unpairedCompletedRuns'])==1
    assert len(subset['threeSeedGroups'])==3
