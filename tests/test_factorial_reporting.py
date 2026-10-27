"""Noise selection cannot depend on defense wins; reports retain seed strata."""
import json
import pytest


def test_noise_selection_uses_predeclared_clean_ce_targets():
    from scripts.screen_factorial_noise import choose_levels
    rows=[dict(sigma=s,relativeCEIncrease=v) for s,v in
        ((.001,.0),(.002,.05),(.004,.25),(.008,.75),(.016,2.))]
    assert choose_levels(rows)==[.002,.004,.008]
    assert choose_levels(list(reversed(rows)))==[.002,.004,.008]
    with pytest.raises(ValueError):choose_levels(rows[:2])
    with pytest.raises(ValueError):choose_levels(rows+rows[:1])


def test_full_report_preserves_same_noise_three_seed_averages(tmp_path,monkeypatch):
    import numpy as np
    from basil_core.factorial_study import matrix,SEEDS
    from scripts.report_factorial_study import report
    import scripts.report_factorial_study as reporting
    from scripts.run_cart_memory_smoke import save_json
    configs=matrix((.005,.01,.02))
    # Artifact validation is exercised with real optimizer/checkpoint evidence
    # by test_factorial_recovery; here isolate complete-matrix aggregation.
    monkeypatch.setattr(reporting,'verify_checkpoint',lambda folder,cfg,source,values,**kwargs:
        dict(hash='a'*64,initialModelHash=str(cfg['seed']),counts=[4900]*10))
    save_json(tmp_path/'plan.json',dict(configs=configs,sigmas=[.005,.01,.02],source={},data={'fitImages':'same'}))
    for cfg in configs:
        folder=tmp_path/cfg['runId']
        value=.2+.1*SEEDS.index(cfg['seed'])+cfg['channelNoiseSigmaAbsolute']
        arr=np.full((100,10),value,np.float32)
        save_json(folder/'run.json',dict(config=cfg,status='COMPLETED',source={},data={'fitImages':'same'},
            officialTestUsed=False,roundMean=arr.mean(axis=1).tolist(),runtime=dict(device='GPU',
                tensorFloat32=False,deterministicOps='1',tensorflow='test',numpy='test',software={},
                gpuMemoryLimitMiB=6144,intraOpThreads=1,interOpThreads=1,oneDnn='0',gpu=dict(name='test'))))
        np.savez_compressed(folder/'metrics.npz',roundNodeAccuracy=arr,perClass=np.repeat(arr[:,:,None],10,axis=2))
    result=report(tmp_path)
    rows=[r for r in result['groups'] if r['partition']=='dirichlet' and r['condition']=='noise_ebm']
    assert len(rows)==3
    for row in rows:
        assert row['mean']==pytest.approx(.3+row['sigma'])
        assert row['sampleStd']==pytest.approx(.1)
        assert row['endpointStart']==90
    assert len(result['pairedCartEffects'])==6
    for row in result['pairedCartEffects']:
        assert row['cartDeltaMean']==pytest.approx(0.)
        assert row['cartDeltaSampleStd']==pytest.approx(0.)
        assert row['fullStackCleanGapMean']==pytest.approx(row['sigma'])
    assert (tmp_path/'reports/seed_summary.json').is_file()
    folder=tmp_path/configs[-1]['runId']
    metadata=json.loads((folder/'run.json').read_text());metadata['status']='FAILED'
    save_json(folder/'run.json',metadata)
    with pytest.raises(ValueError,match='incomplete'):report(tmp_path)
    subset=report(tmp_path,partial=True,completed_only=True)
    assert subset['completeSelectedPlan'] is False and len(subset['runs'])==len(configs)-1
    assert subset['pendingOrIncomplete']==[dict(runId=configs[-1]['runId'],status='FAILED')]
    with pytest.raises(ValueError,match='cannot be a full-study'):report(tmp_path,completed_only=True)


def test_plan_default_never_initializes_tensorflow_or_starts_training():
    import subprocess
    import sys
    code="import runpy,sys;sys.argv=['run_factorial_study.py'];runpy.run_module('scripts.run_factorial_study',run_name='__main__');assert 'tensorflow' not in sys.modules"
    response=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    plan=json.loads(response.stdout)
    assert plan['runs']==144 and not plan['trainingStarted'] and not plan['performanceReady']
