import json


def test_cuda_loader_environment_preserves_bound_profile_and_visibility(monkeypatch,tmp_path):
    from dataclasses import asdict
    from basil_core.execution_profile import ExecutionProfile,ENV
    from scripts.run_sharp_study import worker_environment
    import basil_core.execution_profile as profiles
    monkeypatch.setattr(profiles,'cuda_wheel_environment',lambda base:{'LD_LIBRARY_PATH':'/fake/cuda/lib'})
    profile=ExecutionProfile(name='hpc',gpu=3,gpu_uuid='GPU-1234',memory_mib=8192,
        headroom_mib=2048,output_root=str(tmp_path/'runs'))
    environment=worker_environment(profile)
    assert json.loads(environment[ENV])==asdict(profile)
    assert environment['CUDA_VISIBLE_DEVICES']=='GPU-1234'
    assert environment['LD_LIBRARY_PATH']=='/fake/cuda/lib'
    assert environment['PYTHONUNBUFFERED']=='1'


def test_report_averages_seeds_only_within_the_same_sigma(tmp_path,capsys):
    from types import SimpleNamespace
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    from basil_core.sharp_study import configuration
    from scripts.run_sharp_study import report
    configs=[];source={'test':'fixed'}
    for sigma,base in ((.005,.4),(.01,.5),(.02,.6)):
        for seed in (2025,2026,2027):
            for condition in ('attack_noise_defense_rla','attack_noise_defense_rla_cart'):
                cfg=configuration('mnist','iid',seed,condition,sigma,nodes=4,rounds=2)
                configs.append(cfg);folder=tmp_path/cfg['runId'];folder.mkdir()
                (folder/'run.json').write_text(json.dumps(dict(config=cfg,source=source,status='COMPLETED')))
                value=base+(seed-2025)*.01+(.03 if condition.endswith('_cart') else 0)
                np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((2,4),value))
    (tmp_path/'plan.json').write_text(json.dumps(dict(configs=configs,source=source)))
    report(SimpleNamespace(output=tmp_path));capsys.readouterr()
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['groups'])==6 and len(summary['pairedCartEndpoints'])==9
    for group in summary['groups']:
        assert group['seedCount']==3 and abs(group['sampleStd']-.01)<1e-12
    assert all(abs(pair['delta']-.03)<1e-12 for pair in summary['pairedCartEndpoints'])
    assert len({round(g['mean'],6) for g in summary['groups']})==6


def test_completed_monitor_returns_without_sleeping(tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace
    from basil_core.sharp_study import configuration
    from scripts.run_sharp_study import monitor
    import time
    cfg=configuration('mnist','iid',2025,'clean',0.,nodes=4,rounds=2)
    folder=tmp_path/cfg['runId'];folder.mkdir()
    (tmp_path/'plan.json').write_text(json.dumps(dict(configs=[cfg])))
    (tmp_path/'campaign.json').write_text(json.dumps(dict(status='COMPLETED')))
    (folder/'run.json').write_text(json.dumps(dict(status='COMPLETED',completedActivations=8,completedRounds=2,meanAccuracy=.5)))
    def no_sleep(*_):raise AssertionError('Completed monitor must not refresh.')
    monkeypatch.setattr(time,'sleep',no_sleep)
    monitor(SimpleNamespace(output=tmp_path))
    assert 'ALL EXPERIMENTS COMPLETE' in capsys.readouterr().out
