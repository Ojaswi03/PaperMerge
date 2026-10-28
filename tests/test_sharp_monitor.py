import copy
import json

import pytest

from basil_core.sharp_monitor import duration,snapshot,render,monitor


def setup(tmp_path,rows,*,status='RUNNING'):
    configs=[]
    for i,(seed,condition,dataset,row) in enumerate(rows):
        cfg=dict(runId=f'job-{i}',seed=seed,sigma=.01,dataset=dataset,partition='iid',
            model='paper',condition=condition,nodes=10,rounds=10)
        configs.append(cfg);folder=tmp_path/cfg['runId'];folder.mkdir()
        if row is not None:(folder/'run.json').write_text(json.dumps(row))
    (tmp_path/'plan.json').write_text(json.dumps(dict(configs=configs,profiles=[{'gpu':1},{'gpu':3}])))
    (tmp_path/'campaign.json').write_text(json.dumps(dict(status=status,elapsedSeconds=90)))
    return configs


def running(cursor,elapsed):return dict(status='RUNNING',completedActivations=cursor,
    completedRounds=cursor//10,elapsedSeconds=elapsed,meanAccuracy=.8)


def test_eta_uses_each_gpu_sequential_queue_and_parallel_completion(tmp_path):
    setup(tmp_path,[(2025,'clean','mnist',running(25,50)),
        (2025,'noise_rla','mnist',running(50,50)),
        (2026,'clean','mnist',None),(2026,'noise_rla','mnist',None)])
    state=snapshot(tmp_path)
    assert state['entries'][0]['eta']==150 and state['entries'][1]['eta']==50
    assert state['remainingByGpu']=={1:350,3:150}
    assert state['eta']==350 and state['untimedJobs']==0
    assert 'Campaign ETA 00:05:50' in render(state)


def test_same_gpu_timing_is_preferred_when_workers_have_different_speeds(tmp_path):
    setup(tmp_path,[(2025,'clean','mnist',running(20,10)),
        (2025,'clean','mnist',running(20,40)),
        (2026,'clean','mnist',None),(2027,'clean','mnist',None)])
    state=snapshot(tmp_path)
    assert state['remainingByGpu']=={1:90,3:360} and state['eta']==360


@pytest.mark.parametrize('condition,dataset',[('clean','cifar10'),('attack_defense','mnist')])
def test_unknown_workloads_do_not_borrow_clean_mnist_speed(tmp_path,condition,dataset):
    setup(tmp_path,[(2025,'clean','mnist',running(20,40)),(2026,condition,dataset,None)])
    state=snapshot(tmp_path)
    assert state['entries'][0]['eta']==160
    assert state['eta'] is None and state['untimedJobs']==1 and state['remainingByGpu'][3] is None
    assert 'Campaign ETA unavailable' in render(state)


def test_rates_wait_for_one_full_ring_and_pool_only_seed_and_sigma(tmp_path):
    configs=setup(tmp_path,[(2025,'noise_only','mnist',running(5,20)),(2026,'noise_only','mnist',None)])
    assert snapshot(tmp_path)['eta'] is None
    (tmp_path/'job-0'/'run.json').write_text(json.dumps(running(20,40)))
    plan=json.loads((tmp_path/'plan.json').read_text());plan['configs'][1]['sigma']=.02
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    state=snapshot(tmp_path)
    assert state['eta']==200 and state['entries'][0]['eta']==160
    assert state['entries'][1]['config']['sigma']==.02
    assert configs[0]['seed']==2025


def test_completion_exits_without_sleep_and_eta_is_zero(tmp_path,monkeypatch,capsys):
    row=dict(status='COMPLETED',completedActivations=100,completedRounds=10,elapsedSeconds=50,meanAccuracy=.95)
    setup(tmp_path,[(2025,'clean','mnist',row)])
    def no_sleep(_):raise AssertionError('Completed monitor must exit.')
    monkeypatch.setattr('basil_core.sharp_monitor.time.sleep',no_sleep)
    state=snapshot(tmp_path);assert state['eta']==0 and state['status']=='COMPLETED'
    monitor(tmp_path);assert 'ALL EXPERIMENTS COMPLETE' in capsys.readouterr().out


def test_interrupt_closes_only_the_monitor_without_traceback(tmp_path,monkeypatch,capsys):
    setup(tmp_path,[(2025,'clean','mnist',running(20,40))])
    def interrupt(_):raise KeyboardInterrupt
    monkeypatch.setattr('basil_core.sharp_monitor.time.sleep',interrupt)
    monitor(tmp_path)
    assert 'Monitor closed.' in capsys.readouterr().out


def test_stopped_campaign_has_no_completion_eta(tmp_path,monkeypatch,capsys):
    setup(tmp_path,[(2025,'clean','mnist',running(20,40))],status='INTERRUPTED')
    assert snapshot(tmp_path)['eta'] is None
    def no_sleep(_):raise AssertionError('Stopped monitor must exit.')
    monkeypatch.setattr('basil_core.sharp_monitor.time.sleep',no_sleep)
    monitor(tmp_path);assert 'Campaign stopped' in capsys.readouterr().out


def test_render_fits_a_standard_terminal_and_does_not_mutate_data(tmp_path):
    setup(tmp_path,[(2025,'attack_noise_defense_rla_cart','mnist',running(20,40)) for _ in range(8)])
    state=snapshot(tmp_path);before=copy.deepcopy(state)
    text=render(state,width=80,height=24)
    assert len(text.splitlines())<=24 and max(map(len,text.splitlines()))<=80
    assert state==before


def test_invalid_counters_are_not_presented_as_measured_progress(tmp_path):
    setup(tmp_path,[(2025,'clean','mnist',running(101,40))])
    with pytest.raises(ValueError,match='activation counter'):snapshot(tmp_path)


def test_missing_plan_is_explicit_and_duration_supports_days(tmp_path):
    assert snapshot(tmp_path)['status']=='WAITING_FOR_PLAN'
    assert 'Waiting for frozen plan' in render(snapshot(tmp_path))
    assert duration(None)=='unavailable' and duration(90061)=='1d 01:01:01'


def test_cli_reads_active_campaign_without_heavy_imports_or_file_changes(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    setup(tmp_path,[(2025,'clean','mnist',running(20,40))])
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    script=Path(__file__).resolve().parents[1]/'scripts/monitor_sharp_study.py'
    code=(f'import runpy,sys;sys.argv=[{str(script)!r},"--output",{str(tmp_path)!r},"--once"];'
        f'runpy.run_path({str(script)!r},run_name="__main__");'
        'assert not {"tensorflow","numpy"}.intersection(sys.modules)')
    result=subprocess.run([sys.executable,'-I','-c',code],cwd='/tmp',capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert 'ETA 00:02:40' in result.stdout
    assert {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before
