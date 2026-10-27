"""Committed progress, noise/seed labels and terminal completion."""
from basil_core.factorial_study import configuration,CONDITIONS
from basil_core.factorial_dashboard import snapshot
from scripts.run_cart_memory_smoke import save_json


def test_dashboard_shows_seed_noise_device_and_stops_after_completion(tmp_path):
    cfg=configuration(CONDITIONS[0],'iid',2025,0.,stage='pilot')
    save_json(tmp_path/'plan.json',dict(configs=[cfg],profiles=[dict(name='local',gpu=0)]))
    save_json(tmp_path/'campaign.json',dict(status='RUNNING',elapsedSeconds=100))
    folder=tmp_path/cfg['runId']
    save_json(folder/'partition.json',dict(counts=[4900]*10))
    row=dict(status='RUNNING',completedActivations=20,completedRounds=2,
        elapsedSeconds=100,physicalGpuId=0,validationMean=.5)
    save_json(folder/'run.json',row)
    state=snapshot(tmp_path);entry=state['entries'][0]
    assert entry['eta_seconds']==400 and state['eta_seconds']==400
    assert entry['completedOptimizerSteps']==2*10*5*39
    assert 'S2025' in entry['condition'] and 'sigma=0' in entry['condition']
    assert entry['gpu']==0 and state['status']=='RUNNING'
    row.update(status='COMPLETED',completedActivations=100,completedRounds=10)
    save_json(folder/'run.json',row)
    state=snapshot(tmp_path)
    assert state['status']=='COMPLETED' and state['totals']['completed']==1
    assert state['eta_seconds']==0


def test_pending_unknown_runtime_does_not_report_a_fabricated_campaign_eta(tmp_path):
    configs=[configuration(c,'iid',2025,0.,stage='pilot') for c in CONDITIONS[:2]]
    save_json(tmp_path/'plan.json',dict(configs=configs,profiles=[dict(name='local',gpu=0)]))
    save_json(tmp_path/'campaign.json',dict(status='RUNNING',elapsedSeconds=100))
    save_json(tmp_path/configs[0]['runId']/'run.json',dict(status='RUNNING',completedActivations=10,
        completedRounds=1,elapsedSeconds=100,physicalGpuId=0))
    state=snapshot(tmp_path)
    assert state['eta_seconds'] is None and state['totals']['pending']==1
