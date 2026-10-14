"""500-round preparation contracts: zero research execution."""
import copy
import json
from types import SimpleNamespace

import pytest

from basil_core.iid_campaign import campaign_configs, gui_approved, verify_campaign
from gui.services import QueueService
from gui.state import QueueStatus
from gui.state.experiment_state import from_persisted, to_persisted, validate_experiment
from scripts.prepare_iid_500r import PARTITION_HASH, INITIAL_HASH, QUEUE_PRESET, CONFIG_ROOT


@pytest.fixture
def configs():
    return campaign_configs(PARTITION_HASH,INITIAL_HASH,device='GPU',rounds=500)


def test_fixed_500_round_protocol_and_contrasts(configs):
    assert verify_campaign(configs)=={
        'A/B':['actualAttackerCount','attackHidden','attackerCount','attackerIds','resolvedAttackerIds'],
        'B/C':['channelNoiseSigmaAbsolute','useChannelNoise'],
        'C/D':['ebmMode','localObjective']}
    for c in configs:
        assert gui_approved(c) and c['researchValid']
        assert c['nRounds']==500 and c['basilMemorySize']==5 and c['localEpochs']==5
        assert c['partitionStrategy']=='iid' and c['batchSize']==512
        assert c['learningRate']==.05 and c['learningRateSchedule']=='basil_round_decay'
        assert c['momentum']==c['weightDecayCoefficient']==0 and c['gradientClipNorm'] is None
        assert c['expectedPartitionHash']==PARTITION_HASH and c['expectedInitialModelHash']==INITIAL_HASH
        assert '500r' in c['resultDirectory'] and '500r' in c['plotDirectory']
        assert c['executionStartMode']=='fresh' and 'recoverySourceDirectory' not in c
        assert 'R0→R499' in c['displayName']
        assert not validate_experiment(from_persisted(c)) and to_persisted(from_persisted(c))==c
        assert 'R0→R499' in from_persisted(c).dirty_summary
    assert configs[0]['resolvedAttackerIds']==[]
    assert all(c['resolvedAttackerIds']==[0,1,5,7] and c['attackStartRound']==20 for c in configs[1:])
    assert configs[2]['ebmMode']=='none' and configs[2]['localObjective']=='cross_entropy'
    assert configs[3]['channelNoiseSigmaAbsolute']**2==.0001


@pytest.mark.parametrize('flag',['attackGaussian','attackSignFlip','attackModelPoison','attackScaling','attackAlie','attackIpm','attackNoiseAmp'])
def test_other_attack_flags_cannot_replace_hidden(configs,flag):
    assert not gui_approved(dict(configs[1],**{flag:True}))


def test_horizon_and_study_identifier_must_match(configs):
    assert not gui_approved(dict(configs[0],nRounds=100))
    assert not gui_approved(dict(configs[0],nRounds=501))
    assert not gui_approved(dict(configs[0],executionStartMode='resume'))


def test_hidden_outbound_formula_is_not_gaussian_attack():
    import numpy as np
    from basil_core.attacks import applyAttack
    weights=[np.linspace(-1.,1.,12,dtype=np.float32)]
    rng=np.random.default_rng(41)
    z=rng.normal(0.,.05*max(np.std(weights[0]),.01),size=weights[0].shape)
    expected=((1.-.55)*weights[0]+.55*(-1.4*weights[0]+z)).astype(np.float32)
    actual=applyAttack(weights,'hidden',rng=np.random.default_rng(41))
    np.testing.assert_array_equal(actual[0],expected)
    np.testing.assert_array_equal(weights[0],np.linspace(-1.,1.,12,dtype=np.float32))



def test_saved_500_round_queue_is_pending_and_never_launches(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Preparation must never launch a worker')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    raw=json.loads(QUEUE_PRESET.read_text());verify_campaign(raw)
    queue=QueueService(QUEUE_PRESET).load();QueueService.describe_recovery(queue,tmp_path)
    assert len(queue.entries)==4
    for c,e in zip(raw,queue.entries):
        assert json.loads((CONFIG_ROOT/f"{c['experimentName']}.json").read_text())==c
        assert e.status is QueueStatus.PENDING and e.completed_rounds==0
        assert e.execution_hint=='Fresh R0→R499 — GPU'
    target=tmp_path/'queue.json';QueueService(target).save(queue)
    assert [to_persisted(e.config) for e in QueueService(target).load().entries]==raw


def test_500_round_cli_is_dry_run_by_default(configs,tmp_path,capsys):
    from scripts.run_iid_condition import main
    path=tmp_path/'A.json';path.write_text(json.dumps(configs[0]))
    main(['--config',str(path)])
    result=json.loads(capsys.readouterr().out)
    assert result['rounds']==500 and result['executionStarted'] is False


def test_live_progress_uses_500_round_denominator(configs):
    from gui.services.iid_execution_service import IidExecutionHandle
    events=[];handle=IidExecutionHandle.__new__(IidExecutionHandle)
    handle.config=configs[0];handle.sink=events.append
    handle._dispatch({'roundComplete':{'round':499,'averageAccuracy':.5,'worstNodeAccuracy':.4}})
    progress=next(e.payload for e in events if e.kind=='progress_updated')
    assert progress['progress']==1. and progress['completedRounds']==500


def test_500_round_plot_data_and_late_windows_are_not_truncated(monkeypatch,tmp_path):
    import numpy as np
    import reporting.iid_campaign_plots as plotting
    curve=np.linspace(.1,.6,500)
    metrics=dict(fullTestAccuracy=curve,averageAccuracy=curve,worstNodeAccuracy=curve-.01,
        fullTestPerClassAccuracy=np.tile(curve[:,None],(1,10)))
    stats=plotting.convergence(metrics,attack_applicable=True)
    assert stats['horizon']==500 and stats['late_window_final']==[490,499]
    assert stats['late_window_previous']==[480,489]
    assert stats['late_round_mean']==curve[-10:].mean()
    assert stats['mean_post_attack']==curve[20:].mean()
    with pytest.raises(ValueError,match='horizon'):
        plotting.convergence(dict(metrics,fullTestAccuracy=curve[:100]),attack_applicable=True,expected_rounds=500)
    calls=[];monkeypatch.setattr(plotting,'plot',lambda *args,**kw:calls.append((args,kw)))
    manifest=dict(nRounds=500,iidPartitionHash='same',initialModelHash='same',device='GPU')
    loaded={c:(manifest,metrics,[]) for c in 'ABCD'}
    plotting.comparison_plots(loaded,tmp_path)
    assert len(calls)==4 and '500-round' in calls[0][1]['title']
    assert all(len(values)==500 for values in calls[0][0][2].values())
    bad=copy.deepcopy(loaded);bad['D'][1]['averageAccuracy']=curve[:-1]
    with pytest.raises(ValueError,match='truncated'):plotting.comparison_plots(bad,tmp_path)
