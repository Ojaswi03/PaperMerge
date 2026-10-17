"""Mocked observation-only logging: no cloud runs and no model training."""
import random
from types import SimpleNamespace

import pytest

from basil_core.live_tracking import LiveTracker
from scripts.check_publication_policy import forbidden


class FakeRun:
    url='https://wandb.ai/example/IID-Run/runs/test-id'
    def __init__(self):self.logs=[];self.definitions=[];self.summary={};self.finishes=[]
    def define_metric(self,*args,**kwargs):self.definitions.append((args,kwargs))
    def log(self,values):self.logs.append(values)
    def finish(self,**kwargs):self.finishes.append(kwargs)


@pytest.fixture
def setup():
    run=FakeRun();calls=[]
    sdk=SimpleNamespace(util=SimpleNamespace(generate_id=lambda:'test-id'),Settings=lambda **kw:kw,
        init=lambda **kw:(calls.append(kw) or run))
    config=dict(experimentName='D_iid',conditionId='D',experimentProtocol='sequential_basil_iid_v1',
        iidCampaignId='iid_abcd_500r_v1',nRounds=500,localObjective='source_ebm',channelNoiseSigmaAbsolute=.01,
        apiKey='NEVER_UPLOAD',legacyEbmLambda=999999)
    settings=dict(enabled=True,entity='example',project='IID-Run',mode='online')
    return run,calls,sdk,config,settings


def test_round_accuracy_loss_and_diagnostics_use_round_axis(setup):
    run,calls,sdk,cfg,settings=setup
    tracker=LiveTracker(cfg,{},settings=settings,sdk=sdk)
    tracker.round(dict(round=0,full_test_accuracy=.4,full_test_loss=1.5,mean_node_accuracy=.39,
        worst_node_accuracy=.3,class_0_accuracy=.5,ebm_penalty=.0001,channel_noise_norm=3.43))
    assert run.logs[0]['round']==0 and run.logs[0]['round/full_test_accuracy']==.4
    assert run.logs[0]['round/full_test_loss']==1.5
    assert (('round/*',),dict(step_metric='round')) in run.definitions
    assert calls[0]['settings']['disable_git'] and calls[0]['settings']['save_code'] is False
    assert calls[0]['settings']['console']=='off'
    assert 'apiKey' not in calls[0]['config'] and 'legacyEbmLambda' not in calls[0]['config']
    assert calls[0]['config']['ebmCoefficient']==.0001
    assert 'IID RUN' in calls[0]['name']
    assert run.logs[0]['IID RUN/Accuracy']==.4 and run.logs[0]['IID RUN/Loss']==1.5
    assert (('IID RUN/*',),dict(step_metric='round')) in run.definitions
    tracker.finish('completed')
    assert run.summary['research_status']=='completed' and run.finishes==[dict(exit_code=0)]


def test_disabled_logging_never_initializes_sdk(setup):
    run,calls,sdk,cfg,_=setup
    tracker=LiveTracker(cfg,{},settings={'enabled':False},sdk=sdk)
    tracker.round({'round':0,'full_test_accuracy':.2})
    assert not calls and not run.logs and tracker.metadata['status']=='disabled'


def test_noniid_names_and_ebm_coefficient_are_not_iid_or_zero(setup):
    run,calls,sdk,cfg,settings=setup
    cfg.update(partitionStrategy='dirichlet',trackingNamespace='NONIID RUN',
        localObjective='logit_prior_ebm_snapshot_anchor',ebmMode='gradient_norm_objective',
        snapshotAnchorMu=.01,dirichletAlpha=.2)
    tracker=LiveTracker(cfg,{'partitionHash':'paired'},settings=settings,sdk=sdk)
    tracker.round(dict(round=0,full_test_accuracy=.3,full_test_loss=1.))
    assert calls[0]['name'].startswith('NONIID RUN — D')
    assert calls[0]['config']['ebmCoefficient']==.0001
    assert calls[0]['config']['partitionHash']=='paired'
    assert calls[0]['job_type']=='noniid-basil-research'
    assert run.logs[0]['NONIID RUN/Accuracy']==.3
    assert 'IID RUN/Accuracy' not in run.logs[0]


def test_metrics_only_and_no_rng_change(setup):
    import numpy as np
    run,calls,sdk,cfg,settings=setup
    random.seed(13);np.random.seed(13)
    python_state=random.getstate();numpy_state=np.random.get_state()
    tracker=LiveTracker(cfg,{},settings=settings,sdk=sdk)
    row=dict(round=0,full_test_accuracy=.3,full_test_loss=1.4,unknown=float('nan'),secret='NO_UPLOAD')
    original=dict(row);tracker.round(row)
    assert random.getstate()==python_state
    assert np.array_equal(np.random.get_state()[1],numpy_state[1])
    assert row==original
    assert 'round/unknown' not in run.logs[0] and 'round/secret' not in run.logs[0]


def test_recovery_reuses_run_id_and_skips_already_logged_points(setup):
    run,calls,sdk,cfg,settings=setup
    previous=dict(liveTracking=dict(runId='original',entity='example',project='IID-Run',
        lastLoggedRound=12,lastLoggedActivation=129))
    tracker=LiveTracker(cfg,previous,settings=settings,sdk=sdk)
    assert calls[0]['id']=='original' and calls[0]['resume']=='allow'
    tracker.round({'round':12,'full_test_accuracy':.3})
    tracker.activation(dict(round=12,nodeId=9,globalTestAccuracy=.3))
    assert not run.logs
    tracker.round({'round':13,'full_test_accuracy':.4})
    assert len(run.logs)==1


def test_resume_destination_change_is_not_silent(setup):
    run,calls,sdk,cfg,settings=setup
    tracker=LiveTracker(cfg,dict(liveTracking=dict(runId='previous',entity='different',project='private')),
        settings=settings,sdk=sdk)
    assert not calls and tracker.metadata['status']=='unavailable'


def test_fresh_attempts_use_new_group_and_new_id(setup):
    run,calls,sdk,cfg,settings=setup
    settings['group']='iid_fresh_attempt_2'
    tracker=LiveTracker(cfg,{},settings=settings,sdk=sdk)
    assert calls[0]['group']=='iid_fresh_attempt_2'
    assert calls[0]['id']=='test-id' and calls[0]['resume'] is None
    assert tracker.metadata['group']=='iid_fresh_attempt_2'


def test_resume_retains_original_tracking_group(setup):
    _,calls,sdk,cfg,settings=setup
    settings['group']='different_new_attempt'
    previous={'liveTracking':dict(runId='original',entity='example',project='IID-Run',group='original_group')}
    LiveTracker(cfg,previous,settings=settings,sdk=sdk)
    assert calls[0]['group']=='original_group' and calls[0]['id']=='original'


def test_prepared_cloud_id_is_reused_without_fabricated_points(setup):
    run,calls,sdk,cfg,settings=setup
    settings['preparedRuns']={'D':dict(entity='example',project='IID-Run',runId='prepared-id',group='fresh_group')}
    tracker=LiveTracker(cfg,{},settings=settings,sdk=sdk,prepare_only=True)
    assert calls[0]['id']=='prepared-id' and calls[0]['resume']=='allow'
    assert calls[0]['group']=='fresh_group' and not run.logs
    assert run.summary['research_training_started'] is False
    tracker.finish('prepared')
    assert run.finishes==[dict(exit_code=0)]


def test_worker_resume_of_prepared_tracking_marks_training_not_completion(setup):
    run,calls,sdk,cfg,settings=setup
    settings['preparedRuns']={'D':dict(entity='example',project='IID-Run',runId='prepared-id',group='fresh_group')}
    tracker=LiveTracker(cfg,{},settings=settings,sdk=sdk)
    assert run.summary['research_training_started'] is True
    assert run.summary['research_status']=='running'
    assert run.summary['research_rounds_completed']==0 and not run.logs
    tracker.round(dict(round=0,full_test_accuracy=.2))
    assert run.summary['research_rounds_completed']==1


def test_connection_failure_cannot_raise_into_training(setup):
    run,calls,sdk,cfg,settings=setup
    def broken(*args,**kwargs):raise OSError('network down')
    sdk.init=broken
    tracker=LiveTracker(cfg,{},settings=settings,sdk=sdk)
    tracker.round({'round':0,'full_test_accuracy':.3})
    assert tracker.metadata['status']=='unavailable'


def test_logging_failure_keeps_local_run_independent(setup):
    run,calls,sdk,cfg,settings=setup
    tracker=LiveTracker(cfg,{},settings=settings,sdk=sdk)
    def broken(*args,**kwargs):raise OSError('upload failed')
    run.log=broken
    tracker.round({'round':0,'full_test_accuracy':.3})
    assert tracker.metadata['status']=='unavailable' and tracker.run is None


@pytest.mark.parametrize('name',['docs/SessionHandoff.md','RESEARCH_HANDOFF_PROMPT.md','notes/hAnDoFf.txt',
    'CHATGPT_CATCH_UP_PROMPT.md','docs/claude_context.txt','notes/chatgpt_prompt.md'])
def test_publication_guard_rejects_handoff_documents(name):assert forbidden(name)


@pytest.mark.parametrize('name',['basil_core/handoff.py','docs/IID_500_ROUND_PROTOCOL.md','README.md'])
def test_publication_guard_allows_scientific_source_and_guide(name):assert not forbidden(name)
