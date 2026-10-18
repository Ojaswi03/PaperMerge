"""Headless queue control and rendering using synthetic workers and saved events."""
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from basil_core.execution_profile import ENV, ROOT, ExecutionProfile
from basil_core.terminal import controller, dashboard, queues, state
from basil_core.iid_campaign import campaign_configs
from gui.services.execution_service import ExecutionEvent


@pytest.fixture(autouse=True)
def isolated_profile(monkeypatch):
    monkeypatch.delenv(ENV,raising=False)


def configs():
    return campaign_configs('a'*64,'b'*64,device='GPU')


def plan(config,mode='fresh',activation=0):
    return dict(name=config['experimentName'],output=config['resultDirectory'],
        status='COMPLETED' if mode=='completed' else 'STOPPED' if mode=='checkpoint' else 'PENDING',
        problems=[],recovery=dict(mode=mode,startActivation=activation,completedRounds=activation//10))


@pytest.fixture
def synthetic(tmp_path,monkeypatch):
    cfgs=configs()[:2];done=set();handles=[]
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    monkeypatch.setattr(queues,'ROOT',tmp_path)
    monkeypatch.setattr(queues,'science',lambda c:None)
    monkeypatch.setattr(queues,'prerequisites',lambda c:[])
    monkeypatch.setattr(queues,'recovery',lambda c:plan(c)['recovery'])
    monkeypatch.setattr(queues,'completed',lambda c:c['conditionId'] in done)
    class Worker:
        def __init__(self,cfg,sink,policy,directory):
            self.cfg=cfg;self.sink=sink;self.active=True;self.stopping=False;handles.append(self)
        def stop(self):self.stopping=True
        def poll(self):
            if not self.active:return
            self.sink(ExecutionEvent('progress_updated',dict(round=1,completedRounds=1,
                completedActivations=10,totalActivations=1000,progress=.01)))
            self.sink(ExecutionEvent('node_accuracy_updated',dict(node=9,accuracy=.61,round_position=1)))
            self.sink(ExecutionEvent('accuracy_updated',dict(average=.60,worst=.55)))
            self.sink(ExecutionEvent('checkpoint_saved',dict(activation=10)))
            if not self.stopping:done.add(self.cfg['conditionId'])
            self.sink(ExecutionEvent('execution_stopped' if self.stopping else 'execution_completed',{}))
            self.active=False
    def create(*,cfgs_override=None,plans=None,resume=False,factory=Worker):
        cs=cfgs_override or cfgs
        return controller.Controller(tmp_path/'state',cs,plans or [plan(c) for c in cs],
            [ExecutionProfile()],tmp_path/'queue.json',handle_factory=factory,resume=resume,wait=lambda _:None)
    return SimpleNamespace(configs=cfgs,done=done,handles=handles,create=create,Worker=Worker)


def test_sequential_queue_uses_measured_events_and_persists(synthetic,tmp_path):
    c=synthetic.create();assert c.run()==0
    saved=state.read_json(tmp_path/'state/status.json')
    assert saved['status']=='COMPLETED' and saved['totals']['progress']==1.
    assert [e['status'] for e in saved['entries']]==['COMPLETED','COMPLETED']
    assert saved['entries'][0]['metrics']==dict(latest_node_accuracy=.61,mean_node_accuracy=.60,worst_node_accuracy=.55)
    assert saved['entries'][0]['checkpoint']['activation']==10
    assert saved['totals']['completedOptimizerSteps']==100000
    assert len(synthetic.handles)==2
    assert state.read_json(tmp_path/'state/plan.json')['configs']==synthetic.configs


def test_completed_items_are_skipped_not_reexecuted(synthetic):
    plans=[plan(synthetic.configs[0],'completed',1000),plan(synthetic.configs[1])]
    assert synthetic.create(plans=plans).run()==0
    assert len(synthetic.handles)==1 and synthetic.handles[0].cfg['conditionId']=='B'


def test_missing_measured_completion_is_failure_and_blocks_next(synthetic,tmp_path,monkeypatch):
    monkeypatch.setattr(queues,'completed',lambda c:False)
    assert synthetic.create().run()==1
    saved=state.read_json(tmp_path/'state/status.json')
    assert saved['status']=='FAILED' and saved['entries'][0]['status']=='FAILED'
    assert saved['entries'][1]['status']=='PENDING' and len(synthetic.handles)==1
    assert 'complete measured result evidence' in saved['error']


def test_graceful_stop_token_waits_for_worker_and_preserves_checkpoint(synthetic,tmp_path):
    class StopWorker(synthetic.Worker):
        def poll(self):
            state.request_stop(tmp_path/'state')
            self.stop();super().poll()
    assert synthetic.create(factory=StopWorker).run()==130
    saved=state.read_json(tmp_path/'state/status.json')
    assert saved['status']=='STOPPED' and saved['entries'][0]['status']=='STOPPED'
    assert saved['entries'][0]['checkpoint']['activation']==10
    assert saved['entries'][1]['status']=='PENDING'
    assert len(synthetic.handles)==1


def test_resume_requires_compatible_checkpoint_never_round_zero(synthetic,tmp_path,monkeypatch):
    c=synthetic.create();c.run()
    path=tmp_path/'state/status.json';old=state.read_json(path)
    old['entries'][0]['status']='STOPPED';old['status']='STOPPED';state.atomic_json(path,old)
    old['entries'][1]['status']='PENDING';state.atomic_json(path,old)
    with pytest.raises(ValueError,match='no compatible checkpoint'):synthetic.create(resume=True).run()
    ps=[plan(synthetic.configs[0],'checkpoint',10),plan(synthetic.configs[1],'completed',1000)]
    monkeypatch.setattr(queues,'recovery',lambda c:plan(c,'checkpoint',10)['recovery'])
    assert synthetic.create(plans=ps,resume=True).run()==0


def test_duplicate_queue_and_live_or_unverifiable_owner(synthetic,tmp_path):
    with state.lease(tmp_path/'state/queue.lock'):
        with pytest.raises(RuntimeError,match='locked'):synthetic.create().run()
    c=synthetic.create();snapshot=c.snapshot
    snapshot.update(status='RUNNING',owner=state.identity());state.atomic_json(tmp_path/'state/status.json',snapshot)
    with pytest.raises(ValueError,match='owner'):synthetic.create(resume=True).run()
    snapshot['owner']['host']='another-host';state.atomic_json(tmp_path/'state/status.json',snapshot)
    with pytest.raises(ValueError,match='owner'):synthetic.create(resume=True).run()


def test_no_resume_without_saved_state(synthetic):
    with pytest.raises(ValueError,match='No saved terminal queue'):synthetic.create(resume=True).run()


def test_blocked_queue_launches_no_workers(synthetic):
    ps=[dict(plan(c),status='BLOCKED',problems=['missing prerequisite']) for c in synthetic.configs]
    with pytest.raises(ValueError,match='missing prerequisite'):synthetic.create(plans=ps).run()
    assert synthetic.handles==[]


def test_weighted_dirichlet_progress_and_queue_denominator():
    c=dict(configs()[0],experimentProtocol='sequential_basil_dirichlet_v1',nRounds=500,
        optimizerStepsPerNode=[30,45,50,10,95,70,30,55,45,80])
    e=controller.entry_state(c,plan(c,'checkpoint',1),0)
    assert e['completedActivations']==1 and e['completedOptimizerSteps']==30
    assert e['totalOptimizerSteps']==255000 and e['progress']==30/255000
    complete=controller.entry_state(c,plan(c,'completed',5000),1)
    total=controller.totals([e,complete])
    assert total['progress']==255030/510000 and total['completed']==1


def test_queue_presets_load_without_training():
    paths=list((ROOT/'gui/queues').glob('*.json'))
    valid=0
    for path in paths:
        raw=json.loads(path.read_text())
        if not isinstance(raw,list) or not raw:continue
        cs=queues.load_queue(path);assert cs
        for cfg in cs:assert queues.adapter(cfg)[2].startswith('scripts/')
        valid+=1
    assert valid>=14


def test_duplicate_paths_credential_fields_and_unknown_protocol(tmp_path):
    path=tmp_path/'queue.json';cs=configs()
    path.write_text(json.dumps([cs[0],cs[0]]))
    with pytest.raises(ValueError,match='Duplicate output'):queues.load_queue(path)
    cs[0]['api_key']='synthetic-not-a-real-key';path.write_text(json.dumps(cs))
    with pytest.raises(ValueError,match='credential'):queues.load_queue(path)
    with pytest.raises(ValueError,match='No approved'):queues.adapter({})


def test_existing_output_without_manifest_is_never_overwritten(tmp_path,monkeypatch):
    monkeypatch.setattr(queues,'ROOT',tmp_path)
    c=configs()[0];(tmp_path/c['resultDirectory']).mkdir(parents=True)
    with pytest.raises(ValueError,match='overwriting is forbidden'):queues.recovery(c)


def test_checkpoint_replay_is_not_resume(tmp_path,monkeypatch):
    import basil_core.protocol_checkpoint as checkpoints
    monkeypatch.setattr(queues,'ROOT',tmp_path)
    monkeypatch.setattr(checkpoints,'recovery_plan',lambda *args:dict(mode='verified_replay'))
    with pytest.raises(ValueError,match='restart from round zero'):queues.recovery(configs()[0])


def test_completed_manifest_requires_matching_identity_and_rounds(tmp_path,monkeypatch):
    monkeypatch.setattr(queues,'ROOT',tmp_path);c=configs()[0]
    out=tmp_path/c['resultDirectory'];out.mkdir(parents=True)
    state.atomic_json(out/'run.json',dict(status='completed',config=c,completedRounds=100))
    (out/'round_metrics.csv').write_text('round\n'+'\n'.join(str(r) for r in range(100))+'\n')
    assert queues.completed(c)
    (out/'round_metrics.csv').write_text('round\n0\n')
    with pytest.raises(ValueError,match='contiguous'):queues.completed(c)
    state.atomic_json(out/'run.json',dict(status='completed',config={},completedRounds=100))
    with pytest.raises(ValueError,match='different configuration'):queues.completed(c)


def test_staged_gate_deferred_only_for_all_matching_100_conditions(monkeypatch):
    cs=[dict(c,horizonStudyRevision='test',partitionHash='a',initialModelHash='b') for c in configs()]
    long=dict(cs[0],nRounds=500)
    ps=[plan(c) for c in cs]+[dict(plan(long),status='BLOCKED',problems=['stage waiting'])]
    monkeypatch.setattr(controller,'deferred',lambda c:True)
    monkeypatch.setattr(queues,'prerequisites',lambda c:[])
    controller.validate_plan(cs+[long],ps)
    with pytest.raises(ValueError,match='stage waiting'):controller.validate_plan(cs[:3]+[long],ps[:3]+[ps[-1]])
    monkeypatch.setattr(queues,'prerequisites',lambda c:['GPU verification missing'])
    with pytest.raises(ValueError,match='GPU verification missing'):controller.validate_plan(cs+[long],ps)


def test_stale_pid_reuse_and_remote_owner_never_signaled(tmp_path):
    own=state.identity();assert state.alive(own) is True
    reused=dict(own,start=str(int(own['start'])+1));assert state.alive(reused) is False
    saved=dict(status='RUNNING',owner=reused,run_id='synthetic',entries=[dict(status='RUNNING',worker=None)])
    state.atomic_json(tmp_path/'status.json',saved)
    assert state.inspect_state(tmp_path)['status']=='FAILED'
    with pytest.raises(ValueError,match='no process was signaled'):state.request_stop(tmp_path)
    assert not (tmp_path/'stop.json').exists()


def test_monitor_formats_metrics_without_inventing_zeroes(synthetic):
    c=synthetic.create();e=c.snapshot['entries'][0];e['status']='RUNNING'
    c.snapshot['totals']=controller.totals(c.snapshot['entries'])
    text=dashboard.render(c.snapshot,dict(host='synthetic-host'))
    assert 'Latest node accuracy: unavailable  Mean: unavailable  Worst: unavailable' in text
    assert 'Node-9 test CE: unavailable' in text and '1000' in text
    assert 'synthetic-host' in text and 'PENDING' in text and 'Refresh: 1 seconds' in text


def test_monitor_default_one_second_refresh_and_plain_fallback(tmp_path):
    stream=io.StringIO();waits=[];fetches=[]
    def wait(seconds):
        waits.append(seconds)
        if len(waits)==2:raise KeyboardInterrupt
    def fetch(directory):fetches.append(directory);return dict(status='IDLE')
    dashboard.monitor(tmp_path,stream=stream,clock=lambda:0.,wait=wait,fetch=fetch,resources=lambda:{})
    assert waits==[1.,1.] and len(fetches)==2
    assert '\x1b' not in stream.getvalue() and 'queue execution is independent' in stream.getvalue()
    assert not list(tmp_path.iterdir())
    for value in (0,-1,float('nan')):
        with pytest.raises(ValueError):dashboard.monitor(tmp_path,value,once=True,stream=stream)


def test_atomic_state_readable_from_second_process_without_gui(tmp_path):
    value=dict(status='STOPPED',profile='hpc',entries=[],activity=[],run_id='synthetic')
    state.atomic_json(tmp_path/'status.json',value)
    command=[sys.executable,'scripts/hpc.py','status','--json','--state-dir',str(tmp_path)]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==value
    assert state.read_json(tmp_path/'status.json')==value
    code="import sys; from scripts.hpc import main; main(['monitor','--once','--plain','--state-dir',sys.argv[1]]); assert 'tkinter' not in sys.modules; assert 'tensorflow' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '\x1b' not in result.stdout


def test_run_without_execute_is_dry_and_missing_resume_fails(tmp_path,monkeypatch,capsys):
    from scripts import hpc
    cs=configs();ps=[plan(c) for c in cs]
    monkeypatch.setattr(hpc,'print_plan',lambda *a,**k:(cs,ps))
    monkeypatch.setattr(controller.Controller,'run',lambda _:pytest.fail('must not train'))
    assert hpc.main(['run','--profile','local','--queue','unused','--state-dir',str(tmp_path)])==0
    assert 'Validation only' in capsys.readouterr().out
    with pytest.raises(ValueError,match='No saved queue plan'):hpc.main(['resume','--state-dir',str(tmp_path),'--execute'])


def test_required_wandb_missing_sdk_and_verification_is_blocked(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:None)
    missing=queues.prerequisites(dict(liveTrackingEnabled=True,executionDevice='CPU'))
    assert any('W&B SDK' in s for s in missing)


def test_doctor_environment_failure_never_runs_gpu_verification(tmp_path,monkeypatch):
    from basil_core.terminal import doctor
    calls=[]
    monkeypatch.setattr(doctor,'dependencies',lambda:{p:None for p in
        ('tensorflow','keras','numpy','scipy','scikit-learn','h5py','matplotlib','pillow','tqdm','psutil','wandb')})
    monkeypatch.setattr(doctor,'gpu_inventory',lambda:[])
    monkeypatch.setattr(dashboard,'system_info',lambda:{})
    def runner(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=0,stdout='',stderr='')
    result=doctor.diagnose(ExecutionProfile(),tmp_path,runner=runner)
    assert result['status']=='blocked' and not result['gpuVerified'] and result['researchRunsStarted']==0
    assert all(c[1:]==['-m','pip','check'] for c in calls)


def test_log_tail_redaction_rotation_and_error_code(tmp_path):
    assert 'synthetic-secret' not in state.safe_text('api_key=synthetic-secret')
    assert '\x1b' not in state.safe_text('\x1b[31merror')
    path=tmp_path/'queue.log';path.write_text('one\ntwo\nthree\n')
    stream=io.StringIO();dashboard.follow_logs(tmp_path,lines=2,stream=stream)
    assert stream.getvalue()=='two\nthree\n'
    result=subprocess.run([sys.executable,'scripts/hpc.py','logs','--state-dir',str(tmp_path/'missing')],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==1 and 'No execution log' in result.stderr and 'Traceback' not in result.stderr


def test_parallel_scheduling_requires_prior_single_gpu_evidence(synthetic):
    from dataclasses import replace
    c=synthetic.create();c.parallel=True;c.independent_approved=False
    with pytest.raises(ValueError,match='independent-approved'):c.validate_parallel()
    c.independent_approved=True;c.policies=[replace(c.policies[0],output_root=str(c.directory.parent/'hpc'))]
    with pytest.raises(ValueError,match='single-GPU IID execution'):c.validate_parallel()


def test_resume_cannot_change_resource_policy(tmp_path):
    state.atomic_json(tmp_path/'plan.json',dict(queue='missing',configs=configs(),policies=[asdict(ExecutionProfile())]))
    result=subprocess.run([sys.executable,'scripts/hpc.py','resume','--state-dir',str(tmp_path),
        '--gpu-memory-mib','8192','--execute'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==1 and 'cannot change the saved resource policy' in result.stderr


def test_failed_stage_validation_blocks_entry_and_later_jobs(synthetic,tmp_path,monkeypatch):
    c=synthetic.create()
    def rejected(config):raise ValueError('prerequisite audit failed')
    monkeypatch.setattr(c,'stage',rejected)
    with pytest.raises(ValueError,match='prerequisite audit failed'):c.run()
    saved=state.read_json(tmp_path/'state/status.json')
    assert saved['entries'][0]['status']=='BLOCKED'
    assert saved['entries'][1]['status']=='PENDING' and not synthetic.handles


def test_headless_worker_environment_keeps_selected_gpu(tmp_path,monkeypatch):
    from basil_core.execution_profile import ExecutionProfile, worker_environment
    from gui.services import iid_execution_service as service
    p=ExecutionProfile(name='hpc',gpu=2,gpu_uuid='GPU-00000002-abcd',memory_mib=8192,
        headroom_mib=2048,output_root=str(tmp_path/'runs'))
    monkeypatch.setenv(ENV,p.environment()[ENV]);launches=[]
    def launch(command,**kwargs):
        launches.append((command,kwargs));return SimpleNamespace(pid=12345)
    monkeypatch.setattr(service.subprocess,'Popen',launch)
    monkeypatch.setattr(service.threading,'Thread',lambda **kwargs:SimpleNamespace(start=lambda:None))
    service.IidExecutionHandle(configs()[0],lambda e:None,_approved=lambda c:True,
        _plan={'mode':'fresh'},_work_root=tmp_path/'worker',_headless=True,
        _environment=worker_environment('GPU',os.environ,p))
    command,options=launches[0]
    assert '_worker' in command and '--execute' in command
    assert options['env']['CUDA_VISIBLE_DEVICES']==p.gpu_uuid
    assert options['env']['MPLBACKEND']=='Agg'


def test_hpc_checkpoint_from_another_runtime_is_blocked(tmp_path,monkeypatch):
    import basil_core.execution_profile as profiles
    import basil_core.protocol_checkpoint as checkpoints
    p=profiles.ExecutionProfile(name='hpc',gpu=1,output_root=str(tmp_path/'runs'))
    monkeypatch.setenv(ENV,p.environment()[ENV])
    monkeypatch.setattr(profiles,'resource_identity',lambda:dict(host='new-hpc',physicalGpuId=1))
    monkeypatch.setattr(checkpoints,'recovery_plan',lambda *args:dict(mode='checkpoint',startActivation=10))
    cfg=configs()[0];out=profiles.profile_path(ROOT/cfg['resultDirectory']);out.mkdir(parents=True)
    state.atomic_json(out/'run.json',dict(numericalExecution={'resources':{'host':'old-local'}}))
    with pytest.raises(ValueError,match='different hardware'):queues.recovery(cfg)


def test_hpc_run_needs_explicit_gpu_and_no_unapproved_parallelism(tmp_path):
    from scripts import hpc
    args=hpc.parser().parse_args(['run','--queue','unused','--execute'])
    with pytest.raises(ValueError,match='explicit --gpus'):hpc.select_profiles(args,require_selection=True)
    args=hpc.parser().parse_args(['run','--queue','unused','--gpus','1,1'])
    with pytest.raises(ValueError,match='distinct'):hpc.select_profiles(args)


def test_tmux_missing_and_bad_session_name_cannot_launch(tmp_path,monkeypatch):
    from scripts import hpc
    monkeypatch.setattr(hpc.shutil,'which',lambda name:None)
    with pytest.raises(ValueError,match='tmux is unavailable'):
        hpc.tmux_start(SimpleNamespace(tmux='test'),['run'])
    monkeypatch.setattr(hpc.shutil,'which',lambda name:'/usr/bin/tmux')
    with pytest.raises(ValueError,match='alphanumeric'):
        hpc.tmux_start(SimpleNamespace(tmux='test;echo'),['run'])


def test_log_follow_tracks_rotation_without_affecting_worker(tmp_path):
    path=tmp_path/'queue.log';path.write_text('before\n');stream=io.StringIO();calls=[]
    def wait(_):
        calls.append(1)
        if len(calls)==1:
            path.rename(tmp_path/'queue.log.1');path.write_text('after\n')
        if len(calls)==3:raise KeyboardInterrupt
    dashboard.follow_logs(tmp_path,follow=True,stream=stream,wait=wait)
    assert stream.getvalue()=='before\nafter\n'


def test_setup_refuses_legacy_environment_without_installing(tmp_path):
    legacy=tmp_path/'legacy';legacy.mkdir()
    (legacy/'pyvenv.cfg').write_text('synthetic existing environment\n')
    result=subprocess.run(['bash','scripts/setup_hpc.sh','--venv',str(legacy),'--python',sys.executable],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==1 and 'protect legacy dependencies' in result.stderr
    assert sorted(p.name for p in legacy.iterdir())==['pyvenv.cfg']


def test_profile_state_cannot_target_research_or_papers():
    from scripts import hpc
    for name in ('Papers','newResults','gui','basil_core'):
        with pytest.raises(ValueError,match='Operational state must be separate'):
            hpc.main(['status','--state-dir',str(ROOT/name)])


def test_campaign_queue_validation_rejects_unpaired_initialization(tmp_path):
    cs=configs();cs[1]['expectedInitialModelHash']='c'*64
    path=tmp_path/'queue.json';path.write_text(json.dumps(cs))
    with pytest.raises(ValueError,match='Unexpected scientific differences'):queues.inspect_queue(path,False)


def test_independent_scheduler_assigns_distinct_gpus_before_polling(synthetic,tmp_path,monkeypatch):
    from dataclasses import replace
    launched=[]
    class Worker(synthetic.Worker):
        def __init__(self,cfg,sink,policy,directory):
            launched.append(policy.gpu);super().__init__(cfg,sink,policy,directory)
        def poll(self):
            assert launched==[1,2]
            super().poll()
    c=synthetic.create(factory=Worker)
    first=ExecutionProfile(name='hpc',gpu=1,output_root=str(tmp_path/'runs'))
    c.policies=[first,replace(first,gpu=2)];c.parallel=True
    monkeypatch.setattr(c,'validate_parallel',lambda:None)  # Gating is tested separately.
    assert c.run()==0
    assert [e['gpu'] for e in c.snapshot['entries']]==[1,2]


def test_slow_resource_probe_does_not_delay_dashboard_refresh(tmp_path,monkeypatch):
    import threading
    release=threading.Event();waits=[];stream=io.StringIO()
    def slow_resources():release.wait(5);return dict(host='synthetic')
    monkeypatch.setattr(dashboard,'system_info',slow_resources)
    def wait(value):
        waits.append(value)
        if len(waits)==2:raise KeyboardInterrupt
    try:
        dashboard.monitor(tmp_path,stream=stream,clock=lambda:0.,wait=wait,
            resources=slow_resources,fetch=lambda p:dict(status='RUNNING'))
        assert waits==[1.,1.] and stream.getvalue().count('TERMINAL EXECUTION MONITOR')==2
    finally:release.set()


def test_resume_never_repeats_missing_previously_completed_output(synthetic):
    synthetic.create().run()
    with pytest.raises(ValueError,match='refusing to repeat'):
        synthetic.create(resume=True).run()


def test_stop_at_final_completion_keeps_measured_completion(synthetic):
    class FinalStop(synthetic.Worker):
        def poll(self):
            self.stopping=True;synthetic.done.add(self.cfg['conditionId']);super().poll()
    c=synthetic.create(cfgs_override=synthetic.configs[:1],factory=FinalStop)
    assert c.run()==0
    assert c.snapshot['entries'][0]['status']=='COMPLETED'


@pytest.mark.parametrize('option',[['--tmux','test'],['--tmux=test']])
def test_tmux_detach_preserves_relative_paths_and_accepts_equals(tmp_path,monkeypatch,option):
    from scripts import hpc
    import shlex
    launches=[]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hpc.shutil,'which',lambda name:'/usr/bin/tmux')
    def run(command,**kwargs):
        launches.append(command);return SimpleNamespace(returncode=1 if 'has-session' in command else 0)
    monkeypatch.setattr(hpc.subprocess,'run',run)
    argv=['run','--queue','relative/queue.json','--state-dir','relative/state','--execute',*option]
    assert hpc.tmux_start(SimpleNamespace(tmux='test'),argv)==0
    launch=launches[-1]
    assert launch[launch.index('-c')+1]==str(tmp_path)
    forwarded=shlex.split(launch[-1])
    assert all(not part.startswith('--tmux') for part in forwarded)
    assert forwarded[-1]=='--execute' and 'relative/queue.json' in forwarded


def test_doctor_uses_selected_gpu_receipt_not_controller_default(tmp_path,monkeypatch):
    from basil_core.terminal import doctor
    import basil_core.execution_profile as profiles
    from test_hpc_profiles import inventory,policy
    from dataclasses import replace
    p1=policy(tmp_path).bind(inventory());p2=replace(p1,gpu=2,gpu_uuid=inventory()[2]['uuid'])
    monkeypatch.setenv(ENV,p1.environment()[ENV])
    monkeypatch.setattr(profiles,'gpu_inventory',inventory)
    monkeypatch.setattr(doctor,'gpu_inventory',inventory)
    monkeypatch.setattr(doctor.sys,'prefix',str(tmp_path/'venv'))
    monkeypatch.setattr(doctor,'library_probe',lambda *args,**kwargs:dict(status='passed',libraries={}))
    monkeypatch.setattr(doctor,'dependencies',lambda:{name:'synthetic' for name in
        ('tensorflow','keras','numpy','scipy','scikit-learn','h5py','matplotlib','pillow','tqdm','psutil','wandb')})
    monkeypatch.setattr(dashboard,'system_info',lambda:{})
    receipt=dict(status='passed',numericalExecution=dict(cudaVersion='synthetic',cudnnVersion='synthetic',
        resources=profiles.resource_identity(p2)))
    state.atomic_json(profiles.verification_path(p2),receipt)
    # No GPU kernels are launched: only the exact receipt-selection contract is tested.
    runner=lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout='',stderr='')
    report=doctor.diagnose(p2,tmp_path/'state',runner=runner,objective_checks=False)
    assert report['status']=='passed' and report['gpuVerified']
    assert report['numericalExecution']['resources']['physicalGpuId']==2
    assert profiles.current_profile()==p1


def test_parallel_resume_retains_original_gpu_even_when_free_gpu_is_first(synthetic,tmp_path,monkeypatch):
    from dataclasses import replace
    import basil_core.execution_profile as profiles
    first=ExecutionProfile(name='hpc',gpu=1,output_root=str(tmp_path/'runs'))
    second=replace(first,gpu=2)
    cs=synthetic.configs;ps=[plan(c,'checkpoint',10) for c in cs]
    launched=[];checked=[]
    def recovered(config):
        gpu=2 if config['conditionId']=='A' else 1
        checked.append((config['conditionId'],profiles.current_profile().gpu))
        assert profiles.current_profile().gpu==gpu
        return plan(config,'checkpoint',10)['recovery']
    monkeypatch.setattr(queues,'recovery',recovered)
    class Worker(synthetic.Worker):
        def __init__(self,cfg,sink,policy,directory):
            launched.append(policy.gpu);super().__init__(cfg,sink,policy,directory)
    c=controller.Controller(tmp_path/'state',cs,ps,[first,second],tmp_path/'queue.json',
        resume=True,independent_approved=True,handle_factory=Worker,wait=lambda _:None)
    old=dict(c.snapshot,status='STOPPED')
    for entry,gpu in zip(old['entries'],[2,1]):entry.update(status='STOPPED',gpu=gpu)
    state.atomic_json(tmp_path/'state/status.json',old)
    monkeypatch.setattr(c,'validate_parallel',lambda:None)
    assert c.run()==0
    assert launched==[2,1] and checked==[('A',2),('B',1)]


def test_resume_preflight_uses_saved_per_entry_gpu(tmp_path,monkeypatch):
    from scripts import hpc
    from dataclasses import replace
    import basil_core.execution_profile as profiles
    first=ExecutionProfile(name='hpc',gpu=1,output_root=str(tmp_path/'runs'));second=replace(first,gpu=2)
    cs=configs()[:2]
    state.atomic_json(tmp_path/'plan.json',dict(queue=str(tmp_path/'missing-queue.json'),configs=cs,
        policies=[asdict(first),asdict(second)]))
    state.atomic_json(tmp_path/'status.json',dict(entries=[dict(gpu=2),dict(gpu=1)]))
    checked=[]
    def inspect(config):
        checked.append(profiles.current_profile().gpu);return dict(status='STOPPED')
    monkeypatch.setattr(queues,'inspect_config',inspect)
    assert hpc.main(['resume','--state-dir',str(tmp_path)])==0
    assert checked==[2,1]


def test_independent_scheduling_rejects_stale_single_gpu_runtime(synthetic,tmp_path,monkeypatch):
    import basil_core.protocol_checkpoint as checkpoints
    import basil_core.execution_profile as profiles
    c=synthetic.create();c.parallel=True;c.independent_approved=True
    p=ExecutionProfile(name='hpc',gpu=1,output_root=str(tmp_path/'runs'));c.policies=[p]
    monkeypatch.setattr(checkpoints,'checkpoint_implementation',lambda config:{'synthetic':'source'})
    monkeypatch.setattr(queues,'completed',lambda config:True)
    monkeypatch.setattr(profiles,'resource_identity',lambda policy:dict(physicalGpuId=1,software='new'))
    receipt=dict(host=controller.socket.gethostname(),protocol='sequential_basil_iid_v1',status='completed',
        config=c.configs[0],implementation={'synthetic':'source'},resources=dict(physicalGpuId=1,software='old'))
    path=tmp_path/'runs/verification/single_gpu_execution.json';state.atomic_json(path,receipt)
    with pytest.raises(ValueError,match='current runtime'):c.validate_parallel()
    receipt['resources']['software']='new';state.atomic_json(path,receipt)
    c.validate_parallel()
