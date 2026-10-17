"""Configuration/GUI gating only; no learning or real experiment subprocess."""
import json
import subprocess
import sys

import pytest


def configs():
    from basil_core.dirichlet_prior_study import configurations
    return configurations('a'*64,'b'*64,'c'*64)


def test_confirmation_is_separate_paired_and_not_automatically_approved():
    from basil_core.dirichlet_prior_study import validate_configuration,gui_approved
    from basil_core.dirichlet_study import verify_pairing
    cfgs=configs()
    assert verify_pairing(cfgs)['C/D']==['ebmMode','localObjective']
    for cfg in cfgs:
        validate_configuration(cfg)
        assert cfg['nRounds']==30 and not cfg['researchValid'] and not cfg['fullResearchReady']
        assert not gui_approved(cfg)
        assert 'confirmation_30r_v1' in cfg['resultDirectory']
    for key,value in [('nRounds',500),('snapshotAnchorMu',.02),('channelNoiseSigmaAbsolute',.02),('ebmMode','legacy_gradient_scale')]:
        with pytest.raises(ValueError):validate_configuration(dict(cfgs[3],**{key:value}))


def test_gui_loading_unapproved_confirmation_cannot_enable_run():
    from gui.state.experiment_state import from_persisted,to_persisted,validate_experiment
    from gui.services.queue_service import QueueService
    from gui.state.queue_state import QueueState,QueueEntry
    from pathlib import Path
    state=from_persisted(configs()[0])
    assert to_persisted(state)==configs()[0]
    assert validate_experiment(state)
    queue=QueueState([QueueEntry(state)])
    QueueService.describe_recovery(queue,Path.cwd())
    assert queue.entries[0].status.value=='pending'
    assert queue.entries[0].execution_hint.startswith('Recovery blocked:')


def test_cli_dry_run_validates_without_launching(tmp_path):
    file=tmp_path/'D.json';file.write_text(json.dumps(configs()[3]))
    result=subprocess.run([sys.executable,'scripts/run_dirichlet_prior_condition.py','--config',str(file)],
        check=True,capture_output=True,text=True)
    payload=json.loads(result.stdout)
    assert not payload['trainingStarted'] and not payload['executionReady']
    assert payload['rounds']==30 and payload['condition']=='D'


@pytest.mark.parametrize('module',['basil_core.dirichlet_prior_study','scripts.run_dirichlet_prior_condition'])
def test_confirmation_imports_lazy(module):
    subprocess.run([sys.executable,'-c',f'import importlib,sys;importlib.import_module({module!r});'
        'assert not ({"numpy","tensorflow","wandb","matplotlib"} & sys.modules.keys())'],check=True)


def test_gui_launch_route_keeps_prior_worker_distinct(monkeypatch,tmp_path):
    import io
    import basil_core.dirichlet_prior_study as study
    import gui.services.dirichlet_execution_service as service
    import gui.services.iid_execution_service as common
    monkeypatch.setattr(service,'ROOT',tmp_path)
    monkeypatch.setattr(study,'validate_evidence',lambda cfg:{'status':'unit_test_only'})
    monkeypatch.setattr(study,'recovery_plan',lambda *a:dict(mode='fresh',completedRounds=0,startActivation=0))
    captured=[]
    class Process:
        pid=999999
        stdout=io.StringIO('')
        def poll(self):return 0
        def wait(self):return 0
    def launch(command,**kwargs):
        captured.append(command)
        return Process()
    monkeypatch.setattr(common.subprocess,'Popen',launch)
    handle=service.DirichletExecutionHandle(configs()[3],lambda event:None)
    handle.reader.join(timeout=2);handle.poll()
    assert captured[0][1].endswith('scripts/run_dirichlet_prior_condition.py')
    assert '--execute' in captured[0] and '--recover' not in captured[0]


def test_single_condition_entrypoint_calls_approved_runner_and_plotter(monkeypatch,tmp_path):
    import scripts.run_dirichlet_prior_condition as cli
    import scripts.run_dirichlet_calibration as runner
    import basil_core.dirichlet_prior_study as study
    import reporting.noniid_study_plots as plotting
    cfg=configs()[3];file=tmp_path/'D.json';file.write_text(json.dumps(cfg))
    calls=[]
    monkeypatch.setattr(study,'validate_evidence',lambda _:None)
    monkeypatch.setattr(study,'recovery_plan',lambda *a:dict(mode='fresh'))
    monkeypatch.setattr(runner,'run_pair',lambda device,**kwargs:calls.append((device,kwargs)))
    monkeypatch.setattr(plotting,'generate_condition_plots',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',['runner','--config',str(file),'--execute'])
    cli.main()
    assert calls[0][1]['configs_override']==[cfg]
    assert calls[0][1]['stage']=='threat' and calls[0][1]['gui_events'] is True
    assert str(calls[1][0]).endswith(cfg['resultDirectory'])
    assert str(calls[1][1]).endswith(cfg['plotDirectory'])


@pytest.mark.parametrize('fallback',[False,True])
def test_confirmation_recovery_uses_verified_immutable_checkpoint_not_jsonl(monkeypatch,tmp_path,fallback):
    from pathlib import Path
    import basil_core.dirichlet_prior_study as study
    import basil_core.protocol_checkpoint as checkpoint
    cfg=configs()[0]
    (tmp_path/'run.json').write_text(json.dumps(dict(config=cfg,status='interrupted')))
    (tmp_path/'checkpoint.npz').write_bytes(b'unit-test-placeholder')
    if fallback:(tmp_path/'checkpoint.previous.npz').write_bytes(b'unit-test-placeholder')
    monkeypatch.setattr(study,'validate_evidence',lambda _:None)
    def load(path,*args,**kwargs):
        assert kwargs==dict(partition_hash=cfg['partitionHash'],initial_hash=cfg['initialModelHash'])
        if fallback and Path(path).name=='checkpoint.npz':raise ValueError('corrupt checkpoint')
        return dict(next_activation=17,completed_rounds=1)
    monkeypatch.setattr(checkpoint,'load_checkpoint',load)
    plan=study.recovery_plan(tmp_path,cfg)
    assert plan['mode']=='checkpoint' and plan['startActivation']==17 and plan['completedRounds']==1
    assert plan['checkpoint'].endswith('checkpoint.previous.npz' if fallback else 'checkpoint.npz')
    assert not (tmp_path/'partial_activation_telemetry.jsonl').exists()


def test_confirmation_recovery_refuses_missing_checkpoint_and_failed_run(monkeypatch,tmp_path):
    import basil_core.dirichlet_prior_study as study
    monkeypatch.setattr(study,'validate_evidence',lambda _:None)
    cfg=configs()[0]
    (tmp_path/'run.json').write_text(json.dumps(dict(config=cfg,status='interrupted')))
    with pytest.raises(ValueError,match='No verified'):study.recovery_plan(tmp_path,cfg)
    (tmp_path/'run.json').write_text(json.dumps(dict(config=cfg,status='failed')))
    with pytest.raises(ValueError,match='investigated'):study.recovery_plan(tmp_path,cfg)


def test_terminal_manifest_cannot_hide_incomplete_checkpoint(monkeypatch,tmp_path):
    import basil_core.dirichlet_prior_study as study
    import basil_core.protocol_checkpoint as checkpoint
    monkeypatch.setattr(study,'validate_evidence',lambda _:None)
    monkeypatch.setattr(checkpoint,'load_checkpoint',lambda *a,**k:dict(completed_rounds=29,next_activation=300))
    cfg=configs()[0]
    (tmp_path/'run.json').write_text(json.dumps(dict(config=cfg,status='completed')))
    (tmp_path/'checkpoint.npz').write_bytes(b'unit-test-placeholder')
    with pytest.raises(ValueError,match='disagrees'):study.recovery_plan(tmp_path,cfg)


def test_live_csv_and_restored_gui_history_use_committed_rounds(tmp_path,capsys):
    import csv
    import numpy as np
    from scripts.run_dirichlet_calibration import write_round_metrics,saved_round_rows,emit_restored
    state=dict(completed_rounds=1,next_activation=13,round_node_accuracy=np.full((30,10),.25,np.float32),
        round_node_loss=np.ones((30,10)),round_node_per_class=np.full((30,10,10),.25,np.float32),
        telemetry=[dict(round=1,nodeId=2,globalTestAccuracy=.3)])
    rows=saved_round_rows(state)
    write_round_metrics(tmp_path,rows)
    with (tmp_path/'round_metrics.csv').open() as handle:stored=list(csv.DictReader(handle))
    assert len(stored)==1 and stored[0]['round']=='0' and float(stored[0]['full_test_accuracy'])==.25
    assert not (tmp_path/'round_metrics.csv.tmp').exists()
    emit_restored(state,15.,configs()[0])
    restored=json.loads(capsys.readouterr().out)['restored']
    assert restored['round']==1 and restored['progress']==pytest.approx(13/300)
    assert restored['accuracyHistory']==[[1,.25,.25]]
    assert restored['activationHistory']==[[1.3,.3]]


def test_unapproved_multicondition_confirmation_cannot_bypass_gui_gate():
    from scripts.run_dirichlet_calibration import run_pair
    with pytest.raises(ValueError,match='evidence-approved'):
        run_pair('GPU',stage='threat')


def test_plot_identity_reads_older_diagnostic_metadata_without_rewriting():
    from reporting.sequential_basil_plots import plot_identity
    metadata=dict(config=configs()[3],researchValid=False)
    original=json.dumps(metadata,sort_keys=True)
    assert plot_identity(metadata)==('D','diagnostic')
    assert json.dumps(metadata,sort_keys=True)==original
    assert plot_identity(dict(metadata,runId='canonical-D'))==('canonical-D','diagnostic')
    with pytest.raises(ValueError,match='identifier'):plot_identity(dict(config={},researchValid=False))


def test_real_runner_io_interrupt_resume_and_live_events_without_learning(monkeypatch,tmp_path,capsys):
    """Real runner/checkpoint I/O, synthetic engine: zero SGD or research data."""
    import contextlib
    import hashlib
    import numpy as np
    # Import before mocking subprocess.check_output: Keras may query fontconfig
    # during its first import and expects real bytes, not the fake Git string.
    import tensorflow as tf
    import scripts.run_dirichlet_calibration as runner
    import basil_core.dirichlet_prior_study as study
    import basil_core.protocol_checkpoint as checkpoint
    import basil_core.research_protocol as protocol
    import basil_core.dirichlet_study as partitioner
    import basil_core.dirichlet_diagnosis as diagnosis
    import basil_core.iid_runtime as runtime
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(study,'validate_evidence',lambda _:None)
    monkeypatch.setattr(runtime,'prepare_environment',lambda _:None)
    monkeypatch.setattr(runtime,'configure_tensorflow',lambda *a:None)
    monkeypatch.setattr(runtime,'runtime_metadata',lambda *a:dict(device='CPU',numericalExecution={'device':'CPU'}))
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:'synthetic-unit-test')
    params=[np.ones(3,np.float32)];initial=protocol.params_hash(params)
    class Model:
        def get_weights(self):return params
    monkeypatch.setattr(protocol,'build_model',lambda *a:Model())
    audit=dict(partitionHash='a'*64,nodes=[dict(optimizerStepsPerActivation=5) for _ in range(10)])
    monkeypatch.setattr(partitioner,'partition_audit',lambda *a:(audit,None))
    monkeypatch.setattr(diagnosis,'array_identity_hash',lambda _: 'synthetic-data')
    source=tmp_path/'scientific.py';source.write_text('# synthetic checkpoint fixture\n')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(checkpoint,'checkpoint_implementation',lambda _: {'scientific.py':digest})
    cfg=study.configurations('a'*64,initial,'c'*64)[0];cfg['executionDevice']='CPU'
    original_load=np.load
    arrays={
        'x_train_float32.npy':np.broadcast_to(np.zeros((1,32,32,3),np.float32),(50000,32,32,3)),
        'x_test_float32.npy':np.broadcast_to(np.zeros((1,32,32,3),np.float32),(10000,32,32,3)),
        'y_train_int32.npy':np.repeat(np.arange(10,dtype=np.int32),5000),
        'y_test_int32.npy':np.repeat(np.arange(10,dtype=np.int32),1000)}
    def load(path,*a,**k):
        from pathlib import Path
        return arrays[Path(path).name] if Path(path).name in arrays else original_load(path,*a,**k)
    monkeypatch.setattr(np,'load',load)
    invocations=[]
    def engine(config,*data,**kw):
        assert config==cfg and callable(kw['worker_factory'])
        state=kw['resume_state'];invocations.append(state is not None)
        if state is None:
            state=dict(next_activation=0,completed_rounds=0,telemetry=[],
                node_params=[params for _ in range(10)],memories=protocol.initialize_memories(params,10,5))
            for name in checkpoint.STATE_ARRAYS:
                shape=(30,10,10) if name in ('activation_per_class','round_node_per_class') else (30,10)
                state[name]=np.zeros(shape,np.float32)
        for cursor in range(state['next_activation'],300):
            r,node=divmod(cursor,10)
            record=dict(round=r,nodeId=node,inputSourceNode=(node-1)%10,inputSnapshotRound=r-1,
                candidates=[],trainedOutputHash=initial,inputHash=initial,snapshotAnchorReferenceHash=initial,
                configuredByzantine=False,attackActive=False,attackRelativeParameterChange=0.,
                baseGradientNorm=1.,robustObjectiveGradientNorm=1.,optimizerSteps=5,
                ebmMode='none',ebmCoefficient=0.,effectiveCoordinateSigma=0.,outgoingLinks=[],
                globalTestAccuracy=.25,perClassGlobalTestAccuracy=[.25]*10)
            state['telemetry'].append(record);kw['activation_callback'](record)
            state['next_activation']=cursor+1;kw['checkpoint_callback'](state)
            if len(invocations)==1:raise InterruptedError('Synthetic stop after one committed activation')
            if node==9:
                state['round_node_accuracy'][r]=.25;state['round_node_per_class'][r]=.25
                state['round_node_loss'][r]=1.
                kw['round_callback'](dict(round=r,averageAccuracy=.25,worstNodeAccuracy=.25,
                    fullTestAccuracy=.25,fullTestLoss=1.,perClassAccuracy=[.25]*10))
                state['completed_rounds']=r+1;kw['checkpoint_callback'](state)
        return dict(finalNodeParams=state['node_params'],telemetry=state['telemetry'],
            fullTestAccuracy=state['round_node_accuracy'][:,9],fullTestLoss=state['round_node_loss'][:,9],
            averageAccuracy=state['round_node_accuracy'].mean(axis=1),
            worstNodeAccuracy=state['round_node_accuracy'].min(axis=1),
            fullTestPerClassAccuracy=state['round_node_per_class'][:,9])
    monkeypatch.setattr(protocol,'run_research_protocol',engine)
    root=(tmp_path/cfg['resultDirectory']).parent
    kwargs=dict(stage='threat',output_root=root,configs_override=[cfg],gui_events=True)
    with pytest.raises(InterruptedError):runner.run_pair('CPU',**kwargs)
    output=tmp_path/cfg['resultDirectory']
    assert json.loads((output/'run.json').read_text())['status']=='interrupted'
    assert study.recovery_plan(output,cfg)['startActivation']==1
    runner.run_pair('CPU',recover=True,**kwargs)
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert invocations==[False,True]
    assert len([e for e in events if 'networkUpdate' in e])==300
    assert [e['roundComplete']['round'] for e in events if 'roundComplete' in e]==list(range(30))
    assert next(e['restored'] for e in events if 'restored' in e)['progress']==pytest.approx(1/300)
    assert study.recovery_plan(output,cfg)['mode']=='completed'
    assert not (root/'comparison.json').exists()
    assert not (output/'partial_activation_telemetry.jsonl').exists()
    assert len((output/'round_metrics.csv').read_text().splitlines())==31
