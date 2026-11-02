"""Boundary preservation, server ownership, resource admission and mixed exports."""
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from basil_core import basil_sharp_thesis as study
from basil_core import round_continuation as continuation
from basil_core import continuation_campaign as campaign
from scripts import run_fedsharp_500r as launcher


def test_all_150_configs_extend_horizon_without_changing_scientific_parameters():
    spec,items=continuation.catalog()
    report=continuation.description(spec,items)
    assert report['configurations']==150 and report['fedSharpPlus']==12
    assert report['additionalActivations']==6000000 and report['additionalOptimizerSteps']==30000000
    assert report['datasets']=={'mnist':78,'cifar10':72}
    assert len({i['config']['runId'] for i in items})==150
    for item in items:
        parent=continuation.source_record(item)
        assert item['config']['rounds']==500 and item['sourceCursor']==10000
        assert item['config']['runId']!=item['sourceRunId']
        assert item['config']==continuation.extended_view(parent['config'],500)
    allocation=campaign.assignments(items,{'themachine':58,'hal9k':9})
    assert len(allocation)==150
    assert sum(Counter(allocation.values()).values())==150
    assert 120<=sum(h=='themachine' for h in allocation.values())<=135
    for item in items:
        siblings=[i for i in items if (i['sourceArchive'],i['config']['dataset'],i['config']['partition'],i['config']['comparisonLabel'])==
            (item['sourceArchive'],item['config']['dataset'],item['config']['partition'],item['config']['comparisonLabel'])]
        assert len({allocation[i['config']['runId']] for i in siblings})==1


def test_description_creates_no_hardware_data_or_training_state(tmp_path,monkeypatch,capsys):
    def forbidden(*a,**kw):raise AssertionError('Description touched execution resources.')
    monkeypatch.setattr(campaign,'probe',forbidden)
    monkeypatch.setattr(campaign,'initialize',forbidden)
    monkeypatch.setattr(launcher,'execute',forbidden)
    assert launcher.main(['--describe','--write-configs',str(tmp_path/'configs.json')])==0
    report=json.loads(capsys.readouterr().out)
    assert report['trainingStarted'] is False
    assert len(json.loads((tmp_path/'configs.json').read_text())['items'])==150


def test_live_probe_reserves_gpu_context_headroom_and_global_ram(monkeypatch):
    from basil_core import execution_profile
    from scripts import run_cart_v2_scaled as scaled
    spec=continuation.read(continuation.PRESET)
    monkeypatch.setattr(campaign,'host_name',lambda:'themachine')
    free=[97887-v for v in (22189,15801,28673,44681)]
    inventory=[dict(index=g,uuid=f'GPU-{g:08x}',**{'memory.free':value}) for g,value in enumerate(free)]
    monkeypatch.setattr(execution_profile,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(scaled,'available_ram_mib',lambda:900000)
    result=campaign.probe(spec)
    assert [d['slots'] for d in result['devices']]==[16,18,14,10]
    assert result['maximumWorkersNow']==58
    monkeypatch.setattr(scaled,'available_ram_mib',lambda:90*1024)
    assert campaign.probe(spec)['maximumWorkersNow']==9


@pytest.fixture(scope='module',autouse=True)
def cpu():
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()


def parent_fixture(tmp_path,monkeypatch,arm='basil_joint',attack='hidden'):
    from tests.test_cart_comparison_runtime import data
    from basil_core.sharp_protocol import save_state,vector_hash
    from scripts.run_cart_v2_study import runtime
    import tensorflow as tf
    import basil_core.sharp_cart_v2_protocol as protocol
    family='basil' if arm.startswith('basil') else 'sharp'
    env='noise_only' if arm.endswith('noise_only') else 'clean' if arm.endswith('clean') else 'joint'
    args=SimpleNamespace(datasets=['mnist'],partitions=['dirichlet'],seeds=[2025],families=[family],environments=[env],
        nodes=4,rounds=2,local_steps=1,alpha=.5,sigma=.01,attack_start=1,attack_type=attack,gaussian_std=1.)
    old=next(v for v in study.selected(args) if v['comparisonLabel']==arm);new=continuation.extended_view(old,4)
    x,y,chunks=data();held=np.arange(20);audit={'fixture':'same private fitting/holdout allocation'}
    from basil_core.sharp_data import digest
    dataset=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
    worker=study.configure_worker(old,'CPU');initial=vector_hash(worker.export())
    with study.intervention(old,worker) as cfg:state=protocol.run(cfg,worker,x,y,chunks,x[held],y[held])
    source=study.source();numerical=runtime(tf);numerical['device']='/CPU:0'
    contract=dict(config=cfg,comparisonCase=old['comparisonCase'],methodRevision=study.REVISION,
        source=source,runtime=numerical,initialHash=initial,data=dataset,partition=audit)
    source_root=tmp_path/'docs/results/parent';folder=source_root/old['runId'];folder.mkdir(parents=True)
    old_path=folder/'checkpoint.npz';save_state(old_path,state,contract)
    record=dict(status='COMPLETED',config=old,source=source,contract=contract,methodRevision=study.REVISION,
        data=dataset,partition=audit,initialHash=initial,runtime=numerical,completedActivations=8,
        completedRounds=2,optimizerSteps=8,elapsedSeconds=state['elapsedSeconds'],meanAccuracy=state['roundMetrics'][-1]['meanAccuracy'])
    (folder/'run.json').write_text(json.dumps(record))
    np.savez(folder/'metrics.npz',roundNodeAccuracy=np.array([m['nodeAccuracy'] for m in state['roundMetrics']]),
        roundNodePerClassAccuracy=np.array([m['perClass'] for m in state['roundMetrics']]),
        roundHonestMeanAccuracy=np.array([m['honestMeanAccuracy'] for m in state['roundMetrics']]))
    queue=tmp_path/'shared';seed=queue/'seeds/parent.npz';seed_hash=continuation.copy_seed(old_path,seed)
    item=dict(config=new,sourceArchive='docs/results/parent',sourceRunId=old['runId'],
        sourceRunSha256=continuation.sha(folder/'run.json'),sourceMetricsSha256=continuation.sha(folder/'metrics.npz'),
        sourceCursor=8,sourceRounds=2,sourceElapsedSeconds=state['elapsedSeconds'],
        seedRelativePath='seeds/parent.npz',seedSha256=seed_hash,assignedHost='hal9k',initialAccuracy=record['meanAccuracy'])
    output=tmp_path/'continued'
    plan=dict(revision=study.REVISION,configs=[new],source=source,data={'mnist':dataset},
        partitions={new['runId']:audit},continuations={new['runId']:item},
        profiles=[{'output_root':str(output)}],sharedQueue=str(queue),host='hal9k',continuationQueueId='fixture')
    monkeypatch.setattr(continuation,'ROOT',tmp_path)
    return SimpleNamespace(old=old,new=new,cfg=cfg,x=x,y=y,chunks=chunks,held=held,audit=audit,dataset=dataset,
        source=source,contract=contract,state=state,old_path=old_path,item=item,plan=plan,output=output)


@pytest.mark.parametrize('arm,attack',[
    ('basil_clean','hidden'),('basil_noise_only','hidden'),('basil_joint','gaussian'),
    ('basil_acds','hidden'),('sharp_joint','hidden'),('sharp_cart','hidden'),('sharp_cart','gaussian')])
def test_real_model_fork_matches_uninterrupted_training_and_preserves_original(tmp_path,monkeypatch,arm,attack):
    from basil_core.sharp_protocol import load_state
    import basil_core.sharp_cart_v2_protocol as protocol
    fixture=parent_fixture(tmp_path,monkeypatch,arm,attack)
    worker=study.configure_worker(fixture.new,'CPU')
    target=dict(fixture.contract,config=study.effective_config(fixture.new),comparisonCase=fixture.new['comparisonCase'])
    # Explicitly allow a recorded host/device assignment transition, not a runtime-version change.
    target['runtime']=copy.deepcopy(target['runtime']);target['runtime']['host']='themachine'
    before=continuation.sha(fixture.old_path)
    with pytest.raises(ValueError,match='identity differs'):
        load_state(fixture.old_path,target,worker.dimension)
    state,receipt=continuation.bootstrap(fixture.plan,fixture.new,fixture.output/fixture.new['runId'],worker,
        target,fixture.x[fixture.held],fixture.y[fixture.held])
    assert receipt['sourceCursor']==8 and state['cursor']==8
    assert state['telemetry']==fixture.state['telemetry'] and state['roundMetrics']==fixture.state['roundMetrics']
    assert receipt['maximumBoundaryInferenceDifference']==0
    with study.intervention(fixture.new,worker) as cfg:
        actual=protocol.run(cfg,worker,fixture.x,fixture.y,fixture.chunks,fixture.x[fixture.held],fixture.y[fixture.held],state=state)
    fresh=study.configure_worker(fixture.new,'CPU')
    with study.intervention(fixture.new,fresh) as cfg:
        expected=protocol.run(cfg,fresh,fixture.x,fixture.y,fixture.chunks,fixture.x[fixture.held],fixture.y[fixture.held])
    np.testing.assert_array_equal(actual['nodeParams'],expected['nodeParams'])
    np.testing.assert_array_equal(actual['outbound'],expected['outbound'])
    assert actual['telemetry']==expected['telemetry'] and actual['roundMetrics']==expected['roundMetrics']
    assert len(actual['roundMetrics'])==4 and len(actual['telemetry'])==16
    for node,reference in expected['references'].items():
        np.testing.assert_array_equal(actual['references'][node]['params'],reference['params'])
    assert continuation.sha(fixture.old_path)==before


def test_fork_rejects_runtime_change_and_corrupt_seed_before_target_checkpoint(tmp_path,monkeypatch):
    fixture=parent_fixture(tmp_path,monkeypatch)
    worker=study.configure_worker(fixture.new,'CPU')
    contract=dict(fixture.contract,config=study.effective_config(fixture.new),comparisonCase=fixture.new['comparisonCase'])
    contract['runtime']=dict(contract['runtime'],tensorflow='different')
    folder=fixture.output/fixture.new['runId']
    with pytest.raises(ValueError,match='runtime changed'):
        continuation.bootstrap(fixture.plan,fixture.new,folder,worker,contract,fixture.x[:20],fixture.y[:20])
    assert not (folder/'checkpoint.npz').exists()
    contract['runtime']=fixture.contract['runtime']
    seed=Path(fixture.plan['sharedQueue'])/fixture.item['seedRelativePath'];seed.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum differs'):
        continuation.bootstrap(fixture.plan,fixture.new,folder,worker,contract,fixture.x[:20],fixture.y[:20])
    assert not (folder/'checkpoint.npz').exists()


def test_worker_bootstrap_resumption_and_export_prefix_are_end_to_end(tmp_path,monkeypatch):
    fixture=parent_fixture(tmp_path,monkeypatch,'sharp_cart')
    from basil_core import thesis_raw_data
    from scripts import cart_comparison_worker
    from reporting import round_continuation as reporting
    monkeypatch.setattr(thesis_raw_data,'training_data',lambda dataset:(fixture.x,fixture.y))
    monkeypatch.setattr(study,'partitions_for',lambda view,y:(fixture.chunks,fixture.held,fixture.audit))
    monkeypatch.setattr(study,'ROOT',tmp_path)
    # Source fingerprints still point to real installed files.
    monkeypatch.setattr(study,'source',lambda:fixture.source)
    output=fixture.output;output.mkdir()
    (output/'plan.json').write_text(json.dumps(fixture.plan))
    for name in fixture.source:
        path=output/'source_snapshots'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
    with study.installed():cart_comparison_worker.run(output,0,cpu=True)
    run=output/fixture.new['runId'];record=json.loads((run/'run.json').read_text())
    assert record['status']=='COMPLETED' and record['completedActivations']==16 and record['optimizerSteps']==16
    assert record['runtimeAdapter']['startedCursor']==8
    assert reporting.collect(output)[1][0]['curve'].shape==(4,)
    assert reporting.generate(output,output/'thesis_figures')['prefixMetricsExactlyPreserved']
    (output/'campaign.json').write_text(json.dumps(dict(status='COMPLETED',completed=1)))
    from scripts.export_basil_sharp_thesis import export
    assert export(output,tmp_path/'export')==1
    assert (tmp_path/'export'/fixture.new['runId']/'continuation.json').is_file()


def test_shared_status_waiting_taken_server_columns_and_inherited_work_eta(tmp_path,monkeypatch):
    spec,items=continuation.catalog();items=copy.deepcopy(items[:2])
    for item,host in zip(items,('hal9k','themachine')):
        item.update(assignedHost=host,initialAccuracy=.8)
    campaign.atomic(tmp_path/'continuation_plan.json',dict(items=items,probes={'hal9k':{},'themachine':{}}))
    state=campaign.public_snapshot(tmp_path)
    assert state['completedActivations']==20000 and state['totalActivations']==100000
    assert state['etaSeconds'] is None
    assert all(':WAITING:' in r['phase'] for r in state['runs'])
    row=dict(state['runs'][0],status='RUNNING',gpu=1,completedActivations=10100,completedRounds=101,phase='TRAINING')
    campaign.atomic(tmp_path/'hosts/hal9k.json',dict(host='hal9k',updatedAt=100.,startedAt=90.,
        runIds=[items[0]['config']['runId']],snapshot=dict(status='RUNNING',runs=[row])))
    monkeypatch.setattr(campaign.time,'time',lambda:100.)
    state=campaign.public_snapshot(tmp_path)
    assert state['activationsPerSecond']==10 and state['etaSeconds']==7990
    assert state['runs'][0]['phase'].startswith('shared:hal9k:TAKEN:')
    campaign.refresh_csv(tmp_path,state)
    assert 'SERVER,GPU,CLAIM,STATUS' in (tmp_path/'configuration_status.csv').read_text()
    assert 'themachine' in campaign.render_status(tmp_path)
    raw=json.dumps(state)
    assert str(tmp_path) not in raw and 'checkpoint' not in raw
    monkeypatch.setattr(campaign.time,'time',lambda:200.)
    stale=campaign.public_snapshot(tmp_path)
    assert stale['status']=='SUPERVISOR_RECOVERY_NEEDED' and stale['etaSeconds'] is None


def test_host_status_cannot_claim_a_config_assigned_to_other_server(tmp_path):
    _,items=continuation.catalog();item=dict(items[0],assignedHost='themachine',initialAccuracy=.8)
    campaign.atomic(tmp_path/'continuation_plan.json',dict(items=[item],probes={'hal9k':{},'themachine':{}}))
    row=campaign.public_snapshot(tmp_path)['runs'][0]
    campaign.atomic(tmp_path/'hosts/hal9k.json',dict(updatedAt=0.,startedAt=0.,
        runIds=[item['config']['runId']],snapshot=dict(status='RUNNING',runs=[row])))
    with pytest.raises(ValueError,match='ownership collision'):campaign.public_snapshot(tmp_path)


def test_two_server_preparation_is_disjoint_and_output_binding_prevents_duplicate_namespace(tmp_path,monkeypatch):
    from basil_core import execution_profile
    spec,items=continuation.catalog()
    for item in items:
        item.update(assignedHost='hal9k' if item['config']['dataset']=='mnist' else 'themachine',
            seedSha256='a'*64,seedRelativePath='seeds/fixture.npz',initialAccuracy=.8)
    source=study.source();queue=tmp_path/'queue'
    probes={host:dict(devices=[dict(gpu=g,uuid=f'GPU-{g:08x}',slots=10) for g in spec['hosts'][host]['gpus']])
        for host in spec['hosts']}
    master=dict(queueId='shared-fixture',source=source,items=items,resources=spec['resources'],probes=probes)
    campaign.atomic(queue/'continuation_plan.json',master)
    for name in source:
        target=queue/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((study.ROOT/name).read_bytes())
    inventory=[dict(index=g,uuid=f'GPU-{g:08x}',**{'memory.free':97887}) for g in range(4)]
    monkeypatch.setattr(execution_profile,'gpu_inventory',lambda:inventory)
    prepared=[]
    for host in ('hal9k','themachine'):
        monkeypatch.setattr(campaign,'host_name',lambda h=host:h)
        monkeypatch.setattr(campaign.socket,'gethostname',lambda h=host:h)
        plan=campaign.prepare_host(queue,tmp_path/host)
        assert plan==campaign.prepare_host(queue,tmp_path/host)
        prepared.append(plan)
        for cfg in plan['configs']:
            record=json.loads((tmp_path/host/cfg['runId']/'run.json').read_text())
            assert record['completedActivations']==10000 and record['status']=='PENDING'
            assert 'contract' not in record  # Unstarted continuations can use any verified GPU on their server.
        with pytest.raises(ValueError,match='already bound'):
            campaign.prepare_host(queue,tmp_path/(host+'-duplicate'))
    a,b=({c['runId'] for c in plan['configs']} for plan in prepared)
    assert not a&b and len(a|b)==150
    state=campaign.public_snapshot(queue)
    assert state['completedActivations']==1500000 and state['etaSeconds'] is None


def test_background_controller_detaches_and_starts_permanent_monitor_without_training_locally(tmp_path,monkeypatch):
    queue=tmp_path/'queue';queue.mkdir();output=tmp_path/'local'
    campaign.atomic(queue/'continuation_plan.json',dict(queueId='123456789abcdef',resources={}))
    plan=dict(configs=[dict(runId='case')])
    monkeypatch.setattr(campaign,'prepare_host',lambda *a:plan)
    monkeypatch.setattr(campaign,'host_name',lambda:'hal9k')
    captured={};calls=[]
    def spawn(command,**kwargs):
        captured.update(command=command,**kwargs)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(launcher.subprocess,'Popen',spawn)
    monkeypatch.setattr(launcher.subprocess,'run',lambda command,**kw:calls.append(command))
    args=SimpleNamespace(queue=queue,output=output,background=True,ntfy_topic='papermerge-ojaswi',no_publish=False)
    assert launcher.execute(args)==0
    assert captured['start_new_session'] is True
    assert captured['command'][captured['command'].index('--output')+1]==str(output)
    assert '--background' not in captured['command'] and '--execute' in captured['command']
    assert len(calls)==1 and 'share_cart_monitor.py' in calls[0][1]
    assert calls[0][calls[0].index('--output')+1]==str(queue)


def test_initialization_copies_seeds_without_mutating_parents_and_refuses_second_assignment(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'host_name',lambda:'hal9k')
    spec,all_items=continuation.catalog()
    grouped={}
    for item in all_items:
        c=item['config'];key=(item['sourceArchive'],c['dataset'],c['partition'],c['comparisonLabel'])
        grouped.setdefault(key,[]).append(item)
    items=copy.deepcopy(sum(list(grouped.values())[:2],[]));queue=tmp_path/'queue'
    for index,item in enumerate(items):
        path=tmp_path/f'parent-{index}.npz';path.write_bytes(b'original seed fixture '+str(index).encode())
        item['originalCheckpoint']=str(path)
    before={i['originalCheckpoint']:Path(i['originalCheckpoint']).read_bytes() for i in items}
    for host in ('hal9k','themachine'):
        campaign.atomic(queue/'probes'/f'{host}.json',dict(host=host,observedAt=campaign.time.time(),
            maximumWorkersNow=9,resources=spec['resources']))
    monkeypatch.setattr(continuation,'source_record',lambda item:dict(meanAccuracy=.8))
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*1024**3))
    master=campaign.initialize(queue,spec,items,['hal9k','themachine'])
    assert len(master['items'])==6 and Counter(i['assignedHost'] for i in master['items'])=={'hal9k':3,'themachine':3}
    for item in master['items']:
        assert continuation.sha(queue/item['seedRelativePath'])==item['seedSha256']
        assert Path(item['originalCheckpoint']).read_bytes()==before[item['originalCheckpoint']]
    with pytest.raises(FileExistsError,match='already frozen'):
        campaign.initialize(queue,spec,items,['hal9k','themachine'])


def test_seed_copy_refuses_source_as_destination(tmp_path):
    path=tmp_path/'original.npz';path.write_bytes(b'original')
    with pytest.raises(ValueError,match='separate copy'):continuation.copy_seed(path,path)
    assert path.read_bytes()==b'original'


def test_gpu_warmup_heartbeat_keeps_startup_distinct_from_supervisor_failure(tmp_path,monkeypatch):
    _,items=continuation.catalog();item=dict(items[0],assignedHost='hal9k',initialAccuracy=.8)
    campaign.atomic(tmp_path/'continuation_plan.json',dict(items=[item],probes={'hal9k':{}}))
    row=campaign.public_snapshot(tmp_path)['runs'][0]
    campaign.atomic(tmp_path/'hosts/hal9k.json',dict(updatedAt=0.,startedAt=0.,
        runIds=[item['config']['runId']],snapshot=dict(status='VERIFYING_GPUS',runs=[row])))
    campaign.atomic(tmp_path/'heartbeats/hal9k.json',dict(host='hal9k',updatedAt=200.))
    monkeypatch.setattr(campaign.time,'time',lambda:201.)
    state=campaign.public_snapshot(tmp_path)
    assert state['status']=='VERIFYING_GPUS' and state['etaSeconds'] is None
