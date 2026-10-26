"""Fixed replication contracts, seed isolation and independent GPU scheduling."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def arguments(tmp_path,**changes):
    values=dict(output=tmp_path/'.papermerge/local-tests/replication',profile='hpc',gpus='0,1,2,3',
        gpu_memory_mib=None,recover=False)
    return SimpleNamespace(**dict(values,**changes))


@pytest.mark.parametrize('seed',[2025,2026,2027])
def test_replication_seeds_preserve_original_schedule_and_exact_pairing(seed):
    from basil_core.cart_validation import configurations,validate_config
    from basil_core.research_protocol import resolve_attackers
    configs=configurations('a'*64,'b'*64,'CPU','replication_30r',seed=seed)
    assert [c['runId'] for c in configs]==[f'{m}_{letter}' for letter in 'ABCD' for m in ('prior','budgeted')]
    for cfg in configs:
        validate_config(cfg)
        assert (cfg['seed'],cfg['attackerSelectionSeed'])==(seed,seed)
        assert cfg['resolvedAttackerIds']==list(resolve_attackers(seed,cfg['attackerCount']))
        assert (cfg['nRounds'],cfg['attackStartRound'],cfg['channelNoiseStartRound'])==(30,20,0)
        assert (cfg['localEpochs'],cfg['batchSize'],cfg['learningRate'],cfg['snapshotAnchorMu'])==(5,512,.05,.01)
        assert cfg['cartHoldoutSeed']==2025 and cfg['cartReplicationSeeds']==[2025,2026,2027]
        assert cfg['officialTestUsed'] is False and cfg['researchValid'] is False
        for field,value in [('attackStartRound',1),('nRounds',31),('cartCorrectionRatioCap',.2),
                ('cartReplicationSeeds',[2027,2026,2025]),('cartHoldoutSeed',seed+1)]:
            with pytest.raises(ValueError):validate_config(dict(cfg,**{field:value}))
    for a,b in zip(configs[::2],configs[1::2]):
        differences={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        assert differences=={'cartBeta','cartMemoryMode','cartCorrectionRatioCap','runId','experimentName',
            'displayName','resultDirectory','plotDirectory'}


def test_partition_seed_changes_allocation_but_holdout_stays_fixed():
    import numpy as np
    from basil_core.cart_validation import split_training,partition
    labels=np.repeat(np.arange(10,dtype=np.int32),5000)
    fit,heldout=split_training(labels)
    hashes=[]
    for seed in (2025,2026,2027):
        digest,chunks=partition(labels[fit],seed=seed)
        hashes.append(digest)
        assert all(len(c)>0 for c in chunks)
        np.testing.assert_array_equal(np.sort(np.concatenate(chunks)),np.arange(49000))
        assert partition(labels[fit],seed=seed)[0]==digest
        np.testing.assert_array_equal(heldout,split_training(labels)[1])
    assert len(set(hashes))==3


@pytest.mark.parametrize('seed',[True,2024,2028])
def test_undeclared_seeds_rejected(seed):
    from basil_core.cart_validation import configurations
    with pytest.raises(ValueError):configurations('a'*64,'b'*64,'CPU','replication_30r',seed=seed)


def test_only_complete_replication_pairs_can_be_selected():
    from basil_core.cart_validation import selected_configurations
    assert len(selected_configurations('a'*64,'b'*64,'CPU','replication_30r',seed=2027,pair='D'))==2
    with pytest.raises(ValueError):selected_configurations('a'*64,'b'*64,'CPU','clean',pair='A')
    with pytest.raises(ValueError):selected_configurations('a'*64,'b'*64,'CPU','clean',seed=2026)


def test_four_gpu_plan_assigns_one_fixed_pair_per_slot_and_all_seeds(tmp_path,monkeypatch):
    import scripts.run_cart_replication as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    planned=runner.plan(arguments(tmp_path))
    assert planned['totalArms']==24 and planned['totalActivations']==7200
    assert len(planned['jobs'])==12 and len(planned['resources'])==4
    assert [(j['seed'],j['pair'],j['resources']['gpu']) for j in planned['jobs']]==[
        (seed,pair,gpu) for seed in (2025,2026,2027) for gpu,pair in enumerate('ABCD')]
    for job in planned['jobs']:
        command=runner.child_command(job,recover=True)
        assert '--execute' in command and '--recover' in command
        assert command[command.index('--pair')+1]==job['pair']
        assert command[command.index('--campaign-plan')+1]==str(args_root(tmp_path)/'plan.json')
    with pytest.raises(ValueError):runner.plan(arguments(tmp_path,gpus='0,0'))
    with pytest.raises(ValueError):runner.plan(arguments(tmp_path,gpu_memory_mib=0))
    with pytest.raises(ValueError):runner.plan(arguments(tmp_path,profile='local',gpus='0,1,2,3'))


def args_root(tmp_path):
    return tmp_path/'.papermerge/local-tests/replication'


def test_local_child_environment_overrides_an_inherited_hpc_profile(tmp_path,monkeypatch):
    from basil_core.execution_profile import ExecutionProfile,ENV
    from scripts.run_cart_replication import child_environment
    inherited=ExecutionProfile(name='hpc',gpu=3,output_root=str(tmp_path)).environment()[ENV]
    monkeypatch.setenv(ENV,inherited)
    local=ExecutionProfile(gpu_uuid='GPU-abc')
    environment=child_environment(local)
    assert json.loads(environment[ENV])['name']=='local'
    assert json.loads(environment[ENV])['gpu_uuid']=='GPU-abc'
    assert environment['CUDA_VISIBLE_DEVICES']=='0'


def test_pilot_starts_fresh_worker_with_frozen_gpu_and_loader_environment(tmp_path,monkeypatch):
    from dataclasses import asdict
    from basil_core.execution_profile import ExecutionProfile,ENV
    import scripts.run_cart_replication as runner
    import scripts.run_cart_train_validation as worker
    profile=ExecutionProfile(name='hpc',gpu=3,gpu_uuid='GPU-abc',memory_mib=8192,
        headroom_mib=2048,output_root=str(tmp_path/'pilot'))
    prepared=dict(output=str(tmp_path/'pilot'),jobs=[dict(seed=2026,pair='D',resources=asdict(profile))])
    monkeypatch.setattr(runner,'preflight',lambda _:prepared)
    monkeypatch.setattr(runner,'verified_devices',lambda _:None)
    def forbidden(*_):raise AssertionError('Pilot data/GPU work ran in the supervisor process')
    monkeypatch.setattr(worker,'preflight',forbidden)
    monkeypatch.setattr('basil_core.execution_profile.cuda_wheel_paths',lambda:dict(
        libraryDirectories=['/private/cuda/lib'],binaryDirectories=['/private/cuda/bin']))
    calls=[]
    class Process:
        def __init__(self,command,**kwargs):calls.append((command,kwargs))
        def wait(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    runner.pilot(SimpleNamespace())
    command,options=calls[0]
    assert '--pilot-worker' in command and '--pilot' not in command
    assert command[command.index('--gpus')+1]=='3'
    assert command[command.index('--gpu-memory-mib')+1]=='8192'
    assert options['env']['LD_LIBRARY_PATH'].split(':')[0]=='/private/cuda/lib'
    assert options['env']['CUDA_VISIBLE_DEVICES']=='GPU-abc'
    assert json.loads(options['env'][ENV])==asdict(profile)


def test_failed_pilot_child_cannot_be_reported_as_success(tmp_path,monkeypatch):
    from dataclasses import asdict
    from basil_core.execution_profile import ExecutionProfile
    import scripts.run_cart_replication as runner
    profile=ExecutionProfile()
    monkeypatch.setattr(runner,'preflight',lambda _:dict(output=str(tmp_path),
        jobs=[dict(seed=2026,pair='D',resources=asdict(profile))]))
    monkeypatch.setattr(runner,'verified_devices',lambda _:None)
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:SimpleNamespace(wait=lambda:2))
    with pytest.raises(ValueError,match='Pilot worker failed with exit code 2'):
        runner.pilot(SimpleNamespace())


def test_campaign_data_change_is_rejected_before_gpu_environment(tmp_path,monkeypatch):
    import numpy as np
    from dataclasses import asdict
    import scripts.run_cart_train_validation as runner
    import basil_core.cart_validation as study
    import basil_core.protocol_checkpoint as checkpoints
    import basil_core.execution_profile as profiles
    import basil_core.iid_runtime as runtime
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.delenv(profiles.ENV,raising=False)
    cache=tmp_path/'experiments/cache/cifar10';cache.mkdir(parents=True)
    np.save(cache/'x_train_float32.npy',np.zeros((20,2),np.float32))
    np.save(cache/'y_train_int32.npy',np.repeat(np.arange(10,dtype=np.int32),2))
    monkeypatch.setattr(study,'validate_training_arrays',lambda *_:None)
    monkeypatch.setattr(study,'split_training',lambda _: (np.arange(10),np.arange(10,20)))
    monkeypatch.setattr(study,'partition',lambda _,**kwargs: ('a'*64,[np.array([i]) for i in range(10)]))
    monkeypatch.setattr(checkpoints,'checkpoint_implementation',lambda _: {'worker.py':'b'*64})
    def forbidden(*_):raise AssertionError('GPU initialization preceded the frozen data check')
    monkeypatch.setattr(runtime,'prepare_environment',forbidden)
    output=tmp_path/'.papermerge/local-tests/campaign/seed-2026/pair-D'
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(jobs=[dict(seed=2026,pair='D',id='D',output=str(output),
        resources=asdict(profiles.current_profile()))],checks=[dict(job='D',implementationSha256={'worker.py':'b'*64},
        dataSha256={'changed':'input'},partitionHash='a'*64)])))
    with pytest.raises(ValueError,match='frozen campaign plan'):
        runner.execute(SimpleNamespace(output=output,recover=False,stage='replication_30r',seed=2026,pair='D',
            device='GPU',campaign_plan=plan))
    assert not output.exists()


def test_campaign_monitor_keeps_live_arms_visible_and_finishes_once(tmp_path,monkeypatch):
    import io,time
    import scripts.run_cart_replication as runner
    from basil_core.terminal.state import identity
    from basil_core.terminal.dashboard import monitor
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    args=arguments(tmp_path);saved=runner.plan(args)
    saved['checks']=[dict(job=j['id'],nodeSamples=[512]*10,optimizerStepsPerArm=1500) for j in saved['jobs']]
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(saved))
    jobs=[dict(j,status='RUNNING' if i<4 else 'PENDING') for i,j in enumerate(saved['jobs'])]
    state=dict(status='RUNNING',owner=identity(),started_wall=time.time()-10,jobs=jobs)
    for j in jobs[:4]:
        arm=Path(j['output'])/j['arms'][0];arm.mkdir(parents=True)
        (arm/'run.json').write_text(json.dumps(dict(status='running',completedActivations=10,completedRounds=1,
            runtimeSeconds=10.,latestValidationMean=.2,latestValidationWorst=.1)))
        (arm/'checkpoint.npz').touch()
    (args.output/'campaign.json').write_text(json.dumps(state))
    snap=runner.read_snapshot(args.output,display_limit=8)
    assert sum(e['status']=='RUNNING' for e in snap['entries'])==4 and snap['totals']['total']==24
    assert snap['entries'][0]['completedOptimizerSteps']==50 and snap['entries'][0]['checkpoint']['activation']==10
    for j in jobs:
        for name in j['arms']:
            arm=Path(j['output'])/name;arm.mkdir(parents=True,exist_ok=True)
            (arm/'run.json').write_text(json.dumps(dict(status='completed',completedActivations=300,completedRounds=30,
                runtimeSeconds=300.,latestValidationMean=.2,latestValidationWorst=.1)))
    state['status']='COMPLETED';(args.output/'campaign.json').write_text(json.dumps(state))
    stream=io.StringIO()
    def forbidden(*_):raise AssertionError('Completed monitor must not refresh or query hardware')
    status=monitor(args.output,stream=stream,fetch=lambda root:runner.read_snapshot(root,display_limit=8),
        wait=forbidden,resources=forbidden)
    assert status=='COMPLETED' and 'ALL EXPERIMENTS COMPLETE (24/24)' in stream.getvalue()


def test_report_uses_all_three_seeds_and_fixed_post_attack_window():
    from scripts.report_cart_replication import aggregate
    rows=[]
    for seed in (2025,2026,2027):
        for condition in 'ABCD':
            rows.append(dict(seed=seed,condition=condition,controlMean=[.2]*30,
                cartMean=[.9]*20+[.21]*10,controlWorst=[.1]*30,cartWorst=[.11]*30))
    measured=aggregate(rows)
    assert all(c['primaryMeanDifferencePp']==pytest.approx(1.) for c in measured)
    assert all(c['primarySampleStdPp']==pytest.approx(0.) and c['improvedSeeds']==3 for c in measured)
    with pytest.raises(ValueError):aggregate(rows[:-1])
    with pytest.raises(ValueError):aggregate(rows[:-1]+[rows[0]])


def test_report_allows_gpu_identity_changes_but_preserves_numerical_policy():
    from scripts.report_cart_replication import numerical_policy
    a=dict(profile=dict(gpu=0,gpu_uuid='GPU-a',output_root='/a',memory_mib=8192,threads=1),
        gpu=dict(index=0,uuid='GPU-a',name='Blackwell',driver_version='580'),tensorflow='2.21',host='same')
    b=dict(profile=dict(a['profile'],gpu=3,gpu_uuid='GPU-b',output_root='/b'),
        gpu=dict(a['gpu'],index=3,uuid='GPU-b'),tensorflow='2.21',host='same')
    assert numerical_policy(a)==numerical_policy(b)
    b['profile']['memory_mib']=16384
    assert numerical_policy(a)!=numerical_policy(b)


def test_complete_evidence_can_be_reviewed_after_copying_from_hpc(tmp_path,monkeypatch):
    import hashlib
    import scripts.report_cart_train_validation as paired
    from scripts.report_cart_replication import review
    remote=Path('/remote/PaperMerge/.papermerge/local-tests/study')
    jobs=[]
    for seed in (2025,2026,2027):
        for letter in 'ABCD':
            jobs.append(dict(seed=seed,pair=letter,output=str(remote/f'seed-{seed}'/f'pair-{letter}')))
            folder=tmp_path/f'seed-{seed}'/f'pair-{letter}'/('prior_'+letter);folder.mkdir(parents=True)
            cfg=dict(seed=seed,partitionHash=hashlib.sha256(f'partition{seed}'.encode()).hexdigest(),
                initialModelHash=hashlib.sha256(f'init{seed}'.encode()).hexdigest())
            (folder/'run.json').write_text(json.dumps(dict(config=cfg,implementationSha256={'source':'same'},
                runtime={'host':'hpc','profile':{'gpu':ord(letter)-65}},dataSha256={'data':'same'})))
    (tmp_path/'plan.json').write_text(json.dumps(dict(revision='cart_replication_30r_three_seed_v1',
        seeds=[2025,2026,2027],rounds=30,attackStartRoundIndex=20,totalArms=24,officialTestUsed=False,
        primaryEndpoint='mean_node_validation_accuracy_round_indices_20_to_29',jobs=jobs,output=str(remote))))
    def checked_pair(root,plot=False):
        seed=int(root.parent.name.split('-')[1]);letter=root.name[-1]
        return dict(seed=seed,stage='replication_30r',methods=[dict(method=m+'_'+letter,
            validationMean=[value]*30,validationWorst=[.1]*30) for m,value in [('prior',.2),('budgeted',.21)]])
    # Each pair's scientific verifier is covered separately. Here the contract
    # under test is portable location resolution and complete three-seed grouping.
    monkeypatch.setattr(paired,'review',checked_pair)
    assert review(tmp_path)['status']=='all_saved_pairs_verified'


@pytest.mark.parametrize('failure',[False,True])
def test_supervisor_bounds_concurrency_and_stops_scheduling_after_failure(tmp_path,monkeypatch,failure):
    import scripts.run_cart_replication as runner
    import scripts.report_cart_replication as reporting
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'verified_devices',lambda _:None)
    monkeypatch.setattr(runner,'frozen_sources',lambda _:None)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    monkeypatch.setattr(runner.signal,'signal',lambda *_:None)
    reviewed=[];monkeypatch.setattr(reporting,'review',lambda root:reviewed.append(root))
    active=set();started=[];peak=[]
    class Process:
        def __init__(self,command,**kwargs):
            self.gpu=kwargs['env']['CUDA_VISIBLE_DEVICES'];self.ticks=0;self.returncode=None;self.pid=1234
            assert self.gpu not in active
            active.add(self.gpu);started.append(command);peak.append(len(active))
            self.fail=failure and len(started)==1
        def poll(self):
            if self.returncode is None:
                self.ticks+=1
                if self.ticks>=2:self.returncode=1 if self.fail else 0;active.remove(self.gpu)
            return self.returncode
        def terminate(self):
            if self.returncode is None:self.returncode=143;active.remove(self.gpu)
        def wait(self):return self.poll()
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    args=arguments(tmp_path);prepared=runner.plan(args)
    if failure:
        with pytest.raises(ValueError,match='stopped before complete'):runner.execute(args,prepared)
    else:runner.execute(args,prepared)
    state=json.loads((args.output/'campaign.json').read_text())
    assert max(peak)==4 and not active
    assert state['status']==('FAILED' if failure else 'COMPLETED')
    assert len(started)==(4 if failure else 12)
    assert bool(reviewed) is not failure


def test_replication_engine_rejects_wrong_initialization_and_partition_before_updates(monkeypatch):
    import numpy as np
    import tensorflow as tf
    import basil_core.research_protocol as protocol
    from basil_core.cart_validation import configurations,make_worker,partition
    def tiny(architecture,seed):
        return tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))])
    monkeypatch.setattr(protocol,'build_model',tiny)
    x=np.ones((1000,32,32,3),np.float32);y=np.repeat(np.arange(10,dtype=np.int32),100)
    actual=protocol.params_hash(tiny('',2026).get_weights());digest,_=partition(y,seed=2026)
    cfg=configurations(digest,actual,'CPU','replication_30r',seed=2026)[0]
    for altered,message in [(dict(cfg,initialModelHash='a'*64),'initialization'),
                            (dict(cfg,partitionHash='b'*64),'partition')]:
        with pytest.raises(ValueError,match=message):
            protocol.run_research_protocol(altered,x,y,x[:20],y[:20],worker_factory=lambda m:make_worker(m,altered))


@pytest.mark.parametrize('module',['scripts.run_cart_replication','scripts.report_cart_replication'])
def test_replication_modules_import_without_heavy_libraries(module):
    import subprocess,sys
    subprocess.run([sys.executable,'-c',f'import {module};import sys;'
        'assert not ({"tensorflow","numpy","wandb","matplotlib"}&sys.modules.keys())'],check=True)
