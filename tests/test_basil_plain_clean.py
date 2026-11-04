from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from basil_core import basil_plain_clean as plain,basil_sharp_thesis as base
from scripts import run_basil_plain_clean as launcher
from tests.test_basil_sharp_thesis import options


def opts(**kwargs):return options(families=['basil'],environments=['clean'],rounds=500,**kwargs)


def test_matrix_and_frozen_effective_plain_policy():
    from basil_core import fedsharp_clean_controls as previous
    before=base.source();old=previous.source();views=plain.selected(opts())
    assert len(views)==len({v['runId'] for v in views})==12
    for v in views:
        cfg=plain.effective_config(v)
        assert cfg['proximalMu']==v['proximalMu']==cfg['sigma']==cfg['actualAttackers']==cfg['byzantineBudget']==0
        assert cfg['window']==v['window']==1 and not cfg['cart'] and not any(cfg['switches'].values())
        assert (cfg['rounds'],cfg['nodes'],cfg['localSteps'])==(500,100,5)
        assert cfg['receiverRule']=='immediate_predecessor'
        assert v['plainCleanControl']['start']=='initialization'
        assert v['comparisonCase']['family']=='basil' and v['comparisonCase']['addon'] is None
    with plain.installed():
        assert base.selected(opts())==views and base.source()==plain.source()
        assert len(plain.source())==len(before)+8
        for v in views:assert base.effective_config(v)==plain.effective_config(v)
    assert base.source()==before and previous.source()==old


def test_all_twelve_fedsharp_pairs_have_same_initialization_and_partition(cpu):
    from basil_core import fedsharp_plain_clean as sharp
    from basil_core.sharp_data import prepare_partition
    from basil_core.sharp_protocol import vector_hash
    labels=np.tile(np.arange(10,dtype=np.int32),1000)
    for view in plain.selected(opts()):
        paired=sharp.selected(plain.options_for(view,'sharp'))[0]
        cfg=plain.effective_config(view);paired_cfg=sharp.effective_config(paired)
        for field in ('dataset','partition','seed','nodes','rounds','localSteps','batchSize','model',
                      'learningRate','learningRateSchedule','momentum','weightDecay','gradientClipping',
                      'proximalMu','sigma','actualAttackers','window','byzantineBudget','switches'):
            assert cfg[field]==paired_cfg[field],field
        a,held,audit=prepare_partition(labels,cfg);b,other,other_audit=prepare_partition(labels,paired_cfg)
        assert audit==other_audit
        np.testing.assert_array_equal(held,other)
        for first,second in zip(a,b):np.testing.assert_array_equal(first,second)
        assert vector_hash(plain.configure_worker(view,'CPU').export())==vector_hash(sharp.configure_worker(paired,'CPU').export())


@pytest.mark.parametrize('field,value',[('proximalMu',1.),('window',66),('byzantineBudget',33)])
def test_policy_tampering_refused(field,value):
    view=plain.selected(opts())[0];view[field]=value
    with pytest.raises(ValueError):plain.effective_config(view)


def test_direct_predecessor_and_bootstrap_never_score_average_or_screen():
    from basil_core.sharp_protocol import initial_state
    class ForbiddenScoring:
        def __getattr__(self,name):pytest.fail('No model scoring allowed: '+name)
    view=plain.selected(opts(nodes=4,datasets=['mnist'],seeds=[2025],partitions=['iid']))[0]
    cfg=plain.effective_config(view);initial=np.array([1.,2.],np.float32);state=initial_state(cfg,initial)
    got,record=plain.select_input(cfg,state,ForbiddenScoring(),None,None,0,0,None)
    np.testing.assert_array_equal(got,initial);assert record['bootstrap']
    got[0]=99;np.testing.assert_array_equal(state['nodeParams'][0],initial)
    for cursor in range(1,9):
        r,node=divmod(cursor,4);sender=(node-1)%4
        state['cursor']=cursor;state['senderRounds'][sender]=r if node else r-1
        state['outbound'][:]=np.arange(8).reshape(4,2)
        expected=state['outbound'][sender].copy()
        got,record=plain.select_input(cfg,state,ForbiddenScoring(),None,None,node,r,None)
        np.testing.assert_array_equal(got,expected)
        assert record['selectedSenders']==[sender] and record['selectedCount']==1 and not record['ranking']
        assert not record['screenApplied'] and not record['screenGuardFallback']
        got[0]=999;np.testing.assert_array_equal(state['outbound'][sender],expected)
    state['senderRounds'][sender]=-1
    with pytest.raises(ValueError,match='stale'):plain.select_input(cfg,state,None,None,None,node,r,None)


def test_description_and_four_gpu_preset_never_load_data(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(launcher.driver,'prepare',lambda *a:pytest.fail('must not prepare'))
    target=tmp_path/'configs.json'
    assert launcher.main(['--output',str(tmp_path/'run'),'--describe','--write-configs',str(target)])==0
    row=json.loads(capsys.readouterr().out)
    assert row['configurations']==12 and row['requestedGpuCount']==4
    assert row['proximalMu']=={'IID':0,'nonIID':0} and row['activations']==600000 and row['optimizerSteps']==3000000
    assert not row['screening'] and not row['ranking'] and not row['averaging'] and not row['noiseRegularization']
    assert not (tmp_path/'run').exists() and len(json.loads(target.read_text())['configs'])==12


def test_execution_refuses_other_hosts(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher.socket,'gethostname',lambda:'hal9k')
    with pytest.raises(ValueError,match='restricted to themachine'):
        launcher.main(['--output',str(tmp_path/'run'),'--execute'])
    assert not (tmp_path/'run').exists()


@pytest.fixture
def cpu():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()


def test_both_models_both_splits_match_independent_five_step_plain_sgd(cpu,tmp_path):
    from scripts.verify_basil_plain_clean_gpu import verify
    views=plain.selected(opts(seeds=[2025]))
    (tmp_path/'plan.json').write_text(json.dumps(dict(configs=views,source=plain.source())))
    receipt=verify(tmp_path,0,None,'CPU')
    assert receipt['status']=='PASSED' and len(receipt['checks'])==4
    assert all(c['mu']==0 and c['exactPredecessorTransfer'] and c['exactFiveStepOrdinarySGD'] and c['exactFedSharpUpdate'] for c in receipt['checks'])


@pytest.mark.parametrize('dataset',['mnist','cifar10'])
@pytest.mark.parametrize('split',['iid','dirichlet'])
def test_full_trajectory_chain_and_mid_round_resume(cpu,tmp_path,split,dataset):
    from tests.test_cart_comparison_runtime import data
    from basil_core.sharp_protocol import vector_hash,save_state,load_state
    import basil_core.sharp_cart_v2_protocol as protocol
    args=opts(datasets=[dataset],partitions=[split],seeds=[2025],nodes=4,local_steps=5);args.rounds=3
    view=plain.selected(args)[0];cfg=plain.effective_config(view);x,y,chunks=data()
    if dataset=='cifar10':x=np.random.default_rng(918).random((len(y),32,32,3),dtype=np.float32)
    worker=plain.configure_worker(view,'CPU');initial=vector_hash(worker.export())
    with plain.intervention(view,worker):expected=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20])
    from basil_core import fedsharp_plain_clean as sharp
    paired=sharp.selected(plain.options_for(view,'sharp'))[0];paired_worker=sharp.configure_worker(paired,'CPU')
    with sharp.intervention(paired,paired_worker) as paired_cfg:
        counterpart=protocol.run(paired_cfg,paired_worker,x,y,chunks,x[:20],y[:20])
    np.testing.assert_array_equal(counterpart['nodeParams'],expected['nodeParams'])
    np.testing.assert_array_equal(counterpart['outbound'],expected['outbound'])
    assert counterpart['telemetry']==expected['telemetry'] and counterpart['roundMetrics']==expected['roundMetrics']
    previous=initial
    for i,record in enumerate(expected['telemetry']):
        assert record['inputHash']==previous
        previous=record['trainedHash'];assert record['outboundHash']==previous
        assert record['selection']['selectedSenders']==([] if i==0 else [(i-1)%4])
        assert not record['selection']['screenApplied'] and not record['selection']['ranking']
        assert record['training']['optimizerSteps']==5 and record['training']['proximalMu']==0
        assert record['training']['rlaCoefficient']==0 and not record['attackActive'] and record['cart'] is None
    worker=plain.configure_worker(view,'CPU')
    with plain.intervention(view,worker):partial=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20],max_activations=5)
    contract=dict(config=cfg,source=plain.source());path=tmp_path/'checkpoint.npz';save_state(path,partial,contract)
    state=load_state(path,contract,worker.dimension);worker=plain.configure_worker(view,'CPU')
    with plain.intervention(view,worker):actual=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20],state=state)
    np.testing.assert_array_equal(actual['nodeParams'],expected['nodeParams'])
    np.testing.assert_array_equal(actual['outbound'],expected['outbound'])
    assert actual['telemetry']==expected['telemetry'] and actual['roundMetrics']==expected['roundMetrics']


def test_four_gpu_gate_scheduler_handoff(tmp_path,monkeypatch):
    from basil_core import live_gpu_scheduler as live,execution_profile
    from basil_core.execution_profile import ExecutionProfile
    from basil_core.cart_campaign_io import read_json,write_json
    root=tmp_path/'run';root.mkdir()
    views=plain.selected(opts())
    hardware={g:ExecutionProfile(name='hpc',gpu=g,gpu_uuid=f'GPU-{g:08x}',memory_mib=3072,
        headroom_mib=8192,ram_mib=8192,output_root=str(root)) for g in range(4)}
    inventory=[dict(index=g,uuid=p.gpu_uuid,**{'memory.free':90000}) for g,p in hardware.items()]
    monkeypatch.setattr(execution_profile,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(launcher.adaptive.scaled,'available_ram_mib',lambda:1000000)
    monkeypatch.setattr(launcher.adaptive.scaled,'acquire_leases',lambda profiles:{g:(tmp_path/f'lease{g}').open('a') for g in hardware})
    checked=[]
    def gate(command,**kwargs):
        g=int(command[command.index('--gpu')+1]);checked.append(g);assert kwargs['pass_fds']
        plan=read_json(root/'plan.json')
        write_json(root/f'revision_gpu_{g}.json',dict(status='PASSED',source=plan['source'],
            planSha256=launcher.adaptive.sha(root/'plan.json'),runtime=dict(host=launcher.socket.gethostname(),profile=asdict(hardware[g]))))
    monkeypatch.setattr(launcher.subprocess,'run',gate)
    def scheduling(args):
        assert live.WORKER==Path(launcher.__file__).resolve() and live.WORKER.name in live.OWNERS
        assert live.controller(root) is None
        actual,devices,assigned,digest=live.verified(root)
        assert len(actual['configs'])==12 and len(devices)==4 and len(assigned)==12
        return 0
    monkeypatch.setattr(live,'execute',scheduling)
    args=SimpleNamespace(output=root,wait_for_resources=False,poll_seconds=3,ntfy_topic=None)
    with plain.installed():
        plan=dict(revision=plain.REVISION,source=plain.source(),configs=views,
            profiles=[asdict(hardware[i%4]) for i in range(12)],host=launcher.socket.gethostname(),maximumWorkersPerGpu=24,
            resourceBudget=dict(context_overhead_mib=1024,ram_headroom_mib=16384,ram_mib_per_worker=8192))
        write_json(root/'plan.json',plan)
        assert launcher.execute(args,plan)==0
    assert checked==[0,1,2,3]


def test_real_worker_checkpoint_metrics_and_fresh_inference(cpu,tmp_path,monkeypatch):
    from tests.test_cart_comparison_runtime import data
    from basil_core import cart_comparison_study as comparison,sharp_data
    from basil_core.sharp_protocol import load_state
    from scripts.cart_comparison_worker import run
    from scripts.verify_cart_comparison_results import verify
    args=opts(datasets=['mnist'],partitions=['iid'],seeds=[2025],nodes=4,local_steps=5);args.rounds=2
    view=plain.selected(args)[0];x,y,chunks=data();held=np.arange(20);audit=dict(fixture='synthetic')
    source=plain.source();identity=dict(imagesSha256=sharp_data.digest(x),labelsSha256=sharp_data.digest(y),
                                      samples=len(y),officialTestLoaded=False)
    plan=dict(revision=plain.REVISION,configs=[view],source=source,data=dict(mnist=identity),
              partitions={view['runId']:audit})
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    for name in source:
        target=tmp_path/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((plain.ROOT/name).read_bytes())
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with plain.installed():
            monkeypatch.setattr(sharp_data,'training_data',lambda dataset:(x,y))
            monkeypatch.setattr(comparison,'partitions_for',lambda v,labels:(chunks,held,audit))
            run(tmp_path,0,cpu=True)
            row=json.loads((tmp_path/view['runId']/'run.json').read_text())
            assert row['status']=='COMPLETED' and row['optimizerSteps']==40 and row['completedActivations']==8
            worker=plain.configure_worker(view,'CPU')
            state=load_state(tmp_path/view['runId']/'checkpoint.npz',row['contract'],worker.dimension)
            assert state['cursor']==8 and not state['references']
            receipt=verify(tmp_path)
            assert receipt['status']=='ALL_COMPARISON_MODELS_VERIFIED' and len(receipt['verified'])==1
            assert receipt['verified'][0]['maximumValidationAccuracyDifference']==0
    finally:
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
