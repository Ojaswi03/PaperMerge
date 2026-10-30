from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest
from basil_core import basil_sharp_thesis as study


def options(**changes):
    return SimpleNamespace(**(dict(datasets=['mnist','cifar10'],partitions=['iid','dirichlet'],seeds=[2025,2026,2027],
        nodes=100,rounds=100,local_steps=5,alpha=.5,sigma=.01,attack_start=20)|changes))


def test_requested_matrix_has_no_duplicate_condition_and_true_baseline():
    views=study.selected(options())
    assert len(views)==len({v['runId'] for v in views})==84
    assert Counter(v['partition'] for v in views)==dict(iid=36,dirichlet=48)
    for view in views:
        c=study.effective_config(view);case=view['comparisonCase']
        assert c['nodes']==c['rounds']==100 and c['localSteps']==5
        assert c['actualAttackers']==(0 if case['environment']=='clean' else 33)
        assert c['sigma']==(0 if case['environment']!='joint' else .01)
        assert c['switches']['byzantine_mitigation']
        assert c['window']==(34 if case['family']=='basil' else 66)
        assert c['switches']['noise_mitigation']==(case['family']=='sharp')
        assert c['proximalMu']==(0 if case['family']=='basil' else 1)
        assert c['cart']==(case['addon']=='cart')
        if case['addon']:assert c['partition']=='dirichlet'


def test_small_and_full_case_configs_have_valid_independent_source_identities():
    for view in study.selected(options(nodes=4,rounds=2,local_steps=1,seeds=[2025],datasets=['mnist'])):
        assert study.effective_config(view)['rounds']==2


@pytest.fixture(scope='module',autouse=True)
def cpu():
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()


@pytest.mark.parametrize('split',['iid','dirichlet'])
def test_every_condition_resumes_full_model_identically(tmp_path,split):
    from tests.test_cart_comparison_runtime import data
    from basil_core.sharp_protocol import save_state,load_state
    import basil_core.sharp_cart_v2_protocol as protocol
    x,y,chunks=data()
    for view in study.selected(options(datasets=['mnist'],partitions=[split],seeds=[2025],nodes=4,rounds=2,local_steps=1,attack_start=1)):
        worker=study.configure_worker(view,'CPU')
        with study.intervention(view,worker) as cfg:expected=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20])
        worker=study.configure_worker(view,'CPU')
        with study.intervention(view,worker) as cfg:partial=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20],max_activations=3)
        contract=dict(config=cfg,comparisonCase=view['comparisonCase']);path=tmp_path/(view['runId']+'.npz')
        save_state(path,partial,contract);worker=study.configure_worker(view,'CPU');state=load_state(path,contract,worker.dimension)
        with study.intervention(view,worker) as cfg:actual=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20],state=state)
        np.testing.assert_array_equal(expected['nodeParams'],actual['nodeParams'])
        np.testing.assert_array_equal(expected['outbound'],actual['outbound'])
        assert expected['roundMetrics']==actual['roundMetrics']
        assert sum(r['training']['optimizerSteps'] for r in actual['telemetry'])==8


def test_basil_fast_selection_is_exactly_the_unmodified_single_model_rule():
    from tests.test_cart_comparison_runtime import data
    from basil_core.sharp_protocol import select_input,initial_state
    x,y,_=data()
    for view in study.selected(options(datasets=['mnist'],partitions=['iid'],seeds=[2025],nodes=4,rounds=2,local_steps=1)):
        if view['comparisonCase']['family']!='basil':continue
        cfg=study.effective_config(view);worker=study.configure_worker(view,'CPU');state=initial_state(cfg,worker.export())
        state['senderRounds'][:]=0
        state['outbound']+=np.random.default_rng(91).normal(0,.001,state['outbound'].shape).astype(np.float32)
        first,row=select_input(cfg,state,worker,x,y,0,1,np.arange(14))
        second,other=study.basil_select(cfg,state,worker,x,y,0,1,np.arange(14))
        np.testing.assert_array_equal(first,second)
        assert row['selectedSenders']==other['selectedSenders'] and other['selectedCount']==1


def test_basil_acds_expands_data_without_changing_holdout_or_becoming_sharp():
    from basil_core.sharp_data import prepare_partition
    views=study.selected(options(datasets=['mnist'],partitions=['dirichlet'],seeds=[2025],nodes=4))
    view=next(v for v in views if v['comparisonCase']['addon']=='acds');cfg=study.effective_config(view)
    labels=np.tile(np.arange(10,dtype=np.int32),200)
    original,held,_=prepare_partition(labels,cfg);expanded,other,audit=study.partitions_for(view,labels)
    np.testing.assert_array_equal(held,other)
    assert cfg['method']=='basil' and cfg['window']==2
    assert audit['acds']['sharedUniqueSamples']>0
    assert all(set(a)<=set(b) for a,b in zip(original,expanded))
    assert not np.intersect1d(np.concatenate(expanded),held).size


def test_full_concurrency_budget_on_four_hpc_gpus():
    assert 21*(3072+1024)+8192==94208<97887
    assert 84*16384+32768==1409024


def test_fresh_hpc_preparation_freezes_84_cases_across_four_devices(tmp_path,monkeypatch):
    from scripts import run_basil_sharp_thesis as runner
    from basil_core import execution_profile
    args=options(output=tmp_path/'hpc',resume=False,gpus=None,gpu_count=4,gpu_memory_mib=3072,
        ram_mib_per_worker=16384,ram_headroom_mib=32768)
    inventory=[dict(index=g,uuid=f'GPU-{g:08x}',**{'memory.free':97887}) for g in range(4)]
    monkeypatch.setattr(execution_profile,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(runner.adaptive.scaled,'available_ram_mib',lambda:1500000)
    monkeypatch.setattr(study,'data_contract',lambda views:({}, {v['runId']:{} for v in views}))
    plan=runner.prepare(args)
    assert Counter(p['gpu'] for p in plan['profiles'])=={g:21 for g in range(4)}
    assert len(plan['configs'])==84 and plan['revision']==study.REVISION
    readiness=runner.readiness(args.output,plan)
    assert readiness['gpuCapacityNow']=={g:21 for g in range(4)}
    assert readiness['hostRequiredForAllMiB']==1409024


def test_two_gpu_small_host_queues_cases_and_preserves_scientific_configs(tmp_path,monkeypatch):
    from scripts import run_basil_sharp_thesis as runner
    from basil_core import execution_profile
    from basil_core.cart_worker_admission import admission,Resident,placement
    inventory=[dict(index=g,uuid=f'GPU-{g:08x}',**{'memory.free':97887}) for g in range(2)]
    monkeypatch.setattr(execution_profile,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(runner.controller,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(runner.adaptive.scaled,'available_ram_mib',lambda:90*1024)
    monkeypatch.setattr(study,'data_contract',lambda views:({}, {v['runId']:{} for v in views}))
    args=options(output=tmp_path/'small',resume=False,gpus=None,gpu_count=2,gpu_memory_mib=3072,
        ram_mib_per_worker=8192,ram_headroom_mib=16384)
    plan=runner.prepare(args);report=runner.readiness(args.output,plan)
    assert report['maximumConcurrentNow']==report['hostWorkerCapacityNow']==9
    assert report['casesPerGpu']=={0:42,1:42}
    hardware,assigned=placement(plan);residents=[]
    for p in assigned:
        decision=admission(p,residents,inventory,[],90*1024,context_mib=1024,
            ram_headroom_mib=16384,maximum=24)
        if decision['admit']:residents.append(Resident(len(residents)+100,p.gpu,0))
    assert len(residents)==9
    # Recovery uses the saved RAM budget even though CLI defaults may differ.
    args.resume=True
    assert runner.prepare(args)==plan
    args.resume=False;args.output=tmp_path/'large';args.ram_mib_per_worker=16384;args.ram_headroom_mib=32768
    larger=runner.prepare(args)
    assert larger['configs']==plan['configs']
    assert runner.readiness(args.output,larger)['maximumConcurrentNow']==3


def test_complete_figure_matrix_uses_plain_names_and_preserves_archive(tmp_path):
    import json
    from reporting.basil_sharp_thesis import generate,sha
    root=tmp_path/'data';root.mkdir();(root/'source_snapshots').mkdir()
    (root/'source_snapshots/frozen.py').write_text('# frozen producer\n')
    views=study.selected(options(datasets=['mnist'],seeds=[2025],nodes=4,rounds=2,local_steps=1,attack_start=1))
    plan=dict(revision=study.REVISION,configs=views,source={'frozen.py':sha(root/'source_snapshots/frozen.py')},
        data={'mnist':{}},partitions={v['runId']:{} for v in views})
    (root/'plan.json').write_text(json.dumps(plan))
    for i,v in enumerate(views):
        folder=root/v['runId'];folder.mkdir()
        record=dict(status='COMPLETED',config=v,source=plan['source'],methodRevision=study.REVISION,
            completedActivations=8,optimizerSteps=8,data={},partition={})
        (folder/'run.json').write_text(json.dumps(record))
        np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((2,4),.3+i*.01))
    before={p:sha(p) for p in root.rglob('*') if p.is_file()}
    result=generate(root,tmp_path/'figures')
    assert result['configurations']==14
    assert sorted(result['svgFiles'])==['mnist_iid_accuracy.svg','mnist_non_iid_accuracy.svg']
    for name in result['svgFiles']:
        svg=(tmp_path/'figures'/name).read_text()
        assert 'BASIL' in svg and 'SHARP' in svg and 'Validation accuracy' in svg
        assert 'Dirichlet' not in svg and 'macro' not in svg.lower() and 'commit' not in svg.lower()
        assert 'Official TEST' not in svg
    assert 'ACDS' in (tmp_path/'figures/mnist_non_iid_accuracy.svg').read_text()
    assert 'ACDS' not in (tmp_path/'figures/mnist_iid_accuracy.svg').read_text()
    assert all(sha(p)==value for p,value in before.items())
    assert generate(root,tmp_path/'figures')==result
