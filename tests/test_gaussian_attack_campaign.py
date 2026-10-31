"""Matched attack selection, seeded corruption, recovery and partial exports."""
from collections import Counter
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from basil_core import basil_sharp_thesis as study
from scripts import run_basil_sharp_thesis as runner

ROOT=Path(__file__).resolve().parents[1]
PRESET=ROOT/'configs/research/hal9k_gaussian_100r.json'


def options(tmp_path, *extra):
    return runner.parse_options(['--config',str(PRESET),'--output',str(tmp_path),*extra])


def test_sixty_matched_cases_and_terminal_attack_override(tmp_path):
    args=options(tmp_path)
    gaussian=study.selected(args)
    hidden=study.selected(options(tmp_path,'--attack-type','hidden'))
    assert len(gaussian)==60 and len({c['runId'] for c in gaussian})==60
    assert Counter(c['partition'] for c in gaussian)=={'iid':24,'dirichlet':36}
    assert not ({c['runId'] for c in gaussian}&{c['runId'] for c in hidden})
    assert args.gpu_count==2 and args.scheduler=='shared'
    assert args.ram_mib_per_worker==8192 and args.ram_headroom_mib==16384
    for gauss,previous in zip(gaussian,hidden):
        a=study.effective_config(gauss);b=study.effective_config(previous)
        assert a['attackType']=='gaussian' and b['attackType']=='hidden'
        assert a['gaussianAttackMode']=='pure_model_replacement' and a['gaussianAttackStd']==1.
        assert a['actualAttackers']==33 and a['attackStart']==20
        assert a['sigma']==(0.01 if gauss['comparisonCase']['environment']=='joint' else 0.)
        for key in ('attackType','gaussianAttackMode','gaussianAttackStd','runId','baseRunId','guardedRunId'):a.pop(key,None);b.pop(key,None)
        assert a==b  # The attack is the only numerical-setting difference.


def test_original_hidden_configuration_identities_are_unchanged(tmp_path):
    plan=json.loads((ROOT/'docs/results/hal9k_basil-sharp-thesis-100r-20261030/plan.json').read_text())
    args=runner.parse_options(['--output',str(tmp_path)])
    assert study.selected(args)==plan['configs']


def test_description_needs_no_gpu_data_or_output_directory(tmp_path,monkeypatch,capsys):
    def forbidden(*a,**kw):raise AssertionError('Describe tried to access hardware or training data.')
    monkeypatch.setattr(runner,'prepare',forbidden)
    assert runner.main(['--config',str(PRESET),'--output',str(tmp_path/'absent'),'--describe'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['configurations']==60 and report['activations']==600000
    assert report['curvesPerDataset']=={'IID':4,'nonIID':6}
    assert not (tmp_path/'absent').exists()


@pytest.mark.parametrize('settings',[
    {'attack_type':'not_an_attack'}, {'gpu_count':0}, {'execute':True}, {'unknown_setting':1},
    {'environments':[]}, {'gpu_count':True}])
def test_invalid_presets_fail_before_training(tmp_path,settings):
    preset=tmp_path/'invalid.json';preset.write_text(json.dumps(dict(schema=1,settings=settings)))
    with pytest.raises(SystemExit):runner.parse_options(['--config',str(preset),'--output',str(tmp_path)])


def test_resume_rejects_a_changed_attack_or_preset_before_accessing_the_plan(tmp_path):
    with pytest.raises(SystemExit,match='frozen plan'):
        runner.main(['--output',str(tmp_path),'--resume','--execute','--attack-type','gaussian'])


def test_seeded_gaussian_formula_ignores_global_rng_and_is_distinct_from_hidden():
    from basil_core.attacks import applyAttack
    from basil_core.research_protocol import keyed_rng
    weights=[np.linspace(-1,1,97,dtype=np.float32).reshape(1,97),np.zeros(10,np.float32)]
    rng=keyed_rng(2025,'sharp_attack',20,7)
    expected=[rng.normal(0,1.,w.shape).astype(np.float32) for w in weights]
    np.random.seed(1)
    first=applyAttack(weights,'gaussian',rng=keyed_rng(2025,'sharp_attack',20,7))
    np.random.seed(999)
    second=applyAttack(weights,'gaussian',rng=keyed_rng(2025,'sharp_attack',20,7))
    hidden=applyAttack(weights,'hidden',rng=keyed_rng(2025,'sharp_attack',20,7))
    for a,b,c,h in zip(first,second,expected,hidden):
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)
        assert a.dtype==np.float32 and not np.array_equal(a,h)


def test_gaussian_is_pure_replacement_not_blend_or_weight_scaled_noise():
    from basil_core.attacks import applyAttack
    shapes=[(100000,),(100000,)]
    zero=[np.zeros(shape,np.float32) for shape in shapes]
    shifted=[np.full(shape,200.,np.float32) for shape in shapes]
    first=applyAttack(zero,'gaussian',rng=np.random.default_rng(123))
    second=applyAttack(shifted,'gaussian',rng=np.random.default_rng(123))
    for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    flat=np.concatenate(first)
    assert abs(float(flat.mean()))<.01
    assert abs(float(flat.std())-1.)<.01
    assert not np.array_equal(first[0],first[1])  # separate parameter draws


def test_attack_strength_is_frozen_separately_from_channel_noise(tmp_path):
    standard=study.selected(options(tmp_path))[0]
    strong=study.selected(options(tmp_path,'--gaussian-std','2'))[0]
    assert standard['runId']!=strong['runId']
    a=study.effective_config(standard);b=study.effective_config(strong)
    assert a['sigma']==b['sigma']==0. and a['gaussianAttackStd']==1. and b['gaussianAttackStd']==2.
    from basil_core.sharp_study import configuration
    for value in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='standard deviation'):
            configuration('mnist','iid',2025,'attack_defense',0.,attack_type='gaussian',gaussian_std=value)


def test_hidden_transformation_still_matches_the_archived_producer():
    import importlib.util
    from basil_core.attacks import applyAttack
    from basil_core.research_protocol import keyed_rng
    path=ROOT/'docs/results/hal9k_basil-sharp-thesis-100r-20261030/source_snapshots/basil_core/attacks.py'
    spec=importlib.util.spec_from_file_location('archived_attacks',path)
    archived=importlib.util.module_from_spec(spec);spec.loader.exec_module(archived)
    weights=[np.linspace(-.3,.2,113,dtype=np.float32),np.zeros(10,np.float32)]
    actual=applyAttack(weights,'hidden',rng=keyed_rng(2025,'sharp_attack',20,7))
    expected=archived.applyAttack(weights,'hidden',rng=keyed_rng(2025,'sharp_attack',20,7))
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('module',['sharp_protocol','sharp_guarded_protocol','sharp_cart_v2_protocol'])
def test_every_ring_dispatches_gaussian_and_resumes_identically(tmp_path,module):
    import importlib
    from basil_core.sharp_study import configuration
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration as cart_configuration
    from basil_core.sharp_protocol import attacked_outbound,save_state,load_state
    from tests.test_sharp_protocol import FakeWorker
    cfg=configuration('mnist','iid',2025,'attack_noise_defense_rla',.01,
        nodes=4,rounds=3,attack_start=1,attack_type='gaussian')
    if module!='sharp_protocol':cfg=guarded_configuration(cfg)
    if module=='sharp_cart_v2_protocol':cfg=cart_configuration(cfg)
    protocol=importlib.import_module('basil_core.'+module)
    worker=FakeWorker();x=np.zeros((40,1),np.float32);y=np.arange(40,dtype=np.int32)%10
    chunks=np.array_split(np.arange(40),4)
    whole=protocol.run(cfg,worker,x,y,chunks,x[:10],y[:10])
    partial=protocol.run(cfg,worker,x,y,chunks,x[:10],y[:10],max_activations=7)
    path=tmp_path/'checkpoint.npz';contract=dict(config=cfg)
    save_state(path,partial,contract);restored=load_state(path,contract,worker.dimension)
    np.random.seed(7654)
    resumed=protocol.run(cfg,worker,x,y,chunks,x[:10],y[:10],state=restored)
    assert whole['telemetry']==resumed['telemetry'] and whole['roundMetrics']==resumed['roundMetrics']
    np.testing.assert_array_equal(whole['nodeParams'],resumed['nodeParams'])
    np.testing.assert_array_equal(whole['outbound'],resumed['outbound'])
    assert not any(r['attackActive'] for r in whole['telemetry'][:4])
    active=[r for r in whole['telemetry'][-4:] if r['attackActive']]
    assert len(active)==1
    node=active[0]['node']
    np.testing.assert_array_equal(whole['outbound'][node],
        attacked_outbound(cfg,worker,whole['nodeParams'][node],2,node))


def test_shared_scheduler_starts_after_verification_lock_and_leases_release(tmp_path,monkeypatch):
    from basil_core import execution_profile,live_gpu_scheduler
    from basil_core.cart_campaign_io import write_json
    args=options(tmp_path);configs=study.selected(args);held=[];closed=[]
    profile=execution_profile.ExecutionProfile(name='hpc',gpu=0,gpu_uuid='GPU-00000000',
        memory_mib=3072,headroom_mib=8192,ram_mib=8192,output_root=str(tmp_path))
    plan=dict(configs=configs,profiles=[],scheduler='shared',maximumWorkersPerGpu=24,
        resourceBudget=dict(ram_headroom_mib=16384),source={})
    write_json(tmp_path/'plan.json',plan)
    write_json(tmp_path/'revision_gpu_0.json',dict(status='PASSED',planSha256=runner.adaptive.sha(tmp_path/'plan.json')))
    @contextmanager
    def lock(*a,**kw):
        held.append(True)
        try:yield
        finally:held.pop()
    class Lease:
        def close(self):closed.append(True)
    monkeypatch.setattr(runner.adaptive,'acquire_supervisor',lock)
    monkeypatch.setattr(runner.adaptive,'ensure_source_snapshots',lambda *a:None)
    monkeypatch.setattr(runner,'placement',lambda p:({0:profile},[profile]*60))
    monkeypatch.setattr(execution_profile,'gpu_inventory',lambda:[dict(index=0,uuid=profile.gpu_uuid,**{'memory.free':97000})])
    monkeypatch.setattr(runner.adaptive.scaled,'available_ram_mib',lambda:90000)
    monkeypatch.setattr(runner.adaptive.scaled,'acquire_leases',lambda p:{0:Lease()})
    monkeypatch.setattr(runner.adaptive,'validate_plan',lambda r:(plan,{0:profile},[profile]*60))
    def start(actual):
        assert not held and closed==[True]
        assert actual.output==tmp_path and actual.maximum==24
        return 0
    monkeypatch.setattr(live_gpu_scheduler,'execute',start)
    assert runner.execute(args,plan)==0


def test_gaussian_partial_matrix_exports_all_expected_curves_without_clean_runs(tmp_path):
    from reporting.basil_sharp_thesis import generate,collect,sha
    args=options(tmp_path,'--datasets','mnist','--seeds','2025','--nodes','4','--rounds','2','--local-steps','1')
    configs=study.selected(args);root=tmp_path/'runs';root.mkdir()
    (root/'source_snapshots').mkdir();(root/'source_snapshots/frozen.py').write_text('# test source\n')
    plan=dict(revision=study.REVISION,configs=configs,source={'frozen.py':sha(root/'source_snapshots/frozen.py')},
        attackType='gaussian',environments=['attack_only','joint'],data={'mnist':{}},
        partitions={v['runId']:{} for v in configs})
    (root/'plan.json').write_text(json.dumps(plan))
    for v in configs:
        path=root/v['runId'];path.mkdir()
        (path/'run.json').write_text(json.dumps(dict(status='COMPLETED',config=v,source=plan['source'],
            methodRevision=study.REVISION,completedActivations=8,optimizerSteps=8,data={},partition={})))
        np.savez(path/'metrics.npz',roundNodeAccuracy=np.full((2,4),.3))
    result=generate(root,tmp_path/'figures')
    assert result['configurations']==10
    assert not list((tmp_path/'figures').glob('*.pdf'))
    for name in result['svgFiles']:
        svg=(tmp_path/'figures'/name).read_text()
        assert 'Gaussian attack' in svg and 'FedSharp' in svg and '\u2014' not in svg
        assert ': clean' not in svg
    # A campaign started directly with the shared scheduler has no legacy
    # adaptive_execution receipt. It still exports all measured runs correctly.
    (root/'campaign.json').write_text(json.dumps(dict(status='COMPLETED',total=10,completed=10)))
    for v in configs:(root/v['runId']/'mechanisms.json').write_text('{}')
    from scripts.export_basil_sharp_thesis import export
    assert export(root,tmp_path/'export')==10
    exported=json.loads((tmp_path/'export/SHA256.json').read_text())
    assert 'adaptive_execution.json' not in exported
    assert all(sha(tmp_path/'export'/name)==value for name,value in exported.items())
    missing=configs[-1];(root/missing['runId']/'run.json').unlink()
    with pytest.raises(FileNotFoundError):collect(root)


@pytest.mark.parametrize('dataset,family,addon',[
    ('mnist','basil',None),('mnist','sharp',None),('mnist','basil','acds'),
    ('mnist','sharp','cart'),('cifar10','basil',None),('cifar10','sharp',None)])
def test_real_workers_gaussian_checkpoint_and_gpu_gate_formula_on_cpu(tmp_path,dataset,family,addon):
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()
    from basil_core.sharp_protocol import save_state,load_state,attacked_outbound
    from scripts.verify_basil_sharp_thesis_gpu import verify_attack_transform
    import basil_core.sharp_cart_v2_protocol as protocol
    args=options(tmp_path,'--datasets',dataset,'--partitions','nonIID','--seeds','2025',
        '--nodes','4','--rounds','2','--local-steps','1','--attack-start','1')
    view=next(v for v in study.selected(args) if v['comparisonCase']['family']==family
        and v['comparisonCase']['addon']==addon and v['comparisonCase']['environment']=='joint')
    shape=(28,28,1) if dataset=='mnist' else (32,32,3)
    x=np.random.default_rng(918).uniform(size=(80,*shape)).astype(np.float32)
    y=np.tile(np.arange(10,dtype=np.int32),8);chunks=np.array_split(np.arange(80),4)
    worker=study.configure_worker(view,'CPU')
    assert verify_attack_transform(study.effective_config(view),worker)['seededOutboundAttackVerified']
    with study.intervention(view,worker) as cfg:whole=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20])
    worker=study.configure_worker(view,'CPU')
    with study.intervention(view,worker) as cfg:partial=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20],max_activations=5)
    path=tmp_path/'checkpoint.npz';contract=dict(config=cfg)
    save_state(path,partial,contract);worker=study.configure_worker(view,'CPU')
    state=load_state(path,contract,worker.dimension)
    with study.intervention(view,worker) as cfg:resumed=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20],state=state)
    np.testing.assert_array_equal(whole['nodeParams'],resumed['nodeParams'])
    np.testing.assert_array_equal(whole['outbound'],resumed['outbound'])
    assert whole['roundMetrics']==resumed['roundMetrics']
    assert sum(r['training']['optimizerSteps'] for r in resumed['telemetry'])==8
    for r in resumed['telemetry'][-4:]:
        params=resumed['nodeParams'][r['node']]
        expected=attacked_outbound(cfg,worker,params,1,r['node']) if r['attackActive'] else params
        np.testing.assert_array_equal(resumed['outbound'][r['node']],expected)


def test_cifar_full_extension_warm_gate_rejects_pure_gaussian_peers(tmp_path):
    from unittest.mock import patch
    from basil_core import cart_contribution_study as old_study
    from scripts.verify_cart_contribution_gpu import warm
    args=options(tmp_path,'--datasets','cifar10','--partitions','nonIID','--seeds','2025',
        '--nodes','4','--rounds','2','--local-steps','1','--attack-start','1')
    view=next(v for v in study.selected(args) if v['comparisonCase']['addon']=='cart')
    with study.installed(),patch.object(old_study,'variant',lambda v:'cart_full'), \
            patch.object(old_study,'configure_worker',study.configure_worker), \
            patch.object(old_study,'intervention',study.intervention):
        receipt=warm(view,'CPU',gaussian_peers=True)
    assert receipt['gaussianReplacementPeersVerified'] and receipt['recognitionContext']
    assert receipt['composedEmpiricalBoundVerified'] and receipt['optimizerSteps']==1


def test_early_gaussian_cold_start_overflow_is_detected_not_reported_as_accuracy(tmp_path):
    """Keep a real adverse result: do not soften N(0,1) to make a test pass.

    This deliberately attacks after one synthetic round, before receiver class
    recognition, rather than the research campaign's twenty-round warm-up.
    """
    import tensorflow as tf
    import basil_core.sharp_cart_v2_protocol as protocol
    args=options(tmp_path,'--datasets','cifar10','--partitions','nonIID','--seeds','2025',
        '--nodes','4','--rounds','2','--local-steps','1','--attack-start','1')
    view=next(v for v in study.selected(args) if v['comparisonCase']['addon']=='cart')
    worker=study.configure_worker(view,'CPU')
    x=np.random.default_rng(918).uniform(size=(80,32,32,3)).astype(np.float32)
    y=np.tile(np.arange(10,dtype=np.int32),8);chunks=np.array_split(np.arange(80),4)
    with study.intervention(view,worker) as cfg:
        with pytest.raises(tf.errors.InvalidArgumentError,match='Nonfinite prior-adjusted objective'):
            protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20])
