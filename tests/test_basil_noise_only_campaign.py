"""Noise-only controls keep BASIL selection while disabling every outbound attack."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from basil_core import basil_sharp_thesis as study
from scripts import run_basil_sharp_thesis as runner

ROOT=Path(__file__).resolve().parents[1]
PRESET=ROOT/'configs/research/hal9k_mnist_basil_noise_only_100r.json'


def options(root,*extra):
    return runner.parse_options(['--config',str(PRESET),'--output',str(root),*extra])


def test_six_noise_only_controls_match_existing_basil_numerical_settings(tmp_path):
    views=study.selected(options(tmp_path))
    assert len(views)==len({v['runId'] for v in views})==6
    assert Counter(v['partition'] for v in views)=={'iid':3,'dirichlet':3}
    archived=json.loads((ROOT/'docs/results/hal9k_basil-sharp-thesis-100r-20261030/plan.json').read_text())
    for view in views:
        cfg=study.effective_config(view)
        reference=next(v for v in archived['configs'] if v['dataset']=='mnist'
            and v['partition']==cfg['partition'] and v['seed']==cfg['seed']
            and v['comparisonCase']['arm']=='basil_joint')
        previous=study.effective_config(reference)
        assert cfg['actualAttackers']==0 and not cfg['switches']['attack']
        assert cfg['switches']['noise'] and cfg['sigma']==.01
        assert cfg['window']==34 and cfg['proximalMu']==0
        assert not cfg['switches']['noise_mitigation'] and not cfg['cart']
        assert view['comparisonCase']['addon'] is None
        assert view['comparisonCase']['shareFraction']==0
        for key in ('actualAttackers','switches'):cfg.pop(key);previous.pop(key)
        assert cfg==previous


def test_describe_needs_no_data_gpu_or_output(tmp_path,monkeypatch,capsys):
    def forbidden(*a,**kw):raise AssertionError('Description accessed training resources.')
    monkeypatch.setattr(runner,'prepare',forbidden)
    target=tmp_path/'absent'
    assert runner.main(['--config',str(PRESET),'--output',str(target),'--describe'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['configurations']==6 and report['activations']==60000
    assert report['families']==['basil'] and report['effectiveAttackers']==[0]
    assert not report['attacksEnabled'] and report['attackStartsAtDisplayedRound'] is None
    assert report['attackType'] is None
    assert report['curvesPerDataset']=={'IID':1,'nonIID':1}
    assert not report['trainingStarted'] and not target.exists()


def test_preparation_freezes_six_controls_on_two_gpus(tmp_path,monkeypatch):
    from basil_core import execution_profile
    args=options(tmp_path/'runs')
    inventory=[dict(index=g,uuid=f'GPU-{g:08x}',**{'memory.free':97887}) for g in range(2)]
    monkeypatch.setattr(execution_profile,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(runner.controller,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(study,'data_contract',lambda views:({}, {v['runId']:{} for v in views}))
    monkeypatch.setattr(runner.adaptive.scaled,'available_ram_mib',lambda:90*1024)
    plan=runner.prepare(args)
    assert plan['families']==['basil'] and plan['environments']==['noise_only']
    assert Counter(p['gpu'] for p in plan['profiles'])=={0:3,1:3}
    assert runner.readiness(args.output,plan)['maximumConcurrentNow']==6
    args.resume=True
    assert runner.prepare(args)==plan


@pytest.mark.parametrize('split',['IID','nonIID'])
def test_real_cpu_worker_never_attacks_past_onset_and_resumes(tmp_path,monkeypatch,split):
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    import basil_core.sharp_cart_v2_protocol as protocol
    from basil_core.sharp_protocol import save_state,load_state,link_vector
    from tests.test_cart_comparison_runtime import data
    x,y,chunks=data()
    view=study.selected(options(tmp_path,'--partitions',split,'--seeds','2025',
        '--nodes','4','--rounds','3','--local-steps','1','--attack-start','1'))[0]
    def forbidden(*a,**kw):raise AssertionError('Noise-only run applied a Byzantine attack.')
    monkeypatch.setattr(protocol,'attacked_outbound',forbidden)
    worker=study.configure_worker(view,'CPU')
    with study.intervention(view,worker) as cfg:
        whole=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20])
    worker=study.configure_worker(view,'CPU')
    with study.intervention(view,worker) as cfg:
        partial=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20],max_activations=5)
    contract=dict(config=cfg,comparisonCase=view['comparisonCase'])
    path=tmp_path/'checkpoint.npz';save_state(path,partial,contract)
    state=load_state(path,contract,worker.dimension)
    with study.intervention(view,worker):
        resumed=protocol.run(cfg,worker,x,y,chunks,x[:20],y[:20],state=state)
    assert whole['cursor']==12
    assert not any(r['attackActive'] or r['configuredByzantine'] for r in whole['telemetry'])
    assert all(r['selection']['selectedCount']==1 for r in whole['telemetry'])
    assert whole['roundMetrics']==resumed['roundMetrics']
    np.testing.assert_array_equal(whole['nodeParams'],resumed['nodeParams'])
    np.testing.assert_array_equal(whole['outbound'],whole['nodeParams'])
    received=link_vector(whole['outbound'][0],2025,2,0,1,cfg['sigma'])
    assert not np.array_equal(received,whole['outbound'][0])
    from basil_core.cart_display_labels import comparison_label
    assert comparison_label(view,include_environment=True)=='BASIL / noise only'


def test_noise_only_report_and_export_use_single_family_without_attack_labels(tmp_path,monkeypatch):
    from reporting.basil_sharp_thesis import generate,collect,sha
    from matplotlib.axes import Axes
    def forbidden(*a,**kw):raise AssertionError('Noise-only plot drew an attack-onset reference.')
    monkeypatch.setattr(Axes,'axvline',forbidden)
    root=tmp_path/'runs';root.mkdir()
    configs=study.selected(options(root,'--nodes','4','--rounds','2','--local-steps','1','--attack-start','0'))
    (root/'source_snapshots').mkdir();(root/'source_snapshots/frozen.py').write_text('# test source\n')
    plan=dict(revision=study.REVISION,configs=configs,families=['basil'],environments=['noise_only'],
        attackType='hidden',source={'frozen.py':sha(root/'source_snapshots/frozen.py')},
        data={'mnist':{}},partitions={v['runId']:{} for v in configs})
    (root/'plan.json').write_text(json.dumps(plan))
    (root/'campaign.json').write_text(json.dumps(dict(status='COMPLETED',total=6,completed=6)))
    for view in configs:
        path=root/view['runId'];path.mkdir()
        (path/'run.json').write_text(json.dumps(dict(status='COMPLETED',config=view,source=plan['source'],
            methodRevision=study.REVISION,completedActivations=8,optimizerSteps=8,data={},partition={})))
        (path/'mechanisms.json').write_text('{}')
        np.savez(path/'metrics.npz',roundNodeAccuracy=np.full((2,4),.3))
    result=generate(root,tmp_path/'figures')
    assert result['configurations']==6
    for name in result['svgFiles']:
        svg=(tmp_path/'figures'/name).read_text()
        assert 'BASIL: noise only' in svg and 'No attack' in svg
        assert 'Hidden attack' not in svg and 'FedSharp' not in svg
    from scripts.export_basil_sharp_thesis import export
    assert export(root,tmp_path/'export')==6
    assert len(collect(tmp_path/'export')[1])==6
