import copy
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def revision(tmp_path,monkeypatch):
    from basil_core.cart_campaign_io import read_json
    from basil_core.sharp_study import configuration
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration as cart_configuration
    import reporting.cart_revision as module
    # Reporting regression fixtures are synthetic; they must not require a
    # complete superseded research archive in a fresh source checkout.
    reference=tmp_path/'reference';reference.mkdir()
    original=dict(configs=[],source={'fixture':'synthetic legacy reporting'},data={d:{'fixture':d}
        for d in ('mnist','cifar10')},partitions={})
    conditions=('clean','attack_noise','attack_noise_defense','attack_noise_rla',
        'attack_noise_defense_rla','attack_noise_defense_rla_cart')
    for dataset in ('mnist','cifar10'):
        for partition in ('iid','dirichlet'):
            for seed in (2025,2026,2027):
                for condition in conditions:
                    cfg=cart_configuration(guarded_configuration(configuration(dataset,partition,seed,
                        condition,0. if condition=='clean' else .01)))
                    original['configs'].append(cfg)
                    audit={'fixture':f'{dataset}-{partition}-{seed}'}
                    original['partitions'][cfg['runId']]=audit
                    folder=reference/cfg['runId'];folder.mkdir()
                    row=dict(status='COMPLETED',config=cfg,source=original['source'],
                        data=original['data'][dataset],partition=audit,completedActivations=10000,
                        initialHash=f'fixture-{dataset}-{seed}',
                        runtime=dict(tensorflow='fixture',numpy='fixture',device='/GPU:0',tf32=False))
                    (folder/'run.json').write_text(json.dumps(row))
                    np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((100,100),.30))
    (reference/'plan.json').write_text(json.dumps(original))
    (reference/'SHA256.json').write_text(json.dumps({'fixture':'not experimental evidence'}))
    monkeypatch.setattr('basil_core.sharp_cart_v2_protocol.check_source',lambda _:None)
    root=tmp_path/'revision';root.mkdir()
    configs=[c for c in original['configs'] if c['dataset']=='cifar10' and c['cart']]
    plan=dict(configs=configs,source={'fixture':'unit test only'},baseSource=original['source'],variant='capped',revision='fixture',
        data=original['data'],partitions={c['runId']:original['partitions'][c['runId']] for c in configs})
    (root/'plan.json').write_text(json.dumps(plan))
    for cfg in configs:
        folder=root/cfg['runId'];folder.mkdir();old=read_json(reference/cfg['runId']/'run.json')
        old.update(methodRevision='fixture',source=plan['source'])
        (folder/'run.json').write_text(json.dumps(old));np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((100,100),.33))
    monkeypatch.setattr('basil_core.cart_revision_study.check_source',lambda _:None)
    return module,root,reference,plan


def test_complete_cart_groups_replace_only_matching_series_and_preserve_reference(revision):
    module,root,reference,plan=revision
    before=(reference/'SHA256.json').read_bytes()
    _,groups,rows,hashes,paired=module.collect(root,reference)
    assert len(groups)==24 and len(rows)==72 and len(paired)==2
    for g in groups:
        if g['config']['dataset']=='cifar10' and g['config']['cart']:
            np.testing.assert_allclose(g['mean'],.33);assert g['revision']=='fixture'
    assert all(g['revision']!='fixture' for g in groups if not g['config']['cart'])
    assert (reference/'SHA256.json').read_bytes()==before


def test_two_new_seeds_never_mix_with_one_old_seed(revision):
    module,root,reference,plan=revision
    plan['configs']=plan['configs'][:-1];(root/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='all three seeds'):module.collect(root,reference)


def test_reused_control_requires_identical_initialization(revision):
    module,root,reference,plan=revision
    path=root/plan['configs'][0]['runId']/'run.json';row=json.loads(path.read_text())
    row['initialHash']='different';path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='Unpaired'):module.collect(root,reference)


def test_reference_numerical_runtime_must_match(revision):
    module,root,reference,plan=revision
    path=root/plan['configs'][0]['runId']/'run.json';row=json.loads(path.read_text())
    row['runtime']['tensorflow']='different';path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='numerical runtime'):module.collect(root,reference)


def test_invalid_metric_values_cannot_replace_the_latest_view(revision):
    module,root,reference,plan=revision;folder=root/plan['configs'][0]['runId']
    np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((100,100),np.nan))
    dest=root.parent/'latest';dest.mkdir();(dest/'keep.txt').write_text('preserve')
    with pytest.raises(ValueError,match='Invalid'):module.generate(root,dest,reference)
    assert (dest/'keep.txt').read_text()=='preserve'
