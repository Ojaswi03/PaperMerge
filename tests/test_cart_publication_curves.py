import csv
import json
from pathlib import Path
import numpy as np
import pytest

from reporting import cart_publication_curves as plots
from scripts.plot_cart_publication import readiness, wait_for_completion


@pytest.fixture
def complete(tmp_path):
    root=tmp_path/'results';root.mkdir()
    source=root/'source_snapshots/frozen.py';source.parent.mkdir();source.write_text('# original producer\n')
    plan=dict(configs=[],source={'frozen.py':plots.sha(source)},revision='private-campaign-v2',
        data={d:{'split':'development'} for d in ('mnist','cifar10')},partitions={})
    for dataset in ('mnist','cifar10'):
        for part in ('iid','dirichlet'):
            for seed in (2025,2026,2027):
                for arm in ('acds','prior_fallback'):
                    run_id=f'{dataset}-{part}-{seed}-{arm}'
                    view=dict(runId=run_id,dataset=dataset,partition=part,seed=seed,nodes=4,rounds=4,
                        attackStart=2,dirichletAlpha=.5,sigma=.01,
                        comparisonCase=dict(arm=arm,environment='joint',shareFraction=.05 if arm=='acds' else 0.))
                    plan['configs'].append(view);plan['partitions'][run_id]={'holdout':'fixed'}
    for i,view in enumerate(plan['configs']):
        folder=root/view['runId'];folder.mkdir()
        row=dict(status='COMPLETED',config=view,source=plan['source'],methodRevision=plan['revision'],
            completedActivations=16,data=plan['data'][view['dataset']],partition=plan['partitions'][view['runId']])
        (folder/'run.json').write_text(json.dumps(row))
        values=np.arange(16,dtype=float).reshape(4,4)/100 + .3 + .01*(view['seed']-2025)
        np.savez(folder/'metrics.npz',roundNodeAccuracy=values)
    (root/'plan.json').write_text(json.dumps(plan))
    return root,plan


def test_curve_mean_averages_clients_then_seeds_and_uses_sample_sd(complete):
    root,_=complete
    _,rows,_=plots.collect(root)
    result=plots.aggregate(rows)
    assert len(rows)==24 and len(result)==8
    for values in result.values():
        np.testing.assert_allclose(values['mean'],[.325,.365,.405,.445])
        np.testing.assert_allclose(values['sd'],[.01]*4)
        assert values['seeds']==[2025,2026,2027]


@pytest.mark.parametrize('problem',['incomplete','nan','source','data','checksum','seed'])
def test_rejects_incomplete_or_corrupt_results_before_creating_any_figures(complete,tmp_path,problem):
    root,plan=complete;view=plan['configs'][0];folder=root/view['runId']
    row=json.loads((folder/'run.json').read_text())
    if problem=='incomplete':row['status']='RUNNING'
    elif problem=='data':row['partition']={'holdout':'changed'}
    elif problem=='nan':np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((4,4),np.nan))
    elif problem=='source':(root/'source_snapshots/frozen.py').write_text('# overwritten\n')
    elif problem=='checksum':(root/'SHA256.json').write_text(json.dumps({'plan.json':'invalid'}))
    elif problem=='seed':
        plan['configs'].pop(0);(root/'plan.json').write_text(json.dumps(plan))
    (folder/'run.json').write_text(json.dumps(row))
    dest=tmp_path/'figures'
    with pytest.raises(ValueError):plots.generate(root,dest)
    assert not dest.exists()


def test_single_seed_is_not_given_a_fictitious_sd_band(complete):
    root,plan=complete
    plan['configs']=[c for c in plan['configs'] if c['seed']==2025]
    (root/'plan.json').write_text(json.dumps(plan))
    _,rows,_=plots.collect(root)
    assert all(v['sd'] is None for v in plots.aggregate(rows).values())


def test_publication_export_is_line_only_and_contains_every_seed_and_round(complete,tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    original_save=Figure.savefig
    def verify_layout(fig,*args,**kwargs):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        legend=fig.legends[0].get_window_extent(renderer)
        labels=[ax.xaxis.label.get_window_extent(renderer) for ax in fig.axes if ax.get_visible()]
        assert legend.y1 < min(label.y0 for label in labels), 'Legend overlaps the x-axis labels.'
        return original_save(fig,*args,**kwargs)
    monkeypatch.setattr(Figure,'savefig',verify_layout)
    root,plan=complete;dest=tmp_path/'figures'
    before={p:plots.sha(p) for p in root.rglob('*') if p.is_file()}
    manifest=plots.generate(root,dest)
    assert manifest['configurations']==24 and not manifest['smoothing']
    assert len([n for n in manifest['outputSha256'] if n.endswith('.svg')])==5
    assert len([n for n in manifest['outputSha256'] if n.endswith('.pdf')])==5
    assert len([n for n in manifest['outputSha256'] if n.endswith('.png')])==5
    assert not any('bar' in n for n in manifest['outputSha256'])
    overview=(dest/'comparison_joint_sigma_0.01_accuracy_curves.svg').read_text()
    for label in ('Validation accuracy','Communication round','SHARP + CART','SHARP + ACDS','Attack onset'):
        assert label in overview
    assert 'private-campaign-v2' not in overview and 'Official TEST' not in overview
    assert '#0072b2' in overview and '#d55e00' in overview
    with (dest/'accuracy_by_round_and_seed.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==24*4 and {int(r['round']) for r in rows}=={1,2,3,4}
    with (dest/'accuracy_mean_and_sd.csv').open() as f:summary=list(csv.DictReader(f))
    assert len(summary)==8*4 and all(int(r['seedCount'])==3 for r in summary)
    assert all(plots.sha(p)==v for p,v in before.items())
    assert all(plots.sha(dest/n)==v for n,v in manifest['outputSha256'].items())
    assert plots.sha(dest/'comparison_joint_sigma_0.01_accuracy_curves.png')
    with pytest.raises(FileExistsError):plots.generate(root,dest)


def test_wait_requires_metrics_files_and_stops_on_incomplete_failed_run(complete,monkeypatch):
    root,plan=complete
    assert readiness(root)[:2]==(24,24)
    (root/plan['configs'][0]['runId']/'metrics.npz').unlink()
    assert readiness(root)[:2]==(23,24)
    (root/'campaign.json').write_text(json.dumps({'status':'COMPLETED_WITH_FAILURES'}))
    with pytest.raises(RuntimeError,match='incomplete'):wait_for_completion(root,0)


def test_wait_wakes_only_when_all_metrics_are_committed(monkeypatch,tmp_path):
    import scripts.plot_cart_publication as cli
    states=iter([(8,24,'RUNNING'),(20,24,'RUNNING'),(24,24,'COMPLETED')])
    monkeypatch.setattr(cli,'readiness',lambda _:next(states))
    sleeps=[];monkeypatch.setattr(cli.time,'sleep',sleeps.append)
    cli.wait_for_completion(tmp_path,15)
    assert sleeps==[15,15]


def test_contribution_aliases_do_not_merge_distinct_variants(complete,tmp_path):
    root,plan=complete
    for view in plan['configs']:
        arm='acds_checked' if view['comparisonCase']['arm']=='acds' else 'cart_full'
        view['comparisonCase']['studyVariant']=dict(name=arm,revision='contribution')
        path=root/view['runId']/'run.json'
        row=json.loads(path.read_text());row['config']=view;path.write_text(json.dumps(row))
    (root/'plan.json').write_text(json.dumps(plan))
    _,rows,_=plots.collect(root)
    assert {r['arm'] for r in rows}=={'acds_checked','cart_full'}
    manifest=plots.generate(root,tmp_path/'contribution-figures',arms=['cart_full'])
    assert manifest['configurations']==12
    svg=(tmp_path/'contribution-figures/comparison_joint_sigma_0.01_accuracy_curves.svg').read_text()
    assert 'SHARP + CART' in svg and 'ACDS' not in svg


def test_contribution_report_and_export_preserve_inputs_and_require_complete_groups(complete,tmp_path):
    from reporting.cart_contribution import report
    from scripts.export_cart_contribution import export
    root,plan=complete
    for view in plan['configs']:
        name='acds_checked' if view['comparisonCase']['arm']=='acds' else 'cart_full'
        view['comparisonCase']['studyVariant']=dict(name=name,revision='contribution')
        folder=root/view['runId'];path=folder/'run.json'
        row=json.loads(path.read_text());row['config']=view;path.write_text(json.dumps(row))
        with np.load(folder/'metrics.npz',allow_pickle=False) as z:accuracy=z['roundNodeAccuracy']
        np.savez(folder/'metrics.npz',roundNodeAccuracy=accuracy,roundHonestMeanAccuracy=accuracy.mean(axis=1))
        (folder/'mechanisms.json').write_text('{}')
    (root/'plan.json').write_text(json.dumps(plan))
    (root/'campaign.json').write_text(json.dumps(dict(status='COMPLETED')))
    (root/'adaptive_execution.json').write_text('{}')
    before={p:plots.sha(p) for p in root.rglob('*') if p.is_file()}
    assert report(root)==24
    assert all(plots.sha(p)==value for p,value in before.items())
    with (root/'publication_curves/seed_endpoints.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==24 and all(float(r['finalAccuracy'])>=float(r['lastTenRoundAccuracy']) for r in rows)
    destination=tmp_path/'export'
    assert export(root,destination)==24
    assert len(plots.collect(destination)[1])==24
    assert not list(destination.rglob('checkpoint.npz'))
