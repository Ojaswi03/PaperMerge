import json
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_acds_comparison import arguments
from tests.test_cart_v2_resilient import queue,children


def test_comparison_presets_and_modular_noise_require_no_training(tmp_path):
    base=[sys.executable,'scripts/run_cart_comparison.py','--output',str(tmp_path)]
    for preset,count in [('screen',20),('isolation',32),('confirm',12)]:
        row=json.loads(subprocess.check_output(base+['--preset',preset],text=True))
        assert row['configs']==count and not row['trainingStarted']
    row=json.loads(subprocess.check_output(base+['--preset','easy','--partitions','dirichlet'],text=True))
    assert row['configs']==8 and row['rounds']==5 and not row['trainingStarted']
    row=json.loads(subprocess.check_output(base+['--preset','confirm','--datasets','mnist','cifar10','--noise-levels','3'],text=True))
    assert row['configs']==72 and not tmp_path.joinpath('plan.json').exists()
    bad=subprocess.run(base+['--resume','--execute','--rounds','30'],capture_output=True,text=True)
    assert bad.returncode==2 and 'frozen plan' in bad.stderr


def test_comparison_failures_do_not_stop_healthy_workers(queue,monkeypatch):
    old,args,plan=queue
    import scripts.run_cart_comparison as module
    plan.update(revision='fixture',reference=None);args.ntfy_topic=None
    monkeypatch.setattr(module,'prepared_for_execution',old.prepare)
    monkeypatch.setattr(module,'check_source',lambda _:None)
    plots=[];monkeypatch.setitem(sys.modules,'reporting.cart_comparison',SimpleNamespace(generate=lambda *a,**k:plots.append(a)))
    started,_=children(monkeypatch,module,args.output,bad=(1,6))
    assert module.execute(args)==2 and sorted(started)==list(range(8)) and not plots
    assert json.loads((args.output/'campaign.json').read_text())['completed']==6
    started,_=children(monkeypatch,module,args.output,bad=())
    assert module.execute(args)==0 and sorted(started)==[1,6] and plots


def test_report_rejects_incomplete_and_changed_data_and_keeps_seed_endpoints(tmp_path,monkeypatch):
    from basil_core.cart_comparison_study import selected,effective_config
    import basil_core.cart_comparison_study as study
    from reporting.cart_comparison import collect
    monkeypatch.setattr(study,'check_source',lambda _:None)
    _,configs=selected(arguments(arms=['cart','acds'],partitions=['iid'],rounds=10,nodes=4))
    plan=dict(configs=configs,source={},revision='test',data={'cifar10':{}},partitions={})
    for view in configs:
        cfg=effective_config(view);folder=tmp_path/view['runId'];folder.mkdir()
        plan['partitions'][view['runId']]={'partitionHash':'owner','holdoutHash':'holdout'}
        row=dict(status='COMPLETED',config=view,source={},methodRevision='test',completedActivations=40,
            data={},partition=plan['partitions'][view['runId']],initialHash='initial',runtime={})
        (folder/'run.json').write_text(json.dumps(row))
        np.savez(folder/'metrics.npz',roundNodeAccuracy=np.full((10,4),.2 if view['comparisonCase']['arm']=='cart' else .3))
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    _,rows,curves,_=collect(tmp_path)
    assert len(rows)==6 and all(r['dataSharing']==(r['arm']=='acds') for r in rows)
    assert all(r['primary']==pytest.approx(r['final']) for r in rows)
    target=tmp_path/configs[0]['runId']/'run.json';row=json.loads(target.read_text());row['status']='FAILED';target.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='complete'):collect(tmp_path)
    row['status']='COMPLETED';row['partition']['partitionHash']='different';target.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='identity'):collect(tmp_path)
