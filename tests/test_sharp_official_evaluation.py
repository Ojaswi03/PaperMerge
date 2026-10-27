import json
from pathlib import Path
from types import SimpleNamespace


def test_official_mnist_loader_reads_only_test_keys(tmp_path,monkeypatch):
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    from scripts.evaluate_sharp_official_test import load_official_arrays
    folder=tmp_path/'.keras/datasets';folder.mkdir(parents=True)
    np.savez(folder/'mnist.npz',x_test=np.zeros((10000,28,28),np.uint8),
        y_test=np.arange(10000,dtype=np.int32)%10)
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    x,y,source=load_official_arrays('mnist')
    assert x.shape==(10000,28,28,1) and y.shape==(10000,)
    assert x.dtype==np.float32 and x.max()==0
    assert source.endswith('mnist.npz')


def test_incomplete_campaign_is_rejected_before_test_loading(tmp_path,monkeypatch):
    import pytest
    import scripts.evaluate_sharp_official_test as module
    import scripts.run_sharp_study as runner
    from basil_core.sharp_study import configuration
    cfg=configuration('mnist','iid',2025,'clean',0.,nodes=4,rounds=2)
    folder=tmp_path/cfg['runId'];folder.mkdir()
    (tmp_path/'plan.json').write_text(json.dumps(dict(configs=[cfg],source={'test':'fixed'})))
    (folder/'run.json').write_text(json.dumps(dict(config=cfg,source={'test':'fixed'},status='RUNNING')))
    monkeypatch.setattr(runner,'check_source',lambda _:None)
    def forbidden(*_):raise AssertionError('Official TEST must not load for an incomplete campaign.')
    monkeypatch.setattr(module,'load_official_arrays',forbidden)
    with pytest.raises(ValueError,match='All selected training runs'):
        module.evaluate(SimpleNamespace(output=tmp_path,device='CPU'))


def test_official_summary_never_averages_different_sigmas():
    import pytest
    from basil_core.sharp_study import configuration
    from scripts.evaluate_sharp_official_test import aggregate_final_results
    rows=[]
    for sigma,base in ((.005,.8),(.02,.6)):
        for seed in (2025,2026,2027):
            cfg=configuration('mnist','dirichlet',seed,'attack_noise_defense_rla_cart',sigma)
            rows.append(dict(config=cfg,meanAccuracy=base+.01*(seed-2025)))
    groups=aggregate_final_results(rows)
    assert len(groups)==2 and {g['config']['sigma'] for g in groups}=={.005,.02}
    assert sorted(round(g['meanAccuracy'],2) for g in groups)==[.61,.81]
    assert all(g['sampleStd']==pytest.approx(.01) for g in groups)
    with pytest.raises(ValueError,match='Duplicate'):aggregate_final_results([*rows,rows[0]])


def test_final_evaluation_never_calls_training_and_preserves_training_manifest(tmp_path,monkeypatch):
    import sys
    from basil_core.lazy_imports import lazy_module
    np=lazy_module('numpy')
    import scripts.evaluate_sharp_official_test as module
    import scripts.run_sharp_study as runner
    import basil_core.sharp_worker as workers
    import basil_core.sharp_protocol as protocol
    from basil_core.sharp_study import configuration
    cfg=configuration('mnist','iid',2025,'clean',0.,nodes=4,rounds=2)
    folder=tmp_path/cfg['runId'];folder.mkdir()
    original=dict(config=cfg,source={'test':'fixed'},status='COMPLETED',data={},partition={},runtime={},initialHash='initial')
    (tmp_path/'plan.json').write_text(json.dumps(dict(configs=[cfg],source={'test':'fixed'})))
    (folder/'run.json').write_text(json.dumps(original))
    monkeypatch.setattr(runner,'check_source',lambda _:None)
    config=SimpleNamespace(set_visible_devices=lambda *_:None,
        threading=SimpleNamespace(set_intra_op_parallelism_threads=lambda _:None,set_inter_op_parallelism_threads=lambda _:None),
        experimental=SimpleNamespace(enable_op_determinism=lambda:None))
    monkeypatch.setitem(sys.modules,'tensorflow',SimpleNamespace(config=config))
    class Worker:
        dimension=3
        def __init__(self,*_):pass
        def train(self,*_):raise AssertionError('Final evaluation must never train.')
    monkeypatch.setattr(workers,'SharpWorker',Worker)
    monkeypatch.setattr(protocol,'load_state',lambda *_:dict(cursor=8,nodeParams=np.zeros((4,3),np.float32)))
    monkeypatch.setattr(protocol,'evaluate',lambda *_:(.75,np.full(10,.75)))
    monkeypatch.setattr(module,'load_official_arrays',lambda _:(np.zeros((20,1),np.float32),np.arange(20)%10,'synthetic_TEST_fixture'))
    module.evaluate(SimpleNamespace(output=tmp_path,device='CPU'))
    result=json.loads((folder/'official_test_evaluation.json').read_text())
    assert result['officialTestUsed'] and not result['trainingPerformed'] and not result['cartDecisionsUsedTest']
    assert result['meanAccuracy']==.75 and json.loads((folder/'run.json').read_text())==original
