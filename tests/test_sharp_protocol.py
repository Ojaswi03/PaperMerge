import copy
import json
from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')
import pytest

from basil_core.sharp_study import configuration,validate_config,matrix
from basil_core.sharp_protocol import run,save_state,load_state,link_vector


class FakeWorker:
    dimension=4
    def export(self):return np.array([.1,.2,.3,.4],np.float32)
    def tensors(self,value):return [value]
    def probe(self,params,x,y):return float(np.sum(params.astype(np.float64)**2))
    def train(self,params,x,y,r,node,coefficient,first_batch,should_stop):
        return (params+np.float32(.0001*(node+1))),dict(optimizerSteps=5,learningRate=.03,
            rlaCoefficient=coefficient,localTrainingLoss=1.)
    def outputs(self,params,x):
        return np.repeat(np.arange(10)[None,:]*float(params[0]),len(x),axis=0)


def fixture(nodes=100):
    cfg=configuration('mnist','iid',2025,'attack_noise_defense_rla',.01,nodes=nodes,rounds=2)
    x=np.zeros((2*nodes,1),np.float32);y=np.arange(2*nodes,dtype=np.int32)%10
    chunks=np.array_split(np.arange(len(y)),nodes)
    return cfg,x,y,chunks


def test_full_requested_matrix_and_explicit_defaults():
    configs=matrix();assert len(configs)==432
    assert {c['dataset'] for c in configs}=={'mnist','cifar10','cifar100'}
    for cfg in configs:
        validate_config(cfg)
        assert cfg['nodes']==100 and cfg['byzantineBudget']==33 and cfg['window']==66
        assert cfg['localSteps']==5 and cfg['localEpochs'] is None and cfg['dirichletAlpha']==.5
        assert not cfg['officialTestUsed']
    clean=next(c for c in configs if c['condition']=='clean')
    assert clean['actualAttackers']==0 and clean['proximalMu']==0 and not clean['cart']
    with pytest.raises(ValueError):validate_config(dict(clean,nodes=10))


def test_basil_window_has_a_budget_valid_default_and_explicit_comparators():
    cfg=configuration('cifar10','iid',2025,'attack_defense',0.,method='basil')
    assert cfg['window']==34
    for size in (10,66):
        cfg=configuration('cifar10','iid',2025,'attack_defense',0.,method='basil',basil_window=size)
        validate_config(cfg);assert cfg['window']==size


def test_hundred_nodes_sixtysix_windows_and_exact_checkpoint_resume(tmp_path):
    cfg,x,y,chunks=fixture();worker=FakeWorker()
    whole=run(cfg,worker,x,y,chunks,x[:10],y[:10])
    partial=run(cfg,worker,x,y,chunks,x[:10],y[:10],max_activations=137)
    contract=dict(config=cfg,source={'test':'fixed'})
    path=tmp_path/'checkpoint.npz';save_state(path,partial,contract)
    restored=load_state(path,contract,4)
    resumed=run(cfg,worker,x,y,chunks,x[:10],y[:10],state=restored)
    np.testing.assert_array_equal(whole['nodeParams'],resumed['nodeParams'])
    np.testing.assert_array_equal(whole['outbound'],resumed['outbound'])
    assert whole['telemetry']==resumed['telemetry'] and whole['roundMetrics']==resumed['roundMetrics']
    assert len(whole['roundMetrics'][-1]['nodeAccuracy'])==100
    full=whole['telemetry'][99]['selection']
    assert full['availableCopies']==66 and full['selectedCount']==33
    assert whole['telemetry'][0]['selection']['bootstrap']
    assert whole['telemetry'][1]['selection']['selectedCount']==1
    with pytest.raises(ValueError):load_state(path,dict(contract,source={'test':'changed'}),4)


def test_resume_completes_interrupted_round_evaluation():
    cfg,x,y,chunks=fixture(6);worker=FakeWorker()
    partial=run(cfg,worker,x,y,chunks,x[:10],y[:10],max_activations=6)
    partial['roundMetrics']=[]
    final=run(cfg,worker,x,y,chunks,x[:10],y[:10],state=partial)
    assert [r['round'] for r in final['roundMetrics']]==[0,1]


def test_lazy_noise_is_immutable_per_edge_and_independent_across_receivers():
    base=np.zeros(20000,np.float32)
    a=link_vector(base,2025,3,5,8,.02)
    np.testing.assert_array_equal(a,link_vector(base,2025,3,5,8,.02))
    b=link_vector(base,2025,3,5,9,.02)
    assert not np.array_equal(a,b) and abs(np.corrcoef(a,b)[0,1])<.04
    averages=np.stack([link_vector(base,2025,3,s,99,.02) for s in range(33)]).mean(axis=0)
    assert np.var(averages)==pytest.approx(.02**2/33,rel=.04)


def test_stop_discards_partial_activation():
    cfg,x,y,chunks=fixture(6)
    state=None
    def save(s):
        nonlocal state
        state=copy.deepcopy(s)
    with pytest.raises(InterruptedError):run(cfg,FakeWorker(),x,y,chunks,x[:10],y[:10],checkpoint=save,should_stop=lambda:True)
    assert state['cursor']==0 and state['telemetry']==[]
