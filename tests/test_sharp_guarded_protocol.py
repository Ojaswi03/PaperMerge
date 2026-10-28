import copy

import pytest
from basil_core.lazy_imports import lazy_module
np=lazy_module('numpy')

from basil_core.sharp_study import configuration,validate_config as validate_base
from basil_core.sharp_guarded_protocol import guarded_configuration,validate_config,select_input,run


class SmallWorker:
    dimension=4
    def export(self):return np.array([.1,.2,.3,.4],np.float32)
    def tensors(self,value):return [value]
    def probe(self,params,x,y):return float(np.sum(params.astype(np.float64)**2))
    def train(self,params,x,y,r,node,coefficient,first_batch,should_stop):
        return params+np.float32(.0001*(node+1)),dict(optimizerSteps=5,learningRate=.03,
            rlaCoefficient=coefficient,localTrainingLoss=1.)
    def outputs(self,params,x):return np.repeat(np.arange(10)[None,:]*float(params[0]),len(x),axis=0)


def test_distinct_configuration_rejects_silent_mix_with_original_method():
    base=configuration('mnist','iid',2025,'attack_defense',0.)
    cfg=guarded_configuration(base);validate_config(cfg)
    assert cfg['method']=='sharp_guarded' and cfg['runId']!=base['runId']
    with pytest.raises(ValueError):validate_base(cfg)
    wrong=dict(cfg,survivorGuard='unrecorded_rule')
    with pytest.raises(ValueError):validate_config(wrong)
    assert base['method']=='sharp'


def test_gpu_launcher_selects_the_new_family_for_all_288_cases():
    from types import SimpleNamespace
    from basil_core.sharp_study import CONDITION_NAMES,SIGMAS
    from scripts.run_guarded_sharp_study import selected
    args=SimpleNamespace(datasets=['mnist','cifar10'],partitions=['iid','dirichlet'],
        seeds=[2025,2026,2027],conditions=CONDITION_NAMES,sigmas=SIGMAS,methods=['sharp'],
        nodes=100,rounds=100,model='paper',alpha=.5,local_steps=5,train_per_class=None,
        validation_per_class=None,attack_start=20,hop_denominator='actual_selected',curvature=1.,
        mu=1.,learning_rate=.03,attack_type='hidden',budget=None,basil_window=None,only_sigma=None)
    configs=selected(args)
    assert len(configs)==288 and {c['method'] for c in configs}=={'sharp_guarded'}
    for cfg in configs:validate_config(cfg)


def test_one_dense_attacked_survivor_does_not_exclude_all_benign_copies():
    from basil_core.attacks import applyAttack
    cfg=guarded_configuration(configuration('mnist','iid',2025,'attack_defense',0.))
    base=np.full(1024,.1,np.float32);copies=np.repeat(base[None],100,axis=0)
    copies[np.arange(100),np.arange(100)]+=.001
    copies[0]=applyAttack([base],'hidden',rng=np.random.default_rng(8))[0]
    state=dict(nodeParams=copies.copy(),outbound=copies.copy(),senderRounds=np.zeros(100,np.int32))
    class Receiver:
        def probe(self,params,x,y):return float(np.mean((params-base)**2))
    value,e=select_input(cfg,state,Receiver(),np.zeros((1,1)),np.zeros(1,np.int32),66,1,np.array([0]))
    assert e['screenGuardFallback'] and e['rawSurvivorCount']==1
    assert e['selectedCount']==1 and 0 not in e['selectedSenders']
    assert np.mean((value-base)**2)<1e-6


def test_averaging_count_is_based_on_survivors_not_original_window(monkeypatch):
    import basil_core.sharp_guarded_protocol as protocol
    cfg=guarded_configuration(configuration('mnist','iid',2025,'attack_defense',0.))
    initial=np.zeros((100,4),np.float32)
    state=dict(nodeParams=initial,outbound=initial,senderRounds=np.zeros(100,np.int32))
    def screen(values,**_):
        return np.arange(40),dict(screenAllFailFallback=False,screenApplied=True,
            peakRatios=[0.]*40+[50.]*26,peakThreshold=10.)
    monkeypatch.setattr(protocol,'peak_screen',screen)
    _,e=select_input(cfg,state,SmallWorker(),np.zeros((1,1)),np.zeros(1,np.int32),66,1,np.array([0]))
    assert not e['screenGuardFallback'] and e['selectedCount']==7
    assert e['survivorUnattackedLowerBound']==7


def test_full_surviving_window_matches_original_and_recovers_exactly(tmp_path):
    from basil_core.sharp_protocol import run as base_run,save_state,load_state
    base=configuration('mnist','iid',2025,'attack_noise_defense_rla',.01,nodes=100,rounds=2)
    cfg=guarded_configuration(base);x=np.zeros((200,1),np.float32);y=np.arange(200,dtype=np.int32)%10
    chunks=np.array_split(np.arange(200),100);worker=SmallWorker()
    expected=base_run(base,worker,x,y,chunks,x[:10],y[:10])
    whole=run(cfg,worker,x,y,chunks,x[:10],y[:10])
    np.testing.assert_array_equal(expected['nodeParams'],whole['nodeParams'])
    np.testing.assert_array_equal(expected['outbound'],whole['outbound'])
    partial=run(cfg,worker,x,y,chunks,x[:10],y[:10],max_activations=137)
    contract=dict(config=cfg,source={'test':'same'});path=tmp_path/'state.npz'
    save_state(path,partial,contract)
    restored=load_state(path,contract,worker.dimension)
    resumed=run(cfg,worker,x,y,chunks,x[:10],y[:10],state=restored)
    np.testing.assert_array_equal(whole['nodeParams'],resumed['nodeParams'])
    assert whole['telemetry']==resumed['telemetry'] and whole['roundMetrics']==resumed['roundMetrics']


@pytest.mark.parametrize('condition,sigma',[('attack_defense',0.),('attack_noise_defense_rla_cart',.01)])
def test_real_tensorflow_guarded_rounds_and_checkpoint(condition,sigma,tmp_path):
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import save_state,load_state
    cfg=guarded_configuration(configuration('mnist','iid',2025,condition,sigma,nodes=4,rounds=2,attack_start=0))
    worker=SharpWorker(cfg);y=np.tile(np.arange(2,dtype=np.int32),40)
    x=np.zeros((80,28,28,1),np.float32);x[y==0,:14]=1.;x[y==1,14:]=1.
    chunks=np.array_split(np.arange(80),4);path=tmp_path/'state.npz';contract={'config':cfg}
    state=run(cfg,worker,x,y,chunks,x[:8],y[:8],checkpoint=lambda s:save_state(path,s,contract))
    restored=load_state(path,contract,worker.dimension)
    assert restored['cursor']==8 and len(restored['roundMetrics'])==2
    np.testing.assert_array_equal(state['nodeParams'],restored['nodeParams'])
