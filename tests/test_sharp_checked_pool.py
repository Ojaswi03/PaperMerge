import numpy as np
import pytest
from basil_core import sharp_checked_pool as module


class Worker:
    def score_many(self, vectors, x, y):
        return 3. - vectors[:, 0]


def setup(monkeypatch, pool=0.):
    monkeypatch.setattr(module, 'link_vector', lambda vector, *args: vector.copy())
    cfg=dict(seed=2025,sigma=.01,nodes=4)
    state=dict(outbound=np.array([[2.],[1.],[0.],[0.]],np.float32),senderRounds=np.zeros(4,int))
    evidence=dict(selectedSenders=[0,1],selectedCount=2,targetCount=2,ranking=[dict(sender=0,loss=1.)])
    def original(*args):return np.array([pool],np.float32),dict(evidence)
    args=(cfg,state,Worker(),np.zeros((2,1)),np.zeros(2,int),2,0,np.array([0,1]))
    return original,args


def test_checked_pool_keeps_best_member_when_realized_average_has_higher_loss(monkeypatch):
    import json
    original,args=setup(monkeypatch)
    vector,row=module.selector(original,'checked')(*args)
    np.testing.assert_array_equal(vector,[2.])
    assert row['selectedSenders']==[0] and row['selectedCount']==row['targetCount']==1
    assert row['poolQuality']['originalSelectedCount']==2 and row['poolQuality']['poolExcessCE']==2
    assert row['ranking']==[dict(sender=0,loss=1.)]
    assert json.loads(json.dumps(row))['poolQuality']['fallback'] is True


@pytest.mark.parametrize('pool',[2.,2.5])
def test_useful_or_equal_average_and_its_noise_denominator_are_preserved(monkeypatch,pool):
    original,args=setup(monkeypatch,pool)
    vector,row=module.selector(original,'checked')(*args)
    assert vector[0]==pool and row['selectedCount']==2 and not row['poolQuality']['fallback']


def test_raw_instrumentation_keeps_original_vector_and_selected_count(monkeypatch):
    original,args=setup(monkeypatch)
    vector,row=module.selector(original,'raw')(*args)
    assert vector[0]==0 and row['selectedCount']==2
    assert row['poolQuality']['poolExcessCE']==2 and not row['poolQuality']['fallback']


def test_bootstrap_does_not_invent_a_received_member(monkeypatch):
    original,args=setup(monkeypatch)
    empty=lambda *a:(np.array([.5]),dict(selectedSenders=[],selectedCount=1))
    vector,row=module.selector(empty)(*args)
    assert vector[0]==.5 and not row['poolQuality']['checked']


def test_raw_diagnostics_reproduce_original_acds_parameters_and_metrics_on_cpu(monkeypatch):
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    from tests.test_cart_comparison_runtime import case,data
    from basil_core.cart_comparison_study import configure_worker,intervention
    import basil_core.sharp_cart_v2_protocol as protocol
    view=case('acds');x,y,chunks=data()
    first=configure_worker(view,'CPU')
    with intervention(view,first) as cfg:
        expected=protocol.run(cfg,first,x,y,chunks,x[:20],y[:20])
    second=configure_worker(view,'CPU')
    with intervention(view,second) as cfg:
        original=protocol.select_input
        monkeypatch.setattr(protocol,'select_input',module.selector(original,'raw'))
        actual=protocol.run(cfg,second,x,y,chunks,x[:20],y[:20])
        monkeypatch.setattr(protocol,'select_input',original)
    np.testing.assert_array_equal(actual['nodeParams'],expected['nodeParams'])
    np.testing.assert_array_equal(actual['outbound'],expected['outbound'])
    assert actual['roundMetrics']==expected['roundMetrics']
    assert sum(r['training']['optimizerSteps'] for r in actual['telemetry'])==view['nodes']*view['rounds']*view['localSteps']


def test_local_diagnostic_contract_supports_standard_checkpoint_recovery(tmp_path):
    from tests.test_cart_comparison_runtime import case
    from scripts.diagnose_acds_pool import contract_for
    from basil_core.sharp_protocol import initial_state,save_state,load_state
    view=case('acds');plan=dict(producerSource={'producer':'hash'},diagnosticSource={'local':'hash'})
    contract=contract_for(plan,dict(view=view,mode='checked'),{},'initial',{}, {})
    cfg=contract['config'];assert cfg['nodes']==4 and cfg['rounds']==2 and not cfg['cart']
    state=initial_state(cfg,np.array([1.,2.,3.],np.float32))
    path=tmp_path/'checkpoint.npz';save_state(path,state,contract)
    restored=load_state(path,contract,3)
    np.testing.assert_array_equal(restored['nodeParams'],state['nodeParams'])
    assert restored['cursor']==0
