import copy
from types import SimpleNamespace
import pytest


def test_counterfactual_uses_highest_ranked_selected_copy_without_mutating_evidence():
    import numpy as np
    from scripts.probe_cart_handoff import best_single
    cfg=dict(nodes=4,seed=2025,sigma=0.)
    state=dict(outbound=np.asarray([[1.,2.],[3.,4.],[5.,6.],[7.,8.]],np.float32),senderRounds=np.zeros(4,dtype=int))
    evidence=dict(selectedSenders=[2,1],selectedCount=2,targetCount=2)
    original=copy.deepcopy(evidence)
    selector=best_single(lambda *_:(np.asarray([4.,5.],np.float32),evidence))
    value,row=selector(cfg,state,None,None,None,3,0,None)
    np.testing.assert_array_equal(value,state['outbound'][2])
    assert row['selectedSenders']==[2] and row['selectedCount']==1
    assert row['diagnosticOriginalSelectedCount']==2 and evidence==original


def test_counterfactual_preserves_bootstrap_model():
    import numpy as np
    from scripts.probe_cart_handoff import best_single
    value=np.asarray([1.,2.],np.float32)
    selected=best_single(lambda *_:(value,dict(selectedSenders=[],selectedCount=1)))
    output,row=selected({},None,None,None,None,0,0,None)
    assert output is value and row['selectedSenders']==[] and row['selectedCount']==1


def test_all_diagnostic_arms_are_pre_attack_no_cart_and_no_rla():
    from scripts.probe_cart_handoff import configuration,CASES
    from basil_core.sharp_cart_v2_protocol import validate_config
    args=SimpleNamespace(nodes=100,rounds=5)
    rows=[configuration(args,i) for i in range(len(CASES))]
    assert len(rows)==8 and len({r['name'] for r in rows})==8
    assert sum(r['forceBestSingle'] for r in rows)==2
    for row in rows:
        cfg=row['config'];validate_config(cfg)
        assert cfg['attackStart']>=cfg['rounds'] and not cfg['cart']
        assert cfg['localSteps']==5 and not cfg['switches']['noise_mitigation']
        assert cfg['trainPerClass'] is None and cfg['validationPerClass'] is None
    assert {r['config']['proximalMu'] for r in rows}=={0.,1.}
