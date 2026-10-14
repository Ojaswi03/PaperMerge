"""Adversarial fixtures demonstrate both bounded outliers and remaining failures."""
from dataclasses import replace
import json
import subprocess
import sys

import pytest


def gradients_fixture():
    import numpy as np
    from basil_core.cart_owner_gradients import OwnerGradientResponse
    from basil_core.research_protocol import params_hash
    params = [np.zeros(3,np.float32)]
    responses = [OwnerGradientResponse(c,0,1,0,params_hash(params),16,1.,
                 (np.array([c+1,2*c,-c],np.float32),)) for c in range(10)]
    return params, responses


def score_fixture():
    import numpy as np
    from basil_core.research_protocol import Snapshot, params_hash
    from basil_core.cart_helper_robustness import SelectionScoreResponse
    snapshots=[Snapshot(c,0,[np.array([c],np.float32)]) for c in range(5,10)]
    hashes=tuple(params_hash(s.params) for s in snapshots)
    responses=[SelectionScoreResponse(c,0,1,'a'*64,hashes,(1.,2.,3.,4.,5.)) for c in range(10)]
    return snapshots,responses


def test_four_arbitrary_gradient_rows_stay_in_honest_coordinate_range():
    import numpy as np
    from basil_core.cart_helper_robustness import robust_owner_gradients
    params,responses=gradients_fixture()
    attackers={0,1,5,7}  # Test oracle only; NEVER passed to the defense.
    honest=np.stack([r.gradients[0] for r in responses if r.owner not in attackers])
    for seed in range(30):
        rng=np.random.default_rng(seed)
        poisoned=[replace(r,gradients=(rng.uniform(-1e10,1e10,3).astype(np.float32),))
                  if r.owner in attackers else r for r in responses]
        grads,meta=robust_owner_gradients(params,poisoned,receiver=0,round_id=1,optimizer_step=0,
                                          available_owners=range(10))
        assert np.all(grads[0]>=honest.min(axis=0)) and np.all(grads[0]<=honest.max(axis=0))
        assert meta['retainedValuesPerCoordinate']==2
        assert not meta['researchReady'] and not meta['honestGradientVerified']


def test_legitimate_unique_class_direction_can_be_removed():
    import numpy as np
    from basil_core.cart_helper_robustness import trimmed_coordinates
    rows=np.zeros((10,1));rows[0,0]=1.
    assert trimmed_coordinates(rows,bound=4)[0]==0.
    # Even an honest rare direction can be discarded. Bounded != useful.


def test_robust_score_limits_hold_but_forged_scores_can_reverse_ranking():
    import numpy as np
    from basil_core.cart_helper_robustness import robust_selection
    candidates,responses=score_fixture()
    honest=[(0.,1.,50.,50.,50.)]*3+[(10.,11.,50.,50.,50.)]*3
    forged=[(100.,0.,50.,50.,50.)]*4
    responses=[replace(r,losses=v) for r,v in zip(responses,honest+forged)]
    selected,meta=robust_selection(candidates,responses,receiver=0,round_id=1,request_hash='a'*64,
                                   available_owners=range(10))
    assert np.mean([r[0] for r in honest]) < np.mean([r[1] for r in honest])
    assert selected.sender_id==6  # Honest mean preferred 5; poisoning reverses it.
    scores=np.array(meta['scores']); good=np.array(honest)
    assert np.all(scores>=good.min(axis=0)) and np.all(scores<=good.max(axis=0))
    assert not meta['honestScoresVerified'] and not meta['researchReady']


def test_score_ties_still_select_nearest_and_not_sender_telemetry():
    from basil_core.cart_helper_robustness import robust_selection
    candidates,responses=score_fixture()
    responses=[replace(r,losses=(1.,)*5) for r in responses]
    selected,_=robust_selection(candidates,responses,receiver=0,round_id=1,request_hash='a'*64,
                                available_owners=range(10))
    assert selected.sender_id==9


@pytest.mark.parametrize('n',[1,2,3,4,5,6,7,8])
def test_startup_or_withholding_fails_closed(n):
    from basil_core.cart_helper_robustness import robust_owner_gradients
    params,responses=gradients_fixture()
    with pytest.raises(ValueError,match='quorum'):
        robust_owner_gradients(params,responses[:n],receiver=0,round_id=1,optimizer_step=0,
                                available_owners=range(n))


@pytest.mark.parametrize('fault',['missing','duplicate','stale','shape'])
def test_invalid_gradients_not_silently_dropped(fault):
    import numpy as np
    from basil_core.cart_helper_robustness import robust_owner_gradients
    params,responses=gradients_fixture()
    if fault=='missing':responses.pop()
    if fault=='duplicate':responses[-1]=responses[0]
    if fault=='stale':responses[-1]=replace(responses[-1],optimizer_step=1)
    if fault=='shape':responses[-1]=replace(responses[-1],gradients=(np.zeros(2,np.float32),))
    with pytest.raises(ValueError):
        robust_owner_gradients(params,responses,receiver=0,round_id=1,optimizer_step=0,available_owners=range(10))


@pytest.mark.parametrize('fault',['request','round','candidate','duplicate','missing'])
def test_invalid_score_context_rejected(fault):
    from basil_core.cart_helper_robustness import robust_selection
    candidates,responses=score_fixture()
    if fault=='request':responses[0]=replace(responses[0],request_hash='b'*64)
    if fault=='round':responses[0]=replace(responses[0],round_id=0)
    if fault=='candidate':responses[0]=replace(responses[0],candidate_hashes=tuple(reversed(responses[0].candidate_hashes)))
    if fault=='duplicate':responses[-1]=responses[0]
    if fault=='missing':responses.pop()
    with pytest.raises(ValueError):
        robust_selection(candidates,responses,receiver=0,round_id=1,request_hash='a'*64,available_owners=range(10))


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_rejected_without_repair(value):
    from basil_core.cart_helper_robustness import trimmed_coordinates
    with pytest.raises(FloatingPointError):trimmed_coordinates([[value]]+[[0.]]*9,bound=4)


def test_full_node_gate_remains_closed_and_lazy():
    code=('from basil_core.cart_helper_robustness import full_node_readiness; '
          'import json,sys;print(json.dumps(full_node_readiness()));'
          'assert not ({"numpy","tensorflow","wandb","matplotlib"}&sys.modules.keys())')
    result=subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
    plan=json.loads(result.stdout)
    assert not plan['researchReady'] and not plan['queueStarted'] and plan['trainingRunsStarted']==0
    assert plan['minimumResponses']==9 and plan['firstActivationAvailableResponses']==1
    from basil_core.cart_owner_protocol import require_helper_boundary,FULL_NODE
    with pytest.raises(ValueError,match='blocked'):require_helper_boundary(FULL_NODE,[0,1,5,7])
