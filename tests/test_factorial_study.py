"""Causal switches, equal seed weights and fail-closed study coverage."""
import copy
import pytest

from basil_core.factorial_study import (CONDITIONS, SEEDS, matrix, configuration,
    aggregate_seed_results, validate_config, learning_rate)


def test_matrix_pairs_each_seed_with_all_three_sigmas_without_duplicate_clean():
    configs=matrix((.0025,.005,.01))
    assert len(configs)==144
    assert len({c['runId'] for c in configs})==144
    for cfg in configs:validate_config(cfg)
    for partition in ('iid','dirichlet'):
        for seed in SEEDS:
            rows=[c for c in configs if c['seed']==seed and c['partitionStrategy']==partition]
            assert len(rows)==24
            assert sum(c['conditionId']=='clean' for c in rows)==1
            for c in CONDITIONS:
                assert {r['channelNoiseSigmaAbsolute'] for r in rows if r['conditionId']==c.name}==(
                    {.0025,.005,.01} if c.noise else {0.})


def test_clean_has_no_defenses_and_conditions_have_exact_independent_switches():
    for cfg in matrix((.0025,.005,.01)):
        switches=cfg['factorialSwitches']
        assert cfg['snapshotSelection']==switches['byzantine_mitigation']
        assert (cfg['ebmMode']=='gradient_norm_objective')==switches['noise_mitigation']
        assert cfg['cartEnabled']==switches['cart']
        assert cfg['snapshotAnchorMu']==0 and not cfg['logitCalibrationEnabled']
        assert cfg['attackHidden']==switches['attack']
        assert cfg['useChannelNoise']==switches['noise']
        if cfg['conditionId']=='clean':
            assert not any(v for k,v in switches.items() if k!='name')
            assert cfg['resolvedAttackerIds']==[]
    changed=copy.deepcopy(matrix((.0025,.005,.01))[0]);changed['snapshotAnchorMu']=.01
    with pytest.raises(ValueError):validate_config(changed)


def test_summaries_average_nodes_then_window_then_matching_seeds_only():
    import numpy as np
    sigmas=(.0025,.005,.01)
    rows=[]
    for cfg in matrix(sigmas):
        # Distinct seed effects and sigma effects expose either pooling error.
        value=.2+.1*SEEDS.index(cfg['seed'])+cfg['channelNoiseSigmaAbsolute']
        arr=np.full((100,10),value)
        arr[0]=.99  # Excluded by the predeclared endpoint window.
        rows.append(dict(config=cfg,status='COMPLETED',roundNodeAccuracy=arr))
    result=aggregate_seed_results(rows,sigmas,endpoint_start=1,expected_rounds=100)
    noisy=[r for r in result if r['partition']=='iid' and r['condition']=='noise_only']
    assert [r['sigma'] for r in noisy]==list(sigmas)
    for r in noisy:
        assert r['mean']==pytest.approx(.3+r['sigma'])
        assert r['sampleStd']==pytest.approx(.1)
        assert len(r['seedEndpoints'])==3
    with pytest.raises(ValueError,match='horizon'):aggregate_seed_results(rows,sigmas,endpoint_start=1,expected_rounds=3)
    with pytest.raises(ValueError,match='Missing'):aggregate_seed_results(rows[:-1],sigmas,endpoint_start=1,expected_rounds=100)
    with pytest.raises(ValueError,match='Duplicate'):aggregate_seed_results(rows+rows[:1],sigmas,endpoint_start=1,expected_rounds=100)
    rows[0]['status']='FAILED'
    with pytest.raises(ValueError,match='Incomplete'):aggregate_seed_results(rows,sigmas,endpoint_start=1,expected_rounds=100)


@pytest.mark.parametrize('sigmas',[(.01,.01,.02),(.01,.005,.02),(0,.01,.02),(.01,.02),(.01,.02,float('nan'))])
def test_invalid_levels_rejected(sigmas):
    with pytest.raises(ValueError):matrix(sigmas)


def test_common_optimizer_schedule_is_bounded_and_endpoints_fixed():
    cfg=matrix((.0025,.005,.01))[0]
    assert learning_rate(cfg,0)==pytest.approx(.1)
    assert learning_rate(cfg,99)==pytest.approx(.001)
    assert all(.001<=learning_rate(cfg,i)<=.1 for i in range(100))


def test_residual_model_contains_no_untransmitted_state_and_is_seeded():
    import numpy as np
    from basil_core.research_protocol import build_model,params_hash
    from basil_core.factorial_study import MODEL
    first=build_model(MODEL,2025)
    again=build_model(MODEL,2025)
    other=build_model(MODEL,2026)
    assert not first.non_trainable_variables
    assert params_hash(first.get_weights())==params_hash(again.get_weights())
    assert params_hash(first.get_weights())!=params_hash(other.get_weights())
    assert first(np.zeros((2,32,32,3),np.float32)).shape==(2,10)
