"""Actual nonlinear feasibility, coherent targets and finite-sample scoring."""
import pytest


def test_refresh_can_improve_mean_without_claiming_classwise_dominance():
    from basil_core.cart_feasible_memory import coherent_admission
    before=[1.,2.];after=[1.1,1.7]
    assert not coherent_admission(before,after)
    assert coherent_admission(before,after,bounds=[1.2,2.1])
    assert not coherent_admission(before,[1.3,1.2],bounds=[1.2,2.1])
    assert not coherent_admission(before,[1.2,1.9],bounds=[1.2,2.1])


def test_class_losses_use_only_observed_classes_and_stable_log_softmax():
    import numpy as np
    from basil_core.cart_feasible_memory import class_losses
    labels=np.array([1,1,7],np.int32)
    logits=np.zeros((3,10));logits[0,1]=1000
    classes,loss=class_losses(logits,labels)
    np.testing.assert_array_equal(classes,[1,7])
    np.testing.assert_allclose(loss,[np.log(10)/2,np.log(10)])


def test_envelope_is_one_joint_score_not_unjustified_independent_class_quantiles():
    import numpy as np
    from basil_core.cart_feasible_memory import calibrate_envelope
    reference=np.array([.01,2.])
    samples=np.tile(reference,(19,1));samples[0,0]+=.02;samples[1,1]+=.5
    result=calibrate_envelope(reference,samples)
    assert result['gamma']==pytest.approx(.25)
    np.testing.assert_allclose(result['bounds'],[.035,2.5])
    assert result['marginalReferenceChannelCoverage']==.95
    with pytest.raises(ValueError):calibrate_envelope(reference,samples[:18])


def test_exact_rank_coverage_for_every_possible_held_out_rank():
    import numpy as np
    from basil_core.cart_feasible_memory import calibrate_envelope,within_envelope
    # Enumerate all ranks among 20 exchangeable distinct scores, rather than a
    # flaky Monte Carlo test of the already-standard order-statistic argument.
    covered=[]
    all_scores=np.arange(20)/100
    for heldout in range(20):
        training=np.delete(all_scores,heldout)
        result=calibrate_envelope([1.],1.+training[:,None])
        covered.append(within_envelope([1.+all_scores[heldout]],result['bounds'],tolerance=0.))
    assert sum(covered)==19


def test_nonlinear_nonconvex_path_checks_actual_losses_and_has_a_witness():
    import numpy as np
    from basil_core.cart_feasible_memory import restore_feasible
    # A disconnected feasible set prevents assuming convexity or interpolated
    # losses. Only the zero endpoint is feasible on this particular finite grid.
    def evaluate(params):
        x=float(params[0][0]);return [(x*x-1)**2]
    result,evidence=restore_feasible([np.array([-1.])],[np.array([0.])],[0.],evaluate)
    np.testing.assert_array_equal(result[0],[-1.])
    assert evidence['scale']==0.
    assert evaluate(result)[0]<=0.
    with pytest.raises(ValueError,match='anchor'):
        restore_feasible([np.array([0.])],[np.array([.1])],[0.],evaluate)


def test_admission_preserves_one_model_instead_of_inventing_a_class_minimum_mosaic():
    from basil_core.cart_feasible_memory import coherent_admission
    assert not coherent_admission([.1,2.],[2.,.1])
    assert coherent_admission([1.,2.],[.9,1.9])
    assert not coherent_admission([1.,2.],[1.,2.])


def test_module_import_has_no_numerical_or_gpu_side_effects():
    import subprocess
    import sys
    subprocess.run([sys.executable,'-c','import basil_core.cart_feasible_memory;import sys;'
        'assert not ({"numpy","tensorflow","wandb","matplotlib"}&sys.modules.keys())'],check=True)
