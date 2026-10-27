"""Projection retains safe components and agrees with an independent oracle."""
import pytest


def test_harmful_component_is_removed_without_freezing_safe_direction():
    import numpy as np
    from basil_core.cart_activation_projection import project_displacement
    step,info=project_displacement([2.,-3.],[[1.,0.]])
    np.testing.assert_allclose(step,[0.,-3.],atol=1e-12)
    assert info['status']=='projected'
    step,_=project_displacement([1.,2.],[[1.,1.]])
    np.testing.assert_allclose(step,[-.5,.5],atol=1e-12)
    step,_=project_displacement([1.,2.],[[1.,0.],[-1.,0.],[0.,0.]])
    np.testing.assert_allclose(step,[0.,2.],atol=1e-12)
    scaled,_=project_displacement([1.,2.],[[1000.,0.],[-.01,0.]])
    np.testing.assert_allclose(scaled,step,atol=1e-12)


def test_random_small_problems_match_enumerated_active_set_oracle():
    import itertools
    import numpy as np
    from basil_core.cart_activation_projection import project_displacement
    rng=np.random.default_rng(876)
    for _ in range(20):
        g=rng.normal(size=(3,5));d=rng.normal(size=5);candidates=[]
        for size in range(4):
            for indices in itertools.combinations(range(3),size):
                rows=g[list(indices)]
                dual=np.linalg.solve(rows@rows.T,rows@d) if size else np.empty(0)
                v=d-rows.T@dual
                if np.all(dual>=-1e-10) and np.all(g@v<=1e-9):candidates.append(v)
        expected=min(candidates,key=lambda v:np.linalg.norm(v-d))
        actual,info=project_displacement(d,g)
        assert info['status']=='projected'
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)


def test_numeric_failure_is_explicit_and_zero_is_feasible():
    import numpy as np
    from basil_core.cart_activation_projection import project_displacement
    d=np.array([1.,2.]);g=np.array([[1.,0.],[-1.,.1]])
    v,info=project_displacement(d,g,max_sweeps=1)
    assert info['status']=='fallback_zero'
    np.testing.assert_array_equal(v,[0.,0.])
    assert np.all(g@v<=0)
    v,info=project_displacement(d,np.zeros((2,2)))
    np.testing.assert_array_equal(v,d)
    assert info['status']=='no_nonzero_constraints'
    with pytest.raises(ValueError):project_displacement([float('nan')],[[1.]])
