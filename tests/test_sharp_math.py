from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')
import pytest

from basil_core.sharp_math import averaging_count,peak_screen,ranked_indices,hop_coefficient,class_ce,ridge_transport


@pytest.mark.parametrize('size,expected',[(0,1),(1,1),(33,1),(34,1),(40,7),(65,32),(66,33)])
def test_professor_equation_seven_including_window_warmup(size,expected):
    assert averaging_count(size,33)==expected


def test_spiky_copy_is_screened_but_dense_shift_is_not_certified():
    rng=np.random.default_rng(7);a=rng.normal(0,.01,(66,10000)).astype(np.float32)
    a[-1,143]=100.;a[-2]+=.2
    passed,e=peak_screen(a,full_window=True)
    assert 65 not in passed and 64 in passed
    assert not e['screenAllFailFallback']
    before,e=peak_screen(a,full_window=False)
    assert len(before)==66 and not e['screenApplied']


def test_zero_deviation_and_all_fail_are_explicit():
    passed,e=peak_screen(np.zeros((4,100)),full_window=True)
    assert len(passed)==4 and e['peakRatios']==[0.]*4
    values=np.zeros((4,10000));values[np.arange(4),np.arange(4)]=10.
    passed,e=peak_screen(values,full_window=True)
    assert len(passed)==4 and e['screenAllFailFallback']


def test_ranking_ties_recent_then_nearest_and_nonfinite_last():
    assert ranked_indices([1.,1.,1.,np.nan],[9,8,7,6],[0,1,1,8],0,10,3).tolist()==[1,2,0]
    with pytest.raises(FloatingPointError):ranked_indices([np.inf],[1],[0],0,10,1)


def test_hop_weight_exact_and_decreasing_learning_rate():
    assert hop_coefficient(.01,33,.03)==pytest.approx(.0001/(2*33*.03))
    assert hop_coefficient(.01,1,.03)==pytest.approx(33*hop_coefficient(.01,33,.03))
    assert hop_coefficient(.01,33,.003)>hop_coefficient(.01,33,.03)
    assert hop_coefficient(0.,1,.03)==0
    with pytest.raises(ValueError):hop_coefficient(.01,0,.03)


@pytest.mark.parametrize('classes',[10,100])
def test_generalized_transport_normal_equations_and_zero_gates(classes):
    rng=np.random.default_rng(13);h=rng.normal(size=(37,5));p=rng.normal(size=(37,classes));r=rng.normal(size=p.shape)
    y=np.arange(37,dtype=np.int32)%7;g=np.linspace(0,1,classes)
    delta,e=ridge_transport(h,p,r,y,g)
    a=np.column_stack((h,np.ones(len(h))));weights=1/(7*np.bincount(y)[y])
    target=g[y,None]*((r-r.mean(axis=1,keepdims=True))-(p-p.mean(axis=1,keepdims=True)))
    np.testing.assert_allclose(a.T@(weights[:,None]*(a@delta-target))+e['ridgePenalty']*delta,0,atol=1e-12)
    unchanged,_=ridge_transport(h,p,r,y,np.zeros(classes))
    np.testing.assert_array_equal(unchanged,np.zeros_like(unchanged))
    shifted,_=ridge_transport(h,p+50,r-80,y,g)
    np.testing.assert_allclose(delta,shifted,atol=1e-11)
    assert e['fittedObjective']<=e['zeroObjective']


def test_raw_class_loss_and_nonfinite_input():
    classes,loss=class_ce(np.zeros((3,100)),np.array([0,5,5]))
    assert classes.tolist()==[0,5]
    np.testing.assert_allclose(loss,np.log(100))
    with pytest.raises(ValueError):class_ce(np.full((3,100),np.nan),np.array([0,5,5]))
