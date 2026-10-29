import numpy as np
import pytest
from basil_core.cart_balanced_commit import balanced_loss,commit,transport


def fixture(check_sign=1.,peak=None):
    x=np.arange(8,dtype=np.float32)[:,None]
    y=np.tile([0,1],4)
    context=dict(indices=np.arange(8),split=4,classes=np.array([0,1]),
        ingress=np.array([0.],np.float32))
    class Worker:
        dimension=1;sizes=[0,0]
        def outputs(self,flat,images):
            t=float(flat[0]);strength=t if peak is None else 1-(t-peak)**2
            z=np.zeros((len(images),2))
            for j,index in enumerate(images[:,0].astype(int)):
                z[j,y[index]]=strength*(check_sign if index>=4 else 1)
            return z
        def load(self,flat):self.loaded=flat.copy()
    return Worker(),context,x,y


def test_balanced_ce_gives_each_supported_class_equal_weight_and_is_stable():
    z=np.array([[1000.,999.],[1000.,1001.]])
    value=balanced_loss(z,np.array([0,1]),[0,1])
    repeated=np.repeat(z,[7,1],axis=0)
    assert value==pytest.approx(balanced_loss(repeated,np.array([0]*7+[1]),[0,1]))
    assert np.isfinite(value)
    with pytest.raises(ValueError,match='probe examples'):
        balanced_loss(z,np.array([0,0]),[0,1])


def test_confidence_improvement_is_measured_without_requiring_prediction_change():
    w,c,x,y=fixture();c['ingress'][:]=.25
    candidate,r=commit(w,np.array([1.],np.float32),c,x,y)
    assert r['scale']==1 and r['optimizerStepsAdded']==0
    assert all(a<b for a,b in zip(r['committedProbeLosses'],r['baselineProbeLosses']))
    np.testing.assert_array_equal(candidate,[1.])


def test_both_probes_are_required_and_zero_fallback_preserves_ingress():
    w,c,x,y=fixture(check_sign=-1)
    candidate,r=commit(w,np.array([1.],np.float32),c,x,y)
    assert r['scale']==0 and len(r['trials'])==5
    np.testing.assert_array_equal(candidate,c['ingress'])


def test_backtracking_keeps_reference_for_a_partially_committed_full_update():
    w,c,x,y=fixture(peak=.25);w.quality_context=c
    previous={'id':'previous'};new={'id':'uncommitted proposal'}
    original=lambda *a:(np.array([1.],np.float32),new,{'referenceAdmitted':True})
    chosen,memory,evidence=transport(original)(w,np.array([1.],np.float32),previous,x,y,0,1)
    assert evidence['qualityCommit']['scale']==.5
    assert memory is previous and not evidence['referenceAdmitted']
    np.testing.assert_array_equal(w.loaded,chosen)


def test_nonfinite_proposal_is_rejected_without_mutating_ingress():
    w,c,x,y=fixture();before=c['ingress'].copy()
    result,r=commit(w,np.array([np.inf],np.float32),c,x,y)
    assert r['scale']==0 and r['trials'][0]['probeLosses'] is None
    np.testing.assert_array_equal(c['ingress'],before)
    np.testing.assert_array_equal(result,before)
