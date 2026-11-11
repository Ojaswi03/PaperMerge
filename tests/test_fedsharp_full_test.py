import numpy as np
import pytest

from scripts import evaluate_fedsharp_test as evaluation


def test_full_test_metrics_match_manual_cross_entropy_and_confusion():
    labels=np.tile(np.arange(10,dtype=np.int32),2)
    logits=np.zeros((20,10));logits[np.arange(20),labels]=2
    logits[0]=0;logits[0,1]=2
    row,pred,loss,confidence=evaluation.classify(logits,labels)
    assert row['testCorrect']==19 and row['testAccuracy']==.95
    expected=np.log(np.exp(2)+9)-2
    np.testing.assert_allclose(loss[1:],expected)
    assert loss[0]==pytest.approx(expected+2)
    confusion=np.array(row['confusionMatrix'])
    assert confusion.sum()==20 and confusion[0,1]==1 and confusion[0,0]==1
    assert row['classRecall'][0]==.5 and row['excludingPreviouslyDemonstratedImage0']['accuracy']==1
    assert np.all(confidence>0) and np.all(confidence<=1)
    assert pred.dtype==np.uint8


def test_invalid_arrays_refused_and_extreme_logits_are_stable():
    y=np.arange(10,dtype=np.int32);z=np.full((10,10),-10000.)
    z[np.arange(10),y]=10000
    row,_,loss,_=evaluation.classify(z,y)
    assert row['testAccuracy']==1 and np.all(loss==0)
    with pytest.raises(ValueError):evaluation.classify(z,np.full(10,12))
    with pytest.raises(ValueError):evaluation.classify(np.full((10,10),np.nan),y)
    with pytest.raises(ValueError,match='Every class'):evaluation.classify(np.zeros((10,10)),np.zeros(10,dtype=np.int32))


def test_aggregation_is_across_clients_not_seed_uncertainty():
    y=np.arange(10,dtype=np.int32);z=np.eye(10)*2
    a,*_=evaluation.classify(z,y);b,*_=evaluation.classify(z[:,::-1],y)
    result=evaluation.summarize([a,b],np.array([.8,.7]))
    assert result['meanAccuracy']==.5 and result['worstClientAccuracy']==0
    assert result['finalRoundValidationMeanAccuracy']==.75
    assert result['testMinusFinalValidationPercentagePoints']==-25
    assert 'not seed SD' in result['clientSDInterpretation']
    assert np.array(result['summedClientConfusionMatrix']).sum()==20


def test_existing_result_directory_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):evaluation.evaluate(tmp_path,tmp_path)
