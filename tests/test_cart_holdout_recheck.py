"""Reject internally consistent logs when actual predictions disagree."""
import pytest


def test_prediction_recheck_rejects_an_inflated_node_or_class_score():
    import numpy as np
    from scripts.verify_cart_holdout import compare_predictions
    labels=np.repeat(np.arange(10,dtype=np.int32),2)
    predictions=np.tile(labels,(10,1));predictions[0,0]=1
    nodes=np.ones(10);nodes[0]=.95
    classes=np.ones((10,10));classes[0,0]=.5
    result=compare_predictions(predictions,labels,nodes,classes)
    assert result['meanAccuracy']==pytest.approx(.995)
    wrong=nodes.copy();wrong[0]=1.
    with pytest.raises(ValueError):compare_predictions(predictions,labels,wrong,classes)
    wrong=classes.copy();wrong[0,0]=1.
    with pytest.raises(ValueError):compare_predictions(predictions,labels,nodes,wrong)
