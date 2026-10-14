"""Zero-training controls and mathematical limitations of one-class probes."""
import inspect
import json
import csv
import gzip

import pytest

from basil_core.class_registry import ClassRegistry
from basil_core.noniid_study import anchored_gradients
from scripts.prepare_retention_diagnostic import diagnostic_configs


@pytest.mark.parametrize('rounds', [5, 30])
def test_clean_controls_differ_only_by_anchor_strength(rounds):
    configs = diagnostic_configs(rounds)
    for c in configs:
        assert c['nRounds'] == rounds and c['localEpochs'] == 5
        assert c['batchSize'] == 512 and c['learningRate'] == .05
        assert c['basilMemorySize'] == 5 and c['snapshotSelection']
        assert c['attackerCount'] == 0 and c['resolvedAttackerIds'] == []
        assert not c['attackHidden'] and not c['useChannelNoise']
        assert c['ebmMode'] == 'none' and not c['researchValid']
        assert c['partitionStrategy'] == 'one_class_per_node'
        assert c['resultDirectory'].startswith('newResults/nonIID/retention_diagnostic/')
        assert not c['liveTrackingEnabled']
    assert [c['snapshotAnchorMu'] for c in configs] == [0., .01]


@pytest.mark.parametrize('rounds', [0, 1, 100, 30.0, True])
def test_unapproved_diagnostic_horizon_is_rejected(rounds):
    with pytest.raises(ValueError, match='horizon'):
        diagnostic_configs(rounds)


def test_zero_mu_matches_ce_loss_gradient_and_sgd_update():
    from basil_core.trainer import lossFn
    import numpy as np
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Input((1,)),
                                tf.keras.layers.Dense(10, use_bias=False)])
    model.trainable_variables[0].assign(np.arange(10, dtype=np.float32)[None, :] / 10)
    x = tf.constant([[1.], [2.]])
    y = tf.constant([0, 0])
    start = [tf.identity(w) for w in model.trainable_variables]
    with tf.GradientTape() as tape:
        ce = lossFn(y, model(x))
    ordinary = tape.gradient(ce, model.trainable_variables)
    gradients, values = anchored_gradients(model, x, y, start, mu=0., sigma_e=0., ebm=False)
    assert float(values[2]) == float(ce)
    assert float(values[9]) == 0.
    for expected, actual in zip(ordinary, gradients):
        np.testing.assert_array_equal(expected.numpy(), actual.numpy())
    tf.keras.optimizers.SGD(.05).apply_gradients(zip(gradients, model.trainable_variables))
    for initial, gradient, updated in zip(start, ordinary, model.trainable_variables):
        np.testing.assert_allclose(updated.numpy(), (initial-.05*gradient).numpy(), atol=1e-7)


def test_single_class_ce_suppresses_other_logits_and_initial_anchor_is_zero():
    from basil_core.trainer import lossFn
    import numpy as np
    import tensorflow as tf
    logits = tf.Variable(tf.zeros((1, 10)))
    with tf.GradientTape() as tape:
        ce = lossFn(tf.constant([0]), logits)
    gradient = tape.gradient(ce, logits).numpy()[0]
    np.testing.assert_allclose(gradient, [-.9] + [.1]*9, atol=1e-7)
    # This is the derivative at this toy state, not a claimed full-CNN trajectory.
    before = tf.identity(logits)
    np.testing.assert_array_equal(.01*(logits-before).numpy(), np.zeros((1,10)))


def test_registry_cannot_invent_evidence_for_absent_classes():
    import numpy as np
    registry = ClassRegistry()
    registry.update(9, np.ones(10), support=np.full(10, 10))
    local_support = np.array([512]+[0]*9)
    weights = registry.trustWeights(np.zeros(10), mySupport=local_support)
    assert weights[0] == 1.
    np.testing.assert_array_equal(weights[1:], np.zeros(9))


def test_preparation_dry_run_has_no_training_or_queue_calls(monkeypatch, capsys):
    from scripts import prepare_retention_diagnostic as script
    import sys
    monkeypatch.setattr(sys, 'argv', ['prepare_retention_diagnostic.py'])
    script.main()
    result = json.loads(capsys.readouterr().out)
    assert result['executionStarted'] is False and result['queueModified'] is False
    assert not result['wandbEnabled']


def test_runner_respects_diagnostic_validity_metadata():
    from scripts.run_noniid_basil_study import execute
    # Inspect the metadata assignment without starting TensorFlow research runs.
    assert "researchValid=config.get('researchValid', True)" in inspect.getsource(execute)


def test_saved_diagnostic_review_rejects_missing_round_and_nonfinite_step(tmp_path):
    import numpy as np
    from scripts.report_retention_diagnostic import review_control
    cfg = diagnostic_configs(30)[0]
    metadata = dict(config=cfg,status='completed',completedRounds=30,completedActivations=300,
        researchValid=False,wandbEnabled=False,runtimeSeconds=1.,partitionHash='same',
        initialModelHash='same',memorySafety=dict(maximumGpuPeakMiB=100.,pressureEvents=0))
    (tmp_path/'run.json').write_text(json.dumps(metadata))
    np.savez(tmp_path/'metrics.npz',roundNodePerClassAccuracy=np.broadcast_to(np.eye(10),(30,10,10)),
        averageAccuracy=np.full(30,.1),fullTestAccuracy=np.full(30,.1),worstNodeAccuracy=np.full(30,.1))
    step=dict(baseCrossEntropy=.2,baseGradientNorm=1.,robustObjectiveGradientNorm=1.,
        snapshotDistanceSquared=1.,snapshotAnchorPenalty=0.,anchorGradientNorm=0.,
        anchorCorrectionRatio=0.,snapshotAnchorMu=0.,ebmCoefficient=0.,ebmPenalty=0.,
        ebmCorrectionNorm=0.,ebmObjective=.2)
    records=[]
    for r in range(30):
        for node in range(10):
            senders=[(node-d)%10 for d in range(5,0,-1)]
            records.append(dict(round=r,nodeId=node,optimizerSteps=50,
                epochs=[dict(samplesSeen=5000,batchSizes=[512]*9+[392]) for _ in range(5)],
                optimizerTelemetry=[dict(step) for _ in range(50)],
                snapshotAnchorReferenceHash='input',inputHash='input',trainedOutputHash='updated',
                memorySenderIds=senders,memorySnapshotRounds=[r if s<node else r-1 for s in senders],
                selectedCandidateLoss=.2,candidates=[dict(receiverLocalLoss=.2)],
                attackActive=False,channelNoiseActive=False,outgoingLinks=[dict(noiseL2Norm=0.)]*5,
                meanOtherClassForgetting=.1))
    def write_records():
        with gzip.open(tmp_path/'activation_telemetry.json.gz','wt') as handle:
            json.dump(records,handle)
    write_records()
    def write_rounds(count):
        with (tmp_path/'round_metrics.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=['round','full_test_accuracy','mean_node_accuracy'])
            writer.writeheader()
            writer.writerows(dict(round=i,full_test_accuracy=.1,mean_node_accuracy=.1) for i in range(count))
    write_rounds(30)
    reviewed=review_control(tmp_path)
    assert reviewed['technicalPass'] and reviewed['ownClassCollapse']
    write_rounds(29)
    assert not review_control(tmp_path)['technicalPass']
    write_rounds(30)
    records[0]['optimizerTelemetry'][0]['baseGradientNorm']=float('nan')
    write_records()
    reviewed=review_control(tmp_path)
    assert not reviewed['technicalPass'] and not reviewed['checks']['finiteSteps']
