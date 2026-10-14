"""Analysis-only intervention, stable classification and leakage contracts."""
import pytest


def test_interventions_are_exact_copies_not_averages_and_never_modify_source():
    import numpy as np
    from basil_core.forgetting_audit import INTERVENTIONS, split_parameters
    current = [np.full((2, 2), 2.), np.full((2, 10), 3.), np.full(10, 4.)]
    selected = [np.full((2, 2), 5.), np.full((2, 10), 6.), np.full(10, 7.)]
    expected = {'trained': [2, 3, 4], 'selected': [5, 6, 7],
        'trained_backbone_selected_head': [2, 6, 7], 'selected_backbone_trained_head': [5, 3, 4],
        'trained_zero_output_bias': [2, 3, 0]}
    for name in INTERVENTIONS:
        result = split_parameters(current, selected, name)
        for i, value in enumerate(result):
            assert np.all(value == expected[name][i])
            value[:] = -999
    assert [v.flat[0] for v in current] == [2, 3, 4]
    assert [v.flat[0] for v in selected] == [5, 6, 7]


def test_stable_cross_entropy_and_single_class_bias_gradient():
    import numpy as np
    from basil_core.forgetting_audit import classification_metrics
    logits = np.tile([1000.] + [0.]*9, (20, 1))
    labels = np.ones(20, np.int32)
    result = classification_metrics(logits, labels)
    assert result['crossEntropy'] == pytest.approx(1000.)
    assert result['outputBiasCEGradient'] == [1., -1.] + [0.]*8
    assert result['predictedCounts'] == [20] + [0]*9
    assert result['perClassAccuracy'][0] is None  # unavailable, not zero
    assert result['perClassAccuracy'][1] == 0
    assert sum(result['outputBiasCEGradient']) == pytest.approx(0.)


def test_nonfinite_predictions_are_not_an_accuracy():
    import numpy as np
    from basil_core.forgetting_audit import classification_metrics
    with pytest.raises(ValueError, match='Finite'):
        classification_metrics(np.full((1, 10), np.nan), np.array([0]))


def test_representation_drift_identity_and_zero_norm_are_explicit():
    import numpy as np
    from basil_core.forgetting_audit import representation_metrics
    y = np.repeat(np.arange(10), 2)
    x = np.ones((20, 3))
    result = representation_metrics(x, x, y)
    assert result['meanSampleFeatureDrift'] == 0
    assert result['meanReferenceCosine'] == pytest.approx(1.)
    result = representation_metrics(x*0, x*0, y)
    assert result['meanReferenceCosine'] is None and result['zeroNormFeaturePairs'] == 20


def test_probe_is_training_only_disjoint_from_selection_deterministic():
    import numpy as np
    from scripts.run_forgetting_mechanism_audit import audit_indices
    labels = np.repeat(np.arange(10), 5000)
    ss = np.arange(512)
    a, b = audit_indices(labels, ss), audit_indices(labels, ss)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.bincount(labels[a], minlength=10), [64]*10)
    assert not set(a) & set(ss)


def test_bounded_config_has_no_other_method_threat_or_test_feedback():
    from scripts.run_forgetting_mechanism_audit import diagnostic_config
    c = diagnostic_config('GPU')
    assert c['localEpochs'] == 5 and c['optimizerStepsPerControl'] == 50
    assert c['activationsPerControl'] == 1 and c['basilMemorySize'] == 5
    assert c['actualAttackers'] == [] and c['researchValid'] is False
    assert not c['ebmEnabled'] and not c['channelNoiseEnabled'] and not c['testSetUsed']
    assert c['liveTrackingEnabled'] is False


def test_parameter_drift_separates_head_from_backbone():
    import numpy as np
    from basil_core.forgetting_audit import parameter_drift
    current = [np.ones(2), np.ones((2, 10)), np.ones(10)]
    reference = [v*0 for v in current]
    result = parameter_drift(current, reference)
    assert result['backbone'] == pytest.approx(2**.5)
    assert result['head'] == pytest.approx(30**.5)


def test_single_class_ce_head_gradient_matches_probability_minus_one_hot():
    import numpy as np
    import tensorflow as tf
    from basil_core.forgetting_audit import classification_metrics
    head = tf.keras.layers.Dense(10, kernel_initializer='zeros', bias_initializer='zeros')
    features = tf.constant([[1., 2.], [3., 4.]])
    y = tf.constant([0, 0])
    with tf.GradientTape() as tape:
        logits = head(features)
        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    kernel_grad, bias_grad = tape.gradient(ce, head.trainable_variables)
    result = classification_metrics(logits.numpy(), y.numpy())
    expected_bias = np.array([-.9] + [.1]*9)
    np.testing.assert_allclose(bias_grad, expected_bias, atol=1e-7)
    np.testing.assert_allclose(kernel_grad, np.array([[2.], [3.]])*expected_bias, atol=1e-7)
    np.testing.assert_allclose(result['outputBiasCEGradient'], expected_bias, atol=1e-7)


def test_report_refuses_incomplete_execution(tmp_path):
    import json
    from scripts.report_forgetting_mechanism_audit import review
    (tmp_path/'run.json').write_text(json.dumps({'status': 'running', 'optimizerSteps': 50, 'controlsCompleted': ['ce']}))
    with pytest.raises(ValueError, match='Incomplete'):
        review(tmp_path)


def test_audit_callback_cannot_mutate_training_and_mixed_heads_are_restored():
    import numpy as np
    import tensorflow as tf
    from basil_core.forgetting_audit import split_parameters
    from basil_core.noniid_study import SnapshotAnchorWorker
    from basil_core.research_protocol import params_hash
    outcomes = []
    for callback_enabled in (False, True):
        model = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)), tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.Constant(.1))])
        worker = SnapshotAnchorWorker(model, mu=0., device='CPU')
        initial = worker.export()
        def callback():
            current = worker.export()
            try:
                worker.load(split_parameters(current, initial, 'trained_zero_output_bias'))
                worker.predict(tf.ones((2, 32, 32, 3)))
            finally:
                worker.load(current)
            assert params_hash(worker.export()) == params_hash(current)
            return {'analysisOnly': 123.}
        trained, record = worker.train(initial, np.ones((2, 32, 32, 3), np.float32), np.zeros(2, np.int32),
            seed=2025, round_id=3, node_id=0, epochs=5, batch_size=512, learning_rate=.05/1.15,
            ebm_mode='none', semantics='paper_absolute_gaussian', sigma=0.,
            evaluation=callback if callback_enabled else None)
        outcomes.append((params_hash(trained), record['optimizerTelemetry']))
    assert outcomes[0] == outcomes[1]
