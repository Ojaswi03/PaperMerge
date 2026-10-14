"""No-display contracts for the explicitly revised strict CART adaptation."""
import pytest

from basil_core.class_registry import ClassRegistry
from basil_core.noniid_study import anchored_gradients
from basil_core.strict_cart import StrictCartWorker, anchor_strength
from basil_core.research_protocol import params_hash, initialize_memories, broadcast, Snapshot
from scripts.run_strict_cart_diagnostic import config, expected_candidates


def evidence(value):
    import numpy as np
    return np.array([value] + [np.nan]*9), np.array([512]+[0]*9)


def test_handoff_confidence_is_positive_without_phantom_gap():
    acc, support = evidence(.5)
    result = anchor_strength(.01, 2., acc, acc, support)
    assert result['mu'] == pytest.approx(.01/3)
    assert result['supportedGap'] == 0
    assert result['supportedClasses'] == [0]


def test_class_gap_increases_anchor_but_cap_is_fixed():
    ref, support = evidence(.8)
    current, _ = evidence(.2)
    r = anchor_strength(.01, 2., ref, current, support)
    assert r['mu'] == pytest.approx(.01*(1/3+.6))
    assert anchor_strength(.01, 0., ref, current, support)['mu'] == .01


def test_unknown_classes_cannot_control_anchor():
    import numpy as np
    ref, support = evidence(.8); cur, _ = evidence(.5)
    a = anchor_strength(.01, 2., ref, cur, support)
    ref[1:] = 1.; cur[1:] = 0.
    assert anchor_strength(.01, 2., ref, cur, support) == a
    assert anchor_strength(.01, 2., ref, cur, np.zeros(10))['mu'] == 0


@pytest.mark.parametrize('mu,loss', [(-1, 2), (.01, -1), (float('nan'), 2), (.01, float('inf'))])
def test_nonfinite_or_negative_control_is_rejected(mu, loss):
    ref, support = evidence(.5)
    with pytest.raises(ValueError): anchor_strength(mu, loss, ref, ref, support)


def test_registry_does_not_verify_absent_classes_or_accept_inflated_claim():
    import numpy as np
    incoming = ClassRegistry(); incoming.update(8, np.ones(10), np.full(10, 512), roundId=2)
    local = ClassRegistry(); ref, support = evidence(.5)
    accepted, rejected = local.verifyAndMerge(incoming, ref, receivedSupport=support)
    assert (accepted, rejected) == (0, 1)
    assert not local.class_best_support.any()


def test_exact_ebm_plus_anchor_gradient_against_explicit_tiny_hessian():
    import numpy as np
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(2, use_bias=False)])
    model.layers[0].kernel.assign([[.3, -.2]])
    x, y = tf.constant([[1.]]), tf.constant([0])
    w = model.trainable_variables[0]
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=model(x)))
        base = inner.gradient(ce, w)
    hessian = tf.reshape(outer.jacobian(base, w), (2, 2))
    reference = tf.constant([[.1, -.1]])
    gradients, values = anchored_gradients(model, x, y, [reference], mu=.01, sigma_e=.010, ebm=True)
    correction = 2*.0001*tf.linalg.matvec(hessian, tf.reshape(base, (2,)))
    expected = tf.reshape(base, (2,)) + correction + .01*tf.reshape(w-reference, (2,))
    np.testing.assert_allclose(gradients[0].numpy().ravel(), expected.numpy(), atol=1e-7, rtol=1e-6)
    assert float(values[3]) == pytest.approx(.0001)
    assert float(values[2]) == pytest.approx(float(ce + .0001*tf.reduce_sum(base**2) + .005*tf.reduce_sum((w-reference)**2)))


def tiny_worker():
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)), tf.keras.layers.GlobalAveragePooling2D(),
                                tf.keras.layers.Dense(10, kernel_initializer='zeros', bias_initializer='zeros')])
    return StrictCartWorker(model, mu_max=.01)


def test_legacy_lambda_has_zero_effect_on_loss_gradient_and_update():
    import numpy as np
    import tensorflow as tf
    worker = tiny_worker()
    initial = worker.export()
    for ref, value in zip(worker.reference, initial): ref.assign(value + .02)
    worker.anchor_mu.assign(.007)
    x, y = tf.ones((2, 32, 32, 3))*.3, tf.constant([0, 0])
    results = []
    for legacy in (1., 999999.):
        worker.load(initial)
        grads, values = worker._gradients(x, y, tf.constant([7, 8]), 'gradient_norm_objective',
            'paper_absolute_gaussian', tf.constant(.010), tf.constant(legacy))
        tf.keras.optimizers.SGD(.05).apply_gradients(zip(grads, worker.model.trainable_variables))
        results.append(([g.numpy() for g in grads], [float(v) for v in values], worker.export()))
    for group_a, group_b in zip(*results):
        for a, b in zip(group_a, group_b): np.testing.assert_array_equal(a, b)


def test_epoch_coefficient_changes_the_compiled_anchor_gradient():
    import numpy as np
    import tensorflow as tf
    worker = tiny_worker(); initial = worker.export()
    for ref, value in zip(worker.reference, initial): ref.assign(value + .02)
    x, y = tf.ones((2, 32, 32, 3))*.3, tf.constant([0, 0])
    values = []
    for mu in (.001, .009):
        worker.anchor_mu.assign(mu)
        grads, _ = worker._gradients(x, y, tf.constant([7, 8]), 'none',
            'paper_absolute_gaussian', tf.constant(0.), tf.constant(1.))
        values.append([g.numpy() for g in grads])
    for a, b in zip(*values):
        np.testing.assert_allclose(b-a, np.full_like(a, -.02*(.009-.001)), atol=1e-7)


def test_five_epochs_frozen_reference_registry_and_no_optimizer_extras():
    import numpy as np
    worker = tiny_worker(); initial = worker.export()
    x = np.ones((2, 32, 32, 3), np.float32)*.3; y = np.zeros(2, np.int32)
    loss, _ = worker.probe(initial, x, y)
    trained, record = worker.train(initial, x, y, local_registry=ClassRegistry(), received_registry=ClassRegistry(),
        selection_x=x, selection_y=y, selected_loss=loss, seed=2025, round_id=0, node_id=0,
        epochs=5, batch_size=512, learning_rate=.05, ebm_mode='none',
        semantics='paper_absolute_gaussian', sigma=0.)
    assert record['steps'] == 5 and len(record['epochs']) == 5
    assert record['snapshotAnchorReferenceHash'] == params_hash(initial)
    assert params_hash(trained) != params_hash(initial)
    assert len(record['cartControl']) == 6
    assert record['unverifiedLocalClasses'] == list(range(1, 10))
    assert record['outputRegistry']['class_best_support'][1:] == [0]*9
    assert worker.optimizer.momentum == 0 and worker.optimizer.clipnorm is None
    assert worker.optimizer.weight_decay is None


def test_registry_is_carried_with_selected_snapshot_and_round11_memory():
    import numpy as np
    memories = initialize_memories([np.array([0.], np.float32)])
    for r in range(11):
        for node in range(10):
            broadcast(memories, Snapshot(node, r, [np.array([node], np.float32)], retention_state={'source': node}))
    assert [(s.sender_id, s.round_id) for s in memories[0].candidates()] == [(s, 10) for s in range(5, 10)]
    broadcast(memories, Snapshot(0, 11, [np.array([0.], np.float32)], retention_state={'source': 0}))
    assert [(s.sender_id, s.round_id) for s in memories[1].candidates()] == [(s, 10) for s in range(6, 10)]+[(0, 11)]
    assert memories[1].candidates()[-1].retention_state == {'source': 0}
    assert expected_candidates(11, 0) == [(s, 10) for s in range(5, 10)]


def test_evaluation_only_scores_cannot_change_training_or_mu():
    import numpy as np
    x = np.ones((2, 32, 32, 3), np.float32)*.3; y = np.zeros(2, np.int32)
    outcomes = []
    for fake_global_accuracy in (0., 1.):
        worker = tiny_worker(); initial = worker.export()
        loss, _ = worker.probe(initial, x, y)
        incoming = ClassRegistry(); incoming.update(7, np.ones(10)*.8, np.full(10, 512), roundId=4)
        result, record = worker.train(initial, x, y, local_registry=ClassRegistry(), received_registry=incoming,
            selection_x=x, selection_y=y, selected_loss=loss, seed=2025, round_id=0, node_id=0,
            epochs=5, batch_size=512, learning_rate=.05, ebm_mode='none',
            semantics='paper_absolute_gaussian', sigma=0., evaluation=lambda: {'globalAccuracy': fake_global_accuracy})
        outcomes.append((params_hash(result), record['cartControl']))
        # Historical remote records can travel, but are not locally verified.
        assert record['outputRegistry']['class_best_support'][1:] == [512]*9
        assert record['registryAfter']['class_best_support'][1:] == [0]*9
    assert outcomes[0] == outcomes[1]


def test_diagnostic_configuration_is_full_data_bounded_clean_and_cloud_disabled():
    c = config('GPU')
    assert (c['nRounds'], c['localEpochs'], c['trainingSamplesPerNode'], c['batchSize']) == (3, 5, 5000, 512)
    assert c['basilMemorySize'] == 5 and c['aggregationMode'] == 'strict_sequential_handoff'
    assert not c['useChannelNoise'] and c['ebmMode'] == 'none' and not c['attackerCount']
    assert c['researchValid'] is False and c['liveTrackingEnabled'] is False
    assert c['gradientClipNorm'] is None and c['weightDecayCoefficient'] == 0 and c['momentum'] == 0


def test_report_refuses_partial_results(tmp_path, monkeypatch):
    import json
    import scripts.report_strict_cart_diagnostic as report
    (tmp_path/'run.json').write_text(json.dumps({'status': 'running', 'completedActivations': 30}))
    monkeypatch.setattr(report, 'RESULT', tmp_path)
    with pytest.raises(ValueError, match='Incomplete'): report.review()
