"""Synthetic mathematical/provenance contracts; no CIFAR or cloud runs."""
from dataclasses import FrozenInstanceError, replace

import pytest

from basil_core.cart_feature_memory import (FeaturePacket, ActivationFeatureMemory,
    FeatureReplayWorker, packet_from_features, feature_replay_gradients, encoder_hash,
    owner_refresh_memory)

H = 'a'*64
E = 'b'*64


def packet(owner=1, **changes):
    import numpy as np
    return packet_from_features(np.arange(12, dtype=np.float32).reshape(4, 3),
        **dict(owner=owner, round_id=0, receiver=0, model_hash=H, encoder_id=E, **changes))


def memory(packets=(), allowed=(1, 2)):
    return ActivationFeatureMemory(0, 0, H, E, 3, allowed, tuple(packets))


def test_packet_mean_copy_digest_and_roundtrip():
    import numpy as np
    z = np.arange(60, dtype=np.float32).reshape(20, 3)
    p = packet_from_features(z, owner=1, round_id=0, receiver=0, model_hash=H, encoder_id=E)
    np.testing.assert_allclose(p.prototype, z.mean(axis=0))
    assert len(p.features) == 16 and p.support == 20
    expected = p.digest
    z[:] = -100
    assert p.digest == expected
    assert FeaturePacket.from_dict(p.to_dict()) == p
    with pytest.raises(FrozenInstanceError): p.support = 1


@pytest.mark.parametrize('data_usage', ['test', 'validation', 'evaluation_only'])
def test_no_test_or_validation_feature_source(data_usage):
    with pytest.raises(ValueError, match='Test/validation'):
        packet(data_usage=data_usage)


@pytest.mark.parametrize('shape', [(2, 32, 32, 3), (513, 3), (2, 193), (0, 3)])
def test_raw_images_or_unbounded_packets_rejected(shape):
    import numpy as np
    with pytest.raises(ValueError):
        packet_from_features(np.zeros(shape, np.float32), owner=1,
            round_id=0, receiver=0, model_hash=H, encoder_id=E)


def test_nonfinite_packet_rejected():
    import numpy as np
    with pytest.raises(ValueError):
        packet_from_features([[float('nan')]*3], owner=1, round_id=0,
            receiver=0, model_hash=H, encoder_id=E)


@pytest.mark.parametrize('field,value', [('model_hash', 'c'*64), ('encoder_hash', 'c'*64),
    ('receiver', 2), ('round_id', 1)])
def test_stale_wrong_model_or_receiver_packet_rejected(field, value):
    with pytest.raises(ValueError, match='Stale'):
        memory([replace(packet(), **{field: value})])


def test_no_future_class_duplicates_or_fabricated_missing_scores():
    with pytest.raises(ValueError): memory([packet()], allowed=())
    with pytest.raises(ValueError): memory([packet(), packet()])
    report = memory([packet()]).to_dict()
    assert report['unavailableClasses'] == [2]
    assert 'accuracy' not in report and report['presentClasses'] == [1]
    with pytest.raises(ValueError): memory([], allowed=(0,))


def test_weights_are_class_balanced_and_memory_is_bounded():
    import numpy as np
    second = packet_from_features(np.ones((2, 3), np.float32), owner=2,
        round_id=0, receiver=0, model_hash=H, encoder_id=E)
    z, y, weights = memory([packet(), second]).tensors()
    assert z.shape == (160, 3)
    assert float(weights.sum()) == pytest.approx(1.)
    assert float(weights[y == 1].sum()) == pytest.approx(.5)
    assert float(weights[y == 2].sum()) == pytest.approx(.5)
    assert weights[6:].sum() == 0
    assert memory().tensors()[2].sum() == 0


def tiny_model():
    import tensorflow as tf
    return tf.keras.Sequential([tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(2, use_bias=False, kernel_initializer=tf.keras.initializers.Constant([[.3, .2], [.1, .4]])),
        tf.keras.layers.Dense(2, use_bias=False, kernel_initializer=tf.keras.initializers.Constant([[.2, -.1], [.1, .3]]))])


def objective_inputs(model):
    import tensorflow as tf
    return dict(model=model, x=tf.constant([[.4, .2], [.2, .3]]), y=tf.constant([0, 0]),
        reference=[tf.identity(w)-.01 for w in model.trainable_variables],
        features=tf.constant([[.4, .2], [.2, .5]]), feature_labels=tf.constant([1, 1]),
        row_weights=tf.constant([.5, .5]), mu=.01, sigma_e=.010, ebm=True, beta=1.)


def test_combined_loss_and_gradient_match_nested_objective():
    import numpy as np
    import tensorflow as tf
    model = tiny_model(); inputs = objective_inputs(model)
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=inputs['y'], logits=model(inputs['x'])))
        base = inner.gradient(ce, model.trainable_variables)
        norm_sq = tf.add_n([tf.reduce_sum(g*g) for g in base])
        anchor = .005*tf.add_n([tf.reduce_sum((w-a)**2) for w, a in zip(model.trainable_variables, inputs['reference'])])
        replay = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=inputs['feature_labels'], logits=model.layers[-1](inputs['features'])))
        expected_loss = ce + .0001*norm_sq + anchor + replay
    expected = outer.gradient(expected_loss, model.trainable_variables)
    actual, values = feature_replay_gradients(**inputs)
    assert float(values[2]) == pytest.approx(float(expected_loss), abs=1e-7)
    assert float(values[3]) == pytest.approx(.0001)
    for a, b in zip(actual, expected): np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-7, rtol=1e-6)


def test_beta_zero_matches_existing_loss_and_encoder_has_no_replay_gradient():
    import numpy as np
    from basil_core.noniid_study import anchored_gradients
    model = tiny_model(); inputs = objective_inputs(model)
    base, values = anchored_gradients(model, inputs['x'], inputs['y'], inputs['reference'], mu=.01, sigma_e=.010, ebm=True)
    zero, zvalues = feature_replay_gradients(**dict(inputs, beta=0.))
    one, _ = feature_replay_gradients(**inputs)
    assert float(zvalues[2]) == float(values[2])
    for a, b in zip(base, zero): np.testing.assert_array_equal(a.numpy(), b.numpy())
    np.testing.assert_array_equal(base[0].numpy(), one[0].numpy())
    assert not np.array_equal(base[-1].numpy(), one[-1].numpy())


def test_replay_does_not_change_inputs_reference_or_model():
    import numpy as np
    model = tiny_model(); inputs = objective_inputs(model)
    before = [w.numpy().copy() for w in model.trainable_variables]
    references = [r.numpy().copy() for r in inputs['reference']]
    feature_replay_gradients(**inputs)
    for a, w in zip(before, model.trainable_variables): np.testing.assert_array_equal(a, w.numpy())
    for a, r in zip(references, inputs['reference']): np.testing.assert_array_equal(a, r.numpy())


def tiny_image_worker(beta):
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)),
        tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(3, activation='relu',
            kernel_initializer='ones'), tf.keras.layers.Dense(10, kernel_initializer='zeros')])
    return FeatureReplayWorker(model, beta=beta)


def worker_memory(worker):
    from basil_core.research_protocol import params_hash
    initial = worker.export()
    h, e = params_hash(initial), encoder_hash(initial)
    p = packet_from_features([[.2, .3, .4]], owner=1, round_id=0, receiver=0,
        model_hash=h, encoder_id=e)
    return ActivationFeatureMemory(0, 0, h, e, 3, (1,), (p,))


def test_lambda_has_no_effect_on_loss_gradients_or_update():
    import numpy as np
    import tensorflow as tf
    worker = tiny_image_worker(1.); initial = worker.export()
    for ref, value in zip(worker.reference, initial): ref.assign(value)
    worker.set_memory(worker_memory(worker)); worker.anchor_mu.assign(.01)
    results = []
    for legacy in (1., 999999.):
        worker.load(initial)
        g, v = worker._gradients(tf.ones((2, 32, 32, 3))*.2, tf.zeros(2, tf.int32),
            tf.constant([7, 8]), 'gradient_norm_objective', 'paper_absolute_gaussian',
            tf.constant(.010), tf.constant(legacy))
        tf.keras.optimizers.SGD(.05).apply_gradients(zip(g, worker.model.trainable_variables))
        results.append(([a.numpy() for a in g], [float(a) for a in v], worker.export()))
    for a, b in zip(*results):
        for x, y in zip(a, b): np.testing.assert_array_equal(x, y)


def test_five_synthetic_epochs_keep_reference_and_memory_frozen():
    import numpy as np
    from basil_core.class_registry import ClassRegistry
    from basil_core.research_protocol import params_hash
    worker = tiny_image_worker(1.); initial = worker.export(); m = worker_memory(worker)
    x = np.ones((2, 32, 32, 3), np.float32)*.2; y = np.zeros(2, np.int32)
    selected_loss, _ = worker.probe(initial, x, y)
    trained, record = worker.train(initial, x, y, feature_memory=m,
        local_registry=ClassRegistry(), received_registry=ClassRegistry(), selection_x=x,
        selection_y=y, selected_loss=selected_loss, seed=2025, round_id=0, node_id=0,
        epochs=5, batch_size=512, learning_rate=.05, ebm_mode='none',
        semantics='paper_absolute_gaussian', sigma=0.)
    assert record['steps'] == 5 and len(record['epochs']) == 5
    assert record['snapshotAnchorReferenceHash'] == params_hash(initial)
    assert record['featureMemoryFrozenForActivation']
    assert record['featureMemory']['packetHashes'] == [m.packets[0].digest]
    assert params_hash(trained) != params_hash(initial)
    for step in record['optimizerTelemetry']:
        assert step['featureReplayBeta'] == 1. and step['ebmCoefficient'] == 0.
        assert step['totalObjective'] == pytest.approx(step['baseCrossEntropy'] + step['snapshotAnchorPenalty'] + step['featureReplayPenalty'])
    assert worker.optimizer.momentum == 0 and worker.optimizer.clipnorm is None
    assert worker.optimizer.weight_decay is None


def test_owner_refresh_is_deterministic_and_does_not_mutate_model():
    import numpy as np
    from basil_core.research_protocol import params_hash
    worker = tiny_image_worker(1.)
    x = np.ones((4, 32, 32, 3), np.float32)*.2; y = np.array([0, 0, 1, 1], np.int32)
    assignments = [np.array([0, 1]), np.array([2, 3])] + [np.array([], dtype=np.int64)]*8
    h = params_hash(worker.export())
    a = owner_refresh_memory(worker.model, x, y, assignments, receiver=0, round_id=0, seen_classes={1}, seed=2025)
    b = owner_refresh_memory(worker.model, x, y, assignments, receiver=0, round_id=0, seen_classes={1}, seed=2025)
    assert a == b and a.packets[0].class_id == 1
    assert params_hash(worker.export()) == h
    worker.model.trainable_variables[0].assign_add(np.ones_like(worker.model.trainable_variables[0]))
    with pytest.raises(ValueError, match='exact selected'):
        worker.set_memory(a)


def test_approved_wrapped_model_supports_owner_features_and_new_objective():
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import build_model, params_hash
    model = build_model('basil_paper_cnn', 2025)
    assert sum(int(tf.size(w)) for w in model.trainable_variables) == 117706
    worker = FeatureReplayWorker(model, beta=1.)
    x = np.ones((2, 32, 32, 3), np.float32)*.2
    y = np.ones(2, np.int32)
    pools = [np.array([], dtype=np.int64), np.array([0, 1])] + [np.array([], dtype=np.int64)]*8
    m = owner_refresh_memory(model, x, y, pools, receiver=0, round_id=0, seen_classes={1}, seed=2025)
    assert m.feature_dim == 192 and len(m.packets) == 1
    worker.set_memory(m)
    initial = worker.export()
    for a, p in zip(worker.reference, initial): a.assign(p)
    gradients, values = worker._gradients(tf.constant(x), tf.zeros(2, tf.int32),
        tf.constant([7, 8]), 'none', 'paper_absolute_gaussian', tf.constant(0.), tf.constant(999999.))
    assert len(gradients) == 10 and all(np.isfinite(g.numpy()).all() for g in gradients)
    assert float(values[-1]) == 1. and float(values[3]) == 0.
    assert params_hash(worker.export()) == m.model_hash


def test_default_runner_plan_is_clean_and_does_not_import_ml_libraries():
    import json
    import subprocess
    import sys
    code = "import runpy,sys; sys.argv=['run_cart_feature_diagnostic.py']; " \
        "runpy.run_path('scripts/run_cart_feature_diagnostic.py',run_name='__main__'); " \
        "assert not ({'numpy','tensorflow','matplotlib','wandb'} & sys.modules.keys())"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['trainingRunsStarted'] == 0
    cfg = data['config']
    assert cfg['localEpochs'] == 5 and cfg['samplesPerNode'] == 5000 and cfg['batchSize'] == 512
    assert cfg['featureReplayBeta'] == 1. and cfg['basilMemorySize'] == 5
    assert cfg['actualAttackers'] == [] and not cfg['channelNoise'] and not cfg['ebm']
    assert not cfg['testSetUsed'] and not cfg['liveTrackingEnabled'] and not cfg['researchValid']


def test_report_rejects_incomplete_runs(tmp_path, monkeypatch):
    import json
    from scripts import report_cart_feature_diagnostic as report
    (tmp_path/'run.json').write_text(json.dumps({'status': 'running', 'activationsCompleted': 9}))
    monkeypatch.setattr(report, 'RESULT', tmp_path)
    with pytest.raises(ValueError, match='Incomplete'): report.review()


def test_replay_nonfinite_logits_are_not_converted_to_fake_accuracy():
    import tensorflow as tf
    model = tiny_model(); inputs = objective_inputs(model)
    with pytest.raises(tf.errors.InvalidArgumentError, match='Non-finite'):
        feature_replay_gradients(**dict(inputs, features=tf.fill((2, 2), float('inf'))))
