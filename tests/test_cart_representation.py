"""No CIFAR runs: exact gradients, teacher state, audit isolation and gates."""
import pytest

from basil_core.cart_representation import (FrozenSelectedEncoder,
    RepresentationReplayWorker, representation_replay_gradients,
    preservation_gradients, same_image_audit, calibrate_gamma)


def tiny_model():
    import tensorflow as tf
    return tf.keras.Sequential([tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(3, activation='tanh', kernel_initializer=tf.keras.initializers.Constant(.3)),
        tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.Constant(.2))])


def inputs():
    import tensorflow as tf
    model = tiny_model()
    reference = [tf.identity(w) for w in model.trainable_variables]
    teacher = FrozenSelectedEncoder(model); teacher.freeze([w.numpy() for w in model.trainable_variables])
    model.trainable_variables[0].assign_add(tf.ones_like(model.trainable_variables[0])*.02)
    return dict(model=model, selected_encoder=teacher,
        x=tf.constant([[.2, .5], [.3, .1]]), y=tf.constant([0, 0]),
        reference=reference, features=tf.constant([[.2, .3, .1], [.1, .3, .4]]),
        feature_labels=tf.constant([1, 1]), row_weights=tf.constant([.5, .5]),
        mu=.01, sigma_e=.01, ebm=True, beta=1., gamma=.02)


def test_full_objective_gradient_and_ebm_only_local_ce():
    import numpy as np
    import tensorflow as tf
    data = inputs(); model = data['model']; teacher = data['selected_encoder']
    target = tf.stop_gradient(teacher.teacher(data['x']))
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=data['y'], logits=model(data['x'])))
        base = inner.gradient(ce, model.trainable_variables)
        ebm = .0001*tf.add_n([tf.reduce_sum(g*g) for g in base])
        anchor = .005*tf.add_n([tf.reduce_sum((w-a)**2) for w, a in zip(model.trainable_variables, data['reference'])])
        replay = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=data['feature_labels'], logits=model.layers[-1](data['features'])))
        preservation = .02*tf.reduce_mean(tf.reduce_sum((teacher.student(data['x'])-target)**2, axis=1))
        loss = ce+ebm+anchor+replay+preservation
    expected = outer.gradient(loss, model.trainable_variables)
    actual, values = representation_replay_gradients(**data)
    assert float(values[2]) == pytest.approx(float(loss), rel=1e-6)
    assert float(values[3]) == pytest.approx(.0001)
    for a, b in zip(actual, expected): np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-7, rtol=1e-5)
    teacher.assert_frozen()


def test_preservation_matches_finite_difference_and_never_updates_head():
    import numpy as np
    data = inputs(); model, teacher = data['model'], data['selected_encoder']
    grads, values = preservation_gradients(model, teacher, data['x'], gamma=.02)
    w = model.trainable_variables[0]; initial = w.numpy().copy(); delta = 1e-3
    high = initial.copy(); high[0, 0] += delta; w.assign(high)
    _, a = preservation_gradients(model, teacher, data['x'], gamma=.02)
    low = initial.copy(); low[0, 0] -= delta; w.assign(low)
    _, b = preservation_gradients(model, teacher, data['x'], gamma=.02)
    w.assign(initial)
    assert float(grads[0][0, 0]) == pytest.approx((float(a[1])-float(b[1]))/(2*delta), rel=2e-3, abs=1e-7)
    assert float(values[2]) > 0
    for g in grads[-2:]: np.testing.assert_array_equal(g.numpy(), np.zeros_like(g))


def test_preservation_gradient_step_reduces_same_batch_drift():
    import tensorflow as tf
    data = inputs(); model, teacher = data['model'], data['selected_encoder']
    grads, before = preservation_gradients(model, teacher, data['x'], gamma=.02)
    tf.keras.optimizers.SGD(.05).apply_gradients(zip(grads, model.trainable_variables))
    _, after = preservation_gradients(model, teacher, data['x'], gamma=.02)
    assert float(after[0]) < float(before[0])
    teacher.assert_frozen()


def test_nonfinite_target_aborts_instead_of_producing_fake_measurements():
    import tensorflow as tf
    data = inputs()
    with pytest.raises(tf.errors.InvalidArgumentError, match='Non-finite selected-encoder target'):
        preservation_gradients(data['model'], data['selected_encoder'],
            tf.constant([[float('nan'), .2]]), gamma=.02)


def test_gamma_zero_exact_feature_replay_control_and_zero_handoff_penalty():
    import numpy as np
    from basil_core.cart_feature_memory import feature_replay_gradients
    data = inputs()
    plain = {k: v for k, v in data.items() if k not in {'selected_encoder', 'gamma'}}
    a, av = feature_replay_gradients(**plain)
    b, bv = representation_replay_gradients(**dict(data, gamma=0.))
    for u, v in zip(a, b): np.testing.assert_array_equal(u.numpy(), v.numpy())
    assert float(av[2]) == float(bv[2])
    for w, p in zip(data['model'].trainable_variables, data['reference']): w.assign(p)
    _, values = preservation_gradients(data['model'], data['selected_encoder'], data['x'], gamma=.02)
    assert float(values[0]) == 0. and float(values[2]) == 0.


def test_teacher_rejects_uninitialized_wrong_shape_and_mutation():
    import tensorflow as tf
    model = tiny_model(); teacher = FrozenSelectedEncoder(model)
    with pytest.raises(ValueError, match='not synchronized'): teacher.assert_frozen()
    with pytest.raises(ValueError): teacher.freeze([w.numpy() for w in model.trainable_variables][:-1])
    teacher.freeze([w.numpy() for w in model.trainable_variables])
    teacher.teacher.weights[0].assign_add(tf.ones_like(teacher.teacher.weights[0]))
    with pytest.raises(ValueError, match='changed'): teacher.assert_frozen()


@pytest.mark.parametrize('gamma', [-1., float('nan'), float('inf')])
def test_bad_gamma_rejected(gamma):
    with pytest.raises(ValueError): RepresentationReplayWorker(tiny_model(), beta=1., gamma=gamma)


def test_calibration_is_deterministic_training_only_and_restores_weights():
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import params_hash
    model = tiny_model()
    # Break the equal-logit initializer symmetry so CE reaches the encoder.
    model.layers[-1].kernel.assign(tf.reshape(tf.range(30, dtype=tf.float32)/30, (3, 10)))
    teacher = FrozenSelectedEncoder(model); initial = [w.numpy() for w in model.trainable_variables]
    teacher.freeze(initial)
    x, y = tf.constant([[.2, .5], [.3, .1]]), tf.constant([0, 0])
    a = calibrate_gamma(model, teacher, x, y)
    b = calibrate_gamma(model, teacher, x, y)
    assert a == b and a['gamma'] > 0
    assert a['gamma']*a['unitPreservationGradientNorm']/a['baseGradientNorm'] == pytest.approx(.1)
    assert a['trainingOnly'] and not a['testSetUsed']
    assert params_hash(initial) == params_hash([w.numpy() for w in model.trainable_variables])
    teacher.assert_frozen()


def test_calibration_failure_restores_state_and_does_not_invent_coefficient():
    import tensorflow as tf
    from basil_core.research_protocol import params_hash
    model = tiny_model()
    # Exactly zero encoder gradient, rather than near-zero cancellation of
    # equal float32 head weights (which is legitimately nonzero numerically).
    model.layers[-1].kernel.assign(tf.zeros_like(model.layers[-1].kernel))
    initial = [w.numpy() for w in model.trainable_variables]
    teacher = FrozenSelectedEncoder(model); teacher.freeze(initial)
    with pytest.raises(ValueError, match='Unavailable calibration'):
        calibrate_gamma(model, teacher, tf.zeros((2, 2)), tf.constant([0, 0]))
    assert params_hash(initial) == params_hash([w.numpy() for w in model.trainable_variables])


def test_same_image_audit_preserves_model_identity_and_balanced_confusions():
    import numpy as np
    from basil_core.research_protocol import params_hash
    data = inputs(); model = data['model']; initial = [r.numpy() for r in data['reference']]
    x = np.ones((20, 2), np.float32)*.2; y = np.repeat(np.arange(10), 2); ids = np.arange(20)
    before = params_hash([w.numpy() for w in model.trainable_variables])
    result = same_image_audit(model, data['selected_encoder'], x, y, ids, initial, batch_size=3)
    assert result['trainingOnly'] and result['analysisOnly'] and not result['heldOut']
    assert result['currentParameterHash'] == before
    assert params_hash([w.numpy() for w in model.trainable_variables]) == before
    assert result['perClassDrift'][0]['meanSquaredFeatureDrift'] > 0
    for values in result['interventions'].values():
        assert values['classSupport'] == [2]*10
        assert np.asarray(values['confusion']).sum() == 20
    with pytest.raises(ValueError): same_image_audit(model, data['selected_encoder'], x, y, np.zeros(20, int), initial)
    with pytest.raises(ValueError): same_image_audit(model, data['selected_encoder'], x, np.zeros(20, int), ids, initial)


def tiny_worker():
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)),
        tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(3, activation='relu', kernel_initializer='ones'),
        tf.keras.layers.Dense(10, kernel_initializer='zeros')])
    return RepresentationReplayWorker(model, beta=1., gamma=.01)


def test_five_epochs_teacher_reference_memory_and_optimizer_stay_correct():
    import numpy as np
    from tests.test_cart_feature_memory import worker_memory
    from basil_core.class_registry import ClassRegistry
    worker = tiny_worker(); initial = worker.export(); memory = worker_memory(worker)
    x = np.ones((2, 32, 32, 3), np.float32)*.2; y = np.zeros(2, np.int32)
    loss, _ = worker.probe(initial, x, y)
    trained, record = worker.train(initial, x, y, feature_memory=memory,
        local_registry=ClassRegistry(), received_registry=ClassRegistry(), selection_x=x,
        selection_y=y, selected_loss=loss, seed=2025, round_id=0, node_id=0,
        epochs=5, batch_size=512, learning_rate=.05, ebm_mode='none',
        semantics='paper_absolute_gaussian', sigma=0.)
    assert record['steps'] == 5 and record['selectedTeacherFrozenForActivation']
    assert record['selectedTeacherEncoderHash'] == memory.encoder_hash
    assert record['featureMemory']['packetHashes'] == [memory.packets[0].digest]
    for step in record['optimizerTelemetry']:
        assert step['totalObjective'] == pytest.approx(step['baseCrossEntropy']+
            step['snapshotAnchorPenalty']+step['featureReplayPenalty']+step['representationPenalty'])
    assert worker.optimizer.momentum == 0 and worker.optimizer.clipnorm is None
    assert worker.optimizer.weight_decay is None
    worker.selected_encoder.assert_frozen()


def test_new_worker_lambda_has_zero_effect_on_gradients_loss_and_update():
    import numpy as np
    import tensorflow as tf
    from tests.test_cart_feature_memory import worker_memory
    worker = tiny_worker(); initial = worker.export()
    worker.set_memory(worker_memory(worker)); worker.selected_encoder.freeze(initial)
    for r, p in zip(worker.reference, initial): r.assign(p)
    results = []
    for legacy in (1., 999999.):
        worker.load(initial)
        worker.model.trainable_variables[0].assign_add(tf.ones_like(worker.model.trainable_variables[0])*.001)
        grads, values = worker._gradients(tf.ones((2, 32, 32, 3))*.2, tf.zeros(2, tf.int32),
            tf.constant([7, 8]), 'gradient_norm_objective', 'paper_absolute_gaussian', tf.constant(.010), tf.constant(legacy))
        tf.keras.optimizers.SGD(.05).apply_gradients(zip(grads, worker.model.trainable_variables))
        results.append(([g.numpy() for g in grads], [float(v) for v in values], worker.export()))
    for a, b in zip(*results):
        for u, v in zip(a, b): np.testing.assert_array_equal(u, v)


def test_plan_and_unresolved_gamma_execution_load_no_ml_dependencies():
    import json
    import subprocess
    import sys
    code = "import runpy,sys; sys.argv=['run_cart_representation_diagnostic.py']; " \
        "runpy.run_path('scripts/run_cart_representation_diagnostic.py',run_name='__main__'); " \
        "assert not ({'numpy','tensorflow','matplotlib','wandb'} & sys.modules.keys())"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


    value = json.loads(result.stdout)
    assert value['trainingRunsStarted'] == 0 and value['config']['representationGamma'] is None
    assert value['config']['localEpochs'] == 5 and value['config']['snapshotSelection']
    assert not value['config']['testSetUsed'] and not value['config']['queueStarted']
    code = "import runpy,sys; sys.argv=['run_cart_representation_diagnostic.py','--execute']; " \
        "\ntry:runpy.run_path('scripts/run_cart_representation_diagnostic.py',run_name='__main__')" \
        "\nexcept SystemExit as e:assert e.code==2" \
        "\nassert not ({'numpy','tensorflow','wandb'} & sys.modules.keys())"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_report_rejects_incomplete_learning_without_fabricating_rows(tmp_path, monkeypatch):
    import json
    from scripts import report_cart_representation_diagnostic as report
    monkeypatch.setattr(report, 'RESULT', tmp_path)
    (tmp_path/'run.json').write_text(json.dumps(dict(status='running', activationsCompleted=3, optimizerSteps=150)))
    with pytest.raises(ValueError, match='Incomplete learning'):
        report.review_learning()


def test_30_round_config_is_paired_clean_full_data_and_routes_separately():
    from scripts.run_cart_representation_diagnostic import config, output_paths, RESULT
    one = config('GPU', ratio=.1)
    thirty = config('GPU', ratio=.1, rounds=30)
    assert one['nRounds'] == 1 and not one['testSetUsed']
    assert thirty['nRounds'] == 30 and thirty['testSetUsed']
    assert thirty['representationCalibrationRatio'] == .1
    assert thirty['featureReplayBeta'] == 1 and thirty['cartMuMax'] == .01
    assert thirty['localEpochs'] == 5 and thirty['samplesPerNode'] == 5000
    assert not thirty['researchValid'] and not thirty['liveTrackingEnabled']
    assert not thirty['channelNoise'] and not thirty['ebm'] and thirty['actualAttackers'] == []
    result, plots = output_paths(30)
    assert result != RESULT and 'newResults/nonIID/' in str(result)
    assert 'newPlots/nonIID/' in str(plots) and '30_rounds' in str(plots)
    with pytest.raises(ValueError): output_paths(500)


def test_30_round_ring_contract_rolls_and_round_lr_is_not_activation_lr():
    import numpy as np
    from basil_core.research_protocol import initialize_memories, Snapshot, broadcast, basil_round_learning_rate
    from scripts.run_strict_cart_diagnostic import expected_candidates
    memories = initialize_memories([np.zeros(2, np.float32)])
    for r in range(30):
        for node in range(10):
            assert [(s.sender_id, s.round_id) for s in memories[node].candidates()] == expected_candidates(r, node)
            broadcast(memories, Snapshot(node, r, [np.ones(2, np.float32)*(r*10+node)]))
        assert basil_round_learning_rate(.05, r) == pytest.approx(.05/(1+.05*r))
    assert [(s.sender_id, s.round_id) for s in memories[0].candidates()] == [(s, 29) for s in range(5, 10)]


def test_30_round_dry_plan_never_imports_ml_or_starts_training():
    import json
    import subprocess
    import sys
    code = "import runpy,sys; sys.argv=['run_cart_representation_diagnostic.py','--rounds','30','--calibrate-ratio','0.1']; " \
        "runpy.run_path('scripts/run_cart_representation_diagnostic.py',run_name='__main__'); " \
        "assert not ({'numpy','tensorflow','matplotlib','wandb'} & sys.modules.keys())"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['trainingRunsStarted'] == 0 and value['config']['nRounds'] == 30


def test_round_csv_matches_measured_node9_mean_worst_loss_and_classes():
    from scripts.report_cart_representation_diagnostic import validate_round_metrics
    states = [dict(round=0, node=n, globalTestEvaluationOnly=dict(
        accuracy=n/10, crossEntropy=2+n, perClassAccuracy=[n/10]*10)) for n in range(10)]
    row = dict(round='0', full_test_accuracy='0.9', mean_node_accuracy='0.45',
        worst_node_accuracy='0', full_test_loss='11',
        **{f'class_{c}_accuracy': '0.9' for c in range(10)})
    validate_round_metrics(states, [row], 1)
    for key in ('full_test_accuracy', 'mean_node_accuracy', 'worst_node_accuracy', 'full_test_loss', 'class_3_accuracy'):
        wrong = {**row, key: '999'}
        with pytest.raises(ValueError, match='differs from measured activation evidence'):
            validate_round_metrics(states, [wrong], 1)
    with pytest.raises(ValueError, match='Missing/duplicate'):
        validate_round_metrics(states, [row, row], 1)
    with pytest.raises(ValueError, match='ten measured node states'):
        validate_round_metrics(states[:-1], [row], 1)


def test_fixed_gamma_reporting_checks_exact_float32_not_json_float64():
    import numpy as np
    from scripts.report_cart_representation_diagnostic import validate_fixed_gamma
    configured = .7140174035127002
    measured = float(np.float32(configured))
    assert configured != measured
    validate_fixed_gamma(configured, measured)
    with pytest.raises(ValueError, match='Gamma adapted'):
        validate_fixed_gamma(configured, float(np.nextafter(np.float32(measured), np.float32(1.))))
    with pytest.raises(ValueError, match='Gamma adapted'):
        validate_fixed_gamma(configured, float('nan'))
