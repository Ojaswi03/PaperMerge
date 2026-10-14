"""Synthetic projection/transport contracts; never launch research experiments."""
import copy

from basil_core.lazy_imports import lazy_module
np = lazy_module('numpy')
import pytest

from basil_core.gradient_projection_retention import (
    REVISION, GradientProjectionWorker, project_affine_gradient, update_basis,
    validate_state, state_hash, save_memory, load_memory,
)
from basil_core.research_protocol import initialize_memories, broadcast, Snapshot, params_hash
from scripts.run_gradient_projection_gate import validation_split, gate_decision


def test_projection_matches_analytic_including_bias_and_preserves_old_output():
    import tensorflow as tf
    q = np.array([[1.], [1.], [1.]], np.float32)/np.sqrt(3.)
    kernel = tf.constant([[2., 3.], [4., 1.]])
    bias = tf.constant([5., 6.])
    k, b, residual = project_affine_gradient(kernel, bias, q)
    original = np.concatenate([kernel.numpy(), bias.numpy()[None]], axis=0)
    applied = np.concatenate([k.numpy(), b.numpy()[None]], axis=0)
    np.testing.assert_allclose(applied, original-q@(q.T@original), atol=1e-6)
    np.testing.assert_allclose(q.T@applied, 0., atol=1e-6)
    assert float(residual) < 1e-6
    weights = np.arange(6).reshape(3, 2)
    np.testing.assert_allclose(q.T@(weights-.05*applied), q.T@weights, atol=1e-6)
    assert not np.array_equal(b.numpy(), bias.numpy())


def test_empty_basis_is_exact_ce_gradient_and_full_basis_blocks_update():
    import tensorflow as tf
    kernel, bias = tf.ones((2, 3)), tf.ones(3)
    k, b, _ = project_affine_gradient(kernel, bias, np.zeros((3, 0), np.float32))
    np.testing.assert_array_equal(k, kernel); np.testing.assert_array_equal(b, bias)
    k, b, _ = project_affine_gradient(kernel, bias, np.eye(3, dtype=np.float32))
    np.testing.assert_array_equal(k, np.zeros((2, 3))); np.testing.assert_array_equal(b, np.zeros(3))


def test_convolution_patch_layout_and_projection_idempotence():
    import tensorflow as tf
    rng = np.random.default_rng(6)
    kernel = tf.constant(rng.normal(size=(2, 2, 3, 4)).astype(np.float32))
    bias = tf.constant(rng.normal(size=4).astype(np.float32))
    q, _ = np.linalg.qr(rng.normal(size=(13, 5)))
    q = q.astype(np.float32)
    k, b, residual = project_affine_gradient(kernel, bias, q)
    assert tuple(k.shape) == (2, 2, 3, 4) and tuple(b.shape) == (4,)
    matrix = np.concatenate([k.numpy().reshape(12, 4), b.numpy()[None]], axis=0)
    np.testing.assert_allclose(q.T@matrix, 0., atol=1e-6)
    k2, b2, _ = project_affine_gradient(k, b, q)
    np.testing.assert_allclose(k2, k, atol=1e-6); np.testing.assert_allclose(b2, b, atol=1e-6)
    assert float(residual) < 1e-6


@pytest.mark.parametrize('energy', [0., -1., 1.1, np.nan])
def test_invalid_energy_does_not_change_training_rule(energy):
    with pytest.raises(ValueError):
        update_basis(np.zeros((2, 0)), np.eye(2), energy)


def test_svd_equivalent_energy_expansion_keeps_old_span_and_freezes_input():
    old = np.array([[1.], [0.], [0.]], np.float32)
    r = np.diag([3., 2., 1.])
    before = old.copy()
    new, captured = update_basis(old, r)
    assert captured >= .99
    np.testing.assert_array_equal(old, before)
    np.testing.assert_allclose(new@(new.T@old), old, atol=1e-6)
    np.testing.assert_allclose(new.T@new, np.eye(new.shape[1]), atol=1e-6)
    repeated, _ = update_basis(new, r)
    assert repeated.shape == new.shape


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_representations_abort(bad):
    with pytest.raises(ValueError):
        update_basis(np.zeros((2, 0)), np.array([[bad], [1.]]))


def test_nonorthogonal_or_wrong_revision_state_is_rejected():
    with pytest.raises(ValueError):
        validate_state(dict(revision=REVISION, seenClasses=[0], bases=[np.ones((2, 2))]), [2])
    with pytest.raises(ValueError):
        validate_state(dict(revision='other', seenClasses=[], bases=[]), [])


def test_round11_memory_and_projection_state_lossless_checkpoint(tmp_path):
    memories = initialize_memories([np.ones(2, np.float32)])
    state = dict(revision=REVISION, seenClasses=[0], bases=[np.array([[1.], [0.]], np.float32)])
    original = state_hash(state)
    for round_id in range(11):
        for sender in range(10):
            broadcast(memories, Snapshot(sender, round_id, [np.array([round_id, sender], np.float32)], retention_state=state))
    assert [(s.sender_id, s.round_id) for s in memories[0].candidates()] == [(i, 10) for i in range(5, 10)]
    broadcast(memories, Snapshot(0, 11, [np.array([11, 0], np.float32)], retention_state=state))
    expected = [(6, 10), (7, 10), (8, 10), (9, 10), (0, 11)]
    save_memory(tmp_path/'memory.npz', memories, [2])
    restored = load_memory(tmp_path/'memory.npz', [2])
    assert [(s.sender_id, s.round_id) for s in restored[1].candidates()] == expected
    for left, right in zip(memories, restored):
        for a, b in zip(left.candidates(), right.candidates()):
            assert params_hash(a.params) == params_hash(b.params)
            assert state_hash(b.retention_state) == original
    restored[1].snapshots[0].retention_state['bases'][0][:] = 0
    assert state_hash(state) == original


def image_model():
    import tensorflow as tf
    return tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=3))])


def test_lambda_ignored_and_full_five_epoch_parameter_handoff():
    import tensorflow as tf
    worker = GradientProjectionWorker(image_model())
    incoming = [np.full(v.shape, .01, np.float32) for v in worker.model.trainable_variables]
    state = validate_state(None, worker.dimensions)
    worker.receive_retention_state(state)
    frozen = copy.deepcopy(state)
    for variable in worker.model.trainable_variables:
        variable.assign(tf.ones_like(variable)*9.)
    worker.load(incoming)
    assert params_hash(worker.export()) == params_hash(incoming)
    outcomes = []
    for legacy in (1., 999999.):
        grads, values = worker._gradients(tf.ones((2, 32, 32, 3)), tf.zeros(2, tf.int32),
            tf.constant([7, 8]), 'none', 'paper_absolute_gaussian', tf.constant(0.), tf.constant(legacy))
        outcomes.append(([g.numpy() for g in grads], [float(v) for v in values]))
    for a, b in zip(outcomes[0][0], outcomes[1][0]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(outcomes[0][1], outcomes[1][1])
    _, record = worker.train(incoming, np.zeros((5000, 32, 32, 3), np.float32),
        np.zeros(5000, np.int32), seed=2025, round_id=0, node_id=0, epochs=5,
        batch_size=512, learning_rate=.05, ebm_mode='none', semantics='paper_absolute_gaussian', sigma=0.)
    assert record['steps'] == 50
    assert all(e['samplesSeen'] == 5000 and e['batchSizes'] == [512]*9+[392] for e in record['epochs'])
    assert state_hash(state) == state_hash(frozen)
    assert worker.optimizer.momentum == 0 and worker.optimizer.clipnorm is None and worker.optimizer.weight_decay is None
    assert int(worker.optimizer.iterations) == 50
    assert worker.output_state['seenClasses'] == [0]


def test_validation_split_is_training_only_disjoint_complete_deterministic():
    labels = np.repeat(np.arange(10), 5000)
    training, validation = validation_split(labels)
    other_training, other_validation = validation_split(labels)
    assert all(len(t) == 4500 and len(v) == 500 for t, v in zip(training, validation))
    assert set(np.concatenate(training)).isdisjoint(set(np.concatenate(validation)))
    assert np.array_equal(np.sort(np.concatenate(training+validation)), np.arange(50000))
    for node, (a, b, c, d) in enumerate(zip(training, validation, other_training, other_validation)):
        np.testing.assert_array_equal(a, c); np.testing.assert_array_equal(b, d)
        assert np.all(labels[a] == node) and np.all(labels[b] == node)


def test_preserving_constant_class_without_new_learning_does_not_pass_gate():
    ce, projected = [], []
    for node in range(4):
        own = [float(i == node) for i in range(10)]
        constant = [float(i == 0) for i in range(10)]
        ce.append(dict(beforeTraining=dict(perClassValidationAccuracy=constant),
            epochs=[dict(perClassValidationAccuracy=own)]))
        projected.append(dict(beforeTraining=dict(perClassValidationAccuracy=constant),
            epochs=[dict(perClassValidationAccuracy=constant)]))
    decision = gate_decision(dict(ce=ce, projection=projected))
    assert not decision['passed'] and not decision['checks']['allFirstFourClassesLearned']
    assert decision['validationOnly'] and not decision['testSetUsed']


def test_reporting_refuses_incomplete_gate(tmp_path):
    from scripts.report_gradient_projection_gate import review
    (tmp_path/'run.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError, match='partial gate'):
        review(tmp_path)


def test_constant_class_summary_is_json_serializable():
    import json
    from scripts.report_gradient_projection_gate import constant_class_count
    confusion = np.zeros((10, 10), np.int64)
    confusion[:, 1] = 500
    rows = [dict(epochs=[dict(validationConfusion=confusion.tolist())])]
    count = constant_class_count(rows)
    assert type(count) is int and count == 1
    assert json.loads(json.dumps(dict(count=count))) == {'count': 1}


@pytest.mark.parametrize('mode,sigma', [('gradient_norm_objective', 0.), ('none', .01)])
def test_gate_does_not_silently_enable_noise_or_ebm(mode, sigma):
    import tensorflow as tf
    worker = GradientProjectionWorker(image_model())
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        worker._batch_gradients(tf.ones((2, 32, 32, 3)), tf.zeros(2, tf.int32),
            tf.constant([7, 8]), mode, 'paper_absolute_gaussian', tf.constant(sigma), tf.constant(25.))
