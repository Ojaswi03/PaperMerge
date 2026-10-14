"""Synthetic contracts only: no CIFAR research run or queue wiring."""
from dataclasses import replace

import pytest


def model_and_responses():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_gradients import compute_owner_response
    model = tf.keras.Sequential([tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(3, activation='tanh', kernel_initializer=tf.keras.initializers.Constant(.3)),
        tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.RandomUniform(-.2, .2, seed=3))])
    params = [w.numpy().copy() for w in model.trainable_variables]
    batches = [(np.array([[1., .1], [.7, .2]], np.float32), np.array([0, 0], np.int32)),
               (np.array([[.1, 1.], [.2, .7]], np.float32), np.array([1, 1], np.int32))]
    responses = [compute_owner_response(model, params, x, y, np.array([2*c, 2*c+1]),
        owner=c, receiver=0, round_id=0, optimizer_step=0) for c, (x, y) in enumerate(batches)]
    return model, params, batches, responses


def test_existing_replay_groups_overweight_current_class_vs_each_old_class():
    from basil_core.cart_owner_gradients import feature_supervision_prior
    prior = feature_supervision_prior(9, range(9), beta=1.)
    assert prior[9] == .5
    assert all(prior[c] == pytest.approx(1/18) for c in range(9))
    assert prior[9]/prior[0] == 9
    assert sum(prior.values()) == pytest.approx(1.)
    assert feature_supervision_prior(0, (), beta=1.) == {0: 1.}


def test_stationary_prior_matches_autodiff_constant_logits_not_accuracy_claim():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_gradients import feature_supervision_prior
    prior = feature_supervision_prior(9, range(9), beta=1.)
    logits = tf.Variable(np.log([prior[c] for c in range(10)]).astype(np.float32))
    with tf.GradientTape() as tape:
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[9], logits=logits[None, :])[0]
        old = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.range(9), logits=tf.repeat(logits[None, :], 9, axis=0)))
        loss = ce+old
    np.testing.assert_allclose(tape.gradient(loss, logits).numpy(), 0., atol=1e-7)
    with tf.GradientTape() as tape:
        balanced = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.range(10), logits=tf.repeat(logits[None, :], 10, axis=0)))
    assert float(tf.linalg.norm(tape.gradient(balanced, logits))) > .1


def test_full_owner_aggregate_matches_gradient_of_class_mean_ce_and_keeps_weights():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_gradients import aggregate_balanced_owner_gradients
    model, params, batches, responses = model_and_responses()
    actual, metadata = aggregate_balanced_owner_gradients(params, responses,
        receiver=0, round_id=0, optimizer_step=0, available_classes=(0, 1))
    with tf.GradientTape() as tape:
        loss = tf.add_n([tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=model(x))) for x, y in batches])/2
    for measured, reference in zip(actual, tape.gradient(loss, model.trainable_variables)):
        np.testing.assert_allclose(measured, reference.numpy(), rtol=1e-5, atol=1e-7)
    assert metadata['crossEntropy'] == pytest.approx(float(loss), rel=1e-6)
    assert metadata['classWeights'] == {'0': .5, '1': .5}
    assert metadata['ownerOptimizerSteps'] == 0 and not metadata['rawImagesReturned']
    assert not metadata['modelAveraging']
    for w, p in zip(model.trainable_variables, params): np.testing.assert_array_equal(w.numpy(), p)


def test_old_class_response_reaches_encoder_unlike_detached_feature_replay():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_feature_memory import feature_replay_gradients
    model, params, batches, responses = model_and_responses()
    assert sum(float(np.sum(g*g)) for g in responses[1].gradients[:-2]) > 0
    x, y = batches[0]
    encoder = tf.keras.Model(model.inputs, model.layers[-2].output)
    features = encoder(batches[1][0])
    args = dict(model=model, x=x, y=y, reference=params, features=features,
        feature_labels=batches[1][1], row_weights=tf.constant([.5, .5]),
        mu=0., sigma_e=0., ebm=False)
    off, _ = feature_replay_gradients(**args, beta=0.)
    on, _ = feature_replay_gradients(**args, beta=1.)
    for a, b in zip(off[:-2], on[:-2]): np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_response_arrays_are_immutable_copies_and_order_does_not_change_aggregate():
    import numpy as np
    from basil_core.cart_owner_gradients import aggregate_balanced_owner_gradients, OwnerGradientResponse
    _, params, _, responses = model_and_responses()
    mutable = [g.copy() for g in responses[0].gradients]
    copied = replace(responses[0], gradients=tuple(mutable))
    mutable[0][:] = 999
    np.testing.assert_array_equal(copied.gradients[0], responses[0].gradients[0])
    with pytest.raises(ValueError): copied.gradients[0].setflags(write=True)
    context = dict(receiver=0, round_id=0, optimizer_step=0, available_classes=(0, 1))
    a, _ = aggregate_balanced_owner_gradients(params, responses, **context)
    b, _ = aggregate_balanced_owner_gradients(params, reversed(responses), **context)
    for x, y in zip(a, b): np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('field,value', [('model_hash', 'a'*64), ('round_id', 1),
    ('optimizer_step', 1), ('receiver', 1)])
def test_stale_or_wrong_context_responses_rejected(field, value):
    from basil_core.cart_owner_gradients import aggregate_balanced_owner_gradients
    _, params, _, responses = model_and_responses()
    responses[1] = replace(responses[1], **{field: value})
    with pytest.raises(ValueError, match='Stale'):
        aggregate_balanced_owner_gradients(params, responses, receiver=0,
            round_id=0, optimizer_step=0, available_classes=(0, 1))


def test_missing_duplicate_wrong_shape_nonfinite_and_nontraining_rejected():
    import numpy as np
    from basil_core.cart_owner_gradients import aggregate_balanced_owner_gradients
    _, params, _, responses = model_and_responses()
    context = dict(receiver=0, round_id=0, optimizer_step=0, available_classes=(0, 1))
    for invalid in (responses[:1], (responses[0], responses[0])):
        with pytest.raises(ValueError, match='Missing'):
            aggregate_balanced_owner_gradients(params, invalid, **context)
    with pytest.raises(ValueError, match='layout'):
        aggregate_balanced_owner_gradients(params, (responses[0], replace(responses[1],
            gradients=(np.zeros(1, np.float32),))), **context)
    with pytest.raises(ValueError, match='Finite'):
        replace(responses[1], gradients=(np.array([float('nan')], np.float32),))
    with pytest.raises(ValueError, match='Test/validation'):
        replace(responses[1], data_usage='test')


def test_owner_helper_rejects_wrong_labels_test_indices_and_unsynchronized_state():
    import numpy as np
    from basil_core.cart_owner_gradients import compute_owner_response
    model, params, batches, _ = model_and_responses()
    x, y = batches[0]
    context = dict(owner=0, receiver=0, round_id=0, optimizer_step=0)
    for ids, labels in ((np.array([0, 50000]), y), (np.array([0, 0]), y),
                        (np.array([0, 1]), np.array([1, 1]))):
        with pytest.raises(ValueError, match='owner-local'):
            compute_owner_response(model, params, x, labels, ids, **context)
    with pytest.raises(ValueError, match='Test/validation'):
        compute_owner_response(model, params, x, y, np.array([0, 1]), data_usage='test', **context)
    model.trainable_variables[0].assign_add(np.ones_like(params[0]))
    with pytest.raises(ValueError, match='synchronized'):
        compute_owner_response(model, params, x, y, np.array([0, 1]), **context)


def test_module_import_is_lazy_and_cannot_launch_training():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        "import sys,basil_core.cart_owner_gradients; assert not ({'tensorflow','numpy','wandb','matplotlib'} & sys.modules.keys())"],
        text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_class_balance_is_not_sample_weighting_when_helper_batch_is_smaller():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_gradients import compute_owner_response, aggregate_balanced_owner_gradients
    model, params, batches, _ = model_and_responses()
    small_x, small_y = batches[1][0][:1], batches[1][1][:1]
    local_x, local_y = np.tile(batches[0][0], (3, 1)), np.tile(batches[0][1], 3)
    responses = [compute_owner_response(model, params, x, y, np.arange(len(y))+10*c,
        owner=c, receiver=0, round_id=0, optimizer_step=0)
        for c, (x, y) in enumerate(((local_x, local_y), (small_x, small_y)))]
    measured, _ = aggregate_balanced_owner_gradients(params, responses,
        receiver=0, round_id=0, optimizer_step=0, available_classes=(0, 1))
    with tf.GradientTape() as tape:
        losses = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=model(x))) for x, y in ((local_x, local_y), (small_x, small_y))]
        class_mean = (losses[0]+losses[1])/2
    expected = tape.gradient(class_mean, model.trainable_variables)
    for a, b in zip(measured, expected): np.testing.assert_allclose(a, b.numpy(), atol=1e-7, rtol=1e-5)
    with tf.GradientTape() as tape:
        sample_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=np.concatenate((local_y, small_y)), logits=model(np.concatenate((local_x, small_x)))))
    sample_gradients = tape.gradient(sample_mean, model.trainable_variables)
    assert any(not np.allclose(a, b.numpy(), atol=1e-7) for a, b in zip(measured, sample_gradients))
