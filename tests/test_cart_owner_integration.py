"""Synthetic integration contracts; no CIFAR learning, GUI queue or cloud runs."""
from dataclasses import replace
import json
import subprocess
import sys

import pytest


def probe_fixture():
    import numpy as np
    from basil_core.cart_owner_protocol import training_probes
    from basil_core.research_protocol import Snapshot
    labels = np.repeat(np.arange(10), 5000)
    probes = training_probes(labels, {0: [0, 1], 1: [5000, 5001]})
    candidates = [Snapshot(i, 2, [np.array([i], np.float32)], sender_loss=-1e30,
                           sender_accuracy=1.) for i in range(5, 10)]
    return labels, probes, candidates


def test_balanced_selection_not_single_class_or_sender_telemetry():
    from basil_core.cart_owner_protocol import select_balanced_snapshot, FULL_NODE
    _, probes, candidates = probe_fixture()
    seen = []
    def evaluate(params, owner, indices):
        sender = int(params[0][0])
        seen.append((sender, owner, indices))
        return (.1 if owner == 0 else 9.) if sender == 5 else (1. if sender == 9 else 3.)
    selected, records, own_loss = select_balanced_snapshot(0, candidates, probes, evaluate,
        helper_scope=FULL_NODE, available_owners=(0, 1))
    assert selected.sender_id == 9 and own_loss == 1.
    assert records[0]['balancedTrainingLoss'] == pytest.approx(4.55)
    assert len(seen) == 10
    for owner in (0, 1):
        ids = [entry[2] for entry in seen if entry[1] == owner]
        assert len(ids) == 5 and all(value is ids[0] for value in ids)


def test_balanced_selection_tie_is_nearest_predecessor():
    from basil_core.cart_owner_protocol import select_balanced_snapshot, FULL_NODE
    _, probes, candidates = probe_fixture()
    selected, _, _ = select_balanced_snapshot(0, candidates, probes, lambda *args: 2.,
        helper_scope=FULL_NODE, available_owners=(0, 1))
    assert selected.sender_id == 9


@pytest.mark.parametrize('split', ['test', 'validation'])
def test_nontraining_selection_probes_rejected(split):
    from basil_core.cart_owner_protocol import training_probes
    labels, _, _ = probe_fixture()
    with pytest.raises(ValueError, match='training-side'):
        training_probes(labels, {0: [0]}, source_split=split)


@pytest.mark.parametrize('indices', [[50000], [0, 0], [5000], [-1], [0.5]])
def test_invalid_selection_training_provenance_rejected(indices):
    from basil_core.cart_owner_protocol import training_probes
    labels, _, _ = probe_fixture()
    with pytest.raises(ValueError, match='owner-local'):
        training_probes(labels, {0: indices})


def test_missing_owner_and_nonfinite_selection_rejected():
    from basil_core.cart_owner_protocol import select_balanced_snapshot, FULL_NODE
    _, probes, candidates = probe_fixture()
    with pytest.raises(ValueError, match='coverage'):
        select_balanced_snapshot(0, candidates, probes[:1], lambda *args: 1.,
            helper_scope=FULL_NODE, available_owners=(0, 1))
    with pytest.raises(FloatingPointError, match='selection CE'):
        select_balanced_snapshot(0, candidates, probes, lambda *args: float('nan'),
            helper_scope=FULL_NODE, available_owners=(0, 1))


def test_selection_rejects_mutating_evaluator():
    from basil_core.cart_owner_protocol import select_balanced_snapshot, FULL_NODE
    _, probes, candidates = probe_fixture()
    def evaluate(params, *_):
        params[0][0] += 1
        return 1.
    with pytest.raises(ValueError, match='changed snapshot'):
        select_balanced_snapshot(0, candidates, probes, evaluate,
            helper_scope=FULL_NODE, available_owners=(0, 1))


def test_full_node_threats_fail_closed_not_oracle_filtering():
    from basil_core.cart_owner_protocol import require_helper_boundary, FULL_NODE, TRUSTED_AUXILIARY
    with pytest.raises(ValueError, match='defense is unresolved'):
        require_helper_boundary(FULL_NODE, [0, 1, 5, 7])
    assert not require_helper_boundary(TRUSTED_AUXILIARY, [0, 1, 5, 7])['helperByzantineDefenseVerified']
    _, probes, candidates = probe_fixture()
    from basil_core.cart_owner_protocol import select_balanced_snapshot
    with pytest.raises(ValueError, match='blocked'):
        select_balanced_snapshot(0, candidates, probes, lambda *args: 1., helper_scope=FULL_NODE,
            attacker_ids=[0, 1, 5, 7], available_owners=(0, 1))


def numerical_fixture():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_gradients import compute_owner_response
    model = tf.keras.Sequential([tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(3, activation='tanh', kernel_initializer=tf.keras.initializers.Constant(.3)),
        tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.RandomUniform(-.2, .2, seed=3))])
    params = [w.numpy().copy() for w in model.trainable_variables]
    local = np.array([[1., .2], [.7, .1]], np.float32)
    old = np.array([[.2, 1.], [.1, .7]], np.float32)
    response = compute_owner_response(model, params, old, np.ones(2, np.int32), np.array([5000, 5001]),
        owner=1, receiver=0, round_id=0, optimizer_step=0)
    return model, params, local, old, response


@pytest.mark.parametrize('ebm', [False, True])
def test_combined_objective_matches_independent_nested_tape(ebm):
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_ebm import owner_objective_gradients
    model, params, local, old, response = numerical_fixture()
    reference = [tf.Variable(p+.001) for p in params]
    reference_hash = [a.numpy().copy() for a in reference]
    applied, values, metadata = owner_objective_gradients(model, local, np.zeros(2, np.int32),
        reference, [response], receiver=0, round_id=0, optimizer_step=0,
        mu=.01, sigma_e=.01, ebm=ebm)
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[0, 0], logits=model(local)))
        base = inner.gradient(ce, model.trainable_variables)
        old_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[1, 1], logits=model(old)))
        total = (ce+old_ce)/2 + (.0001 if ebm else 0.)*tf.add_n([tf.reduce_sum(g*g) for g in base])
        total += .01/2*tf.add_n([tf.reduce_sum((w-a)**2) for w, a in zip(model.trainable_variables, reference)])
    expected = outer.gradient(total, model.trainable_variables)
    for a, b in zip(applied, expected):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-5, atol=1e-7)
    assert float(values[2]) == pytest.approx(float(total), rel=1e-6)
    assert metadata['ebmCoefficient'] == pytest.approx(.0001 if ebm else 0.)
    assert metadata['ebmInnerObjective'] == 'receiver_local_ce'
    for a, p in zip(reference, reference_hash):
        np.testing.assert_array_equal(a.numpy(), p)
    for w, p in zip(model.trainable_variables, params):
        np.testing.assert_array_equal(w.numpy(), p)


def test_no_lambda_loss_gradients_and_one_sgd_update_identical():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_ebm import owner_objective_gradients
    model, params, local, _, response = numerical_fixture()
    outputs = []
    for multiplier in (1., 999999.):
        for w, p in zip(model.trainable_variables, params):
            w.assign(p)
        gradients, values, _ = owner_objective_gradients(model, local, np.zeros(2, np.int32), params,
            [response], receiver=0, round_id=0, optimizer_step=0, mu=.01, sigma_e=.01,
            ebm=True, legacy_lambda=multiplier)
        tf.keras.optimizers.SGD(.05, momentum=0.).apply_gradients(zip(gradients, model.trainable_variables))
        outputs.append(([g.numpy() for g in gradients], [float(v) for v in values],
                        [w.numpy() for w in model.trainable_variables]))
    for a, b in zip(outputs[0][0], outputs[1][0]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    for a, b in zip(outputs[0][2], outputs[1][2]):
        np.testing.assert_array_equal(a, b)


def test_local_ebm_matches_analytic_softmax_hessian_not_scalar_scale():
    import numpy as np
    import tensorflow as tf
    from basil_core.cart_owner_ebm import local_ebm_terms
    model = tf.keras.Sequential([tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(3, use_bias=False, kernel_initializer=tf.keras.initializers.Constant([[.2, -.1, .4]]))])
    x = tf.constant([[1.]])
    base, correction, values = local_ebm_terms(model, x, tf.constant([1]), .2, enabled=True)
    p = tf.nn.softmax(model(x))[0].numpy().astype(np.float64)
    gradient = p-np.array([0., 1., 0.])
    hessian = np.diag(p)-np.outer(p, p)
    expected = 2*.2**2*hessian@gradient
    np.testing.assert_allclose(correction[0].numpy()[0], expected, rtol=1e-5, atol=1e-7)
    assert not np.allclose(expected, .2**2*gradient)
    assert float(values[2]) == pytest.approx(.2**2*np.dot(gradient, gradient), rel=1e-6)


def test_stale_helper_and_nonfinite_objective_rejected():
    import numpy as np
    from basil_core.cart_owner_ebm import owner_objective_gradients
    model, params, local, _, response = numerical_fixture()
    context = dict(receiver=0, round_id=0, optimizer_step=0, mu=.01, sigma_e=.01, ebm=True)
    with pytest.raises(ValueError, match='Stale'):
        owner_objective_gradients(model, local, np.zeros(2, np.int32), params,
            [replace(response, model_hash='a'*64)], **context)
    with pytest.raises(ValueError, match='Finite'):
        owner_objective_gradients(model, local, np.zeros(2, np.int32), params,
            [response], **dict(context, sigma_e=float('inf')))


def test_finite_poison_is_not_authenticated_by_a_hash():
    import numpy as np
    from basil_core.cart_owner_gradients import aggregate_balanced_owner_gradients
    model, params, local, _, response = numerical_fixture()
    from basil_core.cart_owner_gradients import compute_owner_response
    own = compute_owner_response(model, params, local, np.zeros(2, np.int32), np.array([0, 1]),
        owner=0, receiver=0, round_id=0, optimizer_step=0)
    poison = replace(response, gradients=tuple(-100*g for g in response.gradients))
    context = dict(receiver=0, round_id=0, optimizer_step=0, available_classes=(0, 1))
    clean, _ = aggregate_balanced_owner_gradients(params, [own, response], **context)
    bad, _ = aggregate_balanced_owner_gradients(params, [own, poison], **context)
    assert any(not np.allclose(a, b) for a, b in zip(clean, bad))
    # Correct context/hash + finite values do NOT make malicious gradients safe.
    from basil_core.cart_owner_protocol import require_helper_boundary, FULL_NODE
    with pytest.raises(ValueError, match='defense is unresolved'):
        require_helper_boundary(FULL_NODE, [1])


def test_attack_before_absolute_noise_links_keyed_and_clean_weights_unchanged():
    import numpy as np
    from basil_core.cart_owner_protocol import snapshot_links
    from basil_core.research_protocol import params_hash, keyed_rng, apply_channel_noise
    from basil_core.attacks import hiddenAttack
    params = [np.ones(4096, np.float32)*.2]
    before = params_hash(params)
    context = dict(seed=2025, sender=0, attacker_ids=[0, 1, 5, 7], sigma_e=.01)
    warm = snapshot_links(params, round_id=19, **context)
    assert all(not r['attackActive'] and r['attackedHash'] == before for r in warm)
    active = snapshot_links(params, round_id=20, **context)
    expected_attack = hiddenAttack(params, rng=keyed_rng(2025, 'hidden_attack', 20, 0))
    for link in active:
        expected, _ = apply_channel_noise(expected_attack, semantics='paper_absolute_gaussian', sigma=.01,
            rng=keyed_rng(2025, 'channel_noise', 20, 0, link['receiver']))
        np.testing.assert_array_equal(link['params'][0], expected[0])
        assert link['attackActive'] and link['attackedHash'] == params_hash(expected_attack)
        assert link['noise']['effectiveCoordinateSigma'] == .01
    assert len({r['receivedHash'] for r in active}) == 5
    paired = snapshot_links(params, round_id=20, **context)
    assert [r['receivedHash'] for r in paired] == [r['receivedHash'] for r in active]
    clean = snapshot_links(params, round_id=20, **dict(context, sigma_e=0.))
    assert all(r['receivedHash'] == r['attackedHash'] for r in clean)
    assert params_hash(params) == before


def test_worker_factory_blocks_full_node_byzantine_use_before_initialization():
    from basil_core.cart_owner_ebm import make_owner_ebm_worker
    from basil_core.cart_owner_protocol import FULL_NODE
    with pytest.raises(ValueError, match='defense is unresolved'):
        make_owner_ebm_worker(None, None, helper_scope=FULL_NODE, attacker_ids=[0, 1, 5, 7])


@pytest.mark.parametrize('ebm', [False, True])
def test_actual_worker_path_matches_augmented_class_mean_and_local_ebm(ebm):
    import numpy as np
    import tensorflow as tf
    from tests.test_cart_owner_gradient_worker import synthetic_worker
    from basil_core.cart_owner_ebm import make_owner_ebm_worker
    from basil_core.cart_owner_protocol import FULL_NODE
    from basil_core.research_audit import augment_batch
    old = synthetic_worker()
    worker = make_owner_ebm_worker(old.model, old.helper_model, helper_scope=FULL_NODE)
    for name in ('source_x', 'source_y', 'owner_pools', 'older_classes', 'receiver', 'seed', 'round_id'):
        setattr(worker, name, getattr(old, name))
    for a, reference in zip(worker.reference, old.reference):
        a.assign(reference)
    worker.anchor_mu.assign(.007)
    x, y, key = tf.constant(worker.source_x[:3]), tf.ones(3, tf.int32), tf.constant([7, 8])
    gradients, values = worker._gradients(x, y, key,
        'gradient_norm_objective' if ebm else 'none', 'paper_absolute_gaussian',
        tf.constant(.01), tf.constant(999999.))
    response = worker.owner_records[0]['owners'][0]
    local = augment_batch(x, key)
    helper = augment_batch(worker.source_x[response['trainingIndices']], response['augmentationKey'])
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=worker.model(local)))
        base = inner.gradient(ce, worker.model.trainable_variables)
        old_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros(16, tf.int32),
                                                                           logits=worker.model(helper)))
        total = (ce+old_ce)/2 + (.0001 if ebm else 0.)*tf.add_n([tf.reduce_sum(g*g) for g in base])
        total += .007/2*tf.add_n([tf.reduce_sum((w-a)**2) for w, a in zip(worker.model.trainable_variables, worker.reference)])
    expected = outer.gradient(total, worker.model.trainable_variables)
    for a, b in zip(gradients, expected):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-5, atol=1e-7)
    assert float(values[2]) == pytest.approx(float(total), rel=1e-6)
    assert worker.owner_records[0]['legacyLambdaIgnored']
    assert worker.owner_records[0]['ownerOptimizerSteps'] == 0


def test_approved_cnn_combined_objective_matches_independent_gradient():
    import numpy as np
    import tensorflow as tf
    from basil_core.research_protocol import build_model, params_hash
    from basil_core.cart_owner_gradients import compute_owner_response
    from basil_core.cart_owner_ebm import owner_objective_gradients
    model = build_model('basil_paper_cnn', 2025)
    params = [w.numpy().copy() for w in model.trainable_variables]
    assert sum(p.size for p in params) == 117706
    images = np.random.default_rng(19).uniform(0, 1, (18, 32, 32, 3)).astype(np.float32)
    response = compute_owner_response(model, params, images[2:], np.ones(16, np.int32), np.arange(5000, 5016),
        owner=1, receiver=0, round_id=0, optimizer_step=0)
    reference = [p+.001 for p in params]
    applied, values, _ = owner_objective_gradients(model, images[:2], np.zeros(2, np.int32), reference,
        [response], receiver=0, round_id=0, optimizer_step=0, mu=.01, sigma_e=.01, ebm=True)
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[0, 0], logits=model(images[:2])))
        local_gradient = inner.gradient(ce, model.trainable_variables)
        old_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones(16, tf.int32), logits=model(images[2:])))
        total = (ce+old_ce)/2 + .0001*tf.add_n([tf.reduce_sum(g*g) for g in local_gradient])
        total += .01/2*tf.add_n([tf.reduce_sum((w-a)**2) for w, a in zip(model.trainable_variables, reference)])
    expected = outer.gradient(total, model.trainable_variables)
    for a, b in zip(applied, expected):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-3, atol=2e-6)
    assert float(values[2]) == pytest.approx(float(total), rel=1e-6)
    assert params_hash([w.numpy() for w in model.trainable_variables]) == params_hash(params)


def test_default_plan_lazy_idle_nonproduction_and_contrasts():
    code = 'from scripts.check_cart_owner_integration import main;main();import sys;assert not ({"numpy","tensorflow","wandb","matplotlib"}&sys.modules.keys())'
    process = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    plan = json.loads(process.stdout)
    assert plan['trainingRunsStarted'] == 0 and not plan['researchReady']
    configs = plan['conditions']
    assert [c['condition'] for c in configs] == list('ABCD')
    for c in configs:
        assert c['localEpochs'] == 5 and c['batchSize'] == 512 and c['basilMemorySize'] == 5
        assert c['nRounds'] is None and not c['researchValid'] and not c['queueStarted']
        assert c['helperChannelNoise'] is False and c['blockers']
    assert configs[2]['ebmMode'] == 'none' and configs[3]['ebmMode'] == 'gradient_norm_objective'
    assert configs[2]['channelNoiseSigmaAbsolute'] == configs[3]['channelNoiseSigmaAbsolute'] == .01
    diff = {k for k in configs[2] if configs[2][k] != configs[3][k]}
    assert diff == {'condition', 'ebmMode'}


def test_new_modules_import_without_heavy_libraries():
    code = 'import basil_core.cart_owner_ebm,basil_core.cart_owner_protocol;import sys;assert not ({"numpy","tensorflow","wandb","matplotlib"}&sys.modules.keys())'
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
