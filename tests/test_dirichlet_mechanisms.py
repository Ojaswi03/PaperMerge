"""Tiny analytical and mocked recovery checks; no CIFAR training or cloud runs."""
import json
from types import SimpleNamespace

import pytest

from scripts.analyze_dirichlet_mechanisms import selected_ancestry


def test_selected_ancestry_telescope_includes_attack_and_skipped_parent():
    # Scalar states: initial=2; train0=3; attacked0=-1; receive1=1;
    # train1=4; receive2=5; train2=6; node3 chooses node0, skipping 1/2.
    def record(node, parent, received, trained, outbound, input_hash):
        return dict(round=0, nodeId=node, inputSourceNode=parent,
            inputSnapshotRound=-1 if parent < 0 else 0, inputHash=input_hash,
            candidates=[dict(senderId=parent, snapshotRound=-1 if parent < 0 else 0,
                             receivedModelNorm=abs(received))],
            modelL2Norm=abs(trained), attackActive=node == 0,
            outgoingLinks=[dict(receiverId=r, modelL2Norm=abs(outbound),
                                transmittedHash=f'from{node}to{r}') for r in range(node+1, 4)])
    records = [record(0, -1, 2, 3, -1, 'initial'),
               record(1, 0, 1, 4, 4, 'from0to1'),
               record(2, 1, 5, 6, 6, 'from1to2'),
               record(3, 0, 2, 7, 7, 'from0to3')]
    result = selected_ancestry(records)
    assert result[2] == dict(initialEnergy=4, attackEnergy=-8, channelEnergy=9,
                            trainingEnergy=31, links=2, attackedLinks=1, closureResidual=0)
    assert result[3]['links'] == 1
    assert result[3]['trainingEnergy'] == 50
    records[2]['inputHash'] = 'wrong'
    with pytest.raises(ValueError, match='hash mismatch'):
        selected_ancestry(records)


@pytest.mark.parametrize('candidate_count',[0,2])
def test_ancestry_refuses_missing_or_duplicate_selected_candidate(candidate_count):
    record=dict(round=0,nodeId=0,inputSourceNode=9,inputSnapshotRound=-1,modelL2Norm=1.,
        candidates=[dict(senderId=9,snapshotRound=-1,receivedModelNorm=1.)]*candidate_count)
    with pytest.raises(ValueError,match='Exactly one selected candidate'):
        selected_ancestry([record])


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf'),1e200])
def test_ancestry_rejects_invalid_parameter_norms(value):
    record=dict(round=0,nodeId=0,inputSourceNode=9,inputSnapshotRound=-1,modelL2Norm=1.,
        candidates=[dict(senderId=9,snapshotRound=-1,receivedModelNorm=value)])
    with pytest.raises(ValueError,match='finite, nonnegative'):
        selected_ancestry([record])


def test_archive_integrity_is_required_but_current_source_can_differ(tmp_path):
    import hashlib
    from scripts.analyze_dirichlet_mechanisms import verify_source_archives
    archive=tmp_path/'run/source_snapshots';archive.mkdir(parents=True)
    original=b'# executed source\n';digest=hashlib.sha256(original).hexdigest()
    (archive/digest).write_bytes(original)
    current=tmp_path/'current';current.mkdir()
    (current/'fixture.py').write_text('# repaired source\n')
    fingerprints={'fixture.py':digest}
    assert verify_source_archives(archive.parent,fingerprints,current)=={'fixture.py':False}
    (archive/digest).write_bytes(b'corrupt executed source')
    with pytest.raises(ValueError,match='archive hash mismatch'):
        verify_source_archives(archive.parent,fingerprints,current)


def test_raw_and_prior_adjusted_ce_can_reverse_candidate_rankings():
    import numpy as np
    from basil_core.dirichlet_calibration import calibration_offsets
    counts = np.array([90, 10] + [0]*8)
    labels = np.repeat(np.arange(10), counts)
    a = np.array([np.log(.9), np.log(.1)] + [-30.]*8)
    b = np.array([np.log(.5), np.log(.5)] + [-30.]*8)
    def ce(logits):
        return np.logaddexp.reduce(logits) - logits[labels].mean()
    offsets = calibration_offsets(counts, method='prior_adjustment')
    assert ce(a) < ce(b)
    assert ce(a-offsets) > ce(b-offsets)


def test_expected_noisy_ce_has_curvature_term_not_gradient_norm_identity():
    import numpy as np
    # Empirical binary CE with 90/10 labels: at w=log(9), g=0, H=.09.
    w, sigma = np.log(9.), .001
    points, weights = np.polynomial.hermite.hermgauss(32)
    def loss(value):
        return .9*np.logaddexp(0., -value) + .1*np.logaddexp(0., value)
    expected = weights @ loss(w+np.sqrt(2)*sigma*points) / np.sqrt(np.pi)
    base = loss(w)
    assert (expected-base)/sigma**2 == pytest.approx(.09/2, rel=1e-6)
    assert not np.isclose((expected-base)/sigma**2, 0.)


def test_minibatch_gradient_norm_penalizes_gradient_variance_too():
    import numpy as np
    # Two equally probable singleton gradients cancel in the full-data mean.
    gradients = np.array([[1., 0.], [-1., 0.]])
    g = gradients.mean(axis=0)
    mean_squared_norm = np.mean(np.sum(gradients**2, axis=1))
    variance_trace = np.mean(np.sum((gradients-g)**2, axis=1))
    assert mean_squared_norm == g@g + variance_trace == 1.
    assert g@g == 0.


def test_received_anchor_has_zero_restoring_gradient_for_incoming_noise():
    import numpy as np
    import tensorflow as tf
    from basil_core.dirichlet_calibration import calibrated_anchor_gradients
    # Balanced constant logits give zero classification gradient even when the
    # shared bias was perturbed. The received anchor cannot identify that noise.
    net = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(10)])
    net.trainable_variables[0].assign(tf.zeros_like(net.trainable_variables[0]))
    net.trainable_variables[1].assign(tf.fill([10], .2))
    reference = [tf.identity(w) for w in net.trainable_variables]
    gradients, values = calibrated_anchor_gradients(net, tf.zeros((10, 1)),
        tf.range(10), reference, np.zeros(10, np.float32), mu=.01, coefficient=.0001)
    assert float(values[8]) == float(values[10]) == 0.
    for gradient in gradients:
        np.testing.assert_allclose(gradient.numpy(), 0., atol=1e-7)


@pytest.mark.parametrize('resuming', [False, True])
def test_tracking_init_failure_preserves_work_and_allows_verified_retry(monkeypatch, tmp_path, resuming):
    """Run fresh/resumed initialization failures twice without scientific work.

    Execute the real runner up to mocked W&B init, with synthetic checkpoint
    metadata. Dataset loading, scientific gates and hardware setup are mocked;
    entry into the protocol engine is forbidden. All writes stay in tmp_path.
    """
    import numpy as np
    import basil_core.dirichlet_diagnosis as diagnosis
    import basil_core.dirichlet_horizon_study as horizon
    import basil_core.dirichlet_study as study
    import basil_core.iid_runtime as runtime
    import basil_core.live_tracking as tracking
    import basil_core.memory_safety as memory
    import basil_core.protocol_checkpoint as checkpoint
    import basil_core.research_protocol as protocol
    import scripts.run_dirichlet_calibration as runner

    cfg = horizon.configurations('a'*64, 'b'*64, 'c'*64, 500)[0]
    cfg['executionDevice'] = 'CPU'  # Synthetic fixture only; no production gate bypass.
    directory = tmp_path/cfg['resultDirectory']
    directory.mkdir(parents=True)
    checkpoint_path = directory/'checkpoint.npz'
    if resuming:checkpoint_path.write_bytes(b'synthetic checkpoint sentinel')
    csv_path = directory/'round_metrics.csv'
    if resuming:csv_path.write_text('saved CSV sentinel\n')
    previous = dict(status='interrupted' if resuming else 'tracking_unavailable',
        config=cfg, implementationSha256={},
        dataSha256={k:'data' for k in ('trainImages','trainLabels','testImages','testLabels')},
        numericalExecution={}, runtimeSeconds=1.,
        completedActivations=123 if resuming else 0, completedRounds=12 if resuming else 0)
    if resuming:
        previous['liveTracking']=dict(runId='original',entity=cfg['wandbEntity'],
            project=cfg['wandbProject'],group='original-group',lastLoggedRound=9,lastLoggedActivation=99)
    (directory/'run.json').write_text(json.dumps(previous))
    state = dict(telemetry=[{} for _ in range(123)], next_activation=123, completed_rounds=12)
    original_plan = horizon.recovery_plan
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    monkeypatch.setattr(horizon, 'validate_evidence', lambda _:None)
    monkeypatch.setattr(horizon, 'execution_fingerprints', lambda _:{})
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runtime, 'prepare_environment', lambda _:None)
    monkeypatch.setattr(runtime, 'configure_tensorflow', lambda *_:None)
    monkeypatch.setattr(runtime, 'runtime_metadata', lambda *_:dict(numericalExecution={}))
    monkeypatch.setattr(memory, 'check_host_memory', lambda:None)
    monkeypatch.setattr(memory, 'RuntimeMemoryGuard', lambda *_:None)
    monkeypatch.setattr(study, 'partition_audit', lambda *_:(dict(partitionHash='a'*64), []))
    monkeypatch.setattr(diagnosis, 'array_identity_hash', lambda _:'data')
    monkeypatch.setattr(checkpoint, 'checkpoint_implementation', lambda _:{})
    monkeypatch.setattr(checkpoint, 'load_checkpoint', lambda *_a, **_k:state)
    monkeypatch.setattr(protocol, 'build_model', lambda *_:SimpleNamespace(get_weights=lambda:[]))
    monkeypatch.setattr(protocol, 'params_hash', lambda _:'b'*64)
    def forbidden(*_a, **_k):
        raise AssertionError('Scientific execution must never be reached')
    monkeypatch.setattr(protocol, 'run_research_protocol', forbidden)
    def fake_load(path, **_):
        return SimpleNamespace(shape=(10000 if 'test' in path.name else 50000, 32, 32, 3))
    monkeypatch.setattr(np, 'load', fake_load)
    # Keep git/subprocess access out of this synthetic runner invocation.
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *_a, **_k:'synthetic')
    attempts=[]
    def offline(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError('synthetic offline')
    sdk = SimpleNamespace(util=SimpleNamespace(generate_id=lambda:'synthetic'),
                          Settings=lambda **kw:kw, init=offline)
    actual_tracker = tracking.LiveTracker
    monkeypatch.setattr(tracking, 'LiveTracker', lambda cfg, manifest, **kw:
                        actual_tracker(cfg, manifest, sdk=sdk, **kw))
    # run_pair installs signal handlers; restore them when the test exits.
    monkeypatch.setattr(runner.signal, 'signal', lambda *_:None)
    for _ in range(2):
        with pytest.raises(ValueError, match='could not initialize'):
            runner.run_pair('CPU', recover=True, stage='threat', configs_override=[cfg],
                            output_root=directory.parent)
        after = json.loads((directory/'run.json').read_text())
        assert after['status'] == ('interrupted' if resuming else 'tracking_unavailable')
        assert after['startupFailure'] is not resuming
        assert after['trackingInitializationFailed']
        assert after['completedActivations'] == (123 if resuming else 0)
        assert after['completedRounds'] == (12 if resuming else 0)
        plan=original_plan(directory, cfg)
        assert plan['mode'] == ('checkpoint' if resuming else 'startup_retry')
        assert plan['startActivation'] == (123 if resuming else 0)
        if resuming:
            assert checkpoint_path.read_bytes() == b'synthetic checkpoint sentinel'
            assert csv_path.read_text() == 'saved CSV sentinel\n'
            assert after['liveTracking']['lastLoggedRound'] == 9
            assert after['liveTracking']['lastLoggedActivation'] == 99
        else:
            assert not checkpoint_path.exists() and not csv_path.exists()
    assert [a['id'] for a in attempts] == (['original']*2 if resuming else ['synthetic']*2)
    assert attempts[-1]['resume'] == 'allow'
