"""Mathematical and non-IID protocol gates; no full experiment is launched."""
import csv
import gzip
import json
import pytest

from basil_core.noniid_study import anchored_gradients, SnapshotAnchorWorker, study_configs, verify_study, partition_audit
from basil_core.research_protocol import params_hash, gradient_norm_objective_gradients, validate_config


def toy_model():
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(2, use_bias=False)])
    model.trainable_variables[0].assign([[.4, -.2]])
    return model


@pytest.mark.parametrize('ebm', [False, True])
def test_full_gradient_matches_hessian_and_frozen_anchor(ebm):
    from basil_core.trainer import lossFn
    import numpy as np
    import tensorflow as tf
    model = toy_model()
    x, y = tf.constant([[1.], [2.]]), tf.constant([0, 1])
    reference = [tf.constant([[-.1, .3]])]
    mu, sigma = .01, .01
    variable = model.trainable_variables[0]
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            ce = lossFn(y, model(x))
        base = inner.gradient(ce, variable)
    hessian = outer.jacobian(base, variable).numpy().reshape(2, 2)
    grads, values = anchored_gradients(model, x, y, reference, mu=mu, sigma_e=sigma, ebm=ebm)
    drift = variable.numpy() - reference[0].numpy()
    expected = base.numpy().reshape(-1) + mu*drift.reshape(-1)
    if ebm:
        expected += 2*sigma**2*hessian @ base.numpy().reshape(-1)
    np.testing.assert_allclose(grads[0].numpy().reshape(-1), expected, rtol=1e-5, atol=1e-6)
    expected_loss = float(ce) + (sigma**2*float(tf.reduce_sum(base**2)) if ebm else 0.) + mu/2*np.sum(drift**2)
    assert float(values[2]) == pytest.approx(expected_loss, rel=1e-6)
    assert float(values[9]) == pytest.approx(mu/2*np.sum(drift**2), rel=1e-6)


def test_zero_anchor_matches_source_ebm_without_lambda():
    import numpy as np
    import tensorflow as tf
    model = toy_model()
    x, y = tf.constant([[1.], [2.]]), tf.constant([0, 1])
    reference = [tf.identity(w) for w in model.trainable_variables]
    source, values = gradient_norm_objective_gradients(model, x, y, .0001, details=True)
    anchored, combined = anchored_gradients(model, x, y, reference, mu=0., sigma_e=.01, ebm=True)
    np.testing.assert_array_equal(source[0].numpy(), anchored[0].numpy())
    assert float(values[2]) == float(combined[2])


def test_predeclared_mu_and_paired_conditions_disable_cloud():
    configs = study_configs(.01)
    assert verify_study(configs)['C/D'] == ['ebmMode', 'localObjective']
    for cfg in configs:
        assert cfg['nRounds'] == 50 and cfg['localEpochs'] == 5 and cfg['batchSize'] == 512
        assert cfg['partitionStrategy'] == 'one_class_per_node' and cfg['snapshotSelection']
        assert cfg['basilMemorySize'] == 5 and cfg['snapshotAnchorMu'] == .01
        assert not cfg['liveTrackingEnabled']
        assert cfg['attackType'] == 'delayed_hidden_parameter_attack_v1'
        assert cfg['attackStartRound'] == 20
        assert cfg['resultDirectory'].startswith('newResults/nonIID/')
    assert configs[0]['resolvedAttackerIds'] == []
    assert all(c['resolvedAttackerIds'] == [0,1,5,7] for c in configs[1:])
    configs[2]['snapshotAnchorMu'] = .02
    with pytest.raises(ValueError, match='Unpaired'):
        verify_study(configs)


def test_one_class_full_training_coverage_and_separate_test_counts():
    import numpy as np
    labels = np.repeat(np.arange(10), 5000)
    test_labels = np.repeat(np.arange(10), 1000)
    audit, indices = partition_audit(labels, test_labels)
    assert audit['assignedSamples'] == audit['uniqueSamples'] == 50000
    for node, chunk in enumerate(indices):
        assert len(chunk) == 5000 and np.all(labels[chunk] == node)
    assert audit['testClassCounts'] == [1000]*10
    assert partition_audit(labels, test_labels)[0]['partitionHash'] == audit['partitionHash']


@pytest.mark.parametrize('mu', [-1., float('nan'), float('inf')])
def test_bad_mu_is_rejected(mu):
    with pytest.raises(ValueError): study_configs(mu)


def test_all_five_epochs_keep_reference_and_include_partial_batch():
    import numpy as np
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Input((32,32,3)), tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(10)])
    worker = SnapshotAnchorWorker(model, mu=.01)
    incoming = worker.export()
    x = np.zeros((5000,32,32,3), np.float32)
    y = np.zeros(5000, np.int32)
    _, record = worker.train(incoming, x, y, seed=2025, round_id=0, node_id=0,
        epochs=5, batch_size=512, learning_rate=.05, ebm_mode='none', semantics='paper_absolute_gaussian', sigma=0.)
    assert record['steps'] == 50
    assert all(e['batchSizes'] == [512]*9+[392] and e['samplesSeen'] == 5000 for e in record['epochs'])
    assert params_hash([r.numpy() for r in worker.reference]) == params_hash(incoming)
    assert record['snapshotAnchorReferenceHash'] == params_hash(incoming)
    assert record['optimizerTelemetry'][0]['snapshotDistanceSquared'] == 0.
    assert record['optimizerTelemetry'][-1]['snapshotDistanceSquared'] > 0.


def test_legacy_lambda_cannot_change_new_gradient_or_update():
    import numpy as np
    import tensorflow as tf
    x = tf.zeros((2,32,32,3)); y = tf.constant([0,1]); key = tf.constant([7,8])
    def create():
        model = tf.keras.Sequential([tf.keras.layers.Input((32,32,3)), tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, kernel_initializer='ones', bias_initializer='zeros')])
        worker = SnapshotAnchorWorker(model, mu=.01)
        for a, w in zip(worker.reference, model.trainable_variables): a.assign(w)
        return worker
    workers = [create(), create()]
    outcomes = []
    for worker, legacy in zip(workers, [1.,999999.]):
        grads, values = worker._gradients(x, y, key, 'gradient_norm_objective', 'paper_absolute_gaussian', tf.constant(.01), tf.constant(legacy))
        tf.keras.optimizers.SGD(.05).apply_gradients(zip(grads,worker.model.trainable_variables))
        outcomes.append((grads, values, worker.export()))
    for first, second in zip(outcomes[0], outcomes[1]):
        for a, b in zip(first, second): np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


def test_iid_still_rejects_anchor():
    from basil_core.iid_campaign import campaign_configs
    cfg = campaign_configs('a'*64,'b'*64)[0]
    cfg['snapshotAnchorMu'] = .01
    with pytest.raises(ValueError, match='No anchor'): validate_config(cfg)


def test_completion_review_separates_correctness_from_learning(tmp_path):
    import numpy as np
    from reporting.noniid_study_plots import review_run
    cfg=study_configs()[0]
    metadata=dict(status='completed',config=cfg,completedRounds=50,completedActivations=500,
        legacyLambdaIgnored=True,wandbEnabled=False,runtimeSeconds=1.,partitionHash='partition',initialModelHash='initial')
    (tmp_path/'run.json').write_text(json.dumps(metadata))
    states=np.broadcast_to(np.eye(10,dtype=np.float32),(50,10,10)).copy()
    np.savez(tmp_path/'metrics.npz',roundNodePerClassAccuracy=states,averageAccuracy=np.full(50,.1),
        fullTestAccuracy=np.full(50,.1),worstNodeAccuracy=np.full(50,.1))
    step=dict(baseCrossEntropy=.2,gradientNormSquared=1.,ebmObjective=.21,ebmPenalty=0.,
        snapshotAnchorPenalty=.01,snapshotDistanceSquared=2.,baseGradientNorm=1.,
        robustObjectiveGradientNorm=1.,ebmCorrectionNorm=0.,anchorGradientNorm=.01,
        anchorCorrectionRatio=.01,ebmCorrectionRatio=0.,ebmCoefficient=0.)
    record=dict(epochs=[dict(samplesSeen=5000,batchSizes=[512]*9+[392]) for _ in range(5)],
        optimizerSteps=50,snapshotAnchorReferenceHash='input',inputHash='input',trainedOutputHash='changed',
        optimizerTelemetry=[step]*50,meanOtherClassForgetting=.1,selectedSenderByzantine=False,inputSnapshotRound=0)
    records=[]
    for r in range(50):
        for node in range(10):
            senders=[(node-distance)%10 for distance in range(5,0,-1)]
            records.append({**record,'round':r,'nodeId':node,'memorySenderIds':senders,
                'memorySnapshotRounds':[r if s<node else r-1 for s in senders],
                'selectedCandidateLoss':.2,'candidates':[{'receiverLocalLoss':.2}],
                'attackActive':False,'channelNoiseActive':False,
                'outgoingLinks':[{'coordinateSigma':0.,'noiseL2Norm':0.}]*5})
    with gzip.open(tmp_path/'activation_telemetry.json.gz','wt') as handle: json.dump(records,handle)
    with (tmp_path/'round_metrics.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['round']);writer.writeheader();writer.writerows([{'round':r} for r in range(50)])
    review=review_run(tmp_path)
    assert review['technicalPass'] and review['constantOwnClassCollapse']
    assert review['finalMean']==pytest.approx(.1)
    assert review['lateChangePp']==0.
    metadata['completedRounds']=49
    (tmp_path/'run.json').write_text(json.dumps(metadata))
    assert not review_run(tmp_path)['technicalPass']


def test_preparation_does_not_start_worker_or_tracking(monkeypatch, capsys):
    from scripts import run_noniid_basil_study as runner
    import sys
    monkeypatch.setattr(sys,'argv',['run_noniid_basil_study.py'])
    def forbidden(*args, **kwargs): raise AssertionError('Dry-run launched execution')
    monkeypatch.setattr(runner,'execute',forbidden)
    monkeypatch.setattr(runner.subprocess,'run',forbidden)
    runner.main()
    payload=json.loads(capsys.readouterr().out)
    assert payload['executionStarted'] is False and payload['wandbEnabled'] is False
