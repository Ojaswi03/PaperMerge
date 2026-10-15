"""Zero-update saved-state diagnosis; never a new training protocol."""
import copy
import subprocess
import sys

import pytest

from basil_core.dirichlet_diagnosis import actually_corrupted, confusion_summary, summarize_activations


def fixture_records():
    cfg = dict(nRounds=1, attackHidden=True, resolvedAttackerIds=[0, 1, 5, 7], attackStartRound=20)
    partition = dict(nodes=[dict(samples=3, classCounts=[3] + [0]*9) for _ in range(10)])
    records = []
    for node in range(10):
        senders = [(node - distance) % 10 for distance in range(5, 0, -1)]
        records.append(dict(round=0, nodeId=node,
            candidates=[dict(senderId=s, snapshotRound=0 if s < node else -1,
                             receiverLocalLoss=2.) for s in senders],
            inputSourceNode=(node-1)%10, inputSnapshotRound=0 if node else -1,
            inputHash='selected', snapshotAnchorReferenceHash='selected',
            beforeTraining=dict(globalTestAccuracy=.1, perClassGlobalTestAccuracy=[.1]*10),
            globalTestAccuracy=.2, perClassGlobalTestAccuracy=[.2]*10,
            optimizerSteps=5,
            epochs=[dict(epoch=i+1, samplesSeen=3, batchSizes=[3]) for i in range(5)],
            optimizerTelemetry=[dict(anchorCorrectionRatio=.01, ebmCorrectionRatio=0.) for _ in range(5)]))
    return records, partition, cfg


def test_saved_history_contracts_and_signed_forgetting():
    records, partition, cfg = fixture_records()
    original = copy.deepcopy(records)
    result = summarize_activations(records, partition, cfg)
    assert result['checkedActivations'] == 10 and result['optimizerSteps'] == 50
    assert result['actuallyCorruptedSelections'] == 0
    assert result['nodes'][0]['meanAbsentClassForgettingPp'] == pytest.approx(-10.)
    assert records == original


def test_streamed_identity_matches_executed_array_hash():
    import numpy as np
    from basil_core.research_audit import array_hash
    from basil_core.dirichlet_diagnosis import array_identity_hash
    for array in [np.arange(100).reshape(10, 10), np.arange(100).reshape(10, 10)[:, ::2],
                  np.array(3.), np.array([], dtype=np.float32)]:
        assert array_identity_hash(array, chunk_rows=3) == array_hash(array)
    with pytest.raises(ValueError): array_identity_hash(np.ones(2), chunk_rows=0)


@pytest.mark.parametrize('defect', ['memory', 'selection', 'anchor', 'epoch', 'batch', 'steps', 'finite', 'order'])
def test_audit_rejects_invalid_saved_evidence(defect):
    records, partition, cfg = fixture_records()
    first = records[0]
    if defect == 'memory': first['candidates'][0]['snapshotRound'] = 4
    if defect == 'selection': first['inputSourceNode'] = 5
    if defect == 'anchor': first['snapshotAnchorReferenceHash'] = 'wrong'
    if defect == 'epoch': first['epochs'].pop()
    if defect == 'batch': first['epochs'][0]['batchSizes'] = [2]
    if defect == 'steps': first['optimizerSteps'] = 4
    if defect == 'finite': first['optimizerTelemetry'][0]['anchorCorrectionRatio'] = float('inf')
    if defect == 'order': first['round'] = 1
    with pytest.raises(ValueError): summarize_activations(records, partition, cfg)


def test_attacker_designation_is_not_corruption_during_warmup():
    _, _, cfg = fixture_records()
    assert not actually_corrupted(0, 19, cfg)
    assert actually_corrupted(0, 20, cfg)
    assert not actually_corrupted(2, 20, cfg)
    assert not actually_corrupted(0, 20, dict(cfg, attackHidden=False))


def test_constant_class_prediction_not_inferred_from_accuracy_alone():
    import numpy as np
    collapsed = np.zeros((10, 10), np.int64); collapsed[:, 2] = 1000
    summary = confusion_summary(collapsed)
    assert summary['accuracy'] == .1 and summary['constantPredictionClass'] == 2
    assert summary['predictedCounts'][2] == 10000
    mixed = np.full((10, 10), 100, np.int64)
    assert confusion_summary(mixed)['accuracy'] == .1
    assert confusion_summary(mixed)['constantPredictionClass'] is None
    with pytest.raises(ValueError): confusion_summary(np.full((10, 10), float('nan')))


def test_retained_hidden_dispatch_is_negative_scale_not_gaussian_replacement():
    import numpy as np
    from basil_core.attacks import applyAttack
    class ZeroNoise:
        def normal(self, mean, std, size): return np.zeros(size)
    weights = [np.array([-2., 0., 3.], np.float32)]
    result = applyAttack(weights, 'hidden', rng=ZeroNoise())
    np.testing.assert_allclose(result[0], -.32 * weights[0], atol=1e-7)
    np.testing.assert_array_equal(weights[0], [-2., 0., 3.])


def test_local_loss_can_prefer_uniform_model_to_confident_wrong_model():
    import numpy as np
    from basil_core.forgetting_audit import classification_metrics
    from basil_core.research_protocol import Snapshot, select_snapshot
    y = np.full(10, 1, np.int32)
    wrong = np.tile([10.] + [0.]*9, (10, 1)).astype(np.float32)
    uniform = np.zeros_like(wrong)
    def evaluate(params):
        metrics = classification_metrics(params[0], y)
        return metrics['crossEntropy'], metrics['accuracy']
    selected, _ = select_snapshot(2, [Snapshot(1, 20, [wrong]), Snapshot(0, 20, [uniform])], evaluate)
    assert selected.sender_id == 0  # Faithful CE selection does not certify useful features.


def test_probe_restores_model_and_detects_zero_hidden_features():
    import numpy as np
    import tensorflow as tf
    from basil_core.models import BasilPaperCifarModel
    from basil_core.research_protocol import params_hash
    from basil_core.dirichlet_diagnosis import probe_saved_parameters
    model = BasilPaperCifarModel(seed=1)
    original_hash = params_hash(model.get_weights())
    params = [np.zeros_like(v) for v in model.get_weights()]
    probe = probe_saved_parameters(model, params, np.zeros((2, 32, 32, 3), np.float32), np.array([1, 2], np.int32))
    assert probe['optimizerSteps'] == 0
    assert all(layer['zeroFraction'] == 1. for layer in probe['hiddenActivations'])
    assert all(gradient['l2'] == 0. for gradient in probe['gradients'][:-1])
    assert probe['gradients'][-1]['l2'] > 0.
    assert params_hash(model.get_weights()) == original_hash
    assert all(not np.any(p) for p in params)


def test_full_local_metrics_count_every_partial_batch_and_restore_parameters():
    import numpy as np
    import tensorflow as tf
    from basil_core.models import BasilPaperCifarModel
    from basil_core.dirichlet_diagnosis import full_local_training_metrics
    from basil_core.research_protocol import params_hash
    model = BasilPaperCifarModel(seed=1)
    original = params_hash(model.get_weights())
    params = [np.zeros_like(v) for v in model.get_weights()]
    params[-1][2] = 1.
    x = np.zeros((6, 32, 32, 3), np.float32)
    y = np.array([0, 0, 1, 2, 2, 2], np.int32)
    metrics = full_local_training_metrics(model, params, x, y, np.arange(6), batch_size=4)
    assert metrics['samples'] == 6 and metrics['accuracy'] == .5 and metrics['optimizerSteps'] == 0
    assert metrics['predictedCounts'][2] == 6
    assert params_hash(model.get_weights()) == original
    with pytest.raises(ValueError): full_local_training_metrics(model, params, x, y, np.array([0, 0]))


@pytest.mark.parametrize('module', ['basil_core.dirichlet_diagnosis', 'scripts.audit_dirichlet_learning',
                                  'reporting.dirichlet_diagnosis'])
def test_audit_imports_are_lazy(module):
    subprocess.run([sys.executable, '-c', f'import importlib,sys;importlib.import_module({module!r});'
        'assert not ({"numpy","tensorflow","wandb","matplotlib"} & sys.modules.keys())'], check=True)


def test_audit_cli_help_does_not_load_tensorflow():
    result = subprocess.run([sys.executable, 'scripts/audit_dirichlet_learning.py', '--help'],
                            capture_output=True, text=True, check=True)
    assert '--probe' in result.stdout and 'tensorflow' not in result.stderr.lower()


def test_diagnostic_plots_refuse_overwrite(tmp_path):
    from reporting.dirichlet_diagnosis import generate_plots
    records, partition, cfg = fixture_records()
    summary = summarize_activations(records, partition, cfg)
    summary['node9NormByRound'] = [10., 11.]
    diagnosis = dict(conditions={letter: copy.deepcopy(summary) for letter in 'ABCD'})
    output = tmp_path / 'diagnosis'
    assert generate_plots(diagnosis, output) == ['clean_local_class_retention.png', 'parameter_norm_collapse.png']
    assert all(p.stat().st_size > 0 for p in output.iterdir())
    with pytest.raises(ValueError): generate_plots(diagnosis, output)
