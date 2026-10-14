"""Three-round orchestration/report screens; no research training in tests."""
import copy
import json

import pytest

from scripts.run_gradient_projection_warm_gate import config_for, expected_candidates, learning_screen
from scripts.report_gradient_projection_warm_gate import review
from basil_core.research_protocol import Snapshot, initialize_memories, broadcast


def fake_controls():
    rows = []
    for round_id in range(3):
        for node in range(10):
            accuracy = [0.]*10
            accuracy[node] = 1.
            rows.append(dict(round=round_id, nodeId=node,
                epochs=[dict(globalValidationAccuracy=.1, perClassValidationAccuracy=accuracy)]))
    return dict(ce=copy.deepcopy(rows), projection=rows)


def test_three_round_configuration_keeps_declared_protocol_and_no_production():
    config = config_for('GPU')
    assert config['nRounds'] == 3 and config['localEpochs'] == 5 and config['batchSize'] == 512
    assert config['trainingSamplesPerNode'] == 4500 and config['validationSamplesPerClass'] == 500
    assert config['projectionEnergy'] == .99 and config['basilMemorySize'] == 5
    assert config['representationImages'] == 32 and config['maximumPatchesPerLayer'] == 2048
    assert config['researchValid'] is config['useChannelNoise'] is config['liveTrackingEnabled'] is False
    assert config['attackerCount'] == config['snapshotAnchorMu'] == config['momentum'] == config['weightDecayCoefficient'] == 0
    assert config['ebmMode'] == 'none' and config['gradientClipNorm'] is None
    assert config['learningRate'] == .05 and config['learningRateSchedule'] == 'basil_round_decay'


def test_memory_expectations_match_actual_three_traversals():
    import numpy as np
    memories = initialize_memories([np.ones(2, np.float32)])
    for round_id in range(3):
        for node in range(10):
            assert [(s.sender_id, s.round_id) for s in memories[node].candidates()] == expected_candidates(round_id, node)
            if round_id:
                assert all(s.round_id >= 0 for s in memories[node].candidates())
            broadcast(memories, Snapshot(node, round_id, [np.ones(2, np.float32)]))


def test_screen_excludes_cold_start_and_rejects_only_constant_class_retention():
    controls = fake_controls()
    for row in controls['projection'][:10]:
        row['epochs'][-1]['perClassValidationAccuracy'] = [1.]*10
        row['epochs'][-1]['globalValidationAccuracy'] = 1.
    screen = learning_screen(controls)
    assert screen['passed'] is False and screen['ownClassLearnedFraction'] == 1.
    assert screen['twoOtherClassesRetainedFraction'] == 0.
    assert screen['meanValidationImprovement'] == 0.
    assert screen['automaticPromotion'] is screen['testSetUsed'] is False


def test_warm_screen_requires_new_and_previous_class_learning_together():
    controls = fake_controls()
    for row in controls['projection'][10:]:
        node = row['nodeId']
        accuracy = [.5]*10
        accuracy[node] = 0.
        row['epochs'][-1].update(globalValidationAccuracy=.45, perClassValidationAccuracy=accuracy)
    assert not learning_screen(controls)['passed']
    assert learning_screen(controls)['checks']['ownClassLearnedInAtLeast80Percent'] is False
    for row in controls['projection'][10:]:
        row['epochs'][-1]['perClassValidationAccuracy'][row['nodeId']] = .5
        row['epochs'][-1]['globalValidationAccuracy'] = .5
    assert learning_screen(controls)['passed']


def test_screen_refuses_missing_warmed_activations():
    controls = fake_controls()
    controls['projection'].pop()
    with pytest.raises(ValueError, match='20 warmed'):
        learning_screen(controls)


def test_saved_report_refuses_interrupted_diagnostic(tmp_path):
    (tmp_path/'run.json').write_text(json.dumps(dict(status='stopped')))
    with pytest.raises(ValueError, match='Partial'):
        review(tmp_path)
