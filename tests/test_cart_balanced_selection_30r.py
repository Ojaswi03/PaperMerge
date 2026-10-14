"""Prespecified longer diagnostic: same mathematics, no automatic promotion."""
import json
import subprocess
import sys

import pytest


def test_thirty_round_dry_plan_is_lazy_and_paired():
    p = subprocess.run([sys.executable, '-c',
        'from scripts.run_cart_balanced_selection_30r import main; main(); '
        "import sys; assert not ({'tensorflow','numpy','wandb','matplotlib'} & sys.modules.keys())"],
        text=True, capture_output=True, check=True)
    plan = json.loads(p.stdout)
    assert plan['trainingRunsStarted'] == 0
    assert plan['differences'] == ['condition','snapshotSelectionRule','startingModelRule']
    from scripts.run_cart_balanced_selection_diagnostic import config as old
    for cfg in plan['configs']:
        reference = old('GPU', cfg['condition'])
        changes = {k for k in cfg.keys() | reference.keys() if cfg.get(k) != reference.get(k)}
        assert changes == {'nRounds','purpose','automaticExtension','stopOnAccuracy','activationStorage'}
        assert cfg['nRounds'] == 30 and cfg['localEpochs'] == 5
        assert cfg['samplesPerNode'] == 5000 and cfg['batchSize'] == 512
        assert cfg['snapshotAnchorMu'] == .01 and cfg['basilMemorySize'] == 5
        assert not cfg['actualAttackers'] and not cfg['channelNoise'] and not cfg['ebm']
        assert not cfg['liveTrackingEnabled'] and not cfg['automaticExtension']


def test_thirty_round_owner_horizon_has_no_future_owner_leakage():
    from basil_core.cart_owner_gradient_worker import available_owner_classes
    for r in range(30):
        for node in range(10):
            expected = tuple(range(node)) if r == 0 else tuple(c for c in range(10) if c != node)
            assert available_owner_classes(r,node,diagnostic_rounds=30) == expected
    for horizon in (0,2,29,31,100,True):
        with pytest.raises(ValueError): available_owner_classes(0,0,diagnostic_rounds=horizon)
    with pytest.raises(ValueError): available_owner_classes(30,0,diagnostic_rounds=30)
    with pytest.raises(ValueError): available_owner_classes(1,0)  # Default remains one traversal.


def test_long_report_rejects_missing_activations(tmp_path):
    from scripts.report_cart_balanced_selection_diagnostic import load_records, validate_records
    with pytest.raises(ValueError,match='Missing'): load_records(tmp_path,30,True)
    with pytest.raises(ValueError,match='Three hundred'): validate_records([], 'balanced_owner_ss', rounds=30)
    with pytest.raises(ValueError,match='Unsupported'): validate_records([], 'balanced_owner_ss', rounds=100)


def test_streaming_trace_pairing_and_freshness(tmp_path):
    from scripts.report_cart_balanced_selection_diagnostic import trace_signature
    base = dict(round_id=0,node_id=0,parameterHashBefore='a',indicesHash='samples',
                augmentationKey=[1,2],imagesHash='images',ownerGradientContext=dict(modelHash='a',owners=[
                    dict(owner=1,trainingIndices=[4,5],augmentationKey=[3,4],parameterHash='a')]))
    path = tmp_path/'trace.jsonl'
    path.write_text(json.dumps(base)+'\n')
    first = trace_signature(path,1)
    # Trajectories can differ; pairing compares sample keys, not learned weights.
    base['parameterHashBefore']='b';base['ownerGradientContext']['modelHash']='b'
    base['ownerGradientContext']['owners'][0]['parameterHash']='b'
    path.write_text(json.dumps(base)+'\n')
    assert trace_signature(path,1) == first
    base['indicesHash']='different';path.write_text(json.dumps(base)+'\n')
    assert trace_signature(path,1) != first
    with pytest.raises(ValueError,match='Missing'): trace_signature(path,2)
    base['ownerGradientContext']['modelHash']='stale';path.write_text(json.dumps(base)+'\n')
    with pytest.raises(ValueError,match='Stale'): trace_signature(path,1)
