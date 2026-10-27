"""Diagnosis must distinguish received corrupted snapshots from attacker labels."""
import subprocess
import sys

import pytest


def test_phase_statistics_uses_snapshot_age_and_measured_correction():
    from scripts.diagnose_cart_replication import phase_statistics
    history=[]
    for snapshot,sender,ratio,active in [(19,0,0.,False),(19,0,.1,True),(20,0,.1,True),(20,1,0.,False)]:
        step=dict(cartBarrierGradientNorm=ratio,baseGradientNorm=1.,cartActiveClassFraction=.5 if ratio else 0.,
            ebmCorrectionRatio=.01,trainingBatchAccuracy=.4)
        history.append(dict(inputSourceNode=sender,inputSnapshotRound=snapshot,attackActive=active,optimizerTelemetry=[step]))
    result=phase_statistics(history,dict(attackHidden=True,resolvedAttackerIds=[0],attackStartRound=20))
    assert result['corruptedSelections']==1
    assert result['outboundAttacks']==2 and result['optimizerSteps']==4
    assert result['meanRetentionBaseRatio']==pytest.approx(.05)
    assert result['capSaturatedStepFraction']==.5 and result['retentionActiveStepFraction']==.5
    assert phase_statistics(history,dict(attackHidden=False,resolvedAttackerIds=[0],attackStartRound=20))['corruptedSelections']==0


def test_diagnosis_rejects_missing_step_evidence():
    from scripts.diagnose_cart_replication import phase_statistics
    with pytest.raises(ValueError,match='telemetry'):
        phase_statistics([],dict(attackHidden=False,resolvedAttackerIds=[]))


def test_diagnostic_import_does_not_initialize_training_libraries():
    subprocess.run([sys.executable,'-c','import scripts.diagnose_cart_replication;import sys;'
        'assert not ({"tensorflow","numpy","wandb","matplotlib"}&sys.modules.keys())'],check=True)
