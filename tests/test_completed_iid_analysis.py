"""Post-run calculations only; no training or display dependency."""
import numpy as np
import pytest

from scripts.analyze_completed_iid import accuracy_summary, corrupted_selection, summary


def test_round_windows_and_zero_based_round_50():
    a = np.arange(100, dtype=float)/100
    stats = accuracy_summary(a)
    assert stats['round49'] == .49
    assert stats['bestRound'] == 99
    assert stats['mean80_89'] == pytest.approx(.845)
    assert stats['mean90_99'] == pytest.approx(.945)
    assert stats['lateChange'] == pytest.approx(.1)
    assert stats['lateSlope'] == pytest.approx(.01)


@pytest.mark.parametrize('values', [np.zeros(99), np.full(100, np.nan)])
def test_incomplete_or_nonfinite_runs_cannot_be_reported_as_complete(values):
    with pytest.raises(ValueError):
        accuracy_summary(values)


def test_corruption_depends_on_source_round_not_receiver_round():
    assert not corrupted_selection(dict(selectedSenderByzantine=True, inputSnapshotRound=19, round=20))
    assert corrupted_selection(dict(selectedSenderByzantine=True, inputSnapshotRound=20, round=20))
    assert not corrupted_selection(dict(selectedSenderByzantine=False, inputSnapshotRound=20, round=20))


def test_absent_metric_is_not_fabricated_as_zero():
    assert summary([]) is None
    assert summary([0, 1])['mean'] == .5
