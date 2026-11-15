import pytest

from scripts.report_training_timings import hms, load_runs, summarize


def test_round_estimate_divides_each_complete_run_and_uses_three_seeds():
    rows = [dict(dataset='mnist', partition='iid', method='FedSharp', environment='joint',
        attack='hidden', seed=seed, recorded500RoundSeconds=seconds, hosts='hal9k',
        optimizedDuringRun=False) for seed, seconds in zip((2025, 2026, 2027), (500, 1500, 1000))]
    summary = summarize(rows)[0]
    assert summary['median500RoundSeconds'] == 1000
    assert summary['medianAverageRoundSeconds'] == 2
    assert (summary['minimum500RoundSeconds'], summary['maximum500RoundSeconds']) == (500, 1500)
    with pytest.raises(ValueError, match='three original seeds'):
        summarize(rows[:2])


def test_current_timings_include_150_unique_completed_runs_with_provenance():
    rows, sources = load_runs()
    assert len(rows) == len({r['runId'] for r in rows}) == 150
    assert len(summarize(rows)) == 50
    assert all(r['averageRoundSeconds'] == r['recorded500RoundSeconds'] / 500 for r in rows)
    assert len([r for r in rows if r['method'] == 'Clean (FedSharp control)']) == 12
    assert len([r for r in rows if r['method'] == 'Clean (BASIL control)']) == 12
    assert all(r['source'] in sources for r in rows)
    assert hms(3601.2) == '01:00:01'
