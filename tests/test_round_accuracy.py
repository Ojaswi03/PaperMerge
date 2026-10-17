"""Overall accuracy reports never alter raw evidence or trigger training."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from reporting.round_accuracy import FIELDS, read_view, save_view, source_directory


def make_run(root, distribution='nonIID', horizon=30, condition='A', count=None, status='completed'):
    path = source_directory(root, distribution, horizon, condition)
    path.mkdir(parents=True, exist_ok=True)
    count = horizon if count is None else count
    target = 500 if distribution == 'IID' else horizon
    manifest = dict(status=status, completedRounds=count,
                    config=dict(conditionId=condition, nonIID=distribution=='nonIID', nRounds=target))
    (path/'run.json').write_text(json.dumps(manifest))
    with (path/'round_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(*FIELDS, 'full_test_accuracy', 'class_0_accuracy'))
        writer.writeheader()
        writer.writerows(dict(round=i, mean_node_accuracy=.4+i*.001,
                              worst_node_accuracy=.3, full_test_loss=1.2,
                              full_test_accuracy=.9, class_0_accuracy=.99) for i in range(count))
    return path


def hashes(path):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}


def test_mean_node_not_node9_or_per_class(tmp_path):
    make_run(tmp_path)
    record = read_view(tmp_path, 'nonIID', 30)['conditions']['A']
    assert record['rows'][0]['mean_node_accuracy'] == .4
    assert 'full_test_accuracy' not in record['rows'][0]
    assert 'class_0_accuracy' not in record['rows'][0]


def test_iid_prefix_is_explicit_not_a_separate_run(tmp_path):
    make_run(tmp_path, distribution='IID', count=500)
    record = read_view(tmp_path, 'IID', 30)['conditions']['A']
    assert len(record['rows']) == 30
    assert record['viewType'] == 'trajectory_prefix'
    assert record['sourceTargetRounds'] == 500


def test_missing_conditions_not_invented(tmp_path):
    make_run(tmp_path)
    view = read_view(tmp_path, 'nonIID', 30)
    assert not view['complete']
    assert view['conditions']['B']['rows'] == []
    assert view['conditions']['B']['status'] == 'not_started'


def test_only_committed_rounds_from_active_run(tmp_path):
    path = make_run(tmp_path, horizon=500, count=7, status='running')
    manifest = json.loads((path/'run.json').read_text())
    manifest['completedRounds'] = 6
    (path/'run.json').write_text(json.dumps(manifest))
    record = read_view(tmp_path, 'nonIID', 500)['conditions']['A']
    assert len(record['rows']) == 6
    assert record['status'] == 'incomplete'


def test_missing_committed_round_is_error(tmp_path):
    path = make_run(tmp_path)
    lines = (path/'round_metrics.csv').read_text().splitlines()
    (path/'round_metrics.csv').write_text('\n'.join(lines[:3]+lines[4:])+'\n')
    with pytest.raises(ValueError, match='Missing committed'):
        read_view(tmp_path, 'nonIID', 30)


def test_duplicate_round_is_error(tmp_path):
    path = make_run(tmp_path)
    lines = (path/'round_metrics.csv').read_text().splitlines()
    lines[2] = lines[1]
    (path/'round_metrics.csv').write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError, match='duplicated'):
        read_view(tmp_path, 'nonIID', 30)


def test_nonfinite_accuracy_is_error(tmp_path):
    path = make_run(tmp_path)
    (path/'round_metrics.csv').write_text((path/'round_metrics.csv').read_text().replace('0.4,', 'nan,', 1))
    with pytest.raises(ValueError, match='Non-finite'):
        read_view(tmp_path, 'nonIID', 30)


def test_output_routing_and_raw_hashes_unchanged(tmp_path):
    paths = [make_run(tmp_path, condition=c) for c in 'ABCD']
    before = {str(p):hashes(p) for p in paths}
    view = read_view(tmp_path, 'nonIID', 30)
    output = save_view(tmp_path, view)
    assert output == tmp_path/'newResults/nonIID/30round'
    assert view['complete']
    assert {str(p):hashes(p) for p in paths} == before
    with (output/'round_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 120
    assert all(not key.startswith('class_') for key in rows[0])


def test_plot_outputs_overall_only_and_read_only(tmp_path):
    from reporting.round_accuracy import generate_plots
    path = make_run(tmp_path)
    before = hashes(path)
    names = generate_plots(tmp_path, read_view(tmp_path, 'nonIID', 30))
    assert len(names) == 3
    assert all('class' not in name for name in names)
    assert all((tmp_path/'newPlots/nonIID/30round'/name).stat().st_size > 1000 for name in names)
    assert hashes(path) == before


def test_reporting_imports_are_lazy():
    result = subprocess.run([sys.executable, '-c',
        "import reporting.round_accuracy,sys; assert 'tensorflow' not in sys.modules; "
        "assert 'matplotlib.pyplot' not in sys.modules; assert 'wandb' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_training_or_scientific_imports():
    import ast
    from reporting import round_accuracy
    tree = ast.parse(Path(round_accuracy.__file__).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(n.name for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or '')
    assert not any(name.startswith(('tensorflow', 'basil_core', 'wandb')) for name in names)
