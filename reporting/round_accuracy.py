"""Read-only round-count views of overall A/B/C/D accuracy; no TensorFlow."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

CONDITIONS = ('A', 'B', 'C', 'D')
HORIZONS = (30, 100, 500)
COLORS = {'A': '#0072B2', 'B': '#D55E00', 'C': '#009E73', 'D': '#CC79A7'}
LABELS = {'A': 'A — clean / no attack', 'B': 'B — Hidden / clean channel',
          'C': 'C — Hidden / noise', 'D': 'D — Hidden / noise / EBM'}
FIELDS = ('round', 'mean_node_accuracy', 'worst_node_accuracy', 'full_test_loss')
METRIC = 'Arithmetic mean of full-test accuracy across all ten trained node models'


def source_directory(root, distribution, horizon, condition):
    if distribution not in ('IID', 'nonIID') or horizon not in HORIZONS or condition not in CONDITIONS:
        raise ValueError('Expected IID/nonIID, 30/100/500 rounds and A/B/C/D.')
    base = Path(root)/'newResults'/distribution
    if distribution == 'IID':
        names = {'A': 'A_clean_no_attack_ce_500r_gpu', 'B': 'B_attack_clean_ce_500r_gpu',
                 'C': 'C_attack_noise_ce_500r_gpu', 'D': 'D_attack_noise_ebm_500r_gpu'}
        return base/names[condition]
    study = ('dirichlet_prior_anchor_confirmation_30r_v1' if horizon == 30
             else f'dirichlet_prior_anchor_{horizon}r_v1')
    return base/study/condition


def read_condition(root, distribution, horizon, condition):
    """Read only committed round rows; never inspect/load model checkpoints."""
    root = Path(root)
    directory = source_directory(root, distribution, horizon, condition)
    manifest_path, csv_path = directory/'run.json', directory/'round_metrics.csv'
    result = dict(condition=condition, source=str(directory.relative_to(root)), rows=[],
                  status='not_started', requestedRounds=horizon, metric=METRIC)
    if not manifest_path.exists():
        return result
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    config = manifest['config']
    if config.get('conditionId') != condition or bool(config.get('nonIID')) != (distribution == 'nonIID'):
        raise ValueError(f'Unexpected condition/distribution in {manifest_path}')
    target = int(config['nRounds'])
    if target < horizon:
        raise ValueError('A shorter source cannot provide the requested horizon.')
    committed = int(manifest.get('completedRounds', 0))
    if not 0 <= committed <= target:
        raise ValueError('Invalid completed-round cursor.')
    if manifest['status'] == 'completed' and committed != target:
        raise ValueError('Completed manifest has an incomplete round cursor.')
    raw_csv = csv_path.read_bytes() if csv_path.exists() else b''
    # A writer may have committed the next CSV row before publishing its
    # manifest. Ignore such rows, using the earlier manifest cursor.
    rows = list(csv.DictReader(io.StringIO(raw_csv.decode('utf-8'))))[:min(horizon, committed)]
    if len(rows) != min(horizon, committed):
        raise ValueError(f'Missing committed CSV rounds: {csv_path}')
    selected = []
    for index, row in enumerate(rows):
        if int(row['round']) != index:
            raise ValueError(f'Missing or duplicated round: {csv_path}')
        values = {key: float(row[key]) for key in FIELDS if key != 'round'}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f'Non-finite saved metrics: {csv_path}')
        if not 0 <= values['worst_node_accuracy'] <= values['mean_node_accuracy'] <= 1:
            raise ValueError(f'Invalid node accuracy values: {csv_path}')
        selected.append(dict(round=index, **values))
    complete = len(selected) == horizon and (target != horizon or manifest['status'] == 'completed')
    result.update(rows=selected, status='complete' if complete else 'incomplete',
                  sourceStatus=manifest['status'], sourceTargetRounds=target,
                  sourceCompletedRounds=committed, viewType='trajectory_prefix' if target != horizon else 'native_horizon',
                  sourceManifestSha256=hashlib.sha256(raw_manifest).hexdigest(),
                  sourceCsvSha256=hashlib.sha256(raw_csv).hexdigest(), config=config)
    return result


def read_view(root, distribution, horizon):
    records = {name: read_condition(root, distribution, horizon, name) for name in CONDITIONS}
    return dict(distribution=distribution, requestedRounds=horizon, metric=METRIC,
                conditions=records, complete=all(r['status'] == 'complete' for r in records.values()))


def _write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_view(root, view):
    """Write derived summaries only; original run artifacts are never moved."""
    root = Path(root)
    dist, horizon = view['distribution'], view['requestedRounds']
    output = root/'newResults'/dist/f'{horizon}round'
    output.mkdir(parents=True, exist_ok=True)
    summary, combined = [], []
    for name, record in view['conditions'].items():
        target = output/name
        target.mkdir(exist_ok=True)
        _write_csv(target/'round_metrics.csv', record['rows'], FIELDS)
        metadata = {k: v for k, v in record.items() if k != 'rows'}
        (target/'source.json').write_text(json.dumps(metadata, indent=2)+'\n')
        rows = record['rows']
        combined.extend(dict(condition=name, **row) for row in rows)
        summary.append(dict(condition=name, status=record['status'], measured_rounds=len(rows),
            last_round=rows[-1]['round'] if rows else '',
            latest_mean_node_accuracy=rows[-1]['mean_node_accuracy'] if rows else '',
            latest_worst_node_accuracy=rows[-1]['worst_node_accuracy'] if rows else ''))
    _write_csv(output/'round_metrics.csv', combined, ('condition', *FIELDS))
    _write_csv(output/'summary.csv', summary, tuple(summary[0]))
    (output/'view.json').write_text(json.dumps({k: v for k, v in view.items() if k != 'conditions'}, indent=2)+'\n')
    lines = [f'# {dist}: {horizon}-round overall accuracy', '', METRIC+'.', '',
             'These are read-only derived views, not new training runs or resume checkpoints.',
             'Raw artifacts remain at their original paths because queue validation and resume depend on them.',
             'IID 30/100-round views are prefixes of the saved 500-round IID runs, not independent experiments.',
             'Only manifest-committed rounds are shown. Missing rounds/conditions are never filled.', '',
             '| Condition | Measured rounds | Status | Source |', '|---|---:|---|---|']
    lines += [f"| {name} | {len(r['rows'])}/{horizon} | {r['status']} | `{r['source']}` |"
              for name, r in view['conditions'].items()]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return output


def generate_plots(root, view):
    # Lazy, headless plotting: no GPU, TensorFlow, worker or W&B imports.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(root)/'newPlots'/view['distribution']/f"{view['requestedRounds']}round"
    directory.mkdir(parents=True, exist_ok=True)
    title = f"{view['distribution']} — {view['requestedRounds']}-round view"
    partial = not view['complete']
    generated = []

    def setup(ax):
        ax.set(xlabel='Round (zero-based)', ylabel='Mean-node overall test accuracy (%)',
               ylim=(0, 100), xlim=(0, view['requestedRounds']-1))
        ax.grid(alpha=.2)
        if view['requestedRounds'] > 20:
            ax.axvline(20, color='#555555', ls=':', lw=1.5, label='Hidden starts, B/C/D')

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(directory/filename, dpi=160)
        plt.close(fig)
        generated.append(filename)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, record in view['conditions'].items():
        rows = record['rows']
        label = f"{LABELS[name]} ({len(rows)}/{view['requestedRounds']} rounds)"
        ax.plot([r['round'] for r in rows], [100*r['mean_node_accuracy'] for r in rows],
                color=COLORS[name], lw=2, marker='o' if len(rows) == 1 else None, label=label)
    setup(ax)
    ax.set_title(title+' — overall average accuracy'+(' [INCOMPLETE]' if partial else ''))
    ax.legend(fontsize=9, loc='lower right')
    save(fig, 'ABCD_overall_average_accuracy_vs_round.png')

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (name, record) in enumerate(view['conditions'].items()):
        rows = record['rows']
        if rows:
            value = rows[-1]['mean_node_accuracy']*100
            ax.bar(index, value, color=COLORS[name], width=.65)
            ax.text(index, value+1, f"{value:.2f}%\nR{rows[-1]['round']}", ha='center', fontsize=10)
        else:
            ax.text(index, 2, 'Not started', ha='center', fontsize=9)
    ax.set(xticks=range(4), xticklabels=CONDITIONS, ylim=(0, 100), xlim=(-.6, 3.6),
           ylabel='Mean-node overall test accuracy (%)',
           title=title+(' — latest available endpoints [UNEQUAL/INCOMPLETE]' if partial else ' — final average accuracy'))
    ax.grid(axis='y', alpha=.2)
    save(fig, 'ABCD_final_average_accuracy_bar.png')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (name, record) in zip(axes.flat, view['conditions'].items()):
        rows = record['rows']
        rounds = [r['round'] for r in rows]
        mean = [100*r['mean_node_accuracy'] for r in rows]
        worst = [100*r['worst_node_accuracy'] for r in rows]
        setup(ax)
        ax.plot(rounds, mean, color=COLORS[name], lw=2, label='Node mean')
        ax.plot(rounds, worst, color=COLORS[name], lw=1, ls='--', label='Worst node')
        ax.fill_between(rounds, worst, mean, color=COLORS[name], alpha=.15)
        ax.set_title(f"{LABELS[name]} — {len(rows)}/{view['requestedRounds']} rounds", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(title+' — node consistency (shading is not a confidence interval)', fontsize=12)
    save(fig, 'ABCD_mean_and_worst_node_accuracy.png')
    (directory/'README.md').write_text(
        f'# {title}\n\nOverall accuracy only; no per-class plots.\n\n'
        'The line graph is the arithmetic mean of all ten node models’ full-test accuracies.\n'
        'The bar chart uses the final measured endpoint, labeled with its round.\n'
        'Missing conditions/rounds are not invented. Incomplete endpoints are not final research results.\n'
        'IID 30/100-round figures are prefixes of the saved 500-round IID trajectories.\n'
        'Source data: `newResults/'+view['distribution']+f"/{view['requestedRounds']}round/`.\n")
    return generated


def build_views(root, distributions=('IID', 'nonIID'), horizons=HORIZONS):
    outputs = []
    for distribution in distributions:
        for horizon in horizons:
            view = read_view(root, distribution, horizon)
            result = save_view(root, view)
            plots = generate_plots(root, view)
            outputs.append(dict(distribution=distribution, rounds=horizon, complete=view['complete'],
                                measured={k:len(v['rows']) for k,v in view['conditions'].items()},
                                resultDirectory=str(result), plots=plots))
    return outputs
