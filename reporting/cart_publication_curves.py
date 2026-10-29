"""Publication learning curves from complete, source-verified metric archives.

This reader never imports or edits the producer implementation. The exported
roundNodeAccuracy series measures the development holdout, not benchmark TEST.
"""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Invalid archive path.')
    return path


def collect(root):
    import numpy as np
    root = Path(root).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if not plan['configs'] or not plan['source']:
        raise ValueError('A complete source-frozen comparison plan is required.')
    for name, expected in plan['source'].items():
        if sha(root / 'source_snapshots' / safe_relative(name)) != expected:
            raise ValueError('Frozen producer snapshot differs: ' + name)
    hashes = {'plan.json': sha(root / 'plan.json')}
    exported = json.loads((root / 'SHA256.json').read_text()) if (root / 'SHA256.json').exists() else None
    records = []
    seen = set()
    for view in plan['configs']:
        run_id = view['runId']
        if safe_relative(run_id).name != run_id or run_id in seen:
            raise ValueError('Invalid or duplicate run identity.')
        seen.add(run_id)
        folder = root / run_id
        row = json.loads((folder / 'run.json').read_text())
        if (row['status'] != 'COMPLETED' or row['config'] != view or
                row['source'] != plan['source'] or row['methodRevision'] != plan['revision'] or
                row['completedActivations'] != view['nodes'] * view['rounds']):
            raise ValueError('Learning curves require every planned run to complete.')
        if (row['data'] != plan['data'][view['dataset']] or
                row['partition'] != plan['partitions'][run_id]):
            raise ValueError('Evaluation data identity differs from the plan.')
        with np.load(folder / 'metrics.npz', allow_pickle=False) as arrays:
            accuracy = arrays['roundNodeAccuracy']
        if (accuracy.shape != (view['rounds'], view['nodes']) or
                not np.isfinite(accuracy).all() or np.any(accuracy < 0) or np.any(accuracy > 1)):
            raise ValueError('Invalid round-by-client accuracy array.')
        for name in ('run.json', 'metrics.npz'):
            key = f'{run_id}/{name}'
            hashes[key] = sha(folder / name)
        case = view['comparisonCase']
        records.append(dict(dataset=view['dataset'], partition=view['partition'], seed=view['seed'],
            arm=case.get('studyVariant', {}).get('name', case['arm']), environment=case['environment'], sigma=view['sigma'],
            rounds=view['rounds'], nodes=view['nodes'], attackStart=view['attackStart'],
            alpha=view['dirichletAlpha'], shareFraction=case['shareFraction'], runId=run_id,
            curve=accuracy.mean(axis=1)))
    if exported is not None:
        for name, value in hashes.items():
            if exported.get(name) != value:
                raise ValueError('Export checksum differs: ' + name)
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in ('dataset', 'partition', 'environment', 'sigma'))].append(row)
    for items in groups.values():
        if len({(r['rounds'], r['nodes'], r['attackStart'], r['alpha']) for r in items}) != 1:
            raise ValueError('Methods have different learning-curve contracts.')
        seeds = {r['seed'] for r in items}
        for arm in {r['arm'] for r in items}:
            actual = [r['seed'] for r in items if r['arm'] == arm]
            if len(actual) != len(set(actual)) or set(actual) != seeds:
                raise ValueError('Every method requires the same complete seed group.')
    for dataset, env, sigma in {(r['dataset'], r['environment'], r['sigma']) for r in records}:
        context = [r for r in records if (r['dataset'], r['environment'], r['sigma']) == (dataset, env, sigma)]
        expected = {(r['arm'], r['seed']) for r in context}
        for part in {r['partition'] for r in context}:
            if {(r['arm'], r['seed']) for r in context if r['partition'] == part} != expected:
                raise ValueError('Partitions require the same complete method/seed matrix.')
    return plan, records, hashes


COLORS = {'prior_fallback': '#0072B2', 'cart': '#0072B2', 'acds': '#D55E00',
          'sharp': '#555555', 'macro': '#009E73', 'prior_cart': '#56B4E9'}
COLORS.update(cart_full='#0072B2', acds_checked='#D55E00', sharp_checked='#555555',
    cart_macro='#009E73', cart_ingress='#CC79A7', cart_commit='#E69F00',
    cart_no_prior='#56B4E9', cart_no_calibration='#332288', cart_no_repair='#882255')
ORDER = ('cart_full', 'acds_checked', 'sharp_checked', 'prior_fallback', 'cart', 'prior_cart', 'acds', 'sharp', 'macro')
LABELS = {'prior_fallback': 'SHARP + CART', 'cart': 'SHARP + CART',
          'acds': 'SHARP + ACDS', 'sharp': 'SHARP', 'macro': 'CART (macro only)',
          'prior_cart': 'CART (local-prior training)', 'prior_macro': 'CART (local prior, macro only)',
          'prior_wide': 'CART (full peer window)', 'cart_fallback': 'CART (consistent fallback)',
          'ingress_only': 'CART (admission only)', 'commit_only': 'CART (update check only)',
          'local_only': 'Local training', 'fresh_peer': 'Fresh peer',
          'no_calibration': 'CART (no calibration)', 'no_repair': 'CART (no head repair)',
          'balanced_commit': 'CART (balanced-loss check)'}
LABELS.update(cart_full='SHARP + CART', acds_checked='SHARP + ACDS (checked pooling)',
    sharp_checked='SHARP (checked pooling)', cart_macro='CART (macro only)',
    cart_ingress='CART (admission only)', cart_commit='CART (update check only)',
    cart_no_prior='CART (no training prior)', cart_no_calibration='CART (no calibration)',
    cart_no_repair='CART (no head repair)', acds_single='SHARP + ACDS (single model)')
DATASETS = {'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}


def arm_order(arm):
    return (ORDER.index(arm) if arm in ORDER else len(ORDER), arm)


def aggregate(records):
    import numpy as np
    grouped = defaultdict(list)
    for row in records:
        grouped[tuple(row[k] for k in ('dataset', 'partition', 'environment', 'sigma', 'arm'))].append(row)
    return {key: dict(mean=np.mean([r['curve'] for r in rows], axis=0),
                     sd=np.std([r['curve'] for r in rows], axis=0, ddof=1) if len(rows) > 1 else None,
                     seeds=sorted(r['seed'] for r in rows)) for key, rows in grouped.items()}


def _draw(ax, records, title, upper, styles):
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    arms = sorted({r['arm'] for r in records}, key=arm_order)
    for arm in arms:
        items = [r for r in records if r['arm'] == arm]
        values = np.asarray([r['curve'] for r in items]) * 100
        mean = values.mean(axis=0)
        x = np.arange(1, len(mean) + 1)
        color, linestyle = styles[arm]
        label = LABELS.get(arm, arm.replace('_', ' '))
        if arm == 'cart' and 'prior_fallback' in arms:
            label = 'CART (original admission)'
        ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=1.7, zorder=3)
        if len(values) > 1:
            sd = values.std(axis=0, ddof=1)
            ax.fill_between(x, np.clip(mean - sd, 0, 100), np.clip(mean + sd, 0, 100),
                            color=color, alpha=.14, linewidth=0, zorder=2)
    row = records[0]
    onset = row['attackStart'] + 1
    if row['environment'] in ('joint', 'attack_only') and onset <= row['rounds']:
        ax.axvline(onset, color='#777777', linewidth=.9, linestyle=(0, (3, 3)),
                   label='Attack onset', zorder=1)
    rounds = row['rounds']
    step = max(1, int(math.ceil(rounds / 5 / 5) * 5)) if rounds > 10 else max(1, rounds // 5)
    ticks = sorted({1, rounds, *range(step, rounds, step)})
    ax.set(title=title, xlabel='Communication round', ylabel='Validation accuracy (%)',
           xlim=(1, max(2, rounds)), ylim=(0, upper), xticks=ticks)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(axis='y', color='#D9DDE2', linewidth=.6, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#888888')
    ax.tick_params(length=3, width=.6, color='#888888')


def generate(root, destination, *, arms=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root, dest = Path(root).resolve(), Path(destination).resolve()
    plan, records, hashes = collect(root)
    if arms is not None:
        records = [r for r in records if r['arm'] in arms]
        if not records:
            raise ValueError('No matching methods for learning curves.')
    protected = [root / 'source_snapshots', root / 'scheduler_snapshots', *[root / c['runId'] for c in plan['configs']]]
    if dest == root or dest in root.parents or any(dest == p or p in dest.parents for p in protected):
        raise ValueError('Figure destination overlaps experiment inputs.')
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Preserve existing figures; choose an empty destination.')
    dest.mkdir(parents=True, exist_ok=True)
    fallback = ('#CC79A7', '#E69F00', '#009E73', '#56B4E9', '#332288', '#882255')
    arms = sorted({r['arm'] for r in records}, key=arm_order)
    styles = {arm: (COLORS.get(arm, fallback[i % len(fallback)]),
                    (0, (5, 2)) if arm == 'acds' else '-' if i < 2 else (0, (2, 1))) for i, arm in enumerate(arms)}
    style = {'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
             'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
             'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': .6,
             'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42,
             'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'cart-publication-curves'}
    files = []
    captions = []

    def save(panels, columns, name, caption, limits, heading=None):
        rows = math.ceil(len(panels) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(7.3, 2.55 * rows + .55), squeeze=False)
        for i, (ax, (items, title)) in enumerate(zip(axes.flat, panels)):
            _draw(ax, items, f'({chr(97 + i)}) {title}', limits[items[0]['dataset']], styles)
            if i % columns:
                ax.set_ylabel('')
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        handles = {}
        for ax in axes.flat:
            lines, labels = ax.get_legend_handles_labels()
            for handle, label in zip(lines, labels):
                handles.setdefault(label, handle)
        fig.legend(handles.values(), handles.keys(), loc='lower center', frameon=False,
                   ncol=min(3, len(handles)), bbox_to_anchor=(.5, .005), columnspacing=1.6, handlelength=2.7)
        legend_rows = math.ceil(len(handles) / 3)
        height = 2.55 * rows + .55 + .22 * (legend_rows - 1)
        fig.set_size_inches(7.3, height)
        if heading:
            fig.suptitle(heading, y=.995, fontsize=10)
        fig.subplots_adjust(left=.09, right=.985, top=.88 if heading else (.91 if rows == 1 else .94),
                            bottom=(.82 + .22 * (legend_rows - 1)) / height, wspace=.28, hspace=.48)
        for ext in ('svg', 'pdf', 'png'):
            path = dest / f'{name}.{ext}'
            fig.savefig(path, dpi=600, bbox_inches='tight', metadata={'Creator': 'PaperMerge learning-curve exporter'})
            files.append(path.name)
        plt.close(fig)
        captions.append(f'{name}: {caption}')

    with plt.rc_context(style):
        contexts = sorted({(r['environment'], r['sigma']) for r in records})
        for env, sigma in contexts:
            current = [r for r in records if (r['environment'], r['sigma']) == (env, sigma)]
            datasets = sorted({r['dataset'] for r in current}, key=lambda d: (d != 'mnist', d))
            aggregated = aggregate(current)
            limits = {}
            for dataset in datasets:
                peaks = [float(r['curve'].max()) for r in current if r['dataset'] == dataset]
                peaks += [float((v['mean'] + (v['sd'] if v['sd'] is not None else 0)).max())
                          for k, v in aggregated.items() if k[0] == dataset]
                limits[dataset] = min(100, max(20, 10 * math.ceil(100 * max(peaks) / 10)))
            panels = []
            suffix = f'{env}_sigma_{sigma:g}'
            for dataset in datasets:
                items = [r for r in current if r['dataset'] == dataset]
                parts = sorted({r['partition'] for r in items}, key=lambda p: p != 'iid')
                pair = []
                for part in parts:
                    group = [r for r in items if r['partition'] == part]
                    partition = 'IID' if part == 'iid' else f'Dirichlet non-IID ($\\alpha={group[0]["alpha"]:g}$)'
                    title = DATASETS.get(dataset, dataset) + ', ' + partition
                    pair.append((group, title))
                panels.extend(pair)
                seeds = sorted({r['seed'] for r in items})
                caption = (f'{DATASETS.get(dataset, dataset)} mean client-model validation accuracy versus completed '
                    f'communication rounds. Lines average {len(seeds)} seeds after averaging clients within each run. '
                    'Shading is sample standard deviation across seeds when at least two seeds are present. '
                    'All recorded rounds are shown without smoothing or resampling. '
                    f'Channel standard deviation is {sigma:g}; environment is {env}.')
                save(pair, min(2, len(pair)), f'{dataset}_{suffix}_accuracy_curves', caption, limits)
                if len(seeds) > 1:
                    seed_panels = []
                    for group, title in pair:
                        for seed in seeds:
                            selected = [r for r in group if r['seed'] == seed]
                            if not selected:
                                raise ValueError('Partition is missing an expected seed.')
                            part = 'IID' if selected[0]['partition'] == 'iid' else f'Non-IID ($\\alpha={selected[0]["alpha"]:g}$)'
                            seed_panels.append((selected, part + f'\nSeed {seed}'))
                    save(seed_panels, len(seeds), f'{dataset}_{suffix}_seed_curves',
                         caption + ' Each panel shows one seed; no uncertainty band is drawn within a seed.',
                         limits, heading=DATASETS.get(dataset, dataset))
            if len(datasets) > 1:
                save(panels, 2, f'comparison_{suffix}_accuracy_curves',
                     'MNIST and CIFAR-10 learning curves; mean across seeds with sample SD shading. '
                     'Each dataset row uses the same y-axis limits across partitions.', limits)
    summary = aggregate(records)
    with (dest / 'accuracy_by_round_and_seed.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'partition', 'environment', 'sigma', 'arm', 'seed', 'round', 'meanClientAccuracy'])
        for r in records:
            for i, value in enumerate(r['curve'], 1):
                writer.writerow([*[r[k] for k in ('dataset', 'partition', 'environment', 'sigma', 'arm', 'seed')], i, float(value)])
    with (dest / 'accuracy_mean_and_sd.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'partition', 'environment', 'sigma', 'arm', 'round', 'mean', 'sampleSD', 'seedCount'])
        for key, series in summary.items():
            for i, value in enumerate(series['mean']):
                writer.writerow([*key, i + 1, float(value), '' if series['sd'] is None else float(series['sd'][i]), len(series['seeds'])])
    captions.append('ACDS shares fitting examples (5% in the full comparison); CART uses receiver-local fitting data. '
                    'Clients in the same run are not independent seed replicates. The dashed vertical marker denotes '
                    'the first attack-active displayed round. Accuracy arrays measure the development holdout.')
    (dest / 'captions.txt').write_text('\n\n'.join(captions) + '\n')
    files.extend(['accuracy_by_round_and_seed.csv', 'accuracy_mean_and_sd.csv', 'captions.txt'])
    manifest = dict(metric='development-holdout mean client accuracy', smoothing=False,
        uncertainty='sample SD across independent seeds; not a confidence interval',
        configurations=len(records), producerSource=plan['source'], inputSha256=hashes,
        plotterSha256=sha(Path(__file__)), outputSha256={name: sha(dest / name) for name in files},
        destination=str(dest))
    (dest / 'learning_curves_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
