#!/usr/bin/env python3
"""Export verified local pool diagnostics without model checkpoints or examples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(inputs, destination):
    import numpy as np
    records = []; snapshots = {}; producer = None
    for root in map(lambda p: Path(p).resolve(), inputs):
        plan = json.loads((root / 'plan.json').read_text())
        receipt = json.loads((root / 'independent_validation.json').read_text())
        if receipt['status'] != 'COMPLETED_CASES_INDEPENDENTLY_VERIFIED' or receipt['producerSource'] != plan['producerSource']:
            raise ValueError('Independent verification is required.')
        if producer is not None and producer != plan['producerSource']:
            raise ValueError('Producer source differs between paired local cases.')
        producer = plan['producerSource']
        verified = {r['mode']: r for r in receipt['results']}
        for case in plan['cases']:
            folder = root / case['mode']
            if not (folder / 'result.json').exists():
                continue
            row = json.loads((folder / 'result.json').read_text()); view = case['view']
            if row['status'] != 'COMPLETED' or row['contract']['case'] != case or sha(folder / 'metrics.npz') != verified[case['mode']]['metricSha256']:
                raise ValueError('Result or independently verified metric identity differs.')
            with np.load(folder / 'metrics.npz', allow_pickle=False) as arrays:
                accuracy = arrays['roundNodeAccuracy']
            if accuracy.shape != (view['rounds'], view['nodes']) or not np.isfinite(accuracy).all():
                raise ValueError('Invalid completed metric array.')
            records.append(dict(view=view, mode=case['mode'], summary=row,
                validation=verified[case['mode']], metrics=folder / 'metrics.npz', curve=accuracy.mean(axis=1)))
        for name, expected in plan['diagnosticSource'].items():
            path = root / 'source_snapshots' / name
            if sha(path) != expected:
                raise ValueError('Frozen diagnostic source changed.')
            snapshots[f'{expected}-{Path(name).name}'] = path
    groups = {}
    for record in records:
        v = record['view']; key = (v['rounds'], v['attackStart'])
        modes = groups.setdefault(key, {})
        if record['mode'] in modes:
            raise ValueError('Duplicate local control.')
        modes[record['mode']] = record
    for modes in groups.values():
        if set(modes) != {'raw', 'checked', 'single'}:
            raise ValueError('Complete raw, checked and single controls are required.')
        contracts = [r['summary']['contract'] for r in modes.values()]
        for key in ('initialHash', 'partition', 'data'):
            if any(c[key] != contracts[0][key] for c in contracts):
                raise ValueError('Paired initialization or data differs: ' + key)
        if any(r['view'] != next(iter(modes.values()))['view'] for r in modes.values()):
            raise ValueError('Paired physical configuration differs.')
    dest = Path(destination).resolve()
    if any(dest == Path(p).resolve() or dest in Path(p).resolve().parents or Path(p).resolve() in dest.parents for p in inputs):
        raise ValueError('Keep exports separate from diagnostic inputs.')
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Use an empty destination; preserve prior exports.')
    dest.mkdir(parents=True, exist_ok=True)
    exported = []
    with (dest / 'round_accuracy.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['rounds', 'attackStartZeroBased', 'mode', 'round', 'meanClientAccuracy'])
        for (rounds, onset), modes in sorted(groups.items()):
            for mode, record in modes.items():
                name = f'{rounds}r_attack{onset+1}_{mode}_metrics.npz'
                shutil.copyfile(record['metrics'], dest / name)
                for index, accuracy in enumerate(record['curve'], 1):
                    writer.writerow([rounds, onset, mode, index, float(accuracy)])
                exported.append(dict(mode=mode, view=record['view'], validation=record['validation'],
                    contract=record['summary']['contract'], rounds=record['summary']['rounds'], metrics=name,
                    finalAccuracy=float(record['curve'][-1])))
    for name, path in snapshots.items():
        target = dest / 'diagnostic_sources' / name; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'raw': '#D55E00', 'checked': '#0072B2', 'single': '#666666'}
    labels = {'raw': 'SHARP + ACDS', 'checked': 'SHARP + ACDS, checked averaging', 'single': 'SHARP + ACDS, best single'}
    for (rounds, onset), modes in groups.items():
        if rounds <= onset:
            continue
        with plt.rc_context({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9,
                             'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
            fig, ax = plt.subplots(figsize=(7.3, 3.7))
            for mode in ('raw', 'single', 'checked'):
                ax.plot(np.arange(1, rounds+1), 100*modes[mode]['curve'], color=colors[mode],
                    linestyle='--' if mode == 'single' else '-', linewidth=1.7, label=labels[mode])
            ax.axvline(onset+1, color='#888888', linestyle=':', linewidth=1, label='Attack onset')
            ax.set(xlabel='Communication round', ylabel='Validation accuracy (%)', xlim=(1, rounds),
                ylim=(0, 10*np.ceil(max(r['curve'].max() for r in modes.values())*10)),
                title=r'CIFAR-10, Dirichlet non-IID ($\alpha=0.5$)')
            ax.grid(axis='y', alpha=.2);ax.spines[['right','top']].set_visible(False)
            ax.legend(loc='upper center', bbox_to_anchor=(.5, -.23), ncol=2, frameon=False, fontsize=8)
            fig.subplots_adjust(bottom=.3)
            for ext in ('svg', 'pdf', 'png'):
                target = dest / f'acds_checked_averaging_{rounds}r.{ext}'
                fig.savefig(target, dpi=600, bbox_inches='tight')
                if ext == 'svg':
                    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
            plt.close(fig)
    (dest / 'summary.json').write_text(json.dumps(dict(status='VERIFIED_LOCAL_DIAGNOSTICS', cases=exported,
        producerSource=producer, limitation='One seed; diagnostic horizons. Not a full-length multi-seed robustness result.'), indent=2)+'\n')
    (dest / 'caption.txt').write_text('Mean client-model validation accuracy for seed 2025 across 100 clients. '
        'The three selection variants use the same 5% ACDS fitting-data sharing, initial model, Gaussian channel, '
        'Hidden attack and five SGD steps per activation. No CART mechanism is used. No seed uncertainty band is '
        'estimated from one seed. The checked variant updates the RLA denominator to the actual selected count.\n')
    files = [p for p in dest.rglob('*') if p.is_file()]
    (dest / 'SHA256.json').write_text(json.dumps({str(p.relative_to(dest)): sha(p) for p in sorted(files)}, indent=2)+'\n')
    return exported


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args(); result = generate(args.inputs, args.destination)
    print(f'Exported {len(result)} verified local cases to {args.destination}')
