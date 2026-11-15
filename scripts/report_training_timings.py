#!/usr/bin/env python3
"""Report saved 500-round loop runtimes without claiming a matched speed benchmark."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    'E1': 'hal9k_fedsharp-fleet-500r-20261103',
    'E2': 'themachine_fedsharp-fleet-500r-20261103',
    'E4': 'themachine_fedsharp-plain-clean-500r-20261104',
    'E5': 'themachine_basil-plain-clean-500r-20261104',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hms(seconds):
    value = round(seconds)
    return f'{value // 3600:02}:{value // 60 % 60:02}:{value % 60:02}'


def load_runs(repo=ROOT):
    repo = Path(repo)
    evidence = repo / 'docs/thesis/results-evidence-20261104/current_runs.csv'
    selected = list(csv.DictReader(evidence.open(newline='')))
    manifests = {}
    rows = []
    provenance = {str(evidence.relative_to(repo)): sha(evidence)}
    for item in selected:
        if item['archive'] not in ARCHIVES or Path(item['runId']).name != item['runId']:
            raise ValueError('Unknown or unsafe selected archive/run identity.')
        root = repo / 'docs/results' / ARCHIVES[item['archive']]
        if root not in manifests:
            manifests[root] = json.loads((root / 'SHA256.json').read_text())
            provenance[str((root / 'SHA256.json').relative_to(repo))] = sha(root / 'SHA256.json')
        path = root / item['runId'] / 'run.json'
        digest = sha(path)
        if digest != manifests[root][item['runId'] + '/run.json']:
            raise ValueError('Timing source checksum mismatch: ' + str(path))
        record = json.loads(path.read_text())
        cfg = record['contract']['config']
        elapsed = record['elapsedSeconds']
        if (record['status'] != 'COMPLETED' or record['completedRounds'] != 500 or cfg['rounds'] != 500
                or cfg['nodes'] != 100 or cfg['localSteps'] != 5
                or not math.isfinite(elapsed) or elapsed <= 0):
            raise ValueError('Expected a complete 500-round timing source.')
        hosts = [record['runtime']['host']]
        continuation = path.with_name('continuation.json')
        if continuation.exists():
            if sha(continuation) != manifests[root][item['runId'] + '/continuation.json']:
                raise ValueError('Continuation timing provenance mismatch.')
            previous = json.loads(continuation.read_text())
            hosts = list(dict.fromkeys([previous['sourceRuntime']['host'], *hosts]))
            provenance[str(continuation.relative_to(repo))] = sha(continuation)
        method = item['method']
        if item['environment'] == 'clean':
            method = 'Clean (FedSharp control)' if item['family'] == 'sharp' else 'Clean (BASIL control)'
        rows.append(dict(dataset=item['dataset'], partition=item['partition'], method=method,
            environment=item['environment'], attack=item['attack'], seed=int(item['seed']),
            runId=item['runId'], recorded500RoundSeconds=elapsed,
            averageRoundSeconds=elapsed / 500, averageActivationSeconds=elapsed / 50000,
            hosts=' -> '.join(hosts), tensorflow=record['runtime']['tensorflow'],
            numpy=record['runtime']['numpy'], threads=record['runtime']['profile']['threads'],
            optimizedDuringRun=bool(record.get('runtimeAdapter', {}).get('executionOptimization')),
            source=str(path.relative_to(repo)), sourceSha256=digest))
        provenance[str(path.relative_to(repo))] = digest
    if len(rows) != 150 or len({r['runId'] for r in rows}) != 150:
        raise ValueError('The current selected validation matrix must contain 150 unique runs.')
    return rows, provenance


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ('dataset', 'partition', 'method', 'environment', 'attack'))
        groups[key].append(row)
    result = []
    for key, cases in sorted(groups.items()):
        if sorted(r['seed'] for r in cases) != [2025, 2026, 2027]:
            raise ValueError('Timing groups require the three original seeds exactly once.')
        values = [r['recorded500RoundSeconds'] for r in cases]
        result.append(dict(zip(('dataset', 'partition', 'method', 'environment', 'attack'), key),
            seeds=3, median500RoundSeconds=statistics.median(values),
            medianAverageRoundSeconds=statistics.median(values) / 500,
            minimum500RoundSeconds=min(values), maximum500RoundSeconds=max(values),
            hosts='; '.join(sorted({r['hosts'] for r in cases})),
            optimizedRuns=sum(r['optimizedDuringRun'] for r in cases)))
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(destination, repo=ROOT):
    repo = Path(repo)
    rows, inputs = load_runs(repo)
    summary = summarize(rows)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / 'per_run.csv', rows)
    write_csv(destination / 'summary.csv', summary)
    lines = ['# Recorded training and evaluation runtimes', '',
        'These are historical observations from the current 150-run validation matrix, not a controlled speed benchmark.', '',
        'One ring round contains 100 sequential client activations, each with five mini-batch SGD steps. '
        'The reported round time is each completed run\'s recorded protocol elapsed time divided by 500. '
        'It is not a directly measured median of 500 individual round durations.', '',
        'The recorded loop time includes training, selection, calibration, repair, regular validation of all client models, '
        'and checkpoint/status work within that loop. It excludes environment startup, queue waiting, GPU admission gates, '
        'cross-server transfers, final independent verification, and discarded work not retained in a resumed checkpoint. '
        'It is not GPU kernel time or calendar time.', '',
        'Continuation totals include the original 100-round elapsed time. Hosts, concurrent worker load and execution '
        'optimizations differ. Some runs changed host after round 100. Clean controls were fresh later runs. '
        'Consequently, ratios in this table must not be claimed as intrinsic algorithmic speedups.', '',
        'The ring is simulated inside each experiment worker; these are not measurements of communication among 100 physical machines.', '',
        '## Reference comparison', '',
        'Impaired methods below use Hidden attack + noise, sigma 0.01. Clean uses the fresh FedSharp direct-transfer control '
        '(mu zero, no attack/noise/defense). BASIL clean controls are retained separately in the CSV, not pooled as extra seeds.', '',
        '| Dataset/split | Method | Average round, median across seeds | 500 rounds, median | 500-round range | Hosts over trajectory |',
        '| --- | --- | ---: | ---: | --- | --- |']
    order = {'Clean (FedSharp control)': 0, 'BASIL': 1, 'BASIL + ACDS': 2, 'FedSharp': 3, 'FedSharp+': 4}
    chosen = [r for r in summary if r['method'] == 'Clean (FedSharp control)' or
              (r['environment'] == 'joint' and r['attack'] == 'hidden')]
    chosen.sort(key=lambda r: (r['dataset'] != 'mnist', r['partition'] != 'iid', order[r['method']]))
    for row in chosen:
        split = 'IID' if row['partition'] == 'iid' else 'non-IID'
        lines.append(f"| {row['dataset']} / {split} | {row['method']} | {row['medianAverageRoundSeconds']:.2f} s | "
            f"{hms(row['median500RoundSeconds'])} | {hms(row['minimum500RoundSeconds'])} to {hms(row['maximum500RoundSeconds'])} | {row['hosts']} |")
    lines += ['', 'BASIL + ACDS and FedSharp+ have no IID cases in the selected study. '
        'They must not be assigned invented IID timing values.', '',
        '## Existing noise-only runs', '',
        'Only BASIL MNIST noise-only runs existed when this report was prepared. '
        'The new 18-case FedSharp/FedSharp+ noise-only campaign has no measured runtime yet.', '',
        '| Dataset/split | Method | Average round | 500 rounds |', '| --- | --- | ---: | ---: |']
    for row in summary:
        if row['environment'] == 'noise_only':
            lines.append(f"| {row['dataset']} / {row['partition']} | {row['method']} | {row['medianAverageRoundSeconds']:.2f} s | {hms(row['median500RoundSeconds'])} |")
    lines += ['', '## Separate official-test evaluation timing', '',
        'These saved local CPU evaluation times cover 100 client models, each classifying all 10,000 test images '
        'in batches of 256 with four TensorFlow intra-op threads. They include prediction, metric calculation, '
        'progress writes and compressed prediction export after the models are loaded. '
        'They exclude checkpoint loading, are not single-image latency, and cannot predict HPC training time. '
        'Only the non-IID, Hidden attack + noise, seed-2025 final models were tested.', '',
        '| Dataset | Method | Evaluation loop seconds for 1,000,000 client-image predictions |',
        '| --- | --- | ---: |']
    inference = []
    for name in ('full-official-test-20261110', 'basil-full-test-20261110'):
        path = repo / 'docs/results/official-test-20261110' / name / 'results.json'
        inputs[str(path.relative_to(repo))] = sha(path)
        for row in json.loads(path.read_text())['results']:
            inference.append(dict(dataset=row['dataset'], method=row['method'], seconds=row['elapsedSeconds']))
    for row in sorted(inference, key=lambda r: (r['dataset'] != 'mnist', order[r['method']])):
        lines.append(f"| {row['dataset']} | {row['method']} | {row['seconds']:.3f} |")
    lines += ['', '## Reproduce', '',
        'Run `python3 scripts/report_training_timings.py --destination <new-empty-directory>`. '
        'The source manifests, selected run IDs, completion counters and input checksums are checked. '
        'Original source results are read only.', '',
        'All CSV timing quantities are seconds. Clock formatting is hours:minutes:seconds. '
        'The new campaign also saves per-round wall intervals in `round_timings.jsonl`, '
        'explicitly identifying partial rounds after recovery.', '']
    (destination / 'report.md').write_text('\n'.join(lines))
    (destination / 'inputs.json').write_text(json.dumps(dict(inputSha256=inputs,
        selectedRuns=len(rows), groups=len(summary), matchedBenchmark=False), indent=2) + '\n')
    outputs = {p.name: sha(p) for p in sorted(destination.iterdir()) if p.is_file()}
    (destination / 'SHA256.json').write_text(json.dumps(outputs, indent=2) + '\n')
    return chosen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.destination), indent=2))
