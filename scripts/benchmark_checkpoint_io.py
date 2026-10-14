#!/usr/bin/env python3
"""Synthetic checkpoint I/O benchmark. No model, dataset or training is run."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from basil_core.protocol_checkpoint import STATE_ARRAYS, TelemetryStore, save_checkpoint
    from basil_core.research_protocol import RollingMemory, Snapshot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-jsonl', type=Path, help='Read one existing telemetry record; never modify it.')
    parser.add_argument('--activations', type=int, nargs='+', default=[100, 1000])
    args = parser.parse_args()
    if any(n < 10 or n > 5000 or n % 10 for n in args.activations):
        parser.error('Activation counts must be multiples of ten in [10, 5000].')
    if args.sample_jsonl:
        with args.sample_jsonl.open() as handle:
            sample = json.loads(next(handle))
    else:
        sample = dict(optimizerTelemetry=[dict(baseLoss=1., ebmPenalty=.001,
            baseGradientNorm=2., robustObjectiveGradientNorm=2.01) for _ in range(50)])
    params = [[np.full(117706, i / 100, np.float32)] for i in range(10)]
    memories = [RollingMemory(i) for i in range(10)]
    for i, memory in enumerate(memories):
        for distance in range(1, 6):
            sender = (i - distance) % 10
            memory.receive(Snapshot(sender, 0, params[sender], 1., .1))
    rows = []
    with tempfile.TemporaryDirectory(prefix='papermerge-checkpoint-benchmark-') as directory:
        for count in args.activations:
            history = []
            for index in range(count):
                record = copy.deepcopy(sample)
                record.update(round=index // 10, nodeId=index % 10)
                history.append(record)
            state = {name: np.zeros((500, 10), np.float32) for name in STATE_ARRAYS}
            state.update(next_activation=count, completed_rounds=count // 10,
                         node_params=params, memories=memories, telemetry=history)
            old = Path(directory) / f'v1-{count}' / 'checkpoint.npz'
            new = Path(directory) / f'v2-{count}' / 'checkpoint.npz'
            store = TelemetryStore(new)
            start = time.perf_counter()
            save_checkpoint(new, {}, state, partition_hash='synthetic', initial_hash='synthetic',
                            telemetry_store=store)
            initial_seconds = time.perf_counter() - start
            durations = {}
            for label, path, telemetry_store in [('v1', old, None), ('v2', new, store)]:
                measured = []
                for _ in range(3):
                    start = time.perf_counter()
                    save_checkpoint(path, {}, state, partition_hash='synthetic', initial_hash='synthetic',
                                    telemetry_store=telemetry_store)
                    measured.append(time.perf_counter() - start)
                durations[label] = statistics.median(measured)
            rows.append(dict(activations=count, v1Bytes=old.stat().st_size,
                v2Bytes=new.stat().st_size, v1MedianSeconds=durations['v1'],
                v2MedianSeconds=durations['v2'], v2InitialIndexSeconds=initial_seconds,
                steadyWriteSpeedup=durations['v1'] / durations['v2']))
    print(json.dumps(dict(scope='synthetic I/O only; not a research runtime prediction',
                          researchRunsStarted=0, results=rows), indent=2))


if __name__ == '__main__':
    main()
