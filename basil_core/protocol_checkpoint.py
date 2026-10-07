"""Atomic activation-boundary checkpoints and verified legacy reconstruction."""
from __future__ import annotations

import hashlib
import json
import os
from zipfile import BadZipFile
from functools import lru_cache
from pathlib import Path

import numpy as np

from basil_core.artifact_paths import writable_output

STATE_ARRAYS = (
    'activation_accuracy', 'activation_per_class', 'round_node_accuracy',
    'round_node_per_class', 'selected_sources', 'selected_rounds', 'round_node_loss',
)


@lru_cache(maxsize=1)
def implementation_fingerprint():
    root=Path(__file__).resolve().parents[1]
    paths=('basil_core/models.py','basil_core/attacks.py','basil_core/trainer.py',
           'basil_core/data/cifar.py','basil_core/research_protocol.py','basil_core/iid_study.py')
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def read_activation_records(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # An abrupt shutdown can leave one incomplete final write.
            raise ValueError(f'Malformed activation record {index + 1}: {path}')
        if (record['round'], record['nodeId']) != divmod(index, 10):
            raise ValueError(f'Non-contiguous activation history: {path}')
        records.append(record)
    return records


def save_checkpoint(path, config, state, *, partition_hash, initial_hash):
    path = writable_output(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: state[name] for name in STATE_ARRAYS}
    weights = {}

    def pack_params(params):
        from basil_core.research_protocol import params_hash
        digest = params_hash(params)
        if digest not in weights:
            weights[digest] = len(params)
            for index, value in enumerate(params):
                arrays[f'{digest}_{index}'] = value
        return digest

    memories = []
    for memory in state['memories']:
        memories.append([dict(sender=s.sender_id, round=s.round_id, params=pack_params(s.params),
            loss=s.sender_loss, accuracy=s.sender_accuracy, channel=s.channel_metadata)
            for s in memory.candidates()])
    metadata = dict(version=1, config=config, nextActivation=state['next_activation'],
        implementation=implementation_fingerprint(),
        completedRounds=state['completed_rounds'], partitionHash=partition_hash,
        initialHash=initial_hash, nodeParams=[pack_params(p) for p in state['node_params']],
        memories=memories, weights=weights, telemetry=state['telemetry'])
    arrays['metadata'] = np.asarray(json.dumps(metadata, allow_nan=False))
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    # Retain the previous committed checkpoint if the newest file is damaged.
    if path.exists():
        os.replace(path, path.with_name('checkpoint.previous.npz'))
    os.replace(temporary, path)


def load_checkpoint(path, config, *, partition_hash, initial_hash):
    from basil_core.research_protocol import RollingMemory, Snapshot, params_hash
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        if metadata['version'] != 1 or metadata['config'] != config:
            raise ValueError('Checkpoint configuration does not match this experiment.')
        if metadata['implementation'] != implementation_fingerprint():
            raise ValueError('Scientific implementation changed since checkpoint creation.')
        if (metadata['partitionHash'], metadata['initialHash']) != (partition_hash, initial_hash):
            raise ValueError('Checkpoint partition or initialization hash differs.')

        def unpack_params(digest):
            params = [saved[f'{digest}_{i}'].copy() for i in range(metadata['weights'][digest])]
            if params_hash(params) != digest or any(not np.isfinite(p).all() for p in params):
                raise ValueError('Checkpoint parameters are corrupt or non-finite.')
            return params

        memories = []
        for receiver, snapshots in enumerate(metadata['memories']):
            memory = RollingMemory(receiver)
            for s in snapshots:
                memory.receive(Snapshot(s['sender'], s['round'], unpack_params(s['params']),
                    s['loss'], s['accuracy'], s['channel']))
            if len(memory.candidates()) != 5:
                raise ValueError('Checkpoint must contain all five predecessor snapshots.')
            memories.append(memory)
        state = {name: saved[name].copy() for name in STATE_ARRAYS}
        state.update(next_activation=metadata['nextActivation'],
            completed_rounds=metadata['completedRounds'], memories=memories,
            node_params=[unpack_params(p) for p in metadata['nodeParams']],
            telemetry=metadata['telemetry'])
    cursor = state['next_activation']
    if len(state['node_params']) != 10 or len(memories) != 10:
        raise ValueError('Checkpoint requires ten logical nodes.')
    if len(state['telemetry']) != cursor or not 0 <= cursor <= config['nRounds'] * 10:
        raise ValueError('Checkpoint activation cursor is invalid.')
    if state['completed_rounds'] not in {cursor // 10, (cursor - 1) // 10}:
        raise ValueError('Checkpoint round boundary is invalid.')
    for name in STATE_ARRAYS:
        if state[name].shape[0] != config['nRounds'] or not np.isfinite(state[name]).all():
            raise ValueError(f'Invalid checkpoint metric array: {name}')
    return state


def recovery_plan(output, config):
    """Read-only preflight; no process is started and no result is modified."""
    output = writable_output(Path(output))
    path = output / 'run.json'
    if not path.exists():
        return dict(mode='fresh', savedActivations=0, startActivation=0, completedRounds=0)
    manifest = json.loads(path.read_text())
    if manifest['config'] != config:
        raise ValueError('Existing run configuration differs; recovery would change the protocol.')
    if manifest['status'] == 'completed':
        return dict(mode='completed', savedActivations=config['nRounds'] * 10,
                    startActivation=config['nRounds'] * 10, completedRounds=config['nRounds'])
    if manifest['status'] == 'failed' and 'Graceful stop' not in manifest.get('error', ''):
        raise ValueError('Failed research run requires investigation before recovery.')
    records = read_activation_records(output / 'partial_activation_telemetry.jsonl')
    checkpoints = (output / 'checkpoint.npz', output / 'checkpoint.previous.npz')
    for checkpoint in checkpoints:
        if checkpoint.exists():
            try:
                state = load_checkpoint(checkpoint, config, partition_hash=manifest['iidPartitionHash'],
                    initial_hash=manifest['initialModelHash'])
            except (ValueError, KeyError, OSError, EOFError, BadZipFile):
                continue
            if state['next_activation'] > len(records):
                continue
            if state['telemetry'] != records[:state['next_activation']]:
                raise ValueError('Checkpoint and saved activation evidence disagree.')
            return dict(mode='checkpoint', checkpoint=str(checkpoint),
                savedActivations=len(records), startActivation=state['next_activation'],
                completedRounds=state['completed_rounds'])
    return dict(mode='verified_replay', savedActivations=len(records), startActivation=0,
                completedRounds=int(manifest.get('completedRounds', 0)))


def verify_replayed_activation(record, expected):
    # Hashes cover the complete selected/trained/attacked and per-link weights.
    # Full equality also checks losses, gradients, predictions and keyed RNG data.
    if record != expected:
        differing = [key for key in record.keys() | expected.keys()
                     if record.get(key) != expected.get(key)]
        raise ValueError(f'Recovery diverged at round {record["round"]}, node {record["nodeId"]}: '
                         f'{", ".join(sorted(differing))}. Original evidence was not replaced.')
