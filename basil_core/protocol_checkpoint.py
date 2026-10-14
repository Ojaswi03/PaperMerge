"""Atomic activation-boundary checkpoints and verified legacy reconstruction."""
from __future__ import annotations

import hashlib
import gzip
import json
import os
import zlib
from zipfile import BadZipFile
from functools import lru_cache
from pathlib import Path


from basil_core.artifact_paths import writable_output

STATE_ARRAYS = (
    'activation_accuracy', 'activation_per_class', 'round_node_accuracy',
    'round_node_per_class', 'selected_sources', 'selected_rounds', 'round_node_loss',
)


class TelemetryStore:
    """Persist immutable completed records once, instead of rewriting history.

    The engine never edits a completed activation after its checkpoint. A new
    history list (e.g. on resume) is indexed afresh. Parameters, memories and
    the cursor still commit atomically at every activation as before.
    """
    def __init__(self, checkpoint_path):
        self.directory=Path(checkpoint_path).parent/'checkpoint_telemetry'
        self.history=None
        self.refs=[]

    def references(self, history):
        if history is not self.history:
            self.history=history
            self.refs=[]
        if len(history)<len(self.refs):
            raise ValueError('Completed checkpoint telemetry cannot be removed.')
        self.directory.mkdir(parents=True,exist_ok=True)
        for record in history[len(self.refs):]:
            raw=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            digest=hashlib.sha256(raw).hexdigest()
            path=self.directory/f'{digest}.json.gz'
            if path.exists():
                if gzip.decompress(path.read_bytes())!=raw:
                    raise ValueError('Existing checkpoint telemetry is corrupt.')
            else:
                temporary=path.with_suffix('.gz.tmp')
                with temporary.open('wb') as handle:
                    handle.write(gzip.compress(raw,compresslevel=1,mtime=0))
                    handle.flush();os.fsync(handle.fileno())
                os.replace(temporary,path)
                # Commit the directory entry before a checkpoint references it.
                descriptor=os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(descriptor)
                finally:os.close(descriptor)
            self.refs.append(digest)
        return list(self.refs)


def _load_telemetry(path, metadata):
    if metadata['version']==1:
        return metadata['telemetry']
    refs=metadata['telemetryRefs']
    if len(refs)!=metadata['nextActivation']:
        raise ValueError('Checkpoint telemetry reference count is invalid.')
    history=[]
    for digest in refs:
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid checkpoint telemetry reference.')
        try:
            raw=gzip.decompress((Path(path).parent/'checkpoint_telemetry'/f'{digest}.json.gz').read_bytes())
        except (OSError, EOFError, zlib.error) as error:
            raise ValueError('Checkpoint telemetry is missing or corrupt.') from error
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('Checkpoint telemetry hash differs.')
        history.append(json.loads(raw))
    return history


@lru_cache(maxsize=1)
def implementation_fingerprint():
    root=Path(__file__).resolve().parents[1]
    paths=('basil_core/models.py','basil_core/attacks.py','basil_core/trainer.py',
           'basil_core/data/cifar.py','basil_core/research_protocol.py','basil_core/iid_study.py')
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def checkpoint_implementation(config):
    fingerprints=dict(implementation_fingerprint())
    if config.get('experimentProtocol')=='sequential_basil_dirichlet_v1':
        root=Path(__file__).resolve().parents[1]
        for name in ('basil_core/noniid_study.py','basil_core/dirichlet_study.py',
                     'scripts/run_noniid_basil_study.py','scripts/run_dirichlet_condition.py',
                     'basil_core/protocol_checkpoint.py'):
            fingerprints[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if config.get('lossStudyRevision')=='class_gradient_importance_anchor_v1':
        root=Path(__file__).resolve().parents[1]
        for name in ('basil_core/class_aware_retention.py','basil_core/noniid_study.py','basil_core/protocol_checkpoint.py'):
            fingerprints[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    return fingerprints


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


def save_checkpoint(path, config, state, *, partition_hash, initial_hash, telemetry_store=None):
    import numpy as np
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
    retention_states = {}
    def pack_retention(snapshot):
        if snapshot.retention_state is None:
            return None
        from basil_core.class_aware_retention import validate_state, state_hash
        state = validate_state(snapshot.retention_state, snapshot.params)
        digest = state_hash(state)
        if digest not in retention_states:
            retention_states[digest] = dict(revision=state['revision'], visits=state['visits'],
                classes={c: pack_params(p) for c,p in state['classes'].items()})
        return digest
    for memory in state['memories']:
        snapshots=[]
        for s in memory.candidates():
            entry=dict(sender=s.sender_id, round=s.round_id, params=pack_params(s.params),
                loss=s.sender_loss, accuracy=s.sender_accuracy, channel=s.channel_metadata)
            if s.retention_state is not None:
                entry['retentionState']=pack_retention(s)
            snapshots.append(entry)
        memories.append(snapshots)
    metadata = dict(version=1, config=config, nextActivation=state['next_activation'],
        implementation=checkpoint_implementation(config),
        completedRounds=state['completed_rounds'], partitionHash=partition_hash,
        initialHash=initial_hash, nodeParams=[pack_params(p) for p in state['node_params']],
        memories=memories, weights=weights, telemetry=state['telemetry'])
    if retention_states:
        metadata['retentionStates']=retention_states
    if telemetry_store is not None:
        if len(state['telemetry'])!=state['next_activation']:
            raise ValueError('Checkpoint activation cursor differs from telemetry count.')
        metadata.pop('telemetry')
        metadata.update(version=2,telemetryRefs=telemetry_store.references(state['telemetry']))
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
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def load_checkpoint(path, config, *, partition_hash, initial_hash):
    import numpy as np
    from basil_core.research_protocol import RollingMemory, Snapshot, params_hash
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        if metadata['version'] not in (1,2) or metadata['config'] != config:
            raise ValueError('Checkpoint configuration does not match this experiment.')
        if metadata['implementation'] != checkpoint_implementation(config):
            raise ValueError('Scientific implementation changed since checkpoint creation.')
        if (metadata['partitionHash'], metadata['initialHash']) != (partition_hash, initial_hash):
            raise ValueError('Checkpoint partition or initialization hash differs.')

        def unpack_params(digest):
            params = [saved[f'{digest}_{i}'].copy() for i in range(metadata['weights'][digest])]
            if params_hash(params) != digest or any(not np.isfinite(p).all() for p in params):
                raise ValueError('Checkpoint parameters are corrupt or non-finite.')
            return params

        memories = []
        def unpack_retention(digest, params):
            if digest is None:
                return None
            from basil_core.class_aware_retention import validate_state, state_hash
            raw=metadata.get('retentionStates',{}).get(digest)
            if raw is None:
                raise ValueError('Missing checkpoint class-retention state.')
            state=validate_state(dict(revision=raw['revision'],visits=raw['visits'],
                classes={c:unpack_params(p) for c,p in raw['classes'].items()}),params)
            if state_hash(state)!=digest:
                raise ValueError('Checkpoint class-retention hash differs.')
            return state
        for receiver, snapshots in enumerate(metadata['memories']):
            memory = RollingMemory(receiver)
            for s in snapshots:
                params=unpack_params(s['params'])
                retention=unpack_retention(s.get('retentionState'),params)
                if config.get('lossStudyRevision')=='class_gradient_importance_anchor_v1' and s['round']>=0 and retention is None:
                    raise ValueError('Class-aware checkpoint lost an outbound retention state.')
                memory.receive(Snapshot(s['sender'], s['round'], params,
                    s['loss'], s['accuracy'], s['channel'],retention))
            if len(memory.candidates()) != 5:
                raise ValueError('Checkpoint must contain all five predecessor snapshots.')
            memories.append(memory)
        state = {name: saved[name].copy() for name in STATE_ARRAYS}
        state.update(next_activation=metadata['nextActivation'],
            completed_rounds=metadata['completedRounds'], memories=memories,
            node_params=[unpack_params(p) for p in metadata['nodeParams']],
            telemetry=_load_telemetry(path,metadata))
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
