"""Explicit CPU-state to GPU handoff; original CPU evidence is read-only."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
from zipfile import BadZipFile

from basil_core.protocol_checkpoint import (
    recovery_plan, read_activation_records, load_checkpoint, save_checkpoint, verify_replayed_activation,
)


def source_for_transition(root: Path, config: dict):
    source_name=config.get('recoverySourceDirectory')
    if not source_name:return None
    source=root/source_name
    if not (source/'run.json').exists():return None
    manifest=json.loads((source/'run.json').read_text())
    if manifest.get('device','CPU')!='CPU':raise ValueError('The transition source must be CPU evidence.')
    allowed={'experimentName','runId','displayName','resultDirectory','plotDirectory',
        'comparisonResultsDirectory','comparisonPlotsDirectory','executionMode','executionDevice',
        'deviceTransition','recoverySourceDirectory','estimatedCampaignRuntimeSeconds','runtimeEstimateModel'}
    old=manifest['config']
    differences={k for k in old.keys()|config.keys() if old.get(k)!=config.get(k)}
    if differences-allowed:raise ValueError(f'GPU transition changes scientific settings: {sorted(differences-allowed)}')
    return source,manifest,recovery_plan(source,old)


def transfer_cpu_state(root: Path, config: dict, data, audit: dict, runtime: dict) -> None:
    """Called only after manual Run; reconstruct old state before any GPU update."""
    from basil_core import research_protocol as protocol
    from basil_core.research_audit import ProtocolAudit, array_hash
    from basil_core.iid_study import IidWorker
    from basil_core.iid_runtime import runtime_metadata
    import tensorflow as tf
    output=root/config['resultDirectory']
    if (output/'run.json').exists():return
    details=source_for_transition(root,config)
    if details is None:return
    source,manifest,plan=details
    if plan['mode']=='completed':raise ValueError('The CPU condition already completed; there are no remaining rounds.')
    old=manifest['config'];protocol.validate_config(old)
    if manifest['tensorflowVersion']!=tf.__version__:raise ValueError('CPU reconstruction needs the original TensorFlow version.')
    records=read_activation_records(source/'partial_activation_telemetry.jsonl')
    expected=manifest.get('dataSha256') or json.loads((source/'audit/initial.json').read_text())
    for name,value in zip(('trainImagesHash','trainLabelsHash','testImagesHash','testLabelsHash'),data):
        if array_hash(value)!=expected[name]:raise ValueError('Device transition dataset bytes differ from the CPU run.')
    if audit['partitionHash']!=manifest['iidPartitionHash']:raise ValueError('Device transition partition differs.')
    output.mkdir(parents=True,exist_ok=True)
    reconstruction=output/'cpu_reconstruction';reconstruction.mkdir(exist_ok=True)
    checkpoint=reconstruction/'checkpoint.npz'
    state=None
    candidates=[checkpoint,checkpoint.with_name('checkpoint.previous.npz')]
    if plan['mode']=='checkpoint':candidates.append(Path(plan['checkpoint']))
    for path in candidates:
        if not path.exists():continue
        try:
            candidate=load_checkpoint(path,old,partition_hash=audit['partitionHash'],initial_hash=manifest['initialModelHash'])
        except (ValueError,KeyError,OSError,EOFError,BadZipFile):continue
        if candidate['next_activation']<=len(records) and candidate['telemetry']==records[:candidate['next_activation']]:
            if state is None or candidate['next_activation']>state['next_activation']:state=candidate
    with (source/'round_metrics.csv').open(newline='') as handle:
        rows=[{k:float(v) if v else None for k,v in row.items()} for row in csv.DictReader(handle)]
    stop=False;started=time.perf_counter()
    def stop_requested(*_args):
        nonlocal stop
        stop=True
    old_handlers={sig:signal.signal(sig,stop_requested) for sig in (signal.SIGTERM,signal.SIGINT)}
    class StateReady(Exception):pass
    def complete(value):
        return value['next_activation']==len(records) and value['completed_rounds']==len(rows)
    def checkpoint_cpu(value):
        nonlocal state
        state=value
        save_checkpoint(checkpoint,old,value,partition_hash=audit['partitionHash'],initial_hash=manifest['initialModelHash'])
        if complete(value):raise StateReady()
    def activation(record):
        index=record['round']*10+record['nodeId']
        if index>=len(records):raise ValueError('CPU reconstruction exceeded the saved prefix.')
        verify_replayed_activation(record,records[index])
        print(json.dumps({'recovery':dict(completed=index+1,total=len(records))}),flush=True)
    def round_complete(record):
        index=record['round']
        if index>=len(rows):raise ValueError('CPU reconstruction exceeded saved round metrics.')
        for key,value in (('full_test_accuracy',record['fullTestAccuracy']),('full_test_loss',record['fullTestLoss']),
            ('mean_node_accuracy',record['averageAccuracy']),('worst_node_accuracy',record['worstNodeAccuracy'])):
            if rows[index][key]!=value:raise ValueError(f'CPU reconstruction round {index}: {key} differs.')
    observer=None
    try:
        if state is None or not complete(state):
            cpu=runtime_metadata(tf,{})
            print(json.dumps(dict(event='device_ready',**cpu)),flush=True)
            print(json.dumps({'recovery':dict(completed=state['next_activation'] if state else 0,total=len(records))}),flush=True)
            observer=ProtocolAudit(reconstruction/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            with tf.device('/CPU:0'):
                try:
                    protocol.run_research_protocol(old,*data,worker_factory=IidWorker,audit=observer,
                        resume_state=state,activation_callback=activation,round_callback=round_complete,
                        checkpoint_callback=checkpoint_cpu,should_stop=lambda:stop or not records)
                except StateReady:pass
        if state is None or not complete(state):raise ValueError('CPU state is not at the complete saved evidence boundary.')
    finally:
        if observer:observer.close()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    if stop:raise InterruptedError('Graceful stop requested during CPU-state reconstruction.')
    # Seed the new output with unchanged metric bytes, never overwrite originals.
    files=('partial_activation_telemetry.jsonl','round_metrics.csv','partition_indices.npz','iid_partition_audit.json')
    hashes={}
    for name in files:
        path=source/name
        if path.exists():
            hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            shutil.copy2(path,output/name)
            if hashlib.sha256((output/name).read_bytes()).hexdigest()!=hashes[name]:raise ValueError('CPU evidence copy hash differs.')
    provenance=output/'cpu_provenance';provenance.mkdir(exist_ok=True)
    shutil.copy2(source/'run.json',provenance/'run.json')
    if (source/'audit/initial.json').exists():shutil.copy2(source/'audit/initial.json',provenance/'initial.json')
    save_checkpoint(output/'checkpoint.npz',config,state,
        partition_hash=audit['partitionHash'],initial_hash=manifest['initialModelHash'])
    transition=dict(sourceDirectory=str(source.relative_to(root)),sourceManifestSha256=hashlib.sha256((source/'run.json').read_bytes()).hexdigest(),
        sourceEvidenceSha256=hashes,activation=state['next_activation'],
        completedRounds=state['completed_rounds'],fromDevice='CPU',toDevice='GPU',
        exactStateTransfer=True,bitIdenticalCpuContinuation=False,
        reconstructionSeconds=time.perf_counter()-started,
        note='All weights and memories restored exactly. Subsequent GPU arithmetic is not a bit-identical CPU trajectory.')
    seeded=dict(manifest,config=config,experimentName=config['experimentName'],status='stopped',**runtime,
        runtimeSeconds=float(manifest.get('runtimeSeconds',0))+transition['reconstructionSeconds'],
        deviceTransition=transition,executionDeviceSegments=[dict(device='CPU',firstActivation=0,endExclusive=state['next_activation']),
            dict(device='GPU',firstActivation=state['next_activation'],endExclusive=None)])
    temporary=output/'run.json.tmp';temporary.write_text(json.dumps(seeded,indent=2,allow_nan=False)+'\n');temporary.replace(output/'run.json')
