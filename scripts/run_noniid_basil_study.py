#!/usr/bin/env python3
"""Prepare/run the four explicitly authorized non-IID 50-round local-only tests."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basil_core.execution_profile import profile_path
CONFIG_ROOT = ROOT / 'gui/configs/nonIID/sequential_basil_50r'


def save_json(path, payload):
    path = profile_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def execute(config, *, recover=False):
    # Set before any TensorFlow/W&B import. This runner never imports LiveTracker.
    os.environ['PAPERMERGE_LIVE_TRACKING'] = 'disabled'
    os.environ['WANDB_MODE'] = 'disabled'
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow, runtime_metadata
    prepare_environment(config['executionDevice'])
    import numpy as np
    import tensorflow as tf
    from basil_core.noniid_study import SnapshotAnchorWorker, partition_audit
    dirichlet=config.get('experimentProtocol')=='sequential_basil_dirichlet_v1'
    if dirichlet:
        from basil_core.dirichlet_study import validate_configuration, partition_audit
        validate_configuration(config)
    class_aware=config.get('lossStudyRevision')=='class_gradient_importance_anchor_v1'
    if class_aware:
        from basil_core.class_aware_retention import ClassAwareRetentionWorker
        worker_type=ClassAwareRetentionWorker
    else:
        worker_type=SnapshotAnchorWorker
    from basil_core.data.cifar import loadCifar10, _datasetArrays
    from basil_core.research_protocol import build_model, params_hash, run_research_protocol, validate_config, evaluate_params
    from basil_core.protocol_checkpoint import TelemetryStore, save_checkpoint, load_checkpoint, recovery_plan, read_activation_records, verify_replayed_activation
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    from basil_core.research_audit import ProtocolAudit, array_hash, tensor_stats

    class AnchorAudit(ProtocolAudit):
        def begin_batch(self, *args, **kwargs):
            with tf.device('/CPU:0'):
                super().begin_batch(*args, **kwargs)
        def gradients(self, gradients, values):
            super().gradients(gradients, values)
            self.current.update(zip(worker_type.extra_metric_names, map(float, values[6:])))
            self.current['ebmCorrectionRatio'] = float(values[6]) / (float(values[4]) + 1e-12)
        def failure(self, worker, error):
            # Never replay an anchored failure through a different CE-only loss,
            # or allocate a second GPU graph after OOM.
            self.write('failure.json', {**getattr(self, 'current', {}), 'exception': str(error),
                'objective': config['localObjective'], 'mu': worker.mu,
                'referenceHash': params_hash([a.numpy() for a in worker.reference]),
                'graphReplaySkipped': True, 'reason': 'Preserve exact anchor objective; avoid extra allocations.'})
            if self.pending:
                np.savez_compressed(self.directory / 'failure_batch.npz', x=self.pending['x'],
                    y=self.pending['y'], key=self.pending['key'],
                    **{f'weight_{i}': p for i, p in enumerate(self.pending['params'])},
                    **{f'anchor_{i}': a.numpy() for i, a in enumerate(worker.reference)})

    validate_config(config)
    if config.get('liveTrackingEnabled') is not False:
        raise ValueError('Cloud tracking must be explicitly disabled.')
    configure_tensorflow(tf, config['executionDevice'])
    runtime = runtime_metadata(tf, config)
    if config['executionDevice'] == 'GPU':
        verification = json.loads(profile_path(ROOT/'gui/worker_state/SNAPSHOT_ANCHOR_GPU_VERIFICATION.json').read_text())
        if verification['status'] != 'passed' or verification['objectiveSha256'] != hashlib.sha256((ROOT/'basil_core/noniid_study.py').read_bytes()).hexdigest():
            raise ValueError('Current snapshot-anchor GPU objective has not been verified.')
        if class_aware:
            verification=json.loads((ROOT/'gui/worker_state/CLASS_AWARE_GPU_VERIFICATION.json').read_text())
            if verification['status']!='passed' or verification['objectiveSha256']!=hashlib.sha256((ROOT/'basil_core/class_aware_retention.py').read_bytes()).hexdigest():
                raise ValueError('Current class-aware GPU objective has not been verified.')
    check_host_memory()
    train, test = loadCifar10(cacheDir=ROOT / 'experiments/cache/cifar10')
    x, y = _datasetArrays(train)
    tx, ty = _datasetArrays(test)
    audit_data, indices = partition_audit(y, ty, config['seed']) if dirichlet else partition_audit(y, ty)
    with tf.device('/CPU:0'):
        canonical_model = build_model(config['modelArchitecture'], config['seed'])
        initial = [v.numpy() for v in canonical_model.trainable_variables]
    initial_hash = params_hash(initial)
    if dirichlet and (initial_hash!=config['initialModelHash'] or audit_data['partitionHash']!=config['partitionHash']):
        raise ValueError('Actual partition/initialization differs from paired preparation.')
    if dirichlet:
        print(json.dumps(dict(event='device_ready',device=runtime['device'],deviceName=runtime.get('deviceName',''))),flush=True)
    output = profile_path(ROOT / config['resultDirectory'])
    if not output.is_relative_to(profile_path(ROOT / 'newResults/nonIID')):
        raise ValueError('Refusing non-IID output outside its tree.')
    output.mkdir(parents=True, exist_ok=True)
    previous = None
    state = None
    saved = []
    if (output / 'run.json').exists():
        if not recover:
            raise ValueError('Existing output is preserved; explicit --recover required.')
        previous = json.loads((output / 'run.json').read_text())
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
               for name, digest in previous['implementationSha256'].items()):
            raise ValueError('Non-IID scientific implementation changed; refusing resume.')
        plan = recovery_plan(output, config)
        if plan['mode'] == 'completed':
            print(f"{config['conditionId']}: already complete; skipped.", flush=True)
            return previous
        if plan['mode'] != 'checkpoint':
            raise ValueError('No verified checkpoint; refusing a guessed restart.')
        if previous['numericalExecution'] != runtime['numericalExecution']:
            raise ValueError('Resume device/software differs.')
        state = load_checkpoint(plan['checkpoint'], config, partition_hash=audit_data['partitionHash'], initial_hash=initial_hash)
        saved = read_activation_records(output / 'partial_activation_telemetry.jsonl')
        if len(saved) != state['next_activation']:
            raise ValueError('Saved log extends beyond committed checkpoint; investigate before resuming.')
    save_json(output / 'config.json', config)
    save_json(output / 'partition_audit.json', audit_data)
    data_hashes = {name: array_hash(v) for name, v in zip(['trainImages', 'trainLabels', 'testImages', 'testLabels'], [x, y, tx, ty])}
    if previous and previous['dataSha256'] != data_hashes:
        raise ValueError('Dataset bytes changed since checkpoint.')
    metadata = dict(status='running', config=config, runId=config['runId'], experimentProtocol=config['experimentProtocol'],
        researchValid=config.get('researchValid', True), purpose=config['purpose'], **runtime,
        executionMode='deterministic_single_process', multiprocessingUsed=False, wandbEnabled=False,
        tensorflowVersion=tf.__version__, pythonVersion=platform.python_version(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        gitCommit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        gitWorktreeDirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)),
        iidPartitionHash=audit_data['partitionHash'], partitionHash=audit_data['partitionHash'],
        initialModelHash=initial_hash, dataSha256=data_hashes,
        resolvedAttackerIds=config['resolvedAttackerIds'], attackType=config['attackType'],
        attackSourceStatus=config['attackSourceStatus'], hiddenAttackParameters=dict(blendRatio=.55, attackStrength=1.4),
        sigma_e=config['channelNoiseSigmaAbsolute'], sigma_e_squared=config['channelNoiseSigmaAbsolute']**2,
        snapshotAnchorMu=config['snapshotAnchorMu'], legacyLambdaIgnored=True,
        rngDerivation='SHA256(seed|stream|round|node|epoch|batch or sender|receiver)',
        implementationSha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'basil_core/noniid_study.py', ROOT/'basil_core/research_protocol.py', ROOT/'basil_core/attacks.py', Path(__file__)]})
    if class_aware:
        for path in ('basil_core/class_aware_retention.py','basil_core/protocol_checkpoint.py'):
            metadata['implementationSha256'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    if dirichlet:
        from basil_core.protocol_checkpoint import checkpoint_implementation
        metadata['implementationSha256']=checkpoint_implementation(config)
        metadata.update(informationModel='receiver_local_train_only',helpersEnabled=False)
    stop = False
    def stop_requested(_sig, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    started = time.perf_counter()
    prior_runtime = previous.get('runtimeSeconds', 0.) if previous else 0.
    if dirichlet and previous:
        print(json.dumps({'restored':dict(round=state['completed_rounds'],
            progress=state['next_activation']/(config['nRounds']*10),
            priorRuntimeSeconds=prior_runtime,startProgress=state['next_activation']/(config['nRounds']*10),
            recoveryTotal=0,recoveryCompleted=0,
            accuracyHistory=[(i,float(np.mean(state['round_node_accuracy'][i])),float(np.min(state['round_node_accuracy'][i])))
                for i in range(state['completed_rounds'])],
            activationHistory=[(r['round']+(r['nodeId']+1)/10,r['globalTestAccuracy']) for r in saved])}),flush=True)
    session = output / ('audit' if previous is None else f"audit_resume_{state['next_activation']}")
    audit = AnchorAudit(session)
    store = TelemetryStore(output / 'checkpoint.npz')
    guard = RuntimeMemoryGuard(tf, runtime['device'])
    rows = []
    if previous:
        with (output / 'round_metrics.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        rows = [r for r in rows if int(r['round']) < state['completed_rounds']]
    def checkpoint(current):
        save_checkpoint(output / 'checkpoint.npz', config, current,
            partition_hash=audit_data['partitionHash'], initial_hash=initial_hash, telemetry_store=store)
        metadata.update(completedRounds=current['completed_rounds'], completedActivations=current['next_activation'],
            memorySafety=guard.checkpoint(), runtimeSeconds=prior_runtime+time.perf_counter()-started)
        save_json(output / 'run.json', metadata)
        print(json.dumps(dict(event='checkpoint_saved',activation=current['next_activation'],
            completedRounds=current['completed_rounds'],checkpoint=str(output/'checkpoint.npz'))),flush=True)
    def activation(record):
        if params_hash(initial) != initial_hash:
            raise ValueError('Canonical initialization was mutated.')
        if record['snapshotAnchorReferenceHash'] != record['inputHash']:
            raise ValueError('Anchor differs from selected received snapshot.')
        if dirichlet:
            counts=np.asarray(audit_data['nodes'][record['nodeId']]['classCounts'])
            before=np.asarray(record['beforeTraining']['perClassGlobalTestAccuracy'])
            after=np.asarray(record['perClassGlobalTestAccuracy'])
            record['localClassCounts']=counts.tolist()
            record['perClassForgetting']=(before-after).tolist()
            record['meanClassForgetting']=float(np.mean(before-after))
            record['localDistributionWeightedAccuracyBefore']=float(np.dot(counts,before)/counts.sum())
            record['localDistributionWeightedAccuracyAfter']=float(np.dot(counts,after)/counts.sum())
        else:
            record['ownClassAccuracyBefore'] = record['beforeTraining']['perClassGlobalTestAccuracy'][record['nodeId']]
            record['ownClassAccuracyAfter'] = record['perClassGlobalTestAccuracy'][record['nodeId']]
            other = [i for i in range(10) if i != record['nodeId']]
            record['meanOtherClassForgetting'] = float(np.mean(np.asarray(record['beforeTraining']['perClassGlobalTestAccuracy'])[other]
                - np.asarray(record['perClassGlobalTestAccuracy'])[other]))
        record['selectedSnapshotGlobalAccuracyEvaluationOnly'] = record['beforeTraining']['globalTestAccuracy']
        if len(saved) > record['round']*10+record['nodeId']:
            verify_replayed_activation(record, saved[record['round']*10+record['nodeId']])
        else:
            with (output / 'partial_activation_telemetry.jsonl').open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
        if dirichlet:
            from gui.services.research_telemetry import activation_payload
            print(json.dumps(activation_payload(record,config['experimentName']),allow_nan=False),flush=True)
    def round_complete(event):
        r = event['round']
        # These arrays are measured round evaluations, not interpolated points.
        row = dict(round=r, full_test_accuracy=event['fullTestAccuracy'], mean_node_accuracy=event['averageAccuracy'],
            worst_node_accuracy=event['worstNodeAccuracy'], full_test_loss=event['fullTestLoss'])
        row.update({f'class_{i}_accuracy': a for i, a in enumerate(event['perClassAccuracy'])})
        rows.append(row)
        temp = output / 'round_metrics.csv.tmp'
        with temp.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader(); writer.writerows(rows)
        temp.replace(output / 'round_metrics.csv')
        print(json.dumps({'roundComplete':event} if dirichlet else dict(condition=config['conditionId'], completedRounds=r+1, totalRounds=config['nRounds'])), flush=True)
    def factory(model):
        worker = worker_type(model, mu=config['snapshotAnchorMu'], device=runtime['device'])
        worker.load(initial)
        return worker
    save_json(output / 'run.json', metadata)
    try:
        with tf.device(runtime['tensorflowDevice']):
            probe = worker_type(build_model(config['modelArchitecture'], config['seed']), mu=config['snapshotAnchorMu'])
            probe.load(initial)
            metadata['initialGlobalAccuracy'] = evaluate_params(probe, initial, tx, ty)[0]
            del probe
            result = run_research_protocol(config, x, y, tx, ty, worker_factory=factory,
                activation_callback=activation, round_callback=round_complete, audit=audit,
                should_stop=lambda: stop, resume_state=state, checkpoint_callback=checkpoint)
        np.savez_compressed(output / 'metrics.npz', **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        for i, params in enumerate(result['finalNodeParams']):
            np.savez_compressed(output / f'node_{i}_weights.npz', **{f'weight_{j}': w for j, w in enumerate(params)})
        with gzip.open(output / 'activation_telemetry.json.gz', 'wt') as handle:
            json.dump(result['telemetry'], handle, allow_nan=False)
        metadata.update(status='completed', completedRounds=config['nRounds'], completedActivations=config['nRounds']*10,
            runtimeSeconds=prior_runtime+time.perf_counter()-started,
            finalAverageAccuracy=float(result['averageAccuracy'][-1]), finalWorstAccuracy=float(result['worstNodeAccuracy'][-1]),
            finalFullTestAccuracy=float(result['fullTestAccuracy'][-1]))
    except Exception as error:
        metadata.update(status='stopped' if isinstance(error, InterruptedError) else 'failed', error=str(error),
                        runtimeSeconds=prior_runtime+time.perf_counter()-started)
        save_json(output / 'run.json', metadata)
        raise
    finally:
        audit.close()
    save_json(output / 'run.json', metadata)
    if os.environ.get('PAPERMERGE_SKIP_PLOTS')!='1':
        from reporting.noniid_study_plots import generate_condition_plots
        generate_condition_plots(output, profile_path(ROOT / config['plotDirectory']))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mu', type=float, default=.01)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--condition', choices=list('ABCD'))
    parser.add_argument('--config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.config:
        if not args.execute:
            raise ValueError('A config alone never authorizes training; explicit --execute is required.')
        execute(json.loads(args.config.read_text()), recover=args.recover)
        return
    os.environ['PAPERMERGE_LIVE_TRACKING'] = 'disabled'
    from basil_core.noniid_study import study_configs, verify_study
    configs = study_configs(args.mu, device=args.device)
    verify_study(configs)
    for cfg in configs:
        if args.prepare or args.execute:
            save_json(CONFIG_ROOT / f"{cfg['runId']}.json", cfg)
    print(json.dumps(dict(conditions=list('ABCD'), rounds=50, mu=args.mu, wandbEnabled=False,
                          differences=verify_study(configs), executionStarted=False)), flush=True)
    if args.execute:
        # Every condition runs in a fresh process: CUDA graphs and allocator state
        # are released between tests, while OS-mapped CIFAR files remain reusable.
        for cfg in configs:
            if args.condition and cfg['conditionId'] != args.condition:
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), '--execute', '--config', str(CONFIG_ROOT / f"{cfg['runId']}.json")]
            if args.recover:
                cmd.append('--recover')
            subprocess.run(cmd, check=True, env={**os.environ, 'PAPERMERGE_LIVE_TRACKING':'disabled', 'WANDB_MODE':'disabled'})
        if not args.condition:
            from reporting.noniid_study_plots import generate_study_report
            print(json.dumps(generate_study_report([ROOT/c['resultDirectory'] for c in configs],
                ROOT/'newResults/nonIID/ABCD_50r_comparison', ROOT/'newPlots/nonIID/ABCD_50r_comparison')), flush=True)


if __name__ == '__main__':
    main()
