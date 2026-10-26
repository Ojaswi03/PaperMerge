#!/usr/bin/env python3
"""Opt-in, fixed three-round CART A/B/C/D smoke checks plus an A beta=0 control.

Uses a reduced TRAIN subset per node and an evaluation-only TEST subset. The
attack starts at round 1 solely to exercise its branch within the small budget.
Outputs are diagnostics, never production research or a readiness approval.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def require_identity(path, expected):
    previous=json.loads(path.read_text())
    if any(previous.get(k)!=v for k,v in expected.items()):
        raise ValueError('Resume source, data, runtime or configuration identity changed; existing evidence retained.')
    return previous


def keep_json(path, value):
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError(f'Refusing to replace existing diagnostic identity: {path.name}')
    else:
        save_json(path,value)


def sources():
    from basil_core.cart_memory_study import configurations
    from basil_core.protocol_checkpoint import checkpoint_implementation
    return checkpoint_implementation(configurations('a'*64, 'b'*64)[0])


def archive_sources(directory, source):
    directory.mkdir(exist_ok=True)
    for filename,digest in source.items():
        raw=(ROOT/filename).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('Source changed during diagnostic.')
        target=directory/digest
        if target.exists():
            if target.read_bytes()!=raw:
                raise ValueError('Existing source archive is corrupt; refusing replacement.')
        else:
            target.write_bytes(raw)


def verify(tf, device):
    """Real CNN CE/EBM/retention CPU/device gradients; reports stay with this run."""
    import numpy as np
    from basil_core.cart_local_memory import retained_gradients
    from basil_core.dirichlet_calibration import calibration_offsets, calibrated_anchor_gradients
    from basil_core.research_protocol import build_model, params_hash
    x = np.random.default_rng(17).uniform(0, 1, (19, 32, 32, 3)).astype(np.float32)
    y = np.arange(19, dtype=np.int32) % 10
    offsets = calibration_offsets(np.arange(1, 11), method='prior_adjustment')
    outcomes = {}
    for selected in dict.fromkeys(('CPU', device)):
        with tf.device(f'/{selected}:0'):
            model = build_model('basil_paper_cnn', 2025)
            initial = [v.numpy() for v in model.trainable_variables]
            reference = [tf.constant(v+.001) for v in initial]
            teacher = tf.nn.log_softmax(model(x[:10], training=False)+tf.range(10, dtype=tf.float32)*.02)
            weights = tf.fill((10,), .04)
            outcomes[selected] = {}
            for coefficient in (0., .0001):
                grads, values = retained_gradients(model, x, y, reference, offsets,
                    x[:10], teacher, weights, coefficient=coefficient)
                base, old_values = calibrated_anchor_gradients(model, x, y, reference,
                    offsets, coefficient=coefficient)
                zero, zero_values = retained_gradients(model, x, y, reference, offsets,
                    x[:10], teacher, weights, beta=0., coefficient=coefficient)
                for a, b in zip(base, zero):
                    np.testing.assert_array_equal(a.numpy(), b.numpy())
                np.testing.assert_array_equal(np.array(old_values), np.array(zero_values[:15]))
                assert float(values[16]) > 0
                assert all(selected in g.device for g in grads)
                assert params_hash(initial) == params_hash([v.numpy() for v in model.trainable_variables])
                outcomes[selected][str(coefficient)] = [g.numpy() for g in grads]
    comparisons = {}
    for coefficient in ('0.0', '0.0001'):
        a, b = outcomes['CPU'][coefficient], outcomes[device][coefficient]
        first = np.concatenate([g.ravel() for g in a])
        second = np.concatenate([g.ravel() for g in b])
        relative = float(np.linalg.norm(first-second)/max(np.linalg.norm(first), 1e-12))
        update = float(np.max(np.abs(.05*(first-second))))
        assert relative < .01 and update < 1e-4
        comparisons[coefficient] = dict(gradientRelativeL2Difference=relative, maxUpdateDifference=update)
    return dict(status='passed', device=device, productionRunsStarted=0,
        comparisons=comparisons, betaZeroExactlyMatchesBaseline=True)


def run(args):
    from basil_core.iid_runtime import prepare_environment, configure_tensorflow
    from basil_core.execution_profile import current_profile
    from basil_core.terminal.progress import DiagnosticProgress
    from basil_core.cart_memory_study import configurations
    from basil_core.cart_local_memory import LocalMemoryCartWorker
    from basil_core.protocol_checkpoint import load_checkpoint, save_checkpoint, TelemetryStore
    from basil_core.memory_safety import RuntimeMemoryGuard, check_host_memory
    from basil_core.research_protocol import build_model, params_hash, run_research_protocol, keyed_rng
    from basil_core.dirichlet_study import partition_audit
    from basil_core.dirichlet_diagnosis import array_identity_hash
    root = args.output_root.resolve()
    if not root.is_relative_to(ROOT/'.papermerge/local-tests'):
        raise ValueError('Smoke outputs must stay inside .papermerge/local-tests, outside published results.')
    if root.exists() and not args.recover:
        raise FileExistsError('Output exists: use a fresh path or explicit --recover; no overwrite.')
    prepare_environment(args.device)
    os.environ.update(WANDB_MODE='disabled', PAPERMERGE_LIVE_TRACKING='disabled')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf, args.device)
    physical_gpu=current_profile().gpu if args.device=='GPU' else None
    device_label=f'Physical GPU {physical_gpu}' if args.device=='GPU' else 'CPU'
    check_host_memory()
    source = sources()
    runtime = dict(python=sys.version, tensorflow=tf.__version__, numpy=np.__version__,
        keras=tf.keras.__version__, device=args.device, build=tf.sysconfig.get_build_info(),
        gpuNames=[tf.config.experimental.get_device_details(d).get('device_name')
            for d in tf.config.list_physical_devices('GPU')])
    if (root/'verification.json').exists():
        require_identity(root/'verification.json',dict(implementationSha256=source,runtime=runtime))
    stages = ('Synthetic gradients', 'TRAIN partition and diagnostic data') + tuple(args.conditions)
    root.mkdir(parents=True, exist_ok=True)
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    outcomes = {}
    with DiagnosticProgress(f'{device_label} (CART {args.profile})', stages, interval=5.) as progress:
        verified = verify(tf, args.device)
        report_path=root/'verification.json'
        if report_path.exists():
            report_path=root/f'verification-attempt-{time.time_ns()}.json'
        save_json(report_path, dict(verified, implementationSha256=source, runtime=runtime))
        if args.verify_only:
            progress.finish('stopped')
            print(json.dumps(verified), flush=True)
            return
        progress.begin(stages[1])
        cache = ROOT/'experiments/cache/cifar10'
        x, y, tx, ty = [np.load(cache/f, mmap_mode='r', allow_pickle=False) for f in (
            'x_train_float32.npy', 'y_train_int32.npy', 'x_test_float32.npy', 'y_test_int32.npy')]
        if x.shape != (50000, 32, 32, 3) or tx.shape != (10000, 32, 32, 3):
            raise ValueError('Expected the existing complete CIFAR-10 cache.')
        partition, chunks = partition_audit(y, ty)
        if args.profile == 'smoke':
            selected = np.concatenate([keyed_rng(2025, 'cart_smoke_test', c).choice(
                np.flatnonzero(ty==c), 100, replace=False) for c in range(10)])
            test_x, test_y = tx[selected], ty[selected]
        else:
            test_x, test_y = tx, ty
        data_hash = {k:array_identity_hash(v) for k,v in zip(
            ('trainImages', 'trainLabels', 'evaluationImages', 'evaluationLabels'), (x,y,test_x,test_y))}
        with tf.device('/CPU:0'):
            initial_hash = params_hash([v.numpy() for v in build_model('basil_paper_cnn',2025).trainable_variables])
        configs = {c['runId']:c for c in configurations(partition['partitionHash'],initial_hash,args.device,args.profile)}
        # Preflight every existing condition before rewriting any shared plan or
        # partition identity. Incompatible recovery must preserve old evidence.
        for name in args.conditions:
            if (root/name).exists():
                require_identity(root/name/'run.json',dict(config=configs[name],
                    implementationSha256=source,dataSha256=data_hash,runtime=runtime))
        actual_sizes=[min(1024,len(c)) if args.profile=='smoke' else len(c) for c in chunks]
        keep_json(root/'partition.json', partition)
        keep_json(root/'plan.json', dict(configs=[configs[n] for n in args.conditions],
            productionRunStarted=False, profile=args.profile,
            subsetWarning='Reduced TRAIN counts change local priors; not a full-data reproduction.' if args.profile=='smoke' else None,
            actualSamplesPerNode=actual_sizes,
            actualOptimizerStepsPerRound=sum(5*((count+511)//512) for count in actual_sizes)))
        for name in args.conditions:
            if stop:
                raise InterruptedError('Stopped before next condition.')
            progress.begin(name)
            cfg = configs[name]
            directory = root/name
            state = None
            identity = dict(config=cfg, implementationSha256=source, dataSha256=data_hash, runtime=runtime)
            metadata = dict(identity, status='running', researchValid=False, productionRunStarted=False,
                completedActivations=0, completedRounds=0, runtimeSeconds=0.)
            if directory.exists():
                previous = require_identity(directory/'run.json',identity)
                if previous['status']=='completed':
                    outcomes[name] = previous
                    continue
                state = load_checkpoint(directory/'checkpoint.npz',cfg,
                    partition_hash=partition['partitionHash'],initial_hash=initial_hash)
                metadata.update(completedActivations=state['next_activation'],
                    completedRounds=state['completed_rounds'],runtimeSeconds=previous['runtimeSeconds'])
            directory.mkdir(exist_ok=True)
            snapshots = directory/'source_snapshots'
            archive_sources(snapshots,source)
            guard = RuntimeMemoryGuard(tf,args.device)
            store = TelemetryStore(directory/'checkpoint.npz')
            started = time.monotonic()
            prior = metadata['runtimeSeconds']
            rows = [] if state is None else [dict(round=r,
                mean=float(state['round_node_accuracy'][r].mean()),
                worst=float(state['round_node_accuracy'][r].min()),
                node9=float(state['round_node_accuracy'][r,9])) for r in range(state['completed_rounds'])]
            def checkpoint(current):
                save_checkpoint(directory/'checkpoint.npz',cfg,current,
                    partition_hash=partition['partitionHash'],initial_hash=initial_hash,telemetry_store=store)
                metadata.update(completedActivations=current['next_activation'],
                    completedRounds=current['completed_rounds'],runtimeSeconds=prior+time.monotonic()-started,
                    memorySafety=guard.checkpoint())
                save_json(directory/'run.json',metadata)
            def activation(record):
                done = record['round']*10+record['nodeId']+1
                elapsed = time.monotonic()-started
                performed = done-(state['next_activation'] if state else 0)
                eta = elapsed/performed*(30-done)
                print(f'{name} | {device_label} | activation {done}/30 | '
                    f'accuracy {record["globalTestAccuracy"]:.1%} | elapsed {elapsed:.0f}s | ETA ~{eta:.0f}s', flush=True)
            def round_done(row):
                rows.append(dict(round=row['round'],mean=row['averageAccuracy'],
                    worst=row['worstNodeAccuracy'],node9=row['fullTestAccuracy']))
                with (directory/'round_metrics.csv').open('w', newline='') as handle:
                    writer=csv.DictWriter(handle, fieldnames=['round','mean','worst','node9'])
                    writer.writeheader()
                    writer.writerows(rows)
                print(f'{name} | round {len(rows)}/3 | mean {rows[-1]["mean"]:.2%} | '
                    f'worst {rows[-1]["worst"]:.2%}',flush=True)
            save_json(directory/'run.json',metadata)
            try:
                def factory(model):
                    if params_hash([v.numpy() for v in model.trainable_variables]) != initial_hash:
                        raise ValueError('Actual model initialization differs from the canonical CPU hash.')
                    return LocalMemoryCartWorker(model,beta=cfg['cartBeta'],device=args.device)
                with tf.device(f'/{args.device}:0'):
                    result = run_research_protocol(cfg,x,y,test_x,test_y,
                        worker_factory=factory,
                        resume_state=state,checkpoint_callback=checkpoint,
                        activation_callback=activation,round_callback=round_done,should_stop=lambda:stop)
                np.savez_compressed(directory/'metrics.npz', **{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
                (directory/'activation_telemetry.json.gz').write_bytes(gzip.compress(
                    json.dumps(result['telemetry'],allow_nan=False).encode(),mtime=0))
                metadata.update(status='completed', completedRounds=3, completedActivations=30,
                    runtimeSeconds=prior+time.monotonic()-started, meanAccuracy=result['averageAccuracy'].tolist(),
                    worstAccuracy=result['worstNodeAccuracy'].tolist(),
                    retentionActiveSteps=sum(s['cartRetentionGradientNorm']>0 for r in result['telemetry'] for s in r['optimizerTelemetry']),
                    attackActivations=sum(r['attackActive'] for r in result['telemetry']),
                    noisyActivations=sum(r['channelNoiseActive'] for r in result['telemetry']),
                    memorySafety=guard.checkpoint())
                del result
            except BaseException as error:
                metadata.update(status='interrupted' if isinstance(error,InterruptedError) else 'failed',
                    error=str(error),runtimeSeconds=prior+time.monotonic()-started)
                raise
            finally:
                save_json(directory/'run.json',metadata)
            outcomes[name]=metadata
            save_json(root/'campaign.json',dict(status='running',outcomes=outcomes,productionRunStarted=False))
        progress.finish('passed')
    save_json(root/'campaign.json',dict(status='completed',outcomes=outcomes,productionRunStarted=False))
    print('ALL CART DIAGNOSTICS COMPLETE. Three-round results require learning review.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--recover',action='store_true')
    parser.add_argument('--profile',choices=('smoke','clean_learning'),default='smoke')
    parser.add_argument('--device',choices=('CPU','GPU'),default='GPU')
    parser.add_argument('--output-root',type=Path,default=ROOT/'.papermerge/local-tests/cart-memory-v1')
    parser.add_argument('--conditions',nargs='+',choices=('A_beta0','A','B','C','D'))
    args=parser.parse_args()
    allowed=['A_beta0','A','B','C','D'] if args.profile=='smoke' else ['A_beta0','A']
    args.conditions=args.conditions or allowed
    if any(name not in allowed for name in args.conditions):
        parser.error('The clean learning profile contains only A_beta0 and A.')
    if len(set(args.conditions))!=len(args.conditions):
        parser.error('Conditions must be distinct.')
    if not (args.execute or args.verify_only):
        print(json.dumps(dict(trainingStarted=False,rounds=3,localEpochs=5,profile=args.profile,
            maxTrainSamplesPerNode=1024 if args.profile=='smoke' else None,
            testSamplesPerClass=100 if args.profile=='smoke' else 1000,
            attackStartRound=1 if args.profile=='smoke' else 20,
            conditions=args.conditions,researchValid=False)))
        return
    try:
        run(args)
    except (ValueError,FileExistsError,InterruptedError) as error:
        parser.exit(2,str(error)+'\n')


if __name__=='__main__':
    main()
