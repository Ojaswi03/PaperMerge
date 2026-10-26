#!/usr/bin/env python3
"""Fixed TRAIN-only prior/KL/barrier comparisons. No official TEST I/O."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def verify_barrier(tf,device):
    import numpy as np
    from basil_core.cart_barrier import barrier_gradients
    from basil_core.research_protocol import build_model
    from basil_core.dirichlet_calibration import calibrated_anchor_gradients
    x=np.random.default_rng(18).uniform(0,1,(19,32,32,3)).astype(np.float32)
    y=np.arange(19,dtype=np.int32)%10
    average=np.stack([(y==c)/sum(y==c) for c in range(10)]).astype(np.float32)
    outcomes={}
    for selected in dict.fromkeys(('CPU',device)):
        with tf.device(f'/{selected}:0'):
            model=build_model('basil_paper_cnn',2025)
            reference=[tf.identity(v) for v in model.trainable_variables]
            for c in (0.,.0001):
                applied,values=barrier_gradients(model,x,y,reference,tf.zeros(10),x,y,
                    average,tf.ones(10),tf.ones(10),coefficient=c)
                zero,_=barrier_gradients(model,x,y,reference,tf.zeros(10),x,y,
                    average,tf.ones(10),tf.ones(10),coefficient=c,beta=0.)
                base,_=calibrated_anchor_gradients(model,x,y,reference,tf.zeros(10),coefficient=c)
                for first,second in zip(zero,base):np.testing.assert_array_equal(first.numpy(),second.numpy())
                assert float(values[16])>0
                outcomes[selected,str(c)]=np.concatenate([g.numpy().ravel() for g in applied])
                bounded,measure=barrier_gradients(model,x,y,reference,tf.zeros(10),x,y,
                    average,tf.ones(10),tf.ones(10),coefficient=c,correction_ratio_cap=.1)
                assert float(measure[16])<=.1*float(measure[4])+1e-7
                outcomes[selected,str(c)+'budgeted']=np.concatenate([g.numpy().ravel() for g in bounded])
    result={}
    for c in ('0.0','0.0001'):
        a,b=outcomes['CPU',c],outcomes[device,c]
        relative=float(np.linalg.norm(a-b)/np.linalg.norm(a))
        assert relative<.01
        result[c]=relative
        a,b=outcomes['CPU',c+'budgeted'],outcomes[device,c+'budgeted']
        relative=float(np.linalg.norm(a-b)/np.linalg.norm(a))
        assert relative<.01
        result[c+'budgeted']=relative
    return dict(status='passed',barrierGradientRelativeL2=result,betaZeroExact=True)


def runtime_identity(tf,np,device):
    """Bind recovery to static hardware and arithmetic settings, not live load."""
    from dataclasses import asdict
    import platform
    import socket
    from basil_core.execution_profile import current_profile,gpu_inventory,resource_identity
    from basil_core.iid_runtime import gpu_memory_limit
    profile=current_profile()
    gpu=None
    if device=='GPU':
        devices=gpu_inventory()
        profile=profile.bind(devices)
        selected=next(d for d in devices if d['uuid']==profile.gpu_uuid)
        gpu={key:selected[key] for key in ('index','uuid','name','driver_version')}
    return dict(python=sys.version,executable=sys.executable,tensorflow=tf.__version__,numpy=np.__version__,
        device=device,build=tf.sysconfig.get_build_info(),software=resource_identity(profile)['software'],host=socket.gethostname(),
        operatingSystem=platform.system(),architecture=platform.machine(),
        profile=asdict(profile),gpu=gpu,gpuMemoryLimitMiB=gpu_memory_limit() if gpu else None,
        intraOpThreads=tf.config.threading.get_intra_op_parallelism_threads(),
        interOpThreads=tf.config.threading.get_inter_op_parallelism_threads(),
        tensorFloat32=tf.config.experimental.tensor_float_32_execution_enabled() if gpu else None,
        oneDnn=os.environ.get('TF_ENABLE_ONEDNN_OPTS'),deterministicOps=os.environ.get('TF_DETERMINISTIC_OPS'))


def device_receipt(profile):
    return ROOT/'.papermerge/verification/cart-replication'/f'{profile.gpu_uuid}.json'


def verify_replication_device():
    """Synthetic checks only; no CIFAR load or optimizer study is launched."""
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    from basil_core.execution_profile import current_profile,gpu_inventory
    from basil_core.cart_validation import REPLICATION_SEEDS,selected_configurations
    from basil_core.protocol_checkpoint import checkpoint_implementation
    from basil_core.research_protocol import build_model,params_hash
    from scripts.run_cart_memory_smoke import save_json
    prepare_environment('GPU')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,'GPU')
    checked=verify_barrier(tf,'GPU')
    hashes={}
    for seed in REPLICATION_SEEDS:
        with tf.device('/CPU:0'):cpu=params_hash(build_model('basil_paper_cnn',seed).get_weights())
        with tf.device('/GPU:0'):gpu=params_hash(build_model('basil_paper_cnn',seed).get_weights())
        if cpu!=gpu:raise ValueError('CPU/GPU initial weights differ; replication was not approved.')
        hashes[str(seed)]=gpu
    if len(set(hashes.values()))!=3:raise ValueError('Seeded initial models did not differ.')
    profile=current_profile().bind(gpu_inventory())
    source=checkpoint_implementation(selected_configurations('a'*64,'b'*64,'GPU','replication_30r')[0])
    report=dict(status='passed',runtime=runtime_identity(tf,np,'GPU'),implementationSha256=source,
        initialModelHashes=hashes,gradientCheck=checked,researchRunsStarted=0)
    save_json(device_receipt(profile),report)
    return dict(status='passed',physicalGpuId=profile.gpu,uuid=profile.gpu_uuid,researchRunsStarted=0)


def require_device_receipt(runtime,source):
    from basil_core.execution_profile import ExecutionProfile
    profile=ExecutionProfile(**runtime['profile'])
    try:report=json.loads(device_receipt(profile).read_text())
    except (OSError,ValueError) as error:
        raise ValueError('CART GPU receipt is missing/unreadable. Run scripts/run_cart_replication.py --verify-gpus first.') from error
    def comparable(value):
        # File placement does not change arithmetic. GPU UUID, cap and software do.
        import copy
        value=copy.deepcopy(value);value.get('profile',{}).pop('output_root',None)
        return value
    if (report.get('status')!='passed' or report.get('implementationSha256')!=source or
            comparable(report.get('runtime',{}))!=comparable(runtime)):
        raise ValueError('CART device verification differs from source/runtime. Run --verify-gpus before training.')


def preflight(args):
    """Read-only input checks before TensorFlow, GPU leases or output writes."""
    import numpy as np
    from basil_core.cart_validation import selected_configurations,partition,split_training,validate_config,validate_training_arrays
    from basil_core.dirichlet_diagnosis import array_identity_hash
    from basil_core.protocol_checkpoint import checkpoint_implementation
    from scripts.run_cart_memory_smoke import require_identity
    root=args.output.resolve()
    if not root.is_relative_to(ROOT/'.papermerge/local-tests'):
        raise ValueError('Use an isolated .papermerge/local-tests directory.')
    if root.exists() and not args.recover:
        raise FileExistsError('Output exists; no overwrite. Use a fresh directory or --recover.')
    cache=ROOT/'experiments/cache/cifar10'
    names=('x_train_float32.npy','y_train_int32.npy')
    missing=[name for name in names if not (cache/name).is_file()]
    if missing:raise ValueError('Missing TRAIN cache: '+', '.join(missing)+'. Prepare CIFAR before execution.')
    try:full_x,full_y=[np.load(cache/name,mmap_mode='r',allow_pickle=False) for name in names]
    except (OSError,EOFError,ValueError) as error:
        raise ValueError(f'TRAIN cache is unreadable: {error}') from error
    validate_training_arrays(full_x,full_y)
    fit,validation=split_training(full_y)
    x,y=full_x[fit],full_y[fit]
    vx,vy=full_x[validation],full_y[validation]
    seed=getattr(args,'seed',2025)
    partition_hash,chunks=partition(y,seed=seed)
    covered=np.concatenate(chunks)
    if (len(chunks)!=10 or any(len(chunk)==0 for chunk in chunks) or
            not np.array_equal(np.sort(covered),np.arange(len(y)))):
        raise ValueError('Dirichlet partition must cover TRAIN exactly once with ten nonempty nodes.')
    # The actual model hash is filled after numerical initialization. No model
    # or GPU is needed to validate the fixed experiment envelope and workload.
    pair=getattr(args,'pair',None)
    configs=selected_configurations(partition_hash,'b'*64,args.device,args.stage,seed=seed,pair=pair)
    for config in configs:validate_config(config)
    source=checkpoint_implementation(configs[0])
    data={k:array_identity_hash(v) for k,v in zip(
        ('fitImages','fitLabels','validationImages','validationLabels'),(x,y,vx,vy))}
    campaign_plan=getattr(args,'campaign_plan',None)
    if campaign_plan is not None:
        from dataclasses import asdict
        from basil_core.execution_profile import current_profile
        if args.stage!='replication_30r':raise ValueError('Campaign contracts apply only to the replication stage.')
        campaign=json.loads(Path(campaign_plan).read_text())
        job=next((j for j in campaign['jobs'] if j['seed']==seed and j['pair']==pair),None)
        expected=next((c for c in campaign['checks'] if job and c['job']==job['id']),None)
        if (job is None or expected is None or Path(job['output']).resolve()!=root or
                job['resources']!=asdict(current_profile()) or expected['implementationSha256']!=source or
                expected['dataSha256']!=data or expected['partitionHash']!=partition_hash):
            raise ValueError('Source/data/resources differ from the frozen campaign plan; no GPU work started.')
    identity=dict(implementationSha256=source,dataSha256=data)
    if args.stage=='replication_30r':identity.update(stage=args.stage,seed=seed,pair=pair)
    if (root/'identity.json').exists():require_identity(root/'identity.json',identity)
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Existing output has no recovery identity; refusing to reuse its evidence.')
    per_round=sum(5*((len(chunk)+511)//512) for chunk in chunks)
    report=dict(status='inputs_checked',trainingStarted=False,tensorflowImported='tensorflow' in sys.modules,
        gpuExecutionVerified=False,initialModelHashVerified=False,officialTestLoaded=False,
        stage=args.stage,seed=seed,pair=pair,methods=[c['runId'] for c in configs],rounds=configs[0]['nRounds'],
        attackStartRoundIndex=configs[0]['attackStartRound'],trainingSamples=len(y),heldoutTrainSamples=len(vy),
        nodeSamples=[len(chunk) for chunk in chunks],optimizerStepsPerArm=per_round*configs[0]['nRounds'],
        totalOptimizerSteps=per_round*configs[0]['nRounds']*len(configs),
        partitionHash=partition_hash,implementationSha256=source,dataSha256=data)
    return dict(root=root,x=x,y=y,vx=vx,vy=vy,fit=fit,validation=validation,chunks=chunks,
        partition_hash=partition_hash,source=source,data=data,report=report)


def execute(args):
    print('Preflight: checking fixed configuration, TRAIN cache, partition and recovery identity before GPU work.',flush=True)
    prepared=preflight(args)
    print(f'Preflight passed: {len(prepared["report"]["methods"])} fixed arms; '
        f'{prepared["report"]["totalOptimizerSteps"]} optimizer updates. Starting numerical verification.',flush=True)
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    from basil_core.memory_safety import RuntimeMemoryGuard,check_host_memory
    from basil_core.terminal.progress import DiagnosticProgress
    from basil_core.cart_validation import selected_configurations,make_worker
    from basil_core.protocol_checkpoint import checkpoint_implementation,save_checkpoint,load_checkpoint,TelemetryStore
    from basil_core.research_protocol import build_model,params_hash,run_research_protocol
    from scripts.run_cart_memory_smoke import save_json,archive_sources,require_identity,keep_json
    root=prepared['root']
    prepare_environment(args.device)
    os.environ.update(WANDB_MODE='disabled',PAPERMERGE_LIVE_TRACKING='disabled')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,args.device);check_host_memory()
    runtime=runtime_identity(tf,np,args.device)
    if args.stage=='replication_30r' and args.device=='GPU':require_device_receipt(runtime,prepared['source'])
    stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    seed=getattr(args,'seed',2025)
    pair=getattr(args,'pair',None)
    names=tuple(c['runId'] for c in selected_configurations('a'*64,'b'*64,args.device,args.stage,seed=seed,pair=pair))
    stages=('CPU/device gradient check','TRAIN split')+names
    with DiagnosticProgress(f'CART TRAIN validation ({args.device})',stages,interval=5.) as progress:
        checked=verify_barrier(tf,args.device)
        progress.begin(stages[1])
        x,y,vx,vy=(prepared[key] for key in ('x','y','vx','vy'))
        fit,validation,chunks=(prepared[key] for key in ('fit','validation','chunks'))
        partition_hash=prepared['partition_hash']
        with tf.device('/CPU:0'):initial_hash=params_hash(build_model('basil_paper_cnn',seed).get_weights())
        configs=selected_configurations(partition_hash,initial_hash,args.device,args.stage,seed=seed,pair=pair)
        rounds=configs[0]['nRounds']
        source=checkpoint_implementation(configs[0])
        if source!=prepared['source']:raise ValueError('Source changed after preflight; restart before training.')
        data=prepared['data']
        identity=dict(implementationSha256=source,dataSha256=data,runtime=runtime)
        # New replication roots bind selection as well as source/data/runtime.
        # Older completed profiles retain their original identity schema.
        if args.stage=='replication_30r':identity.update(stage=args.stage,seed=seed,pair=pair)
        if (root/'identity.json').exists():require_identity(root/'identity.json',identity)
        root.mkdir(parents=True,exist_ok=True)
        keep_json(root/'identity.json',identity)
        if not (root/'verification.json').exists():save_json(root/'verification.json',checked)
        keep_json(root/'split.json',dict(fitIndices=fit.tolist(),validationIndices=validation.tolist(),
            fitSamples=49000,validationSamples=1000,officialTestLoaded=False,
            partitionHash=partition_hash,counts=[np.bincount(y[c],minlength=10).tolist() for c in chunks]))
        outcomes={}
        for cfg in configs:
            if stop:raise InterruptedError('Stopped before the next fixed condition.')
            name=cfg['runId'];progress.begin(name)
            output=root/name
            contract=dict(identity,config=cfg,officialTestLoaded=False,evaluationOrigin='heldout_TRAIN',researchValid=False)
            state=None;previous={}
            if output.exists():
                previous=require_identity(output/'run.json',contract)
                if previous['status']=='completed':outcomes[name]=previous;continue
                state=load_checkpoint(output/'checkpoint.npz',cfg,partition_hash=partition_hash,initial_hash=initial_hash)
            output.mkdir(exist_ok=True);archive_sources(output/'source_snapshots',source)
            metadata=dict(contract,status='running',completedActivations=state['next_activation'] if state else 0,
                completedRounds=state['completed_rounds'] if state else 0)
            prior=previous.get('runtimeSeconds',0.);started=time.monotonic()
            guard=RuntimeMemoryGuard(tf,args.device);store=TelemetryStore(output/'checkpoint.npz')
            def checkpoint(current):
                save_checkpoint(output/'checkpoint.npz',cfg,current,partition_hash=partition_hash,
                    initial_hash=initial_hash,telemetry_store=store)
                metadata.update(completedActivations=current['next_activation'],completedRounds=current['completed_rounds'],
                    runtimeSeconds=prior+time.monotonic()-started,memorySafety=guard.checkpoint())
                save_json(output/'run.json',metadata)
            def activation(record):
                metadata.update(latestNodeValidationAccuracy=record['globalTestAccuracy'],
                    latestLocalTrainingLoss=record['localTrainingLoss'])
                cursor=record['round']*10+record['nodeId']+1
                elapsed=time.monotonic()-started
                performed=cursor-(state['next_activation'] if state else 0)
                print(f'{name} | {args.device} | activation {cursor}/{rounds*10} | '
                    f'elapsed {elapsed:.0f}s | ETA ~{elapsed/performed*(rounds*10-cursor):.0f}s',flush=True)
            def round_done(row):
                metadata.update(latestValidationMean=row['averageAccuracy'],latestValidationWorst=row['worstNodeAccuracy'])
                print(f'{name} | round {row["round"]+1}/{rounds} | TRAIN holdout mean '
                    f'{row["averageAccuracy"]:.2%} | worst {row["worstNodeAccuracy"]:.2%}',flush=True)
            save_json(output/'run.json',metadata)
            try:
                with tf.device(f'/{args.device}:0'):
                    result=run_research_protocol(cfg,x,y,vx,vy,worker_factory=lambda model:make_worker(model,cfg),
                        resume_state=state,checkpoint_callback=checkpoint,activation_callback=activation,
                        round_callback=round_done,should_stop=lambda:stop)
                np.savez_compressed(output/'validation_metrics.npz',mean=result['averageAccuracy'],
                    worst=result['worstNodeAccuracy'],perNode=result['roundNodeAccuracy'],
                    perClass=result['roundNodePerClassAccuracy'],selectedSenders=result['selectedSources'])
                (output/'activation_telemetry.json.gz').write_bytes(gzip.compress(json.dumps(result['telemetry'],allow_nan=False).encode(),mtime=0))
                metadata.update(status='completed',validationMean=result['averageAccuracy'].tolist(),
                    validationWorst=result['worstNodeAccuracy'].tolist(),runtimeSeconds=prior+time.monotonic()-started,
                    optimizerSteps=sum(r['optimizerSteps'] for r in result['telemetry']))
                del result
            except BaseException as error:
                metadata.update(status='interrupted' if isinstance(error,InterruptedError) else 'failed',error=str(error))
                raise
            finally:save_json(output/'run.json',metadata)
            outcomes[name]=metadata;save_json(root/'summary.json',dict(status='running',outcomes=outcomes))
        progress.finish('passed')
    save_json(root/'summary.json',dict(status='completed',outcomes=outcomes,officialTestLoaded=False,
        horizon=rounds,stage=args.stage,seed=seed,pair=pair,hyperparametersChangedDuringExecution=False))
    print('ALL TRAIN-ONLY COMPARISONS COMPLETE; official TEST remains unused.',flush=True)


def main():
    from basil_core.cart_validation import STAGES,REPLICATION_SEEDS
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--execute',action='store_true')
    action.add_argument('--preflight',action='store_true',help='Read-only config/data/source checks; no TensorFlow or GPU initialization.')
    action.add_argument('--verify-device',action='store_true',help='Synthetic replication GPU checks only; no dataset load or research run.')
    parser.add_argument('--recover',action='store_true')
    parser.add_argument('--stage',choices=STAGES,default='clean')
    parser.add_argument('--seed',type=int,choices=REPLICATION_SEEDS,default=2025)
    parser.add_argument('--pair',choices=tuple('ABCD'),help='Run one complete control/CART pair in the replication stage.')
    parser.add_argument('--campaign-plan',type=Path,help='Frozen supervisor input contract, checked before GPU work.')
    parser.add_argument('--device',choices=('GPU','CPU'),default='GPU')
    parser.add_argument('--output',type=Path,default=ROOT/'.papermerge/local-tests/cart-train-validation-20261026')
    args=parser.parse_args()
    if args.verify_device:
        try:print(json.dumps(verify_replication_device()))
        except (OSError,ValueError,RuntimeError) as error:parser.exit(2,str(error)+'\n')
        return
    if args.preflight:
        try:print(json.dumps(preflight(args)['report'],indent=2))
        except (OSError,ValueError) as error:parser.exit(2,str(error)+'\n')
        return
    if not args.execute:
        from basil_core.cart_validation import selected_configurations
        try:configs=selected_configurations('a'*64,'b'*64,args.device,args.stage,seed=args.seed,pair=args.pair)
        except ValueError as error:parser.error(str(error))
        print(json.dumps(dict(trainingStarted=False,rounds=configs[0]['nRounds'],methods=[c['runId'] for c in configs],
            seed=args.seed,trainingSamples=49000,heldoutTrainSamples=1000,officialTestLoaded=False)))
        return
    try:execute(args)
    except (ValueError,FileExistsError,InterruptedError) as error:parser.exit(2,str(error)+'\n')


if __name__=='__main__':main()
