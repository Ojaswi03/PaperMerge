#!/usr/bin/env python3
"""Independent IID/non-IID development comparisons; default is a read-only plan.

Training uses a TRAIN holdout, never the official CIFAR TEST set. GPU workers
start with their CUDA environment already set. One config owns one GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def selected(args):
    from basil_core.factorial_study import matrix
    configs=matrix(args.sigmas,stage=args.stage,device='GPU')
    return [c for c in configs if c['conditionId'] in args.conditions
        and c['partitionStrategy'] in args.partitions and c['seed'] in args.seeds]


def data():
    import numpy as np
    from basil_core.cart_validation import split_training,validate_training_arrays
    from basil_core.dirichlet_diagnosis import array_identity_hash
    folder=ROOT/'experiments/cache/cifar10'
    x=np.load(folder/'x_train_float32.npy',mmap_mode='r',allow_pickle=False)
    y=np.load(folder/'y_train_int32.npy',mmap_mode='r',allow_pickle=False)
    validate_training_arrays(x,y)
    fit,heldout=split_training(y)
    arrays=x[fit],y[fit],x[heldout],y[heldout]
    hashes={k:array_identity_hash(v) for k,v in zip(('fitImages','fitLabels','validationImages','validationLabels'),arrays)}
    return arrays,hashes


def check_source(source):
    for name,digest in source.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source changed during study; no additional config started.')


def worker(args):
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    from basil_core.protocol_checkpoint import save_checkpoint,load_checkpoint,TelemetryStore
    from basil_core.research_protocol import run_research_protocol,build_model,params_hash,keyed_rng,iid_partition_indices
    from basil_core.execution_profile import current_profile
    from scripts.run_cart_memory_smoke import save_json,keep_json,require_identity
    from scripts.run_cart_train_validation import runtime_identity
    plan=json.loads((args.output/'plan.json').read_text())
    cfg=plan['configs'][args.worker]
    if cfg.get('lossStudyRevision')=='cart_coherent_channel_pilot_v1':
        from basil_core.cart_coherent_study import validate_config,make_worker
    else:
        from basil_core.factorial_worker import make_worker
        from basil_core.factorial_study import validate_config
    validate_config(cfg);check_source(plan['source'])
    arrays,hashes=data()
    if hashes!=plan['data']:raise ValueError('TRAIN data differs from frozen plan.')
    prepare_environment('GPU')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,'GPU')
    runtime=runtime_identity(tf,np,'GPU')
    root=args.output/cfg['runId']
    contract=dict(config=cfg,source=plan['source'],data=hashes,runtime=runtime,officialTestUsed=False)
    previous={};state=None
    root.mkdir(exist_ok=True)
    if (root/'run.json').exists():
        previous=require_identity(root/'run.json',contract)
        if previous['status']=='COMPLETED':return
    x,y,vx,vy=arrays
    initial=params_hash(build_model(cfg['modelArchitecture'],cfg['seed']).get_weights())
    eligible=np.arange(len(y))
    if cfg['diagnosticSamplesPerClass']:
        eligible=np.concatenate([keyed_rng(cfg['seed'],'diagnostic_class',c).choice(np.flatnonzero(y==c),cfg['diagnosticSamplesPerClass'],replace=False) for c in range(10)])
    if cfg['partitionStrategy']=='iid':chunks=iid_partition_indices(y[eligible],cfg['seed'])
    else:
        from basil_core.data.cifar import _dirichletPartition
        chunks=_dirichletPartition(y[eligible],10,.2,keyed_rng(cfg['seed'],'dirichlet_partition'))
    chunks=[eligible[c] for c in chunks]
    partition_hash=hashlib.sha256(b''.join(np.asarray(c,dtype='<i8').tobytes() for c in chunks)).hexdigest()
    if cfg.get('lossStudyRevision')=='cart_coherent_channel_pilot_v1' and (
            cfg['partitionHash']!=partition_hash or cfg['initialModelHash']!=initial):
        raise ValueError('Worker initialization/partition differs from the frozen preflight.')
    keep_json(root/'partition.json',dict(hash=partition_hash,counts=[len(c) for c in chunks],initialModelHash=initial))
    if previous:
        if not (root/'checkpoint.npz').exists():raise ValueError('Interrupted run has no checkpoint; retain it and use a fresh output.')
        state=load_checkpoint(root/'checkpoint.npz',cfg,partition_hash=partition_hash,initial_hash=initial)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    started=time.monotonic();prior=previous.get('elapsedSeconds',0.)
    metadata=dict(contract,status='RUNNING',completedActivations=state['next_activation'] if state else 0,
        completedRounds=state['completed_rounds'] if state else 0,
        physicalGpuId=current_profile().gpu)
    store=TelemetryStore(root/'checkpoint.npz')
    def checkpoint(current):
        save_checkpoint(root/'checkpoint.npz',cfg,current,partition_hash=partition_hash,
            initial_hash=initial,telemetry_store=store)
        metadata.update(completedActivations=current['next_activation'],completedRounds=current['completed_rounds'],
            elapsedSeconds=prior+time.monotonic()-started)
        save_json(root/'run.json',metadata)
    def activation(record):
        cursor=record['round']*10+record['nodeId']+1
        metadata.update(latestValidationNode=record['globalTestAccuracy'],latestTrainCE=record['localTrainingLoss'])
        print(f'{cfg["runId"]} | GPU {current_profile().gpu} | activation {cursor}/{cfg["nRounds"]*10} | '
            f'node holdout {record["globalTestAccuracy"]:.2%}',flush=True)
    def round_done(record):
        metadata.update(validationMean=record['averageAccuracy'],validationWorst=record['worstNodeAccuracy'])
        print(f'ROUND {record["round"]+1}/{cfg["nRounds"]} | mean {record["averageAccuracy"]:.2%} | worst {record["worstNodeAccuracy"]:.2%}',flush=True)
    save_json(root/'run.json',metadata)
    try:
        with tf.device('/GPU:0'):
            result=run_research_protocol(cfg,x,y,vx,vy,worker_factory=lambda model:make_worker(model,cfg),
                activation_callback=activation,round_callback=round_done,checkpoint_callback=checkpoint,
                resume_state=state,should_stop=lambda:stopped)
        np.savez_compressed(root/'metrics.npz',roundNodeAccuracy=result['roundNodeAccuracy'],
            perClass=result['roundNodePerClassAccuracy'],selectedSources=result['selectedSources'])
        metadata.update(status='COMPLETED',roundMean=result['averageAccuracy'].tolist(),
            elapsedSeconds=prior+time.monotonic()-started,
            optimizerSteps=sum(r['optimizerSteps'] for r in result['telemetry']))
    except BaseException as error:
        metadata.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error))
        raise
    finally:save_json(root/'run.json',metadata)


def monitor(root):
    from basil_core.factorial_dashboard import monitor as dashboard_monitor
    return dashboard_monitor(root)


def execute(args,*,configs=None,prepared_hashes=None):
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory,validate_resources,worker_environment,use_profile
    from basil_core.protocol_checkpoint import checkpoint_implementation
    from scripts.run_cart_memory_smoke import save_json,keep_json,archive_sources
    from basil_core.terminal.state import identity
    configs=selected(args) if configs is None else configs
    if not configs:raise ValueError('Empty study selection.')
    # Until local learning/defense gates are met, only bounded smoke/pilot runs.
    if args.stage=='comparison':raise ValueError('Full matrix is not performance validated. Complete local pilot and noise screening first.')
    if not args.output.resolve().is_relative_to(ROOT/'.papermerge/local-tests'):
        raise ValueError('Use an isolated .papermerge/local-tests directory.')
    if args.output.exists() and not args.recover:raise FileExistsError('Output already exists; use a fresh directory or --recover.')
    if args.recover and not (args.output/'plan.json').is_file():raise ValueError('Recovery requires an existing frozen plan.')
    if prepared_hashes is None:_,hashes=data()
    else:hashes=prepared_hashes
    inventory=gpu_inventory()
    gpus=[int(v) for v in args.gpus.split(',')]
    profiles=[ExecutionProfile(name=args.profile,gpu=g,
        memory_mib=8192 if args.profile=='hpc' else 6144,
        headroom_mib=2048 if args.profile=='hpc' else 1024,
        output_root=str(args.output.resolve()) if args.profile=='hpc' else '').bind(inventory) for g in gpus]
    validate_resources(profiles,inventory)
    with use_profile(profiles[0]):source=checkpoint_implementation(configs[0])
    plan=dict(revision=configs[0]['lossStudyRevision'],configs=configs,sigmas=args.sigmas,
        source=source,data=hashes,profiles=[asdict(p) for p in profiles],officialTestUsed=False)
    args.output.mkdir(parents=True,exist_ok=True);keep_json(args.output/'plan.json',plan)
    archive_sources(args.output/'source_snapshots',source)
    import fcntl
    lock=(args.output/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[]
    for index,cfg in enumerate(configs):
        path=args.output/cfg['runId']/'run.json'
        previous=json.loads(path.read_text()) if path.exists() else {}
        if previous.get('status')=='COMPLETED':
            if previous.get('config')!=cfg or previous.get('source')!=source or previous.get('data')!=hashes:
                raise ValueError('Completed result identity differs from frozen recovery plan.')
        else:pending.append(index)
    active={};stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    started=time.monotonic();status='RUNNING';owner=identity()
    try:
        while pending or active:
            for gpu,(process,handle) in list(active.items()):
                code=process.poll()
                if code is not None:
                    handle.close();del active[gpu]
                    if code:raise RuntimeError(f'GPU {gpu} worker failed with exit {code}; inspect its log.')
            if stopped:raise InterruptedError('Operator requested stop.')
            for profile in profiles:
                if pending and profile.gpu not in active:
                    check_source(source)
                    index=pending.pop(0);cfg=configs[index]
                    handle=(args.output/f'{cfg["runId"]}.log').open('a')
                    with use_profile(profile):environment={**os.environ,**profile.environment(),**worker_environment('GPU')}
                    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(index),'--output',str(args.output.resolve())]
                    active[profile.gpu]=(subprocess.Popen(cmd,cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT),handle)
                    print(f'GPU {profile.gpu}: {cfg["runId"]}; {len(pending)} configs pending.',flush=True)
            save_json(args.output/'campaign.json',dict(status=status,elapsedSeconds=time.monotonic()-started,
                activeGpus=list(active),pending=len(pending),total=len(configs),owner=owner,host=owner['host']))
            if active:time.sleep(5)
        status='COMPLETED'
        print('ALL SELECTED EXPERIMENTS COMPLETE.',flush=True)
    except BaseException as error:
        status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED'
        for process,_ in active.values():process.terminate()
        for process,handle in active.values():
            process.wait();handle.close()
        raise
    finally:
        save_json(args.output/'campaign.json',dict(status=status,elapsedSeconds=time.monotonic()-started,
            pending=len(pending),total=len(configs),owner=owner,host=owner['host']))
        lock.close()


def main():
    from basil_core.factorial_study import CONDITIONS,SEEDS,PARTITIONS
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--monitor',action='store_true')
    modes.add_argument('--worker',type=int,help=argparse.SUPPRESS)
    p.add_argument('--stage',choices=('smoke','pilot','comparison'),default='smoke')
    p.add_argument('--sigmas',nargs=3,type=float,default=(.005,.01,.02),help='Candidate values; frozen in the plan before execution.')
    p.add_argument('--conditions',nargs='+',choices=[c.name for c in CONDITIONS],default=[c.name for c in CONDITIONS])
    p.add_argument('--partitions',nargs='+',choices=PARTITIONS,default=list(PARTITIONS))
    p.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    p.add_argument('--profile',choices=('local','hpc'),default='local')
    p.add_argument('--gpus',default='0')
    p.add_argument('--output',type=Path,default=ROOT/'.papermerge/local-tests/factorial-development-v1')
    p.add_argument('--recover',action='store_true')
    args=p.parse_args()
    if args.worker is not None:worker(args)
    elif args.monitor:monitor(args.output)
    elif args.execute:execute(args)
    else:
        configs=selected(args)
        print(json.dumps(dict(runs=len(configs),configs=configs,trainingStarted=False,
            noiseLevelsProvisional=True,performanceReady=False,officialTestUsed=False),indent=2))


if __name__=='__main__':main()
