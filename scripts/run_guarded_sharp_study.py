#!/usr/bin/env python3
"""Survivor-guarded SHARP development family; one config per physical GPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temporary,path)


def fingerprint():
    from basil_core.sharp_guarded_protocol import fingerprint as protocol_fingerprint
    result=protocol_fingerprint()
    result['scripts/run_guarded_sharp_study.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def check_source(expected):
    if fingerprint()!=expected:raise ValueError('SHARP source changed; retain this run and use a new frozen plan.')


def selected(args):
    from basil_core.sharp_study import matrix
    from basil_core.sharp_guarded_protocol import guarded_configuration
    configs=matrix(datasets=args.datasets,partitions=args.partitions,seeds=args.seeds,
        conditions=args.conditions,sigmas=args.sigmas,methods=args.methods,nodes=args.nodes,
        rounds=args.rounds,model=args.model,alpha=args.alpha,local_steps=args.local_steps,
        train_per_class=args.train_per_class,validation_per_class=args.validation_per_class,
        attack_start=args.attack_start,denominator=args.hop_denominator,curvature=args.curvature,
        proximal_mu=args.mu,initial_rate=args.learning_rate,attack_type=args.attack_type,budget=args.budget,basil_window=args.basil_window)
    configs=[guarded_configuration(c) for c in configs]
    if args.only_sigma is not None:configs=[c for c in configs if not c['sigma'] or c['sigma']==args.only_sigma]
    if not configs:raise ValueError('No selected experiments.')
    return configs


def data_contract(configs):
    from basil_core.sharp_data import training_data,prepare_partition,digest
    data={};partitions={}
    for name in dict.fromkeys(c['dataset'] for c in configs):
        x,y=training_data(name)
        data[name]=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
        for cfg in configs:
            if cfg['dataset']==name:
                _,_,audit=prepare_partition(y,cfg)
                partitions[cfg['runId']]=audit
    return data,partitions


def profiles(args):
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory,validate_resources
    inventory=gpu_inventory();ids=[int(x) for x in args.gpus.split(',')]
    if len(set(ids))!=len(ids):raise ValueError('GPU IDs must be distinct.')
    result=[ExecutionProfile(name=args.profile,gpu=g,output_root=str(args.output.resolve()) if args.profile=='hpc' else '',
        memory_mib=8192 if args.profile=='hpc' else 6144,headroom_mib=2048 if args.profile=='hpc' else 1024).bind(inventory) for g in ids]
    validate_resources(result,inventory)
    return result


def worker_environment(profile):
    from basil_core.execution_profile import worker_environment as environment,cuda_wheel_environment
    # CUDA loader paths must be set before the child interpreter starts, also locally.
    values={**os.environ,**profile.environment()}
    values.update(environment('GPU',base=values,profile=profile))
    values.update(cuda_wheel_environment(values))
    return values


def setup_gpu():
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    prepare_environment('GPU')
    import tensorflow as tf
    configure_tensorflow(tf,'GPU')
    if not tf.config.list_logical_devices('GPU'):raise RuntimeError('No GPU execution; CPU fallback refused.')
    return tf


def runtime(tf):
    from basil_core.execution_profile import current_profile
    import numpy as np
    return dict(tensorflow=tf.__version__,numpy=np.__version__,profile=asdict(current_profile()),
        host=socket.gethostname(),device='/GPU:0',tf32=False)


def verify_child(args):
    import numpy as np
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import vector_hash
    plan=json.loads((args.output/'verification_plan.json').read_text());check_source(plan['source'])
    tf=setup_gpu();results=[]
    for cfg in plan['configs']:
        w=SharpWorker(cfg,'GPU');initial=w.export()
        shape=(28,28,1) if cfg['dataset']=='mnist' else (32,32,3)
        x=np.random.default_rng(2025).uniform(size=(4,*shape)).astype(np.float32)
        y=np.arange(4,dtype=np.int32)%cfg['classes'];batch=np.arange(4)
        a,record=w.train(initial,x,y,0,0,cfg['sigma']**2,batch)
        b,_=w.train(initial,x,y,0,0,cfg['sigma']**2,batch)
        np.testing.assert_array_equal(a,b)
        if vector_hash(initial)==vector_hash(a):raise AssertionError('GPU verification did not update weights.')
        results.append(dict(dataset=cfg['dataset'],model=cfg['model'],parameters=w.dimension,
            initialHash=vector_hash(initial),trainedHash=vector_hash(a),steps=record['optimizerSteps']))
    write_json(args.output/f'gpu_verification_{args.verify_worker}.json',dict(status='PASSED',
        source=plan['source'],runtime=runtime(tf),checks=results,researchRuns=0,officialTestUsed=False))
    print(f'GPU {args.verify_worker}: objective, SGD and deterministic repeat PASSED.',flush=True)


def verify(args):
    configs=selected(args);hardware=profiles(args);source=fingerprint()
    args.output.mkdir(parents=True,exist_ok=True)
    chosen=[]
    for d in args.datasets:
        available=[c for c in configs if c['dataset']==d and c['switches']['noise_mitigation']]
        if not available:
            from basil_core.sharp_study import configuration
            available=[configuration(d,'iid',2025,'attack_noise_defense_rla',.01,nodes=args.nodes,model=args.model)]
        chosen.append(available[0])
    write_json(args.output/'verification_plan.json',dict(source=source,configs=chosen))
    for profile in hardware:
        command=[sys.executable,__file__,'--verify-worker',str(profile.gpu),'--output',str(args.output.resolve())]
        process=subprocess.Popen(command,cwd=ROOT,env=worker_environment(profile))
        while process.poll() is None:
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:print(f'GPU {profile.gpu}: verifying; elapsed calculation in progress.',flush=True)
        if process.returncode:raise RuntimeError(f'GPU {profile.gpu} verification failed.')


def run_worker(args):
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import save_state,load_state,vector_hash
    from basil_core.sharp_guarded_protocol import run
    from basil_core.sharp_data import training_data,prepare_partition,digest
    from basil_core.execution_profile import current_profile
    plan=json.loads((args.output/'plan.json').read_text());cfg=plan['configs'][args.worker]
    check_source(plan['source']);tf=setup_gpu();w=SharpWorker(cfg,'GPU')
    x,y=training_data(cfg['dataset']);data=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
    if data!=plan['data'][cfg['dataset']]:raise ValueError('TRAIN cache differs from frozen plan.')
    chunks,heldout,audit=prepare_partition(y,cfg)
    if audit!=plan['partitions'][cfg['runId']]:raise ValueError('Worker partition differs from preflight.')
    folder=args.output/cfg['runId'];folder.mkdir(exist_ok=True)
    contract=dict(config=cfg,source=plan['source'],data=data,partition=audit,runtime=runtime(tf),initialHash=vector_hash(w.export()))
    write_json(folder/'partition.json',audit)
    path=folder/'checkpoint.npz';state=load_state(path,contract,w.dimension) if path.exists() else None
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    metadata=dict(contract,status='RUNNING',physicalGpuId=current_profile().gpu,completedActivations=state['cursor'] if state else 0,
        totalActivations=cfg['nodes']*cfg['rounds'],completedRounds=len(state['roundMetrics']) if state else 0)
    if state and state['roundMetrics']:
        last=state['roundMetrics'][-1];metadata.update(meanAccuracy=last['meanAccuracy'],worstNodeAccuracy=last['worstNodeAccuracy'])
    write_json(folder/'run.json',metadata)
    def checkpoint(s):save_state(path,s,contract)
    def activation(record,s):
        metadata.update(completedActivations=s['cursor'],elapsedSeconds=s['elapsedSeconds'],
            optimizerSteps=s['cursor']*cfg['localSteps'],latestTrainCE=record['training']['localTrainingLoss'],
            latestSelectedCount=record['selection']['selectedCount'],latestRlaCoefficient=record['training']['rlaCoefficient'],
            latestCartScale=record['cart']['scale'] if record['cart'] else None)
        write_json(folder/'run.json',metadata)
    def round_done(record,s):
        metadata.update(completedRounds=len(s['roundMetrics']),meanAccuracy=record['meanAccuracy'],
            worstNodeAccuracy=record['worstNodeAccuracy'],worstClassMean=record['worstClassMean'])
        write_json(folder/'run.json',metadata)
        write_json(folder/'round_metrics.json',s['roundMetrics'])
        print(f'{cfg["runId"]}: round {record["round"]+1}/{cfg["rounds"]}, mean {record["meanAccuracy"]:.2%}, '
            f'worst node {record["worstNodeAccuracy"]:.2%}',flush=True)
    try:
        result=run(cfg,w,x,y,chunks,x[heldout],y[heldout],state=state,on_activation=activation,on_round=round_done,
            checkpoint=checkpoint,should_stop=lambda:stopped)
        checkpoint(result)
        metadata.update(status='COMPLETED',elapsedSeconds=result['elapsedSeconds'])
        write_json(folder/'activation_telemetry.json',result['telemetry'])
        import numpy as np
        np.savez_compressed(folder/'metrics.npz',roundNodeAccuracy=np.asarray([r['nodeAccuracy'] for r in result['roundMetrics']]),
            roundNodePerClassAccuracy=np.asarray([r['perClass'] for r in result['roundMetrics']]))
    except BaseException as error:
        metadata.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error))
        raise
    finally:write_json(folder/'run.json',metadata)


def execute(args):
    import fcntl
    configs=selected(args);source=fingerprint();hardware=profiles(args)
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data,partitions=data_contract(configs)
    plan=dict(configs=configs,source=source,data=data,partitions=partitions,profiles=[asdict(p) for p in hardware])
    path=args.output/'plan.json'
    if path.exists():
        if not args.recover:raise FileExistsError('Run exists. Use a fresh output or --recover with identical arguments.')
        if json.loads(path.read_text())!=plan:raise ValueError('Recovery plan differs; no runs started.')
    elif args.recover:raise ValueError('Recovery requires an existing frozen plan.')
    for p in hardware:
        receipt=args.output/f'gpu_verification_{p.gpu}.json'
        if not receipt.exists():raise ValueError('Run --verify-gpus with these datasets/model/output before executing.')
        evidence=json.loads(receipt.read_text())
        verified={(c['dataset'],c['model']) for c in evidence['checks']}
        if (evidence['status']!='PASSED' or evidence['source']!=source or evidence['runtime']['profile']!=asdict(p)
                or not {(c['dataset'],c['model']) for c in configs}.issubset(verified)):
            raise ValueError('GPU verification differs from current source, model or hardware.')
    if not path.exists():write_json(path,plan)
    snapshots=args.output/'source_snapshots';snapshots.mkdir(exist_ok=True)
    for name in source:
        target=snapshots/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    queues={p.gpu:[] for p in hardware}
    for index,cfg in enumerate(configs):
        run_path=args.output/cfg['runId']/'run.json'
        if run_path.exists() and json.loads(run_path.read_text())['status']=='COMPLETED':continue
        queues[hardware[index%len(hardware)].gpu].append(index)
    active={};stopped=False;status='RUNNING';started=time.monotonic()
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def campaign():
        write_json(args.output/'campaign.json',dict(status=status,total=len(configs),
            pending=sum(map(len,queues.values())),activeGpus=list(active),elapsedSeconds=time.monotonic()-started,
            host=socket.gethostname(),profile=args.profile,ownerPid=os.getpid()))
    try:
        campaign()
        while any(queues.values()) or active:
            for gpu,(process,handle) in list(active.items()):
                if process.poll() is not None:
                    code=process.returncode;handle.close();del active[gpu]
                    if code:raise RuntimeError(f'GPU {gpu} worker failed with exit {code}; inspect its log.')
            if stopped:raise InterruptedError('Supervisor stop requested.')
            for p in hardware:
                if queues[p.gpu] and p.gpu not in active:
                    check_source(source);index=queues[p.gpu].pop(0);cfg=configs[index]
                    handle=(args.output/(cfg['runId']+'.log')).open('a')
                    command=[sys.executable,__file__,'--worker',str(index),'--output',str(args.output.resolve())]
                    active[p.gpu]=(subprocess.Popen(command,cwd=ROOT,env=worker_environment(p),stdout=handle,stderr=subprocess.STDOUT),handle)
                    print(f'Physical GPU {p.gpu}: {cfg["runId"]}',flush=True)
            campaign()
            if active:time.sleep(5)
        status='COMPLETED';print('ALL SELECTED EXPERIMENTS COMPLETE.',flush=True)
    except BaseException as error:
        status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED'
        for process,_ in active.values():process.terminate()
        for process,handle in active.values():process.wait();handle.close()
        raise
    finally:campaign();lock.close()


def monitor(args):
    from basil_core.sharp_monitor import monitor as eta_monitor
    return eta_monitor(args.output)


def report(args):
    import numpy as np
    plan=json.loads((args.output/'plan.json').read_text());groups={};lookup={}
    for cfg in plan['configs']:
        row=json.loads((args.output/cfg['runId']/'run.json').read_text())
        if row['status']!='COMPLETED':raise ValueError('Reporting requires every selected experiment to complete.')
        if row['config']!=cfg or row['source']!=plan['source']:raise ValueError('Result identity mismatch.')
        with np.load(args.output/cfg['runId']/'metrics.npz',allow_pickle=False) as metrics:values=metrics['roundNodeAccuracy']
        if values.shape!=(cfg['rounds'],cfg['nodes']) or not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
            raise ValueError('Invalid round/node metric array.')
        key=(cfg['dataset'],cfg['partition'],cfg['method'],cfg['condition'],cfg['sigma'])
        endpoint=float(values[-min(10,cfg['rounds']):].mean());final=float(values[-1].mean())
        seed_record=dict(seed=cfg['seed'],endpoint=endpoint,final=final)
        if any(r['seed']==cfg['seed'] for r in groups.get(key,[])):raise ValueError('Duplicate seed.')
        groups.setdefault(key,[]).append(seed_record);lookup[(*key,cfg['seed'])]=endpoint
    results=[]
    for key,rows in groups.items():
        endpoints=[r['endpoint'] for r in rows]
        results.append(dict(zip(('dataset','partition','method','condition','sigma'),key),seeds=rows,
            mean=float(np.mean(endpoints)),sampleStd=float(np.std(endpoints,ddof=1)) if len(rows)>1 else None,
            seedCount=len(rows),officialTestUsed=False))
    paired=[]
    for key,value in lookup.items():
        d,p,m,c,s,seed=key
        if c.endswith('_cart'):
            baseline=(d,p,m,c[:-5],s,seed)
            if baseline not in lookup:raise ValueError('CART report requires the paired non-CART condition.')
            paired.append(dict(dataset=d,partition=p,method=m,sigma=s,seed=seed,delta=value-lookup[baseline]))
    output=dict(status='COMPLETE_SELECTED_MATRIX',source=plan['source'],groups=results,pairedCartEndpoints=paired,
        averaging='Nodes and fixed endpoint rounds within seed, then seeds at identical dataset/split/method/condition/sigma.',
        officialTestUsed=False)
    write_json(args.output/'summary.json',output)
    print(json.dumps(output,indent=2))


def main():
    from basil_core.sharp_study import DATASETS,SEEDS,SIGMAS,CONDITION_NAMES
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group()
    for name in ('prepare-data','preflight','verify-gpus','execute','monitor','report'):action.add_argument('--'+name,action='store_true')
    action.add_argument('--worker',type=int);action.add_argument('--verify-worker',type=int)
    p.add_argument('--output',type=Path,default=ROOT/'.papermerge/local-tests/sharp-survivor-guard-v1')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--partitions',nargs='+',choices=('iid','dirichlet'),default=['iid','dirichlet'])
    p.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    p.add_argument('--conditions',nargs='+',choices=CONDITION_NAMES,default=list(CONDITION_NAMES))
    p.add_argument('--sigmas',nargs=3,type=float,default=list(SIGMAS));p.add_argument('--only-sigma',type=float)
    p.add_argument('--methods',nargs='+',choices=('sharp',),default=['sharp'])
    p.add_argument('--nodes',type=int,default=100);p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--budget',type=int);p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--basil-window',type=int,help='BASIL comparison window; default b+1, use 10 for the PDF example or 2b for matched fanout.')
    p.add_argument('--local-steps',type=int,default=5);p.add_argument('--model',choices=('paper','resnet20'),default='paper')
    p.add_argument('--train-per-class',type=int);p.add_argument('--validation-per-class',type=int)
    p.add_argument('--attack-start',type=int,default=20);p.add_argument('--attack-type',choices=('hidden','spike'),default='hidden')
    p.add_argument('--hop-denominator',choices=('actual_selected','budget'),default='actual_selected')
    p.add_argument('--curvature',type=float,default=1.);p.add_argument('--mu',type=float,default=1.)
    p.add_argument('--learning-rate',type=float,default=.03)
    p.add_argument('--gpus',default='0');p.add_argument('--profile',choices=('local','hpc'),default='local')
    p.add_argument('--recover',action='store_true');p.add_argument('--tmux')
    args=p.parse_args();args.output=args.output.resolve()
    if args.prepare_data:
        from basil_core.sharp_data import prepare_cache
        for name in args.datasets:prepare_cache(name);print(f'{name}: TRAIN cache ready.')
    elif args.verify_worker is not None:verify_child(args)
    elif args.worker is not None:run_worker(args)
    elif args.monitor:monitor(args)
    elif args.report:report(args)
    elif args.verify_gpus:verify(args)
    elif args.execute:
        if args.tmux:
            if subprocess.run(['tmux','has-session','-t',args.tmux],capture_output=True).returncode==0:raise ValueError('tmux session already exists.')
            argv=list(sys.argv[1:]);at=argv.index('--tmux');del argv[at:at+2]
            subprocess.run(['tmux','new-session','-d','-s',args.tmux,shlex.join([sys.executable,__file__,*argv])],check=True)
            print(f'Started tmux session {args.tmux}. Monitor with --monitor --output {shlex.quote(str(args.output))}')
        else:execute(args)
    else:
        configs=selected(args)
        if args.preflight:
            data,partitions=data_contract(configs);args.output.mkdir(parents=True,exist_ok=True)
            write_json(args.output/'preflight.json',dict(source=fingerprint(),data=data,partitions=partitions,configs=configs))
        print(json.dumps(dict(configs=len(configs),nodes=args.nodes,rounds=args.rounds,localSteps=args.local_steps,
            datasets=args.datasets,seeds=args.seeds,sigmas=args.sigmas,trainingStarted=False,
            optimizerSteps=sum(c['nodes']*c['rounds']*c['localSteps'] for c in configs),
            fullMatrixNeedsMeasuredValidation=True),indent=2))


if __name__=='__main__':main()

