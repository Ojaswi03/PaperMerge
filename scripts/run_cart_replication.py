#!/usr/bin/env python3
"""Fixed three-seed CART replication; one independent comparison pair per GPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def plan(args,*,bind=False):
    from basil_core.cart_validation import REPLICATION_SEEDS,REPLICATION_STAGE,selected_configurations
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    root=args.output.resolve()
    if not root.is_relative_to(ROOT/'.papermerge/local-tests'):
        raise ValueError('Use an isolated .papermerge/local-tests output directory.')
    if args.profile=='hpc' and not args.gpus:raise ValueError('HPC requires explicit --gpus IDs.')
    raw=args.gpus or '0'
    if not re.fullmatch(r'\d+(,\d+)*',raw):raise ValueError('GPU IDs must be comma-separated integers.')
    ids=[int(v) for v in raw.split(',')]
    if len(set(ids))!=len(ids) or len(ids)>4:raise ValueError('Choose one to four distinct physical GPUs.')
    if args.profile=='local' and ids!=[0]:raise ValueError('The local profile preserves the single GPU 0 setup.')
    devices=gpu_inventory() if bind else None
    resources=[]
    for gpu in ids:
        profile=ExecutionProfile(name=args.profile,gpu=gpu,
            memory_mib=args.gpu_memory_mib if args.gpu_memory_mib is not None else (8192 if args.profile=='hpc' else 6144),
            headroom_mib=2048 if args.profile=='hpc' else 1024,
            output_root=str(root) if args.profile=='hpc' else '').validate()
        if bind:
            profile=profile.bind(devices)
            selected=next(d for d in devices if d['uuid']==profile.gpu_uuid)
            if selected['memory.free']<profile.memory_mib+profile.headroom_mib:
                raise ValueError(f'GPU {gpu} lacks the requested memory cap plus headroom.')
        resources.append(asdict(profile))
    jobs=[]
    for seed in REPLICATION_SEEDS:
        for index,pair in enumerate('ABCD'):
            configs=selected_configurations('a'*64,'b'*64,'GPU',REPLICATION_STAGE,seed=seed,pair=pair)
            jobs.append(dict(id=f'seed-{seed}-{pair}',seed=seed,pair=pair,
                output=str(root/f'seed-{seed}'/f'pair-{pair}'),
                resources=resources[index%len(resources)],arms=[c['runId'] for c in configs]))
    return dict(revision='cart_replication_30r_three_seed_v1',stage=REPLICATION_STAGE,
        seeds=list(REPLICATION_SEEDS),holdoutSeed=2025,rounds=30,attackStartRoundIndex=20,
        primaryEndpoint='mean_node_validation_accuracy_round_indices_20_to_29',
        officialTestUsed=False,researchValid=False,profile=args.profile,output=str(root),jobs=jobs,
        resources=resources,totalArms=24,totalActivations=7200,trainingStarted=False)


def preflight(args):
    from basil_core.execution_profile import ExecutionProfile,use_profile
    from scripts.run_cart_train_validation import preflight as check_inputs
    from scripts.run_cart_memory_smoke import require_identity
    result=plan(args,bind=True)
    root=Path(result['output'])
    if root.exists() and not args.recover:raise FileExistsError('Campaign output exists; use a fresh directory or --recover.')
    checks=[]
    for job in result['jobs']:
        print(f'Preflight: {job["id"]}, physical GPU {job["resources"]["gpu"]}; checking TRAIN inputs and source.',file=sys.stderr,flush=True)
        with use_profile(ExecutionProfile(**job['resources'])):
            prepared=check_inputs(SimpleNamespace(output=Path(job['output']),recover=args.recover,
                device='GPU',stage=result['stage'],seed=job['seed'],pair=job['pair']))
        checks.append(dict(job=job['id'],**prepared['report']))
        del prepared
    if any(c['dataSha256']!=checks[0]['dataSha256'] or c['implementationSha256']!=checks[0]['implementationSha256'] for c in checks):
        raise ValueError('Source or TRAIN inputs changed during campaign preflight.')
    result['checks']=checks
    if (root/'plan.json').exists():require_identity(root/'plan.json',result)
    elif root.exists() and any(root.iterdir()):raise ValueError('Existing campaign has no matching immutable plan.')
    return result


def frozen_sources(plan):
    for check in plan['checks']:
        for name,digest in check['implementationSha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Source changed since preflight: {name}. Training was not launched.')


def child_environment(profile):
    from basil_core.execution_profile import worker_environment
    values=worker_environment('GPU',base=os.environ.copy(),profile=profile)
    values.update(profile.environment())
    return values


def verified_devices(prepared):
    from importlib.metadata import version,PackageNotFoundError
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    from scripts.run_cart_train_validation import device_receipt
    devices=gpu_inventory()
    for resource in prepared['resources']:
        try:report=json.loads(device_receipt(ExecutionProfile(**resource)).read_text())
        except (OSError,ValueError) as error:
            raise ValueError('CART GPU receipt is missing/unreadable. Run --verify-gpus before execution.') from error
        runtime=report.get('runtime',{})
        device=next(d for d in devices if d['uuid']==resource['gpu_uuid'])
        try:software_matches=all(version(name)==value for name,value in runtime.get('software',{}).items()
            if value is not None and name!='python')
        except PackageNotFoundError:software_matches=False
        profile_matches=all(runtime.get('profile',{}).get(key)==value for key,value in resource.items() if key!='output_root')
        if (report.get('status')!='passed' or report.get('implementationSha256')!=prepared['checks'][0]['implementationSha256'] or
                runtime.get('gpu',{}).get('uuid')!=resource['gpu_uuid'] or
                runtime.get('gpu',{}).get('driver_version')!=device['driver_version'] or
                runtime.get('python')!=sys.version or runtime.get('executable')!=sys.executable or
                not software_matches or not profile_matches):
            raise ValueError('CART GPU verification is missing/stale. Run --verify-gpus before execution.')


def verify_devices(args):
    from basil_core.execution_profile import ExecutionProfile
    from scripts.run_cart_train_validation import device_receipt
    prepared=plan(args,bind=True)
    for resource in prepared['resources']:
        profile=ExecutionProfile(**resource)
        log=device_receipt(profile).with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        print(f'Physical GPU {profile.gpu}: checking CART gradients and all three seeded initializations.',flush=True)
        with log.open('w') as handle:
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_cart_train_validation.py'),'--verify-device'],
                cwd=ROOT,env=child_environment(profile),stdout=handle,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f'Physical GPU {profile.gpu}: working, elapsed {time.monotonic()-started:.0f}s; ETA unavailable.',flush=True)
            except BaseException:process.terminate();process.wait();raise
        if process.returncode:raise ValueError(f'Physical GPU {profile.gpu} verification failed. See {log}')
        print(f'Physical GPU {profile.gpu}: PASSED; zero research runs.',flush=True)


def child_command(job,*,recover):
    command=[sys.executable,str(ROOT/'scripts/run_cart_train_validation.py'),'--execute',
        '--stage','replication_30r','--seed',str(job['seed']),'--pair',job['pair'],
        '--output',job['output'],'--campaign-plan',str(Path(job['output']).parents[1]/'plan.json')]
    if recover:command.append('--recover')
    return command


def pilot(args):
    """Start the pilot with CUDA wheel paths present at process startup."""
    from basil_core.execution_profile import ExecutionProfile
    prepared=preflight(args);verified_devices(prepared)
    job=next(j for j in prepared['jobs'] if j['seed']==2026 and j['pair']=='D')
    profile=ExecutionProfile(**job['resources'])
    command=[sys.executable,str(ROOT/'scripts/run_cart_replication.py'),'--pilot-worker',
        '--profile',profile.name,'--gpus',str(profile.gpu),'--gpu-memory-mib',str(profile.memory_mib),
        '--output',str(Path(prepared['output']))]
    print(f'Pilot: starting fresh worker on physical GPU {profile.gpu} with its CUDA loader environment.',flush=True)
    process=subprocess.Popen(command,cwd=ROOT,env=child_environment(profile))
    try:code=process.wait()
    except BaseException:
        process.terminate();process.wait();raise
    if code:raise ValueError(f'Pilot worker failed with exit code {code}; the full study was not launched.')


def pilot_worker(args):
    """Real data/model check of control and CART, including exact resumed state."""
    from basil_core.execution_profile import ExecutionProfile,use_profile
    from scripts.run_cart_train_validation import preflight as prepare,verify_barrier,runtime_identity,require_device_receipt
    from scripts.run_cart_memory_smoke import save_json,archive_sources
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    from basil_core.cart_validation import selected_configurations,make_worker
    from basil_core.protocol_checkpoint import save_checkpoint,load_checkpoint,TelemetryStore
    from basil_core.research_protocol import build_model,params_hash,run_research_protocol
    prepared=preflight(args);verified_devices(prepared)
    job=next(j for j in prepared['jobs'] if j['seed']==2026 and j['pair']=='D')
    profile=ExecutionProfile(**job['resources']);root=Path(prepared['output'])
    with use_profile(profile):
        inputs=prepare(SimpleNamespace(output=Path(job['output']),recover=False,device='GPU',
            stage='replication_30r',seed=2026,pair='D'))
        prepare_environment('GPU')
        import numpy as np
        import tensorflow as tf
        configure_tensorflow(tf,'GPU')
        runtime=runtime_identity(tf,np,'GPU');require_device_receipt(runtime,inputs['source'])
        verify_barrier(tf,'GPU')
        with tf.device('/CPU:0'):initial=params_hash(build_model('basil_paper_cnn',2026).get_weights())
        configs=selected_configurations(inputs['partition_hash'],initial,'GPU','replication_30r',seed=2026,pair='D')
        root.mkdir(parents=True,exist_ok=True)
        save_json(root/'pilot_contract.json',dict(plan=prepared,runtime=runtime,researchRunCompleted=False,
            stoppedActivations=4,officialTestUsed=False))
        for cfg in configs:
            output=root/cfg['runId'];output.mkdir()
            archive_sources(output/'source_snapshots',inputs['source'])
            def partial(path,limit,state=None):
                seen=[];store=TelemetryStore(path)
                def checkpoint(current):
                    save_checkpoint(path,cfg,current,partition_hash=inputs['partition_hash'],initial_hash=initial,telemetry_store=store)
                print(f'Pilot GPU {profile.gpu}: {cfg["runId"]}, {limit} activations, resume={state is not None}',flush=True)
                try:
                    with tf.device('/GPU:0'):
                        run_research_protocol(cfg,inputs['x'],inputs['y'],inputs['vx'],inputs['vy'],
                            worker_factory=lambda model:make_worker(model,cfg),resume_state=state,
                            activation_callback=lambda record:seen.append(record),checkpoint_callback=checkpoint,
                            should_stop=lambda:len(seen)>=limit)
                except InterruptedError:pass
                else:raise ValueError('The bounded pilot unexpectedly completed a full study.')
                return load_checkpoint(path,cfg,partition_hash=inputs['partition_hash'],initial_hash=initial)
            first=partial(output/'checkpoint.npz',3)
            if first['next_activation']!=3:raise ValueError('Pilot interruption cursor differs.')
            resumed=partial(output/'checkpoint.npz',1,first)
            reference=partial(output/'reference.npz',4)
            if resumed['next_activation']!=4 or reference['next_activation']!=4:raise ValueError('Pilot resume cursor differs.')
            with np.load(output/'checkpoint.npz',allow_pickle=False) as a,np.load(output/'reference.npz',allow_pickle=False) as b:
                if set(a.files)!=set(b.files):raise ValueError('Pilot checkpoint fields differ after recovery.')
                for field in a.files:
                    if field=='metadata':
                        if json.loads(str(a[field]))!=json.loads(str(b[field])):raise ValueError('Pilot checkpoint metadata differs.')
                    elif not np.array_equal(a[field],b[field]):raise ValueError(f'Pilot resumed state differs: {field}')
            print(f'Pilot {cfg["runId"]}: exact weights, BASIL history, CART memory and telemetry recovery PASSED.',flush=True)
        result=dict(status='PILOT_PASSED',physicalGpuId=profile.gpu,seed=2026,condition='D',
            controlAndCartChecked=True,exactResumeVerified=True,officialTestUsed=False,researchRunCompleted=False)
        save_json(root/'pilot.json',result)
        print(json.dumps(result),flush=True)


def read_snapshot(root,*,display_limit=None):
    from basil_core.terminal.state import alive
    root=Path(root)
    state=json.loads((root/'campaign.json').read_text())
    saved=json.loads((root/'plan.json').read_text())
    entries=[];now=time.time();remaining={p['gpu']:0 for p in saved['resources']}
    work=0;elapsed_work=0.
    for job in state['jobs']:
        check=next(c for c in saved['checks'] if c['job']==job['id'])
        per_node=[5*((n+511)//512) for n in check['nodeSamples']]
        for name in job['arms']:
            file=Path(job['output'])/name/'run.json'
            run=json.loads(file.read_text()) if file.exists() else {}
            elapsed_work+=run.get('runtimeSeconds',0.)
            cursor=run.get('completedActivations',0)
            batches=(cursor//10)*sum(per_node)+sum(per_node[:cursor%10])
            status={'completed':'COMPLETED','interrupted':'STOPPED','failed':'FAILED','running':'RUNNING'}.get(run.get('status'),'PENDING')
            if not run and job['status'] in ('FAILED','STOPPED'):status=job['status']
            remaining[job['resources']['gpu']]+=300-cursor;work+=cursor
            entries.append(dict(name=f'{job["id"]}/{name}',condition=name,dataset='cifar10',nonIID=True,nodes=10,
                rounds=30,device='GPU',gpu=job['resources']['gpu'],status=status,
                completedRounds=run.get('completedRounds',0),completedActivations=cursor,totalActivations=300,
                completedOptimizerSteps=batches,totalOptimizerSteps=check['optimizerStepsPerArm'],progress=cursor/300.,
                elapsed_seconds=run.get('runtimeSeconds'),eta_seconds=None,
                checkpoint=dict(activation=cursor) if (file.parent/'checkpoint.npz').exists() else None,
                metrics=dict(mean_node_accuracy=run.get('latestValidationMean'),worst_node_accuracy=run.get('latestValidationWorst'),
                    latest_node_accuracy=run.get('latestNodeValidationAccuracy'),local_training_loss=run.get('latestLocalTrainingLoss')),
                output=str(file.parent),error=run.get('error')))
    state.update(entries=entries,resources=saved['resources'],queue='CART TRAIN holdout: 30 rounds, 3 seeds, 24 arms',
        profile=saved['profile'],elapsed_seconds=state.get('prior_elapsed_seconds',0.)+(state.get('finished_wall',now)-state['started_wall']),
        eta_seconds=max(remaining.values())*elapsed_work/work if work else None,
        totals=dict(total=24,completed=sum(e['status']=='COMPLETED' for e in entries),
            failed=sum(e['status']=='FAILED' for e in entries),pending=sum(e['status']=='PENDING' for e in entries),
            blocked=0,progress=work/7200.,completedActivations=work,totalActivations=7200))
    if state['status']=='RUNNING' and not alive(state['owner']):
        state.update(status='STOPPED',error='Supervisor is no longer running. Inspect GPU processes before --recover.')
    if display_limit and state['status']=='RUNNING':
        order={'RUNNING':0,'FAILED':1,'STOPPED':2,'PENDING':3,'COMPLETED':4}
        state['entries']=sorted(entries,key=lambda e:order[e['status']])[:display_limit]
    return state


def execute(args,prepared=None):
    import fcntl
    from basil_core.execution_profile import ExecutionProfile
    from basil_core.terminal.state import identity
    from scripts.run_cart_memory_smoke import keep_json,save_json
    prepared=prepared or preflight(args)
    verified_devices(prepared)
    root=Path(prepared['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:raise ValueError('This campaign already has a supervisor.') from error
        keep_json(root/'plan.json',prepared)
        stop=False
        def request_stop(*_):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        (root/'stop.requested').unlink(missing_ok=True)
        previous=json.loads((root/'campaign.json').read_text()) if (root/'campaign.json').exists() else {}
        jobs=[dict(j,status='PENDING') for j in prepared['jobs']]
        state=dict(status='RUNNING',owner=identity(),host=identity()['host'],started_wall=time.time(),jobs=jobs,
            activity=[],previousRunStatus=previous.get('status'))
        state['prior_elapsed_seconds']=previous.get('prior_elapsed_seconds',0.)+max(0.,
            previous.get('finished_wall',previous.get('updated_wall',0.))-previous.get('started_wall',0.))
        logs=root/'logs';logs.mkdir(exist_ok=True)
        active={}
        def event(message):
            from datetime import datetime,timezone
            state['activity'].append(dict(time=datetime.now(timezone.utc).isoformat(),message=message))
            state['activity']=state['activity'][-12:];state['last_event_wall']=time.time()
            print(message,flush=True)
        def persist():
            state['updated_wall']=time.time()
            save_json(root/'campaign.json',state)
        event('Starting fixed CART replication; official TEST unused.');persist()
        try:
            while True:
                stop=stop or (root/'stop.requested').exists()
                for gpu,(process,handle,job) in list(active.items()):
                    code=process.poll()
                    if code is not None:
                        handle.close();del active[gpu]
                        job.update(status='COMPLETED' if code==0 else 'STOPPED' if stop else 'FAILED',
                            exitCode=code,finished_wall=time.time())
                        event(f'{job["id"]} {job["status"]} on physical GPU {gpu}')
                        if code and not stop:stop=True;state['error']=f'{job["id"]} failed; inspect logs before recovery.'
                if stop:
                    for process,_,job in active.values():
                        if not job.get('stopSent'):process.terminate();job['stopSent']=True
                    if not active:break
                else:
                    for job in jobs:
                        gpu=job['resources']['gpu']
                        if job['status']!='PENDING' or gpu in active:continue
                        frozen_sources(prepared)
                        profile=ExecutionProfile(**job['resources'])
                        environment=child_environment(profile)
                        handle=(logs/f'{job["id"]}.log').open('a')
                        try:process=subprocess.Popen(child_command(job,recover=args.recover),cwd=ROOT,env=environment,
                            stdout=handle,stderr=subprocess.STDOUT)
                        except BaseException:handle.close();raise
                        job.update(status='RUNNING',pid=process.pid,started_wall=time.time())
                        active[gpu]=(process,handle,job)
                        event(f'{job["id"]} working on physical GPU {gpu}; control then CART, 30 rounds each')
                    if not active:break
                persist();time.sleep(5)
            if any(j['status']=='FAILED' for j in jobs):state['status']='FAILED'
            elif stop:state['status']='STOPPED'
            else:
                from scripts.report_cart_replication import review
                review(root)
                state['status']='COMPLETED'
            event(f'Campaign {state["status"]}');state['finished_wall']=time.time();persist()
        except BaseException as error:
            for process,_,_ in active.values():process.terminate()
            for process,handle,_ in active.values():process.wait();handle.close()
            state.update(status='FAILED',error=str(error),finished_wall=time.time());persist()
            raise
    if state['status']!='COMPLETED':raise ValueError('Campaign stopped before complete verification. Inspect status/logs; --recover preserves checkpoints.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    for name in ('execute','preflight','verify-gpus','pilot','status','monitor','stop'):action.add_argument('--'+name,action='store_true')
    action.add_argument('--pilot-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--profile',choices=('local','hpc'),default='local')
    parser.add_argument('--gpus');parser.add_argument('--gpu-memory-mib',type=int)
    parser.add_argument('--output',type=Path,default=ROOT/'.papermerge/local-tests/cart-replication-30r-20261026')
    parser.add_argument('--recover',action='store_true');parser.add_argument('--tmux')
    args=parser.parse_args()
    try:
        if args.status or args.monitor:
            from basil_core.terminal.dashboard import monitor
            monitor(args.output,interval=5.,once=args.status,
                fetch=(lambda root:read_snapshot(root,display_limit=8)) if args.monitor else read_snapshot)
        elif args.stop:
            snapshot=read_snapshot(args.output)
            if snapshot['status']!='RUNNING':raise ValueError('Campaign is not running.')
            (args.output/'stop.requested').touch();print('Graceful stop requested; committed checkpoints are retained.')
        elif args.preflight:print(json.dumps(preflight(args),indent=2))
        elif args.verify_gpus:verify_devices(args)
        elif args.pilot:pilot(args)
        elif args.pilot_worker:pilot_worker(args)
        elif args.execute:
            prepared=preflight(args)
            if args.tmux:
                verified_devices(prepared)
                if not re.fullmatch(r'[A-Za-z0-9_-]+',args.tmux):raise ValueError('Use a plain tmux session name.')
                command=[sys.executable,str(Path(__file__).resolve()),'--execute','--profile',args.profile,
                    '--gpus',args.gpus or '0','--output',str(args.output.resolve())]
                if args.gpu_memory_mib is not None:command+=['--gpu-memory-mib',str(args.gpu_memory_mib)]
                if args.recover:command.append('--recover')
                subprocess.run(['tmux','new-session','-d','-s',args.tmux,shlex.join(command)],check=True)
                print(f'tmux attach-session -t {args.tmux}\nUse --monitor for the live 5-second screen.')
            else:execute(args,prepared)
        else:print(json.dumps(plan(args),indent=2))
    except (OSError,ValueError,subprocess.SubprocessError) as error:parser.exit(2,str(error)+'\n')


if __name__=='__main__':main()
