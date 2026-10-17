#!/usr/bin/env python3
"""Headless PaperMerge queues, resource diagnostics and persistent SSH monitoring."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def parser():
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--profile',choices=('local','hpc'),default='hpc',help='Opt-in resource policy (default: hpc)')
    common.add_argument('--state-dir',type=Path,default=ROOT/'.papermerge/terminal/default',help='Persistent controller state and logs')
    common.add_argument('--output-root',type=Path,help='Separate HPC generated-results namespace (default: .papermerge/runs/default)')
    common.add_argument('--gpus',help='Explicit physical GPU index, or comma-separated indices for gated independent scheduling')
    common.add_argument('--gpu-memory-mib',type=int,help='Per-worker TensorFlow allocator cap (HPC default: 8192 MiB)')
    common.add_argument('--gpu-headroom-mib',type=int,help='Free-memory headroom (HPC default: 2048 MiB)')
    common.add_argument('--cpu-threads',type=int,default=1,help='TensorFlow/BLAS threads per worker (1-8)')
    common.add_argument('--ram-budget-mib',type=int,default=32768,help='Worker-tree RSS budget; gracefully stopped if exceeded')
    common.add_argument('--data-workers',type=int,default=0,help='Deterministic approved workers currently require 0')
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    doctor=commands.add_parser('doctor',parents=[common],help='Check environment and real synthetic GPU kernels; no research run')
    doctor.add_argument('--queue',type=Path,help='Also inspect this queue and its scientific prerequisites')
    doctor.add_argument('--no-gpu-check',action='store_true',help='Discovery only; returns unverified/blocked status')
    doctor.add_argument('--iid-only',action='store_true',help='Verify IID kernels only; separate objective receipts remain required')
    commands.add_parser('queues',parents=[common],help='List presets and recorded progress without training')
    plan=commands.add_parser('plan',parents=[common],help='Validate queue, resources, prerequisites and recovery; zero training')
    plan.add_argument('--queue',type=Path,required=True)
    run=commands.add_parser('run',parents=[common],help='Run a validated queue only with --execute')
    run.add_argument('--queue',type=Path,required=True)
    resume=commands.add_parser('resume',parents=[common],help='Validate and resume a saved controller queue only with --execute')
    for command in (run,resume):
        command.add_argument('--execute',action='store_true',help='Explicitly authorize starting validated workers')
        command.add_argument('--parallel',action='store_true',help='Independent IID conditions on distinct GPUs, gated by prior single-GPU success')
        command.add_argument('--independent-approved',action='store_true',help='Attest administrator permission and independent-condition protocol approval')
        command.add_argument('--tmux',metavar='NAME',help='Start a named detached tmux session, or report its existing attach command')
    status=commands.add_parser('status',parents=[common],help='Read the last queue snapshot from any SSH session')
    status.add_argument('--json',action='store_true',help='Emit the operational snapshot as JSON')
    monitor=commands.add_parser('monitor',parents=[common],help='Read-only terminal dashboard; exiting never stops training')
    monitor.add_argument('--interval',type=float,default=1.,help='Refresh seconds, default 1')
    monitor.add_argument('--once',action='store_true');monitor.add_argument('--plain',action='store_true')
    logs=commands.add_parser('logs',parents=[common],help='Read or follow the bounded execution log')
    logs.add_argument('--follow','-f',action='store_true');logs.add_argument('--lines','-n',type=int,default=30)
    logs.add_argument('--experiment',type=int,help='Zero-based entry index; read its worker log instead')
    stop=commands.add_parser('stop',parents=[common],help='Request graceful stop using the saved owner token; no unrelated process signals')
    stop.add_argument('--wait',type=float,default=0.,help='Wait up to this many seconds for graceful exit')
    commands.add_parser('gpu-status',parents=[common],help='Show physical GPU inventory and external compute processes')
    report=commands.add_parser('report',parents=[common],help='Generate plots from measured results only')
    report.add_argument('--queue',type=Path,required=True)
    data=commands.add_parser('cache-data',parents=[common],help='Prepare the existing CIFAR cache, without training')
    data.add_argument('--download',action='store_true',help='Explicitly authorize the standard dataset download/cache operation')
    worker=commands.add_parser('_worker',help='Internal guarded adapter; never a separate scientific engine')
    worker.add_argument('--runner',required=True,type=Path);worker.add_argument('--config',required=True,type=Path)
    worker.add_argument('--execute',action='store_true');worker.add_argument('--recover',action='store_true')
    return p


def select_profiles(args, *, require_selection=False, resolve=False):
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    if require_selection and args.profile=='hpc' and not args.gpus:
        raise ValueError('HPC execution requires an explicit --gpus physical index.')
    ids=[int(part) for part in (args.gpus or '0').split(',')]
    if len(ids)!=len(set(ids)):raise ValueError('GPU IDs must be distinct.')
    policies=[]
    devices=gpu_inventory() if resolve else None
    for gpu in ids:
        policy=ExecutionProfile(name=args.profile,gpu=gpu,
            memory_mib=args.gpu_memory_mib if args.gpu_memory_mib is not None else (8192 if args.profile=='hpc' else 6144),
            headroom_mib=args.gpu_headroom_mib if args.gpu_headroom_mib is not None else (2048 if args.profile=='hpc' else 1024),
            threads=args.cpu_threads,ram_mib=args.ram_budget_mib,data_workers=args.data_workers,
            output_root=str((args.output_root or ROOT/'.papermerge/runs/default').resolve()) if args.profile=='hpc' else '').validate()
        policies.append(policy.bind(devices) if devices is not None else policy)
    os.environ.update(policies[0].environment())
    return policies


def print_plan(path, policies, *, include_runtime=True):
    from basil_core.terminal.queues import inspect_queue
    configs,items=inspect_queue(path,include_runtime)
    for index,item in enumerate(items):
        item['physicalGpuId']=policies[index%len(policies)].gpu
        item['gpuMemoryLimitMiB']=policies[index%len(policies)].memory_mib
        item['configurationSha256']=hashlib.sha256(json.dumps(configs[index],sort_keys=True).encode()).hexdigest()
    print(json.dumps(dict(queue=str(Path(path).resolve()),resources=[asdict(p) for p in policies],
        experiments=items,trainingStarted=False),indent=2))
    return configs,items


def worker(args):
    if not args.execute:
        print('Worker dry run; --execute is required.');return 0
    from basil_core.terminal.queues import adapter,science
    from basil_core.execution_profile import profile_path
    from basil_core.terminal.state import lease
    cfg=json.loads(args.config.read_text());science(cfg)
    script=(ROOT/adapter(cfg)[2]).resolve()
    if args.runner.resolve()!=script:raise ValueError('Worker runner does not match the approved scientific protocol.')
    output=profile_path(ROOT/cfg['resultDirectory'])
    lock=output.parent/('.'+output.name+'.execution.lock')
    with lease(lock):
        sys.argv=[str(script),'--config',str(args.config.resolve()),'--execute']+(['--recover'] if args.recover else [])
        runpy.run_path(str(script),run_name='__main__')
    return 0


def tmux_start(args, argv):
    if not shutil.which('tmux'):
        raise ValueError('tmux is unavailable. Ask the administrator for tmux, an approved scheduler, or a permitted nohup workflow.')
    if not args.tmux or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.tmux):
        raise ValueError('Use an alphanumeric tmux session name with underscores/hyphens.')
    existing=subprocess.run(['tmux','has-session','-t',args.tmux],capture_output=True)
    if existing.returncode:
        forwarded=[];index=0
        while index<len(argv):
            if argv[index]=='--tmux':index+=2;continue
            if argv[index].startswith('--tmux='):index+=1;continue
            forwarded.append(argv[index]);index+=1
        command=shlex.join([sys.executable,str(Path(__file__).resolve()),*forwarded])
        subprocess.run(['tmux','new-session','-d','-s',args.tmux,'-c',str(Path.cwd()),command],check=True)
    print('tmux attach-session -t '+args.tmux)
    print('Detach using Ctrl+B, then D. Monitor/status remain independent.')
    return 0


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv);args=parser().parse_args(argv)
    from basil_core.terminal.state import inspect_state,read_json,safe_text
    if args.command=='_worker':return worker(args)
    args.state_dir=args.state_dir.resolve()
    if ROOT.is_relative_to(args.state_dir) or any(args.state_dir.is_relative_to(ROOT/name)
        for name in ('Papers','papers','newResults','newPlots','basil_core','scripts','gui','environment','.git','.venv-hpc')):
        raise ValueError('Operational state must be separate from source, environments, papers and research evidence.')
    if args.command=='status':
        data=inspect_state(args.state_dir)
        if args.json:print(json.dumps(data,indent=2))
        else:
            from basil_core.terminal.dashboard import render
            print(render(data,dict(host=data.get('host','unavailable'))))
        return 1 if data['status']=='FAILED' else 0
    if args.command=='monitor':
        from basil_core.terminal.dashboard import monitor
        monitor(args.state_dir,args.interval,once=args.once,plain=args.plain);return 0
    if args.command=='logs':
        from basil_core.terminal.dashboard import follow_logs
        if args.experiment is None:follow_logs(args.state_dir,follow=args.follow,lines=args.lines)
        else:
            if args.experiment<0:raise ValueError('Experiment index must be nonnegative.')
            saved=read_json(args.state_dir/'plan.json')
            if not saved or args.experiment>=len(saved['configs']):raise ValueError('Experiment index is not present.')
            directory=args.state_dir/'workers'/str(args.experiment)
            # The existing worker logger uses the original scientific run ID.
            source=directory/(saved['configs'][args.experiment]['runId']+'.log')
            follow_logs(args.state_dir,follow=args.follow,lines=args.lines,path=source)
        return 0
    if args.command=='stop':
        from basil_core.terminal.state import request_stop
        if args.wait<0:raise ValueError('--wait must be nonnegative.')
        request_stop(args.state_dir);print('Graceful stop requested; waiting for the worker checkpoint boundary.')
        deadline=time.monotonic()+args.wait
        while time.monotonic()<deadline:
            if inspect_state(args.state_dir)['status'] not in ('RUNNING','STOPPING','VALIDATING'):return 0
            time.sleep(.2)
        return 0
    if args.command=='gpu-status':
        from basil_core.execution_profile import gpu_inventory,gpu_processes
        print(json.dumps(dict(gpus=gpu_inventory(),processes=gpu_processes(),
            note='Project locks do not reserve GPUs against other users.'),indent=2));return 0
    if args.command=='resume':
        from basil_core.execution_profile import ExecutionProfile
        saved=read_json(args.state_dir/'plan.json')
        if not saved:raise ValueError('No saved queue plan exists; refusing to invent recovery state.')
        policies=[ExecutionProfile(**p).validate() for p in saved['policies']]
        overrides={'--profile':('name',args.profile),'--output-root':('output_root',str(args.output_root.resolve()) if args.output_root else ''),
            '--gpu-memory-mib':('memory_mib',args.gpu_memory_mib),'--gpu-headroom-mib':('headroom_mib',args.gpu_headroom_mib),
            '--cpu-threads':('threads',args.cpu_threads),'--ram-budget-mib':('ram_mib',args.ram_budget_mib),
            '--data-workers':('data_workers',args.data_workers)}
        for option,(field,value) in overrides.items():
            if any(part==option or part.startswith(option+'=') for part in argv) and any(getattr(p,field)!=value for p in policies):
                raise ValueError('Resume cannot change the saved resource policy: '+option)
        if args.gpus and [int(x) for x in args.gpus.split(',')]!=[p.gpu for p in policies]:
            raise ValueError('Resume must retain the saved physical GPU assignment.')
        os.environ.update(policies[0].environment());args.queue=Path(saved['queue'])
        configs=saved['configs']
        from basil_core.terminal.queues import load_queue
        if args.queue.exists() and load_queue(args.queue)!=configs:raise ValueError('Saved queue file changed; exact resume blocked.')
        from basil_core.terminal.queues import inspect_config
        from basil_core.execution_profile import use_profile
        snapshot=read_json(args.state_dir/'status.json',{})
        entries=snapshot.get('entries',[])
        items=[]
        for index,cfg in enumerate(configs):
            assigned=entries[index].get('gpu') if index<len(entries) else None
            matching=[p for p in policies if p.gpu==assigned] if assigned is not None else [policies[index%len(policies)]]
            if not matching:raise ValueError('Saved worker GPU is absent from the execution policy.')
            with use_profile(matching[0]):items.append(inspect_config(cfg))
        print(json.dumps(dict(resources=saved['policies'],experiments=items,trainingStarted=False),indent=2))
    else:
        # Discovery failure is reported in plans without hiding missing evidence.
        try:policies=select_profiles(args,require_selection=args.command=='run' and args.execute,resolve=bool(args.gpus))
        except RuntimeError:
            if args.command=='run' and args.execute:raise
            policies=select_profiles(args)
        if args.command=='doctor':
            from basil_core.terminal.doctor import diagnose
            reports=[diagnose(p,args.state_dir,verify_gpu=not args.no_gpu_check,objective_checks=not args.iid_only) for p in policies]
            if args.queue:
                from basil_core.terminal.queues import inspect_queue
                _,items=inspect_queue(args.queue)
                for report in reports:report['experiments']=items
                for report in reports:report['researchReady']=not any(i['status']=='BLOCKED' for i in items)
                if any(i['status']=='BLOCKED' for i in items):
                    for report in reports:report['status']='blocked'
            print(json.dumps(reports,indent=2));return 0 if all(r['status']=='passed' for r in reports) else 1
        if args.command=='queues':
            from basil_core.terminal.queues import inspect_queue
            paths=sorted((ROOT/'gui/queues').glob('*.json'))
            if (ROOT/'gui/queue_state.json').exists():paths.append(ROOT/'gui/queue_state.json')
            result=[]
            for path in paths:
                try:
                    _,items=inspect_queue(path,False)
                    result.append(dict(queue=str(path.relative_to(ROOT)),count=len(items),
                        totalRounds=sum(i.get('rounds') or 0 for i in items),experiments=items))
                except (ValueError,OSError,KeyError,TypeError) as error:result.append(dict(queue=str(path),status='BLOCKED',error=safe_text(error)))
            if (args.state_dir/'status.json').exists():
                result.append(dict(savedTerminalQueue=inspect_state(args.state_dir)))
            print(json.dumps(result,indent=2));return 0
        if args.command=='cache-data':
            if not args.download:
                print('Dataset preparation requires --download; no dataset or research files changed.');return 0
            from basil_core.iid_runtime import prepare_environment
            prepare_environment('CPU')
            from basil_core.data.cifar import loadCifar10
            loadCifar10(cacheDir=ROOT/'experiments/cache/cifar10')
            print('CIFAR cache prepared using the existing loader; zero research runs.');return 0
        configs,items=print_plan(args.queue,policies,include_runtime=args.command!='report')
        if args.command=='plan':return 1 if any(p['status']=='BLOCKED' for p in items) else 0
        if args.command=='report':
            return report_queue(configs)
    if args.command in ('run','resume'):
        if not args.execute:
            print('Validation only. --execute is required to start workers.');return 1 if any(i['status']=='BLOCKED' for i in items) else 0
        if len(policies)>1 and not args.parallel:raise ValueError('Multiple GPUs require explicit --parallel and independent-condition approval.')
        from basil_core.execution_profile import gpu_inventory,validate_resources
        inventory=gpu_inventory();policies=[p.bind(inventory) for p in policies]
        if any(i['status']!='COMPLETED' for i in items):validate_resources(policies,inventory)
        print(json.dumps(dict(selectedDevices=[g for g in inventory if g['index'] in {p.gpu for p in policies}],
            note='Per-GPU free-memory checks passed; project locks do not reserve resources against other users.'),indent=2))
        from basil_core.terminal.controller import Controller,validate_plan
        validate_plan(configs,items)
        # Each independently assigned device needs its own real verification.
        from basil_core.terminal.queues import prerequisites,completed
        for policy in policies:
            os.environ.update(policy.environment())
            for cfg in configs:
                if not completed(cfg):
                    problems=prerequisites(cfg)
                    if problems:raise ValueError('; '.join(problems))
        os.environ.update(policies[0].environment())
        if args.tmux:return tmux_start(args,argv)
        controller=Controller(args.state_dir,configs,items,policies,args.queue,
            resume=args.command=='resume',independent_approved=args.independent_approved)
        return controller.run()
    return 0


def report_queue(configs):
    from basil_core.execution_profile import profile_path
    from basil_core.terminal.queues import completed
    os.environ['MPLBACKEND']='Agg'
    for config in configs:
        if not completed(config):raise ValueError('Report requires completed measured results; no training is performed.')
        output=profile_path(ROOT/config['resultDirectory']);plots=profile_path(ROOT/config['plotDirectory'])
        if config.get('iidCampaignId'):
            from reporting.iid_study_plots import generate_run_plots
            generate_run_plots(output,plots)
        elif config.get('experimentProtocol')=='cart_verified_one_class_v1':
            from reporting.noniid_verified_plots import report_condition
            report_condition(output,plots)
        else:
            from reporting.noniid_study_plots import generate_condition_plots
            generate_condition_plots(output,plots)
        print('Saved measured-result plots: '+str(plots))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
    except Exception as error:
        from basil_core.terminal.state import safe_text
        print('PaperMerge: '+safe_text(error),file=sys.stderr,flush=True)
        raise SystemExit(1)
