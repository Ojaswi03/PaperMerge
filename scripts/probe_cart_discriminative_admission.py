#!/usr/bin/env python3
"""Paired continuity candidate screen; never changes an active study.

Continuity is an experimental intervention, not an approved defense or established novelty.
All arms use the same full TRAIN partition, initialization and keyed randomness.
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

CASES=(('discriminative_class_cart',True,'class'),('balanced_guard_cart',True,'guard_only'))


def configuration(args,index):
    from basil_core.sharp_study import configuration as base
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import configuration as cart
    name,cart_on,mode=CASES[index]
    condition='attack_noise_defense_rla_cart' if cart_on else 'attack_noise_defense_rla'
    cfg=cart(guarded_configuration(base('cifar10',args.partition,args.seed,condition,.01,nodes=args.nodes,
        rounds=args.rounds,attack_start=args.attack_start)))
    return dict(name=name,config=cfg,continuityMode=mode,diagnosticOnly=True)


def intervention_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('scripts/probe_cart_discriminative_admission.py','basil_core/cart_continuity_candidate.py',
         'basil_core/cart_qualified_continuity_candidate.py','basil_core/cart_admitted_continuity_candidate.py','basil_core/cart_semantic_admission_candidate.py','basil_core/cart_uncertainty_admission_candidate.py','basil_core/cart_discriminative_continuity_candidate.py','basil_core/cart_discriminative_admission_candidate.py')}


def worker(args):
    from scripts.cart_v2_slot_worker import adopt_lease
    adopt_lease(args.lease_fd)
    from scripts.run_cart_v2_study import setup_gpu,fingerprint,runtime,write_json
    tf=setup_gpu()
    import numpy as np
    import basil_core.sharp_cart_v2_protocol as protocol
    from basil_core.sharp_compiled_transfer import CompiledTransferWorker
    from basil_core.sharp_data import training_data,prepare_partition,digest
    from basil_core.sharp_protocol import vector_hash,save_state,load_state
    case=configuration(args,args.worker);cfg=case['config'];root=args.output/case['name'];root.mkdir(exist_ok=True)
    from basil_core.cart_admitted_continuity_candidate import AdmittedWorker as ContinuityWorker
    model=(ContinuityWorker if case['continuityMode'] else CompiledTransferWorker)(cfg,'GPU');initial=vector_hash(model.export())
    x,y=training_data('cifar10');chunks,held,audit=prepare_partition(y,cfg)
    contract=dict(config=cfg,diagnostic=case,source=fingerprint(),runtime=runtime(tf),initialHash=initial,
        data=dict(imagesSha256=digest(x),labelsSha256=digest(y)),partition=audit,
        diagnosticProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interventionProgramSha256=intervention_hashes(),
        transferProgramSha256=hashlib.sha256((ROOT/'basil_core/sharp_compiled_transfer.py').read_bytes()).hexdigest())
    checkpoint=root/'checkpoint.npz';state=load_state(checkpoint,contract,model.dimension) if checkpoint.exists() else None
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    original=protocol.select_input
    if case['continuityMode']:
        from basil_core.cart_discriminative_admission_candidate import selector
        protocol.select_input=selector(original,case['continuityMode'])
    def round_done(row,current):
        print(f'{case["name"]} R{row["round"]+1}/{cfg["rounds"]}: {row["meanAccuracy"]:.2%}',flush=True)
        write_json(root/'progress.json',dict(rounds=len(current['roundMetrics']),meanAccuracy=row['meanAccuracy'],elapsedSeconds=current['elapsedSeconds']))
    try:
        result=protocol.run(cfg,model,x,y,chunks,x[held],y[held],state=state,
            on_round=round_done,checkpoint=lambda s:save_state(checkpoint,s,contract),should_stop=lambda:stopped)
        assert all(r['training']['optimizerSteps']==cfg['localSteps'] for r in result['telemetry'])
        last=result['roundMetrics'][-1];records=result['telemetry'][cfg['nodes']:]
        hops=[(r['node']-s)%cfg['nodes'] for r in records for s in r['selection']['selectedSenders']]
        routing=[r['selection'].get('continuity',{}) for r in records]
        summary=dict(case=case,status='COMPLETED_DIAGNOSTIC',contract=contract,meanAccuracy=last['meanAccuracy'],
            worstNodeAccuracy=last['worstNodeAccuracy'],roundMetrics=result['roundMetrics'],
            meanSelectedCount=float(np.mean([r['selection']['selectedCount'] for r in records])),
            meanSelectedHop=float(np.mean(hops)) if hops else None,meanLastRoundTrainCE=float(np.mean([
                r['training']['localTrainingLoss'] for r in result['telemetry'][-cfg['nodes']:]])),
            elapsedSeconds=result['elapsedSeconds'],officialTestUsed=False,
            ownRoutes=sum(r.get('route')=='own' for r in routing),peerRoutes=sum(r.get('route')=='peer' for r in routing),
            classBoundChecks=sum(r.get('qualifiedInitializationBound',False) for r in routing),
            meanLearnedClasses=float(np.mean([len(r.get('learnedClasses',[])) for r in routing])),
            calibrationDraws=sum(r.get('calibrationDraws',0) for r in routing),
            actualRouteCounts={name:sum(r.get('actualRoute')==name for r in routing) for name in ('fresh','pool','own','repaired_fresh')},
            interpretation='Single-seed candidate screen; routing bounds concern private probes before SGD, not established robustness or novelty.')
        write_json(root/'result.json',summary)
    finally:protocol.select_input=original


def execute(args):
    from basil_core.execution_profile import ExecutionProfile,gpu_inventory
    from scripts.run_cart_v2_scaled import acquire_leases
    from scripts.run_cart_v2_study import worker_environment,fingerprint,write_json
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'plan.json').exists():raise FileExistsError('Use a fresh diagnostic output directory.')
    profile=ExecutionProfile(name='hpc',gpu=0,memory_mib=3072,headroom_mib=1024,ram_mib=8192,
        output_root=str(args.output)).bind(gpu_inventory())
    free=next(r['memory.free'] for r in gpu_inventory() if r['index']==0)
    if free<args.workers*(profile.memory_mib+1024)+profile.headroom_mib:raise RuntimeError('Insufficient GPU headroom.')
    write_json(args.output/'plan.json',dict(cases=[configuration(args,i) for i in range(len(CASES))],profile=asdict(profile),
        source=fingerprint(),diagnosticProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interventionProgramSha256=intervention_hashes(),officialTestUsed=False))
    leases=acquire_leases([profile]);active={};pending=list(range(len(CASES)));failed=[]
    try:
        while pending or active:
            for index,(process,log) in list(active.items()):
                if process.poll() is not None:
                    if process.returncode:failed.append(index)
                    log.close();del active[index]
            while pending and len(active)<args.workers:
                index=pending.pop(0);log=(args.output/(CASES[index][0]+'.log')).open('a');fd=leases[0].fileno()
                command=[sys.executable,__file__,'--output',str(args.output),'--nodes',str(args.nodes),'--rounds',str(args.rounds),
                    '--partition',args.partition,'--seed',str(args.seed),'--attack-start',str(args.attack_start),
                    '--worker',str(index),'--lease-fd',str(fd)]
                active[index]=(subprocess.Popen(command,cwd=ROOT,env=worker_environment(profile),pass_fds=(fd,),stdout=log,stderr=subprocess.STDOUT),log)
            if active:time.sleep(2)
        results=[json.loads((args.output/name/'result.json').read_text()) for i,(name,*_) in enumerate(CASES) if i not in failed]
        for row in results:print(f'{row["case"]["name"]}: {row["meanAccuracy"]:.2%}, q={row["meanSelectedCount"]:.1f}, hop={row["meanSelectedHop"]}',flush=True)
        write_json(args.output/'summary.json',dict(status='FAILED_CASES' if failed else 'COMPLETE_DIAGNOSTIC',failed=failed,cases=results))
        if failed:raise RuntimeError(f'Diagnostic cases failed: {failed}; inspect their logs.')
    finally:
        for process,log in active.values():
            if process.poll() is None:process.terminate()
            process.wait();log.close()
        for handle in leases.values():handle.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nodes',type=int,default=100);p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--partition',choices=('iid','dirichlet'),default='iid');p.add_argument('--seed',type=int,default=2025)
    p.add_argument('--attack-start',type=int,default=20)
    p.add_argument('--worker',type=int,choices=range(len(CASES)));p.add_argument('--lease-fd',type=int)
    args=p.parse_args();args.output=args.output.resolve()
    if args.worker is not None:worker(args)
    else:execute(args)
