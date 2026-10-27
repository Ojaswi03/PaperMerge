#!/usr/bin/env python3
"""Bounded real-data SHARP+CART recovery check across fresh interpreters.

Explicit CPU or GPU execution; no automatic fallback and no official TEST.
This checks state recovery, not predictive effectiveness or a completed study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def child(args):
    from scripts.run_sharp_study import setup_gpu,runtime,fingerprint,data_contract,write_json
    if args.device=='GPU':
        tf=setup_gpu()
        identity=runtime(tf)
    else:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
        identity=dict(device='/CPU:0',tensorflow=tf.__version__)
    from basil_core.sharp_study import configuration
    from basil_core.sharp_data import training_data,prepare_partition
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import run,save_state,load_state,vector_hash
    cfg=configuration('mnist','dirichlet',2025,'attack_noise_defense_rla_cart',.01,
        nodes=8,rounds=4,attack_start=1,train_per_class=40,validation_per_class=10)
    worker=SharpWorker(cfg,args.device)
    data,_=data_contract([cfg]);x,y=training_data('mnist')
    chunks,heldout,audit=prepare_partition(y,cfg)
    contract=dict(config=cfg,source=fingerprint(),data=data['mnist'],partition=audit,
        runtime=identity,initialHash=vector_hash(worker.export()),
        verificationProgramSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=args.output/'interrupted.npz'
    state=load_state(path,contract,worker.dimension) if args.phase=='resume' else None
    result=run(cfg,worker,x,y,chunks,x[heldout],y[heldout],state=state,
        max_activations=17 if args.phase=='partial' else None)
    target=path if args.phase=='partial' else args.output/f'{args.phase}.npz'
    save_state(target,result,contract)
    write_json(args.output/f'{args.phase}.json',dict(cursor=result['cursor'],runtime=identity,
        nonzeroCartCorrections=sum(r['cart']['scale']>0 for r in result['telemetry']),
        references=len(result['references']),officialTestUsed=False))


def compare(root):
    import numpy as np
    from scripts.run_sharp_study import write_json
    with np.load(root/'whole.npz',allow_pickle=False) as whole, np.load(root/'resume.npz',allow_pickle=False) as resumed:
        if set(whole.files)!=set(resumed.files):raise ValueError('Recovered arrays differ.')
        checks={}
        for key in whole.files:
            if key=='metadata':continue
            if not np.array_equal(whole[key],resumed[key]):raise ValueError(f'Recovery changed {key}.')
            checks[key]=hashlib.sha256(whole[key].tobytes()).hexdigest()
        a=json.loads(whole['metadata'].tobytes());b=json.loads(resumed['metadata'].tobytes())
        a.pop('elapsedSeconds');b.pop('elapsedSeconds')
        if a!=b:raise ValueError('Recovery changed telemetry, references, metrics or identity.')
    result=dict(status='EXACT_FRESH_PROCESS_RECOVERY_VERIFIED',cursor=a['cursor'],
        contract=a['contract'],exactArraySha256=checks,telemetryExact=True,roundMetricsExact=True,
        cartMemoryExact=True,officialTestUsed=False,
        nonzeroCartCorrections=json.loads((root/'resume.json').read_text())['nonzeroCartCorrections'],
        interpretation='Bounded eight-node four-round real TRAIN-data checkpoint check, not an efficacy study.')
    if not result['nonzeroCartCorrections']:raise ValueError('The recovery probe did not exercise a CART correction.')
    write_json(root/'recovery_verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('CPU','GPU'),required=True)
    parser.add_argument('--profile',choices=('local','hpc'),default='local')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--phase',choices=('partial','resume','whole'),help=argparse.SUPPRESS)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.phase:
        child(args)
        return
    if args.output.exists():raise FileExistsError('Use a fresh recovery-check output directory.')
    args.output.mkdir(parents=True)
    environment=dict(os.environ,TF_ENABLE_ONEDNN_OPTS='0',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if args.device=='GPU':
        from scripts.run_sharp_study import profiles,worker_environment
        from types import SimpleNamespace
        profile=profiles(SimpleNamespace(gpus=str(args.gpu),profile=args.profile,output=args.output))[0]
        environment.update(worker_environment(profile))
    else:
        environment['CUDA_VISIBLE_DEVICES']='-1'
    for phase in ('partial','resume','whole'):
        print(f'{args.device}: {phase} in a fresh interpreter',flush=True)
        command=[sys.executable,__file__,'--output',str(args.output),'--device',args.device,'--phase',phase]
        with (args.output/f'{phase}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=compare(args.output)
    print(json.dumps({k:result[k] for k in ('status','cursor','nonzeroCartCorrections','officialTestUsed')},indent=2))


if __name__=='__main__':main()
