#!/usr/bin/env python3
"""Synthetic CPU/GPU derivative and initialization parity for the new model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def verify(output):
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    from basil_core.research_protocol import build_model,params_hash
    from basil_core.factorial_study import MODEL,SEEDS,CONDITIONS,configuration
    from basil_core.factorial_worker import make_worker
    from basil_core.execution_profile import current_profile
    from basil_core.protocol_checkpoint import checkpoint_implementation
    from scripts.run_cart_train_validation import runtime_identity
    from scripts.run_cart_memory_smoke import save_json
    prepare_environment('GPU')
    import numpy as np
    import tensorflow as tf
    configure_tensorflow(tf,'GPU')
    identities={}
    for seed in SEEDS:
        with tf.device('/CPU:0'):cpu=params_hash(build_model(MODEL,seed).get_weights())
        with tf.device('/GPU:0'):gpu=params_hash(build_model(MODEL,seed).get_weights())
        if cpu!=gpu:raise ValueError('Seeded CPU/GPU initialization mismatch.')
        identities[str(seed)]=cpu
    if len(set(identities.values()))!=3:raise ValueError('Different seeds produced identical initial models.')
    x=np.random.default_rng(778).uniform(0,1,(10,32,32,3)).astype(np.float32)
    y=np.arange(10,dtype=np.int32)
    key=tf.constant([17,31],tf.int32)
    outputs={}
    for device in ('CPU','GPU'):
        with tf.device(f'/{device}:0'):
            model=build_model(MODEL,2025)
            for name in ('clean','noise_ebm','attack_noise_basil_ebm_cart'):
                condition=next(c for c in CONDITIONS if c.name==name)
                cfg=configuration(condition,'iid',2025,.01 if condition.noise else 0.,stage='smoke',device=device)
                worker=make_worker(model,cfg)
                if condition.cart:
                    padded=np.zeros((160,32,32,3),np.float32);padded[:10]=x
                    worker.probe_images.assign(padded)
                    worker._set_retention_controls(y,[dict(classId=c,bestLoss=0.) for c in range(10)])
                params=worker.export()
                gradients,values=worker._gradients(tf.constant(x),tf.constant(y),key,cfg['ebmMode'],
                    'paper_absolute_gaussian',tf.constant(cfg['channelNoiseSigmaAbsolute'],tf.float32),tf.constant(0.))
                repeated,_=worker._gradients(tf.constant(x),tf.constant(y),key,cfg['ebmMode'],
                    'paper_absolute_gaussian',tf.constant(cfg['channelNoiseSigmaAbsolute'],tf.float32),tf.constant(0.))
                for first,second in zip(gradients,repeated):
                    np.testing.assert_array_equal(first.numpy(),second.numpy())
                if not all(device in g.device for g in gradients):raise ValueError('Gradient ran on the wrong device.')
                if params_hash(worker.export())!=params_hash(params):raise ValueError('Verification changed model weights.')
                if condition.cart:
                    if not 0 < float(values[16]) <= .1*float(values[4])+1e-6:
                        raise ValueError('CART correction did not satisfy its declared gradient budget.')
                    if float(values[11])!=0 or float(values[14])!=0:
                        raise ValueError('A hidden anchor or logit prior was enabled.')
                outputs[device,name]=np.concatenate([g.numpy().ravel() for g in gradients])
                print(f'{device}: {name} finite gradients, no optimizer updates.',flush=True)
    comparisons={}
    for name in ('clean','noise_ebm','attack_noise_basil_ebm_cart'):
        cpu,gpu=outputs['CPU',name],outputs['GPU',name]
        relative=float(np.linalg.norm(cpu-gpu)/max(np.linalg.norm(cpu),1e-12))
        max_update=float(np.max(np.abs(.1*(cpu-gpu))))
        if relative>=.01 or max_update>=1e-4:
            raise ValueError(f'CPU/GPU derivative mismatch for {name}: {relative:g}, {max_update:g}')
        comparisons[name]=dict(relativeL2=relative,maxUpdateDifference=max_update)
    cfg=configuration(CONDITIONS[0],'iid',2025,0.,stage='smoke')
    save_json(output,dict(status='PASSED',initialModelHashes=identities,gradientChecks=comparisons,
        runtime=runtime_identity(tf,np,'GPU'),source=checkpoint_implementation(cfg),
        physicalGpuId=current_profile().gpu,optimizerSteps=0,researchRunsStarted=0,sameDeviceRepeatability=True))
    print(f'GPU {current_profile().gpu}: all derivative/initialization checks PASSED.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=('local','hpc'),default='local')
    p.add_argument('--gpus',default='0')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:verify(args.output)
    else:
        from basil_core.execution_profile import ExecutionProfile,worker_environment,use_profile,gpu_inventory,validate_resources
        inventory=gpu_inventory();profiles=[]
        for gpu in [int(v) for v in args.gpus.split(',')]:
            profiles.append(ExecutionProfile(name=args.profile,gpu=gpu,
                memory_mib=8192 if args.profile=='hpc' else 6144,
                headroom_mib=2048 if args.profile=='hpc' else 1024,
                output_root=str(args.output.resolve()) if args.profile=='hpc' else '').bind(inventory))
        validate_resources(profiles,inventory)
        for profile in profiles:
            with use_profile(profile):env={**os.environ,**profile.environment(),**worker_environment('GPU')}
            target=args.output/f'gpu-{profile.gpu}.json'
            if target.exists():raise FileExistsError('Verification report exists; choose a fresh output directory.')
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(target.resolve())],
                cwd=ROOT,env=env)
            started=time.monotonic()
            while process.poll() is None:
                time.sleep(5)
                if process.poll() is None:
                    print(f'Physical GPU {profile.gpu}: verifying, elapsed {time.monotonic()-started:.0f}s; ETA unavailable.',flush=True)
            if process.returncode:raise RuntimeError(f'Physical GPU {profile.gpu} verification failed ({process.returncode}).')


if __name__=='__main__':main()
