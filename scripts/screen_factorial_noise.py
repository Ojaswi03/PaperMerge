#!/usr/bin/env python3
"""TRAIN-only channel severity calibration on a completed clean pilot.

Select low/medium/high severities by CE degradation of a fixed clean model,
not by which values make a defense win. No optimizer steps or TEST loading.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CANDIDATES=(.00125,.0025,.005,.01,.02,.04,.08,.16)
TARGETS=(.05,.25,.75)


def choose_levels(records):
    """Predeclared targets: 5%, 25%, 75% increases in clean TRAIN probe CE."""
    import math
    rows=sorted(records,key=lambda r:r['sigma'])
    if (len(rows)<3 or len({r['sigma'] for r in rows})!=len(rows)
            or any(not math.isfinite(r['relativeCEIncrease']) or r['sigma']<=0 for r in rows)):
        raise ValueError('Finite distinct candidate measurements required.')
    # All targets use relative CE, so different baseline CE scales are comparable.
    selected=min(itertools.combinations(rows,3),key=lambda group:
        sum(((r['relativeCEIncrease']-target)/target)**2 for r,target in zip(group,TARGETS)))
    return [r['sigma'] for r in selected]


def screen(args):
    import numpy as np
    from basil_core.iid_runtime import prepare_environment,configure_tensorflow
    from basil_core.protocol_checkpoint import load_checkpoint
    from basil_core.research_protocol import build_model,SequentialWorker,evaluate_params,keyed_rng,apply_channel_noise
    from basil_core.dirichlet_diagnosis import array_identity_hash
    from scripts.run_factorial_study import data,check_source
    from scripts.run_cart_memory_smoke import save_json
    from basil_core.factorial_study import validate_config
    metadata=json.loads((args.input/'run.json').read_text())
    cfg=metadata['config']
    validate_config(cfg)
    if metadata['status']!='COMPLETED' or cfg['conditionId']!='clean' or cfg['factorialStage']=='smoke':
        raise ValueError('Use a completed full-TRAIN clean pilot, not a reduced smoke run.')
    if args.output.exists():raise FileExistsError('Screen report exists; retain it and use a fresh path.')
    if not args.output.resolve().is_relative_to(ROOT/'.papermerge/local-tests'):
        raise ValueError('Place generated screening reports under .papermerge/local-tests.')
    check_source(metadata['source'])
    arrays,hashes=data()
    if hashes!=metadata['data']:raise ValueError('TRAIN data differs from the clean pilot.')
    x,y,_,_=arrays
    # These are already-fit TRAIN images. The heldout validation and TEST splits
    # do not determine sigma selection, memory admission, or optimization.
    probe=np.concatenate([keyed_rng(314159,'factorial_noise_probe',c).choice(np.flatnonzero(y==c),100,replace=False) for c in range(10)])
    px,py=x[probe],y[probe]
    partition=json.loads((args.input/'partition.json').read_text())
    state=load_checkpoint(args.input/'checkpoint.npz',cfg,
        partition_hash=partition['hash'],initial_hash=partition['initialModelHash'])
    prepare_environment('GPU')
    import tensorflow as tf
    configure_tensorflow(tf,'GPU')
    with tf.device('/GPU:0'):
        worker=SequentialWorker(build_model(cfg['modelArchitecture'],cfg['seed']))
        params=state['node_params'][9]  # Fixed endpoint, not best validation node.
        baseline,_,base_ce=evaluate_params(worker,params,px,py,128,details=True)
        if baseline<.7:raise ValueError(f'Clean model TRAIN probe accuracy {baseline:.2%} is too low for severity calibration.')
        rows=[]
        try:
            for sigma in CANDIDATES:
                draws=[]
                for draw in range(3):
                    noisy,noise=apply_channel_noise(params,semantics='paper_absolute_gaussian',sigma=sigma,
                        rng=keyed_rng(314159,'factorial_noise_draw',draw))
                    accuracy,_,ce=evaluate_params(worker,noisy,px,py,128,details=True)
                    draws.append(dict(draw=draw,accuracy=accuracy,ce=ce,**noise))
                relative=float(np.median([r['ce']/base_ce-1 for r in draws]))
                rows.append(dict(sigma=sigma,relativeCEIncrease=relative,draws=draws))
                print(f'sigma {sigma:g} | median TRAIN CE change {relative:+.1%}',flush=True)
        finally:worker.load(params)
    chosen=choose_levels(rows)
    report=dict(selection='fixed_targets_relative_TRAIN_CE',targets=list(TARGETS),candidates=list(CANDIDATES),
        sigmaLevels=chosen,measurements=rows,cleanTrainAccuracy=baseline,cleanTrainCE=base_ce,
        checkpoint=str(args.input.resolve()),checkpointSeed=cfg['seed'],checkpointNode=9,
        probeImagesHash=array_identity_hash(px),probeLabelsHash=array_identity_hash(py),
        source=metadata['source'],data=metadata['data'],noiseSeed=314159,drawsPerCandidate=3,
        officialTestUsed=False,validationUsedForSigmaSelection=False,optimizerSteps=0,
        defenseResultsUsedForSelection=False,performanceReady=False)
    save_json(args.output,report)
    print('Selected low/medium/high sigmas: '+', '.join(f'{v:g}' for v in chosen),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not args.execute:
        print(json.dumps(dict(candidates=CANDIDATES,targets=TARGETS,trainingStarted=False,officialTestUsed=False)))
    elif args.worker:screen(args)
    else:
        from basil_core.execution_profile import ExecutionProfile,worker_environment,use_profile
        profile=ExecutionProfile()
        with use_profile(profile):env={**os.environ,**profile.environment(),**worker_environment('GPU')}
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--execute','--worker',
            '--input',str(args.input.resolve()),'--output',str(args.output.resolve())],env=env,cwd=ROOT)
        started=time.monotonic()
        while process.poll() is None:
            time.sleep(5)
            if process.poll() is None:print(f'GPU 0: screening channel severity, elapsed {time.monotonic()-started:.0f}s.',flush=True)
        if process.returncode:raise RuntimeError(f'Noise screening failed ({process.returncode}).')


if __name__=='__main__':main()
