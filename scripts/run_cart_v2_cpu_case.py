#!/usr/bin/env python3
"""Explicit CPU development executor for correctness-qualified CART v2."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from basil_core.sharp_study import configuration,CONDITION_NAMES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(2025,2026,2027),default=2025)
    parser.add_argument('--dataset',choices=('mnist','cifar10','cifar100'),default='mnist')
    parser.add_argument('--partition',choices=('iid','dirichlet'),default='iid')
    parser.add_argument('--condition',choices=CONDITION_NAMES,default='attack_defense')
    parser.add_argument('--sigma',type=float,default=0.)
    parser.add_argument('--nodes',type=int,default=100)
    parser.add_argument('--rounds',type=int,default=35)
    parser.add_argument('--model',choices=('paper','resnet20'),default='paper')
    parser.add_argument('--attack-start',type=int,default=20)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output=args.output.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/papermerge-mpl')
    import numpy as np
    import tensorflow as tf
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_data import training_data,prepare_partition
    from basil_core.sharp_protocol import save_state,vector_hash
    from basil_core.sharp_guarded_protocol import guarded_configuration
    from basil_core.sharp_cart_v2_protocol import run,configuration as cart_configuration,fingerprint
    from scripts.run_sharp_study import data_contract,write_json
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    base=configuration(args.dataset,args.partition,args.seed,args.condition,args.sigma,
        nodes=args.nodes,rounds=args.rounds,model=args.model,attack_start=args.attack_start)
    cfg=cart_configuration(guarded_configuration(base))
    if args.output.exists():raise FileExistsError('Use a fresh CPU validation output.')
    args.output.mkdir(parents=True);source=fingerprint();data,partitions=data_contract([cfg])
    program={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output/'plan.json',dict(configs=[cfg],source=source,data=data,partitions=partitions,
        executionProgram=program,executionMode='EXPLICIT_CPU_CART_V2_DEVELOPMENT_NOT_GPU_FALLBACK'))
    x,y=training_data(cfg['dataset']);chunks,heldout,audit=prepare_partition(y,cfg)
    worker=SharpWorker(cfg,'CPU');folder=args.output/cfg['runId'];folder.mkdir()
    contract=dict(config=cfg,source=source,data=data[cfg['dataset']],partition=audit,
        runtime=dict(device='/CPU:0',tensorflow=tf.__version__,numpy=np.__version__,host=socket.gethostname()),
        initialHash=vector_hash(worker.export()))
    meta=dict(contract,status='RUNNING',physicalGpuId=None,completedActivations=0,completedRounds=0,
        totalActivations=cfg['nodes']*cfg['rounds'],executionMode='EXPLICIT_CPU_CART_V2_DEVELOPMENT')
    def checkpoint(state):save_state(folder/'checkpoint.npz',state,contract)
    def activation(record,state):
        meta.update(completedActivations=state['cursor'],elapsedSeconds=state['elapsedSeconds'])
        if state['cursor']%10==0:write_json(folder/'run.json',meta)
    def round_done(record,state):
        meta.update(completedRounds=len(state['roundMetrics']),meanAccuracy=record['meanAccuracy'])
        write_json(folder/'run.json',meta);write_json(folder/'round_metrics.json',state['roundMetrics'])
        print(f'guarded seed {args.seed} {args.condition} round {record["round"]+1}/{cfg["rounds"]}: {record["meanAccuracy"]:.2%}',flush=True)
    write_json(folder/'run.json',meta);write_json(args.output/'campaign.json',dict(status='RUNNING',total=1,profile='explicit_cpu'))
    try:
        state=run(cfg,worker,x,y,chunks,x[heldout],y[heldout],on_activation=activation,on_round=round_done,checkpoint=checkpoint)
        checkpoint(state);write_json(folder/'activation_telemetry.json',state['telemetry'])
        np.savez_compressed(folder/'metrics.npz',roundNodeAccuracy=np.asarray([r['nodeAccuracy'] for r in state['roundMetrics']]),
            roundNodePerClassAccuracy=np.asarray([r['perClass'] for r in state['roundMetrics']]))
        meta.update(status='COMPLETED',elapsedSeconds=state['elapsedSeconds'])
        write_json(folder/'run.json',meta)
        write_json(args.output/'per_arm_summary.json',dict(status='COMPLETED_SINGLE_ARM',
            method=cfg['method'],condition=cfg['condition'],seed=cfg['seed'],sigma=cfg['sigma'],
            finalMean=float(np.asarray([r['nodeAccuracy'] for r in state['roundMetrics']])[-1].mean()),
            pairedComparisons='Use matching control and CART outputs for a paired report.',officialTestUsed=False))
        write_json(args.output/'campaign.json',dict(status='COMPLETED',total=1,profile='explicit_cpu'))
    except BaseException as error:
        meta.update(status='FAILED',error=str(error));write_json(folder/'run.json',meta)
        write_json(args.output/'campaign.json',dict(status='FAILED',total=1,profile='explicit_cpu'))
        raise


if __name__=='__main__':main()
