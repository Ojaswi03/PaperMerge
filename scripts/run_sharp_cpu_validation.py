#!/usr/bin/env python3
"""Explicit CPU warm-up/attack study; never an automatic GPU fallback.

Used alongside GPU tests when longer MNIST warm-up is needed to distinguish
an attack on an untrained model from the professor's delayed attack setting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(2025,2026,2027),default=2025)
    parser.add_argument('--nodes',type=int,default=100)
    parser.add_argument('--rounds',type=int,default=25)
    parser.add_argument('--attack-start',type=int,default=20)
    parser.add_argument('--sigma',type=float,default=.01)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output=args.output.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/papermerge-mpl')
    import numpy as np
    import tensorflow as tf
    from basil_core.sharp_study import configuration
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_data import training_data,prepare_partition
    from basil_core.sharp_protocol import run,save_state,vector_hash
    from scripts.run_sharp_study import fingerprint,data_contract,write_json,report
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    configs=[configuration('mnist','dirichlet',args.seed,c,args.sigma,nodes=args.nodes,rounds=args.rounds,
        attack_start=args.attack_start,model='resnet20') for c in ('attack_noise_defense_rla','attack_noise_defense_rla_cart')]
    if args.output.exists():raise FileExistsError('Use a fresh explicitly CPU-only validation output.')
    args.output.mkdir(parents=True)
    source=fingerprint();data,partitions=data_contract(configs)
    program={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output/'plan.json',dict(configs=configs,source=source,data=data,partitions=partitions,
        executionProgram=program,executionMode='EXPLICIT_CPU_VALIDATION_NOT_GPU_FALLBACK'))
    x,y=training_data('mnist');stopped=False;campaign_status='RUNNING'
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        for cfg in configs:
            worker=SharpWorker(cfg,'CPU');chunks,heldout,audit=prepare_partition(y,cfg)
            folder=args.output/cfg['runId'];folder.mkdir()
            contract=dict(config=cfg,source=source,data=data['mnist'],partition=audit,
                runtime=dict(device='/CPU:0',tensorflow=tf.__version__,numpy=np.__version__,host=socket.gethostname()),
                initialHash=vector_hash(worker.export()))
            metadata=dict(contract,status='RUNNING',physicalGpuId=None,completedActivations=0,completedRounds=0,
                totalActivations=cfg['nodes']*cfg['rounds'],executionMode='EXPLICIT_CPU_VALIDATION')
            def checkpoint(state):save_state(folder/'checkpoint.npz',state,contract)
            def activation(record,state):
                metadata.update(completedActivations=state['cursor'],elapsedSeconds=state['elapsedSeconds'])
                if state['cursor']%10==0:write_json(folder/'run.json',metadata)
            def round_done(record,state):
                metadata.update(completedRounds=len(state['roundMetrics']),meanAccuracy=record['meanAccuracy'])
                write_json(folder/'run.json',metadata);write_json(folder/'round_metrics.json',state['roundMetrics'])
                print(f'seed {args.seed} {cfg["condition"]} round {record["round"]+1}/{cfg["rounds"]}: {record["meanAccuracy"]:.2%}',flush=True)
            write_json(folder/'run.json',metadata)
            write_json(args.output/'campaign.json',dict(status='RUNNING',total=len(configs),profile='explicit_cpu'))
            try:
                state=run(cfg,worker,x,y,chunks,x[heldout],y[heldout],on_activation=activation,
                    on_round=round_done,checkpoint=checkpoint,should_stop=lambda:stopped)
                checkpoint(state);write_json(folder/'activation_telemetry.json',state['telemetry'])
                np.savez_compressed(folder/'metrics.npz',roundNodeAccuracy=np.asarray([r['nodeAccuracy'] for r in state['roundMetrics']]),
                    roundNodePerClassAccuracy=np.asarray([r['perClass'] for r in state['roundMetrics']]))
                metadata.update(status='COMPLETED',elapsedSeconds=state['elapsedSeconds'])
            except BaseException as error:
                metadata.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error))
                raise
            finally:write_json(folder/'run.json',metadata)
        report(args);campaign_status='COMPLETED'
    except BaseException as error:
        campaign_status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED'
        raise
    finally:write_json(args.output/'campaign.json',dict(status=campaign_status,total=len(configs),profile='explicit_cpu'))


if __name__=='__main__':main()
