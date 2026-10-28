#!/usr/bin/env python3
"""Resume local guarded CPU cases or reconcile a completed single-arm report.

Keeps the frozen training source intact, archives old manifests and records
this continuation program. Completed checkpoint/metrics bytes are preserved.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def active_writer(output):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any(Path(a).name in ('run_guarded_sharp_case.py','run_guarded_sharp_case_v2.py',
                    'recover_guarded_sharp_case.py') for a in argv):continue
            if '--output' not in argv:continue
            target=Path(argv[argv.index('--output')+1])
            if not target.is_absolute():target=Path(os.readlink(proc/'cwd'))/target
            if target.resolve()==output.resolve():return int(proc.name)
        except (OSError,ValueError,IndexError):continue
    return None


def recover(output,*,should_stop=lambda:False):
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/papermerge-mpl')
    import numpy as np
    import tensorflow as tf
    from basil_core.sharp_guarded_protocol import check_source,validate_config,run
    from basil_core.sharp_protocol import load_state,save_state,vector_hash
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_data import training_data,prepare_partition,digest
    from scripts.run_sharp_study import write_json
    output=Path(output).resolve()
    if active_writer(output) is not None:raise ValueError('An existing CPU worker still owns this output.')
    lock=(output/'cpu_recovery.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        plan=json.loads((output/'plan.json').read_text());check_source(plan['source'])
        if len(plan['configs'])!=1:raise ValueError('This tool is only for single-case CPU development outputs.')
        cfg=plan['configs'][0];validate_config(cfg);folder=output/cfg['runId']
        metadata=json.loads((folder/'run.json').read_text())
        for name,value in plan.get('executionProgram',{}).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value:raise ValueError('A frozen executor changed.')
        if metadata['runtime']['device']!='/CPU:0':raise ValueError('GPU campaigns use their own supervisor recovery.')
        tf.config.set_visible_devices([],'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
        worker=SharpWorker(cfg,'CPU');x,y=training_data(cfg['dataset'])
        chunks,heldout,partition=prepare_partition(y,cfg)
        identity=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
        runtime=dict(device='/CPU:0',tensorflow=tf.__version__,numpy=np.__version__,host=socket.gethostname())
        if (metadata['config']!=cfg or metadata['source']!=plan['source'] or metadata['data']!=identity
                or plan['data'][cfg['dataset']]!=identity or metadata['partition']!=partition
                or plan['partitions'][cfg['runId']]!=partition or metadata['runtime']!=runtime
                or metadata['initialHash']!=vector_hash(worker.export())):
            raise ValueError('Recovery source/data/initialization/runtime identity mismatch.')
        contract={k:metadata[k] for k in ('config','source','data','partition','runtime','initialHash')}
        state=load_state(folder/'checkpoint.npz',contract,worker.dimension)
        archive=output/'recovery_records';archive.mkdir(exist_ok=True)
        for path in (output/'plan.json',output/'campaign.json',folder/'run.json'):
            if not path.exists():continue
            raw=path.read_bytes();destination=archive/(hashlib.sha256(raw).hexdigest()+'-'+path.name)
            if not destination.exists():destination.write_bytes(raw)
        program=str(Path(__file__).relative_to(ROOT));program_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        action=dict(program=program,sha256=program_hash,previousStatus=metadata['status'],
            previousError=metadata.get('error'),checkpointActivation=state['cursor'],
            previousLiveActivation=metadata.get('completedActivations'))
        plan.setdefault('executionProgram',{})[program]=program_hash
        plan.setdefault('continuations',[]).append(action);write_json(output/'plan.json',plan)
        metadata.setdefault('continuations',[]).append(action);metadata.pop('error',None)
        metadata.update(status='RUNNING',completedActivations=state['cursor'],
            completedRounds=len(state['roundMetrics']),ownerPid=os.getpid())
        write_json(folder/'run.json',metadata)
        write_json(output/'campaign.json',dict(status='RUNNING',total=1,profile='explicit_cpu',ownerPid=os.getpid()))
        def checkpoint(s):save_state(folder/'checkpoint.npz',s,contract)
        def activation(record,s):
            metadata.update(completedActivations=s['cursor'],elapsedSeconds=s['elapsedSeconds'])
            if s['cursor']%10==0:write_json(folder/'run.json',metadata)
        def round_done(record,s):
            metadata.update(completedRounds=len(s['roundMetrics']),meanAccuracy=record['meanAccuracy'])
            write_json(folder/'run.json',metadata);write_json(folder/'round_metrics.json',s['roundMetrics'])
            print(f'Recovered {cfg["partition"]} {cfg["condition"]} round {record["round"]+1}/{cfg["rounds"]}: {record["meanAccuracy"]:.2%}',flush=True)
        try:
            if state['cursor']<cfg['nodes']*cfg['rounds'] or len(state['roundMetrics'])<cfg['rounds']:
                state=run(cfg,worker,x,y,chunks,x[heldout],y[heldout],state=state,
                    on_activation=activation,on_round=round_done,checkpoint=checkpoint,should_stop=should_stop)
                checkpoint(state)
            accuracy=np.asarray([r['nodeAccuracy'] for r in state['roundMetrics']])
            classes=np.asarray([r['perClass'] for r in state['roundMetrics']])
            metrics=folder/'metrics.npz'
            if metrics.exists():
                with np.load(metrics,allow_pickle=False) as old:
                    if not np.array_equal(old['roundNodeAccuracy'],accuracy) or not np.array_equal(old['roundNodePerClassAccuracy'],classes):
                        raise ValueError('Existing final metrics disagree with the completed checkpoint.')
            else:np.savez_compressed(metrics,roundNodeAccuracy=accuracy,roundNodePerClassAccuracy=classes)
            if not (folder/'activation_telemetry.json').exists():write_json(folder/'activation_telemetry.json',state['telemetry'])
            metadata.update(status='COMPLETED',completedActivations=state['cursor'],completedRounds=len(state['roundMetrics']),
                elapsedSeconds=state['elapsedSeconds'],meanAccuracy=float(accuracy[-1].mean()))
            write_json(folder/'run.json',metadata)
            write_json(output/'per_arm_summary.json',dict(status='COMPLETED_SINGLE_ARM',
                method=cfg['method'],condition=cfg['condition'],seed=cfg['seed'],sigma=cfg['sigma'],
                finalMean=float(accuracy[-1].mean()),primary=float(accuracy[-10:].mean()),
                pairedComparisons='Require matching control/CART outputs; not inferred from a single arm.',officialTestUsed=False))
            write_json(output/'campaign.json',dict(status='COMPLETED',total=1,profile='explicit_cpu'))
            print('Completed checkpoint reconciled; paired reporting remains a separate action.',flush=True)
        except BaseException as error:
            status='INTERRUPTED' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'FAILED'
            metadata.update(status=status,error=str(error));write_json(folder/'run.json',metadata)
            write_json(output/'campaign.json',dict(status=status,total=1,profile='explicit_cpu'));raise
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    recover(args.output,should_stop=lambda:stopped)


if __name__=='__main__':main()
