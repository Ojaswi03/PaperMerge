#!/usr/bin/env python3
"""Isolated comparison worker, including compact exportable mechanism evidence."""
import argparse
from pathlib import Path
import signal
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run(output,index,lease_fd=None,cpu=False):
    if not cpu:
        from scripts.cart_v2_slot_worker import adopt_lease
        adopt_lease(lease_fd)
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core.cart_comparison_study import check_source,configure_worker,partitions_for,intervention,effective_config
    from scripts.run_cart_v2_study import runtime
    if cpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()
    else:
        from scripts.run_cart_v2_study import setup_gpu
        tf=setup_gpu()
    import numpy as np
    from basil_core.execution_profile import current_profile
    from basil_core.sharp_data import training_data,digest
    from basil_core.sharp_protocol import save_state,load_state,vector_hash
    import basil_core.sharp_cart_v2_protocol as protocol
    from scripts.cart_revision_worker import compact_record
    from basil_core.cart_mechanism_audit import telemetry_summary
    plan=read_json(output/'plan.json');check_source(plan['source']);view=plan['configs'][index];cfg=effective_config(view)
    if not cpu:
        receipt=read_json(output/f'revision_gpu_{current_profile().gpu}.json')
        if receipt['status']!='PASSED' or receipt['source']!=plan['source'] or receipt['runtime']!=runtime(tf):
            raise ValueError('Comparison GPU verification differs from this source/runtime.')
    worker=configure_worker(view,'CPU' if cpu else 'GPU');x,y=training_data(cfg['dataset'])
    chunks,held,audit=partitions_for(view,y)
    data=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
    if data!=plan['data'][cfg['dataset']] or audit!=plan['partitions'][view['runId']]:raise ValueError('TRAIN contract changed.')
    folder=output/view['runId'];folder.mkdir(exist_ok=True);checkpoint=folder/'checkpoint.npz'
    numerical_runtime=runtime(tf)
    if cpu:numerical_runtime['device']='/CPU:0'
    contract=dict(config=cfg,comparisonCase=view['comparisonCase'],methodRevision=plan['revision'],source=plan['source'],
        runtime=numerical_runtime,initialHash=vector_hash(worker.export()),data=data,partition=audit)
    state=load_state(checkpoint,contract,worker.dimension) if checkpoint.exists() else None
    row=dict(status='RUNNING',config=view,methodRevision=plan['revision'],source=plan['source'],contract=contract,
        runtime=contract['runtime'],initialHash=contract['initialHash'],data=data,partition=audit,
        physicalGpuId=None if cpu else current_profile().gpu,totalActivations=cfg['nodes']*cfg['rounds'],
        runtimeAdapter=dict(name=plan['revision']))
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def progress(current,phase='TRAINING'):
        last=current['roundMetrics'][-1] if current['roundMetrics'] else {}
        row.update(phase=phase,completedActivations=current['cursor'],completedRounds=len(current['roundMetrics']),
            elapsedSeconds=current['elapsedSeconds'],meanAccuracy=last.get('meanAccuracy'),worstNodeAccuracy=last.get('worstNodeAccuracy'))
        write_json(folder/'run.json',row)
    def save(current):
        save_state(checkpoint,current,contract)
        write_json(folder/'mechanisms.json',dict(source=plan['source'],runId=view['runId'],
            completedActivations=current['cursor'],rounds=telemetry_summary(current['telemetry'],cfg['nodes'])))
    def activation(record,current):
        compact_record(record)
        if current['cursor']%10==0:progress(current)
    try:
        if state:progress(state)
        else:write_json(folder/'run.json',row)
        with intervention(view,worker) as effective:
            result=protocol.run(effective,worker,x,y,chunks,x[held],y[held],state=state,should_stop=lambda:stopped,
                checkpoint=save,on_activation=activation,on_round=lambda _,s:progress(s,'SAVING_CHECKPOINT'))
        save(result)
        arrays=dict(roundNodeAccuracy=np.asarray([r['nodeAccuracy'] for r in result['roundMetrics']]),
            roundNodePerClassAccuracy=np.asarray([r['perClass'] for r in result['roundMetrics']]),
            roundHonestMeanAccuracy=np.asarray([r['honestMeanAccuracy'] for r in result['roundMetrics']]))
        tmp=folder/'metrics.tmp.npz';np.savez_compressed(tmp,**arrays);tmp.replace(folder/'metrics.npz')
        row.update(status='COMPLETED',optimizerSteps=sum(r['training']['optimizerSteps'] for r in result['telemetry']))
        if row['optimizerSteps']!=cfg['nodes']*cfg['rounds']*cfg['localSteps']:raise ValueError('Optimizer-step budget differs.')
        progress(result,'COMPLETED')
    except BaseException as error:
        row.update(status='INTERRUPTED' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'FAILED',error=f'{type(error).__name__}: {error}')
        write_json(folder/'run.json',row);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=int,required=True);p.add_argument('--lease-fd',type=int);p.add_argument('--cpu',action='store_true')
    a=p.parse_args();run(a.output.resolve(),a.worker,a.lease_fd,a.cpu)
