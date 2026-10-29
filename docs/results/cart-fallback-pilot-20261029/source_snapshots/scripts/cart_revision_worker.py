#!/usr/bin/env python3
"""Source-frozen admission worker with transactional checkpoint recovery."""
import argparse
import hashlib
from pathlib import Path
import signal
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def compact_record(record):
    """Keep complete selected-model evidence, scalar scores for other candidates.

    The optimizer never reads historical candidate vectors from telemetry.
    This reduces checkpoint JSON/I/O without changing model or reference state.
    """
    selection=record.get('selection',{});continuity=selection.get('continuity',{})
    candidates=continuity.get('candidates')
    if not candidates:return
    selected_senders=selection.get('selectedSenders',[]);actual=continuity.get('actualRoute')
    for candidate in candidates:
        selected=candidate.get('kind')==actual and candidate.get('senders')==selected_senders
        if not selected:
            candidate.pop('fitF1',None);candidate.pop('checkF1',None)
            candidate.pop('fitSkill',None);candidate.pop('checkSkill',None)
    continuity['candidateTelemetry']='Selected class vectors retained; other candidate scalar scores retained.'


def run(output,index,lease_fd):
    from scripts.cart_v2_slot_worker import adopt_lease
    adopt_lease(lease_fd)
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core.cart_revision_study import check_source,configure_worker
    from scripts.run_cart_v2_study import setup_gpu,runtime
    from basil_core.execution_profile import current_profile
    plan=read_json(output/'plan.json');check_source(plan['source']);cfg=plan['configs'][index]
    receipt=read_json(output/f'revision_gpu_{current_profile().gpu}.json')
    if receipt['status']!='PASSED' or receipt['source']!=plan['source']:raise ValueError('Missing matching revision GPU gate.')
    tf=setup_gpu()
    if receipt['runtime']!=runtime(tf):raise ValueError('GPU runtime differs from its checked profile.')
    import numpy as np
    import basil_core.sharp_cart_v2_protocol as protocol
    from basil_core.sharp_data import training_data,prepare_partition,digest
    from basil_core.sharp_protocol import vector_hash,save_state,load_state
    model,wrap=configure_worker(cfg,plan['variant']);x,y=training_data(cfg['dataset'])
    chunks,held,audit=prepare_partition(y,cfg)
    data=dict(imagesSha256=digest(x),labelsSha256=digest(y),samples=len(y),officialTestLoaded=False)
    if data!=plan['data'][cfg['dataset']] or audit!=plan['partitions'][cfg['runId']]:raise ValueError('Frozen TRAIN contract changed.')
    folder=output/cfg['runId'];folder.mkdir(exist_ok=True);checkpoint=folder/'checkpoint.npz'
    contract=dict(config=cfg,methodRevision=plan['revision'],source=plan['source'],runtime=runtime(tf),
        initialHash=vector_hash(model.export()),data=data,partition=audit)
    state=load_state(checkpoint,contract,model.dimension) if checkpoint.exists() else None
    row=dict(status='RUNNING',config=cfg,methodRevision=plan['revision'],source=plan['source'],contract=contract,
        runtime=contract['runtime'],initialHash=contract['initialHash'],data=data,partition=audit,
        physicalGpuId=current_profile().gpu,totalActivations=cfg['nodes']*cfg['rounds'],
        runtimeAdapter=dict(name=plan['revision']))
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def progress(current,phase='TRAINING'):
        last=current['roundMetrics'][-1] if current['roundMetrics'] else {}
        row.update(phase=phase,completedActivations=current['cursor'],completedRounds=len(current['roundMetrics']),
            elapsedSeconds=current['elapsedSeconds'],meanAccuracy=last.get('meanAccuracy'),
            worstNodeAccuracy=last.get('worstNodeAccuracy'))
        write_json(folder/'run.json',row)
    import basil_core.sharp_cart_v2 as cart_module
    original_transport=cart_module.transport
    if cfg['cart'] and plan['variant'] in ('prior_neutral_class','prior_neutral_macro'):
        from basil_core.cart_prior_neutral_commit import transport
        cart_module.transport=transport(original_transport)
    if cfg['cart'] and plan['variant'] in ('transactional_class','transactional_macro'):
        from basil_core.cart_quality_commit_candidate import transport
        cart_module.transport=transport(original_transport)
    def activation(record,current):
        compact_record(record)
        if current['cursor']%10==0:
            phase='EVALUATING_ROUND' if current['cursor']%cfg['nodes']==0 else 'TRAINING'
            progress(current,phase)
    original=protocol.select_input
    if wrap:protocol.select_input=wrap(original)
    try:
        if state:progress(state)
        else:write_json(folder/'run.json',row)
        result=protocol.run(cfg,model,x,y,chunks,x[held],y[held],state=state,
            should_stop=lambda:stopped,checkpoint=lambda s:save_state(checkpoint,s,contract),
            on_activation=activation,
            on_round=lambda _,s:progress(s,'SAVING_CHECKPOINT'))
        progress(result,'FINALIZING_METRICS')
        arrays=dict(roundNodeAccuracy=np.asarray([m['nodeAccuracy'] for m in result['roundMetrics']]),
            roundNodePerClassAccuracy=np.asarray([m['perClass'] for m in result['roundMetrics']]))
        temporary=folder/'metrics.tmp.npz';np.savez_compressed(temporary,**arrays);temporary.replace(folder/'metrics.npz')
        row.update(status='COMPLETED',optimizerSteps=sum(r['training']['optimizerSteps'] for r in result['telemetry']))
        progress(result,'COMPLETED')
    except BaseException as error:
        row.update(status='INTERRUPTED' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'FAILED',
            error=f'{type(error).__name__}: {error}')
        write_json(folder/'run.json',row);raise
    finally:
        protocol.select_input=original;cart_module.transport=original_transport


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=int,required=True);p.add_argument('--lease-fd',type=int,required=True)
    args=p.parse_args();run(args.output.resolve(),args.worker,args.lease_fd)
