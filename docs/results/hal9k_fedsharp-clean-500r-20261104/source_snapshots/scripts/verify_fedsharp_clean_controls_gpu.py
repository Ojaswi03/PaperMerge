#!/usr/bin/env python3
"""Device-local selection and five-step objective gates for clean controls."""
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(root,gpu,lease_fd,device='GPU'):
    import numpy as np
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core import fedsharp_clean_controls as study,basil_sharp_thesis as base
    from basil_core.sharp_worker import SharpWorker
    from basil_core.sharp_protocol import initial_state,link_vector,vector_hash
    from basil_core.sharp_guarded_protocol import select_input as reference_select
    from scripts.run_cart_v2_study import setup_gpu,runtime
    from scripts.cart_v2_slot_worker import adopt_lease
    import basil_core.sharp_cart_v2_protocol as protocol
    if device=='GPU':adopt_lease(lease_fd);tf=setup_gpu()
    else:
        import tensorflow as tf
    plan=read_json(root/'plan.json');study.check_source(plan['source']);checks=[];seen=set()
    for view in plan['configs']:
        key=(view['dataset'],view['partition'])
        if key in seen:continue
        seen.add(key);cfg=study.effective_config(view)
        assert cfg['sigma']==0 and cfg['actualAttackers']==0 and not cfg['cart']
        assert cfg['proximalMu']==(0. if cfg['partition']=='iid' else 1.)
        worker=base.configure_worker(view,device);reference=SharpWorker(cfg,device)
        flat=worker.export();np.testing.assert_array_equal(flat,reference.export())
        shape=(28,28,1) if cfg['dataset']=='mnist' else (32,32,3)
        rng=np.random.default_rng(731);x=rng.random((cfg['batchSize'],*shape),dtype=np.float32)
        y=np.arange(len(x),dtype=np.int32)%cfg['classes'];batch=np.arange(len(y))
        state=initial_state(cfg,flat);state['senderRounds'][:]=0;state['cursor']=cfg['nodes']
        state['outbound']+=rng.normal(0,.001,state['outbound'].shape).astype(np.float32)
        expected,old=reference_select(cfg,state,reference,x,y,0,1,batch)
        sender=old['selectedSenders'][0]
        best=link_vector(state['outbound'][sender],cfg['seed'],0,sender,0,0.)
        if reference.probe(expected,x,y)>reference.probe(best,x,y)+1e-12:expected=best
        with base.intervention(view,worker) as effective:
            incoming,selection=protocol.select_input(effective,state,worker,x,y,0,1,batch)
            np.testing.assert_array_equal(incoming,expected)
            proposal,training=worker.train(incoming,x,y,1,0,0.,batch)
        target,record=reference.train(expected,x,y,1,0,0.,batch)
        np.testing.assert_array_equal(proposal,target)
        assert training==record and training['optimizerSteps']==cfg['localSteps'] and training['rlaCoefficient']==0.
        assert training['steps'][0]['proximalPenalty']==0.
        if cfg['proximalMu']==0:assert all(s['proximalPenalty']==0 for s in training['steps'])
        elif cfg['localSteps']>1:assert any(s['proximalPenalty']>0 for s in training['steps'][1:])
        checks.append(dict(dataset=cfg['dataset'],partition=cfg['partition'],mu=cfg['proximalMu'],
            exactSelection=True,exactFiveStepUpdate=True,optimizerSteps=training['optimizerSteps'],
            initialModelHash=vector_hash(flat),noiseCoefficient=0.,cart=False))
    receipt=dict(status='PASSED',source=plan['source'],planSha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
        runtime=runtime(tf),checks=checks,scope='Synthetic device-local exact selection and objective equivalence; no efficacy claim.')
    if device=='CPU':receipt['runtime']['device']='/CPU:0'
    write_json(root/f'revision_gpu_{gpu}.json',receipt)
    print(f'GPU {gpu}: clean-control selection/objective gates PASSED.',flush=True)
    return receipt
