#!/usr/bin/env python3
"""Check direct transfer and five ordinary SGD steps on the actual device."""
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(root,gpu,lease_fd,device='GPU'):
    import numpy as np
    from basil_core import fedsharp_plain_clean as study
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core.sharp_protocol import initial_state,vector_hash
    from basil_core.sharp_worker import SharpWorker
    from basil_core.research_protocol import keyed_rng
    from scripts.run_cart_v2_study import setup_gpu,runtime
    from scripts.cart_v2_slot_worker import adopt_lease
    if device=='GPU':adopt_lease(lease_fd);tf=setup_gpu()
    else:
        import tensorflow as tf
    plan=read_json(root/'plan.json');study.check_source(plan['source']);checks=[];seen=set()
    for view in plan['configs']:
        key=(view['dataset'],view['partition'])
        if key in seen:continue
        seen.add(key);cfg=study.effective_config(view)
        worker=study.configure_worker(view,device);reference=SharpWorker(cfg,device)
        flat=worker.export();np.testing.assert_array_equal(flat,reference.export())
        rng=np.random.default_rng(743);shape=(28,28,1) if cfg['dataset']=='mnist' else (32,32,3)
        x=rng.random((cfg['batchSize'],*shape),dtype=np.float32)
        y=np.arange(len(x),dtype=np.int32)%cfg['classes'];batch=np.arange(len(y))
        state=initial_state(cfg,flat)
        initial,evidence=study.select_input(cfg,state,worker,x,y,0,0,batch)
        np.testing.assert_array_equal(initial,flat);assert evidence['bootstrap']
        state['senderRounds'][:]=0;state['cursor']=cfg['nodes']
        state['outbound']+=rng.normal(0,.001,state['outbound'].shape).astype(np.float32)
        incoming,evidence=study.select_input(cfg,state,worker,x,y,0,1,batch)
        expected=state['outbound'][-1].copy()
        np.testing.assert_array_equal(incoming,expected)
        assert evidence['selectedSenders']==[cfg['nodes']-1] and not evidence['ranking'] and not evidence['screenApplied']
        proposal,training=worker.train(incoming,x,y,1,0,0.,batch)
        reference.load(expected)
        @tf.function(reduce_retracing=True)
        def plain_step(images,labels,eta):
            with tf.GradientTape() as tape:
                loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,logits=reference.model(images,training=True)))
            gradients=tape.gradient(loss,reference.model.trainable_variables)
            for variable,gradient in zip(reference.model.trainable_variables,gradients):
                variable.assign_sub(eta*gradient)
        eta=tf.constant(cfg['learningRate']/(1+cfg['learningRate']),tf.float32)
        for step in range(cfg['localSteps']):
            indices=batch if step==0 else keyed_rng(cfg['seed'],'sharp_batch',1,0,step).choice(
                len(y),min(cfg['batchSize'],len(y)),replace=False)
            plain_step(tf.convert_to_tensor(x[indices]),tf.convert_to_tensor(y[indices],tf.int32),eta)
        np.testing.assert_array_equal(proposal,reference.export())
        assert training['optimizerSteps']==5 and training['proximalMu']==training['rlaCoefficient']==0
        assert all(t['proximalPenalty']==t['gradientSquared']==0 and t['objective']==t['ce'] for t in training['steps'])
        checks.append(dict(dataset=cfg['dataset'],partition=cfg['partition'],mu=0.,
            exactPredecessorTransfer=True,exactFiveStepOrdinarySGD=True,optimizerSteps=5,
            screening=False,ranking=False,averaging=False,noiseCoefficient=0.,cart=False,
            initialModelHash=vector_hash(flat)))
    receipt=dict(status='PASSED',source=plan['source'],
        planSha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),runtime=runtime(tf),checks=checks,
        scope='Synthetic device-local direct transfer and independently computed ordinary SGD; no accuracy claim.')
    if device=='CPU':receipt['runtime']['device']='/CPU:0'
    write_json(root/f'revision_gpu_{gpu}.json',receipt)
    print(f'GPU {gpu}: plain clean predecessor/ordinary-SGD gates PASSED.',flush=True)
    return receipt
