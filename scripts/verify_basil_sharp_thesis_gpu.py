#!/usr/bin/env python3
"""Verify genuine BASIL selection and matched SHARP/CART before HPC training."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(root,gpu,lease_fd):
    import numpy as np
    from basil_core import basil_sharp_thesis as study
    from basil_core.cart_campaign_io import read_json,write_json
    from basil_core import cart_comparison_study as baseline
    from scripts.verify_cart_comparison_gpu import verify as original_verify
    from scripts.run_cart_adaptive import sha
    plan=read_json(root/'plan.json');study.check_source(plan['source'])
    gate=root/'baseline_gate';gate.mkdir(exist_ok=True)
    configs=[]
    for dataset in dict.fromkeys(c['dataset'] for c in plan['configs']):
        args=SimpleNamespace(noise_levels=1,noise_values=[.01],share_fraction=.05,acds_batches=5,
            arms=['prior_fallback'],environments=['joint'],datasets=[dataset],partitions=['iid'],
            seeds=[2025],nodes=100,rounds=2,local_steps=5,model='paper',alpha=.5,attack_start=1,
            train_per_class=None,validation_per_class=None)
        configs.extend(baseline.selected(args)[1])
    write_json(gate/'plan.json',dict(configs=configs,source=baseline.source()))
    original_verify(gate,gpu,lease_fd)
    receipt=read_json(gate/f'revision_gpu_{gpu}.json');checks=receipt['checks']
    from basil_core.sharp_protocol import initial_state,select_input as original_select
    seen=set()
    for view in plan['configs']:
        case=view['comparisonCase'];key=(view['dataset'],case['family'],case['environment'],case['addon'])
        if key in seen:continue
        seen.add(key);cfg=study.effective_config(view);worker=study.configure_worker(view,'GPU')
        shape=(28,28,1) if cfg['dataset']=='mnist' else (32,32,3)
        x=np.random.default_rng(311).uniform(size=(cfg['batchSize'],*shape)).astype(np.float32)
        y=np.arange(len(x),dtype=np.int32)%cfg['classes'];batch=np.arange(len(y))
        flat=worker.export();state=initial_state(cfg,flat);state['senderRounds'][:]=0
        state['outbound']+=np.random.default_rng(41).normal(0,.001,state['outbound'].shape).astype(np.float32)
        state['cursor']=cfg['nodes']
        if case['family']=='basil':
            expected,er=original_select(cfg,state,worker,x,y,0,1,batch)
            actual,ar=study.basil_select(cfg,state,worker,x,y,0,1,batch)
            np.testing.assert_array_equal(actual,expected)
            assert ar['selectedSenders']==er['selectedSenders'] and ar['selectedCount']==1 and not ar['screenApplied']
        import basil_core.sharp_cart_v2_protocol as protocol
        import basil_core.sharp_cart_v2 as repair
        with study.intervention(view,worker) as actual_cfg:
            incoming,row=protocol.select_input(actual_cfg,state,worker,x,y,0,1,batch)
            coefficient=cfg['sigma']**2/(2*row['selectedCount']*(cfg['learningRate']/(1+cfg['learningRate']))) if case['family']=='sharp' else 0.
            proposal,training=worker.train(incoming,x,y,1,0,coefficient,batch)
            if cfg['cart']:proposal,_,_=repair.transport(worker,proposal,None,x,y,0,cfg['nodes'])
        assert training['optimizerSteps']==cfg['localSteps'] and np.isfinite(proposal).all()
        checks.append(dict(dataset=cfg['dataset'],family=case['family'],environment=case['environment'],
            addon=case['addon'],window=cfg['window'],optimizerSteps=training['optimizerSteps'],
            basilSingleSelectionVerified=case['family']=='basil'))
    # Force informative class recognition and exercise full CART's quality gate.
    import basil_core.cart_contribution_study as old_study
    from scripts.verify_cart_contribution_gpu import warm
    for dataset in dict.fromkeys(c['dataset'] for c in plan['configs']):
        candidates=[v for v in plan['configs'] if v['dataset']==dataset and v['comparisonCase']['addon']=='cart']
        if candidates:
            with study.installed(),patch.object(old_study,'variant',lambda v:'cart_full'), \
                    patch.object(old_study,'configure_worker',study.configure_worker),patch.object(old_study,'intervention',study.intervention):
                checks.append(warm(candidates[0]))
    import tensorflow as tf
    import resource
    receipt.update(status='PASSED',source=plan['source'],checks=checks,planSha256=sha(root/'plan.json'),
        interventionGate=True,tensorflowMemory=tf.config.experimental.get_memory_info('GPU:0'),
        peakProcessRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(root/f'revision_gpu_{gpu}.json',receipt)
    print(f'GPU {gpu}: BASIL/SHARP thesis gates PASSED.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--lease-fd',type=int,required=True)
    a=p.parse_args();verify(a.output.resolve(),a.gpu,a.lease_fd)
